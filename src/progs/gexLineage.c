#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gex.h"
#include "pca.h"
#include "brownian.h"

static void usage(const char *progname) {
    fprintf(stderr,
        "Usage: %s --trees <trees.nex> --expr <matrix.tsv> [--pca-var-threshold V] [--moran-perms N] [--moran-fdr Q] [--moran-min-i I] [--seed S]\n",
        progname);
}

int main(int argc, char *argv[]) {
    const char *trees_file = NULL;
    const char *expr_file = NULL;
    TreeNode **trees = NULL;
    GexMatrix *gex = NULL;
    GexMatrix *gex_filtered = NULL;
    Matrix *Sigma = NULL;
    Matrix *W = NULL;
    GexMoransResult *morans = NULL;
    GexPCA *pca = NULL;
    int n_trees = 0;
    int i;
    int moran_perms = 1000;
    double moran_fdr = 0.05;
    double moran_min_i = 0.0;
    double pca_var_threshold = 0.999;
    unsigned int moran_seed = 1u;

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--trees") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            trees_file = argv[++i];
        }
        else if (strcmp(argv[i], "--expr") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            expr_file = argv[++i];
        }
        else if (strcmp(argv[i], "--pca-var-threshold") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            pca_var_threshold = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--moran-perms") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            moran_perms = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--moran-fdr") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            moran_fdr = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--moran-min-i") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            moran_min_i = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--seed") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            moran_seed = (unsigned int)strtoul(argv[++i], NULL, 10);
        }
        else if (strcmp(argv[i], "--help") == 0 ||
                 strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        }
        else {
            fprintf(stderr, "ERROR: unknown argument: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (trees_file == NULL || expr_file == NULL) {
        usage(argv[0]);
        return 1;
    }

    trees = gex_read_nexus(trees_file, &n_trees);
    if (trees == NULL) {
        fprintf(stderr, "ERROR: failed to load trees\n");
        return 1;
    }

    gex = gex_read_labeled_matrix(expr_file);
    if (gex == NULL) {
        fprintf(stderr, "ERROR: failed to load expression matrix\n");
        gex_free_trees(trees, n_trees);
        return 1;
    }

    gex_print_io_summary(trees, n_trees, gex);

    /* Filter genes based on tree correlation statistic */
    if (n_trees < 1 || trees[0] == NULL) {
        fprintf(stderr, "ERROR: no trees available for Brownian covariance\n");
        gex_free_trees(trees, n_trees);
        gex_free_matrix_data(gex);
        return 1;
    }

    /* Use first tree for initial testing.
    First test the autocorrelation filter on simulated data, then
    run the filter on the real input data. */
    if (!brownian_run_simulation_check(trees[0],
                                       gex->cell_names,
                                       gex->n_cells,
                                       1000,
                                       1000,
                                       moran_perms,
                                       moran_fdr,
                                       moran_min_i,
                                       moran_seed)) {
        fprintf(stderr, "WARNING: Brownian simulation check of correlation filter did not perfectly recover all positive/negative genes for the provided tree\n\n");
    } else {
        printf("Brownian simulation check of correlation filter successfully recovered all positive/negative genes for the provided tree\n\n");
    }

    Sigma = brownian_covariance_from_tree(trees[0],
                                          gex->cell_names,
                                          gex->n_cells);
    if (Sigma == NULL) {
        fprintf(stderr, "ERROR: failed to compute Brownian covariance matrix\n");
        gex_free_trees(trees, n_trees);
        gex_free_matrix_data(gex);
        return 1;
    }

    brownian_print_covariance_summary(Sigma,
                                      gex->cell_names,
                                      gex->n_cells);

    W = brownian_weight_matrix_from_covariance(Sigma);
    if (W == NULL) {
        fprintf(stderr, "ERROR: failed to compute Brownian weight matrix\n");
        gex_free_trees(trees, n_trees);
        gex_free_matrix_data(gex);
        mat_free(Sigma);
        return 1;
    }

    brownian_print_weight_summary(W);

    morans = gex_compute_morans_i(gex, W, moran_perms, moran_seed);
    if (morans == NULL) {
        fprintf(stderr, "ERROR: failed to compute Moran's I statistics\n");
        gex_free_trees(trees, n_trees);
        gex_free_matrix_data(gex);
        mat_free(Sigma);
        mat_free(W);
        return 1;
    }

    gex_print_morans_summary(morans, gex, moran_fdr, moran_min_i);

    gex_filtered = gex_filter_genes_by_morans_result(gex, morans,
                                                     moran_fdr, moran_min_i);
    if (gex_filtered == NULL) {
        fprintf(stderr, "ERROR: failed to filter genes by Moran's I\n");
        gex_free_trees(trees, n_trees);
        gex_free_matrix_data(gex);
        gex_free_morans_result(morans);
        mat_free(Sigma);
        mat_free(W);
        return 1;
    }

    printf("Filtered matrix has %d gene(s)\n", gex_filtered->n_genes);

    /* Run PCA on the filtered matrix and retain the smallest number of
    components needed to explain at least the requested variance. */
    pca = gex_compute_pca(gex_filtered, pca_var_threshold);
    if (pca == NULL) {
        fprintf(stderr, "ERROR: PCA failed\n");
        gex_free_trees(trees, n_trees);
        gex_free_matrix_data(gex);
        gex_free_matrix_data(gex_filtered);
        gex_free_morans_result(morans);
        mat_free(W);
        mat_free(Sigma);
        return 1;
    }
    printf("Retained %d PCA component(s) to explain at least %.2f%% of variance\n",
           pca->K, 100.0 * pca_var_threshold);

    gex_print_pca_summary(pca);

    printf("done\n");

    gex_free_trees(trees, n_trees);
    gex_free_matrix_data(gex);
    gex_free_matrix_data(gex_filtered);
    gex_free_morans_result(morans);
    gex_free_pca(pca);
    if (Sigma != NULL)
        mat_free(Sigma);
    if (W != NULL)
        mat_free(W);

    return 0;
}
