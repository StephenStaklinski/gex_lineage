#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gex.h"
#include "pca.h"
#include "brownian.h"

static int parse_filter_mode(const char *s, GexFilterMode *mode_out) {
    if (strcmp(s, "moran") == 0) {
        *mode_out = GEX_FILTER_MORAN;
        return 0;
    }
    if (strcmp(s, "lrt") == 0) {
        *mode_out = GEX_FILTER_LRT;
        return 0;
    }
    if (strcmp(s, "both") == 0) {
        *mode_out = GEX_FILTER_BOTH;
        return 0;
    }
    return -1;
}

static int parse_lrt_null_mode(const char *s, GexLRTNullMode *mode_out) {
    if (strcmp(s, "montecarlo") == 0) {
        *mode_out = GEX_LRT_NULL_MONTECARLO;
        return 0;
    }
    if (strcmp(s, "chi2") == 0) {
        *mode_out = GEX_LRT_NULL_CHI2;
        return 0;
    }
    return -1;
}

static void usage(const char *progname) {
    fprintf(stderr,
        "Usage: %s --trees <trees.nex> --expr <matrix.tsv> --outprefix <prefix> [--tree-total-time T] [--filter-test moran|lrt|both] [--lrt-null montecarlo|chi2] [--pca-var-threshold V] [--moran-perms N] [--moran-fdr Q] [--moran-min-i I] [--seed S]\n",
        progname);
}

int main(int argc, char *argv[]) {
    const char *trees_file = NULL;
    const char *expr_file = NULL;
    const char *outprefix = NULL;
    TreeNode **trees = NULL;
    GexMatrix *gex = NULL;
    GexMatrix *gex_filtered = NULL;
    Matrix *Sigma = NULL;
    Matrix *W = NULL;
    GexMoransResult *morans = NULL;
    GexLRTResult *lrt = NULL;
    GexPCA *pca = NULL;
    GexFilterMode filter_mode = GEX_FILTER_LRT;
    GexLRTNullMode lrt_null_mode = GEX_LRT_NULL_CHI2;
    int n_trees = 0;
    int i;
    int moran_perms = 1000;
    double moran_fdr = 0.05;
    double moran_min_i = 0.0;
    double pca_var_threshold = 0.99;
    double tree_total_time = -1.0;
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
        else if (strcmp(argv[i], "--outprefix") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            outprefix = argv[++i];
        }
        else if (strcmp(argv[i], "--tree-total-time") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            tree_total_time = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--filter-test") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            if (parse_filter_mode(argv[++i], &filter_mode) != 0) {
                fprintf(stderr, "ERROR: --filter-test must be one of moran, lrt, both\n");
                return 1;
            }
        }
        else if (strcmp(argv[i], "--lrt-null") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            if (parse_lrt_null_mode(argv[++i], &lrt_null_mode) != 0) {
                fprintf(stderr, "ERROR: --lrt-null must be one of montecarlo, chi2\n");
                return 1;
            }
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

    if (trees_file == NULL || expr_file == NULL || outprefix == NULL) {
        usage(argv[0]);
        return 1;
    }

    trees = gex_read_nexus(trees_file, &n_trees);
    if (trees == NULL) {
        fprintf(stderr, "ERROR: failed to load trees\n");
        return 1;
    }
    if (gex_check_trees_ultrametric(trees, n_trees, 1e-3) != 0) {
        gex_free_trees(trees, n_trees);
        return 1;
    }

    gex = gex_read_labeled_matrix(expr_file);
    if (gex == NULL) {
        fprintf(stderr, "ERROR: failed to load expression matrix\n");
        gex_free_trees(trees, n_trees);
        return 1;
    }

    gex_print_io_summary(trees, n_trees, gex);
    if (gex_reconcile_tree_and_expression(trees, n_trees, &gex) != 0) {
        fprintf(stderr, "ERROR: failed to reconcile tree tips and expression cell names\n");
        gex_free_trees(trees, n_trees);
        gex_free_matrix_data(gex);
        return 1;
    }
    printf("After reconciliation: %d shared cell/tip name(s)\n\n", gex->n_cells);

    if (tree_total_time > 0.0) {
        if (gex_rescale_trees_total_height(trees, n_trees, tree_total_time) != 0) {
            gex_free_trees(trees, n_trees);
            return 1;
        }
        printf("Rescaled all tree(s) to total height %.6f\n\n", tree_total_time);
    }

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
                                       filter_mode,
                                       lrt_null_mode,
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

    if (filter_mode == GEX_FILTER_MORAN || filter_mode == GEX_FILTER_BOTH) {
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
        {
            char corr_path[4096];
            snprintf(corr_path, sizeof(corr_path), "%s.correlation.moran.tsv", outprefix);
            if (gex_write_morans_tsv(corr_path, morans, gex, moran_fdr, moran_min_i) != 0) {
                fprintf(stderr, "ERROR: failed to write Moran correlation results to %s\n",
                        corr_path);
                gex_free_trees(trees, n_trees);
                gex_free_matrix_data(gex);
                gex_free_morans_result(morans);
                mat_free(Sigma);
                mat_free(W);
                return 1;
            }
            printf("Wrote Moran correlation results to %s\n", corr_path);
        }
    }

    if (filter_mode == GEX_FILTER_LRT || filter_mode == GEX_FILTER_BOTH) {
        lrt = gex_compute_brownian_lrt(gex, Sigma, lrt_null_mode, moran_perms, moran_seed);
        if (lrt == NULL) {
            fprintf(stderr, "ERROR: failed to compute Brownian LRT statistics\n");
            gex_free_trees(trees, n_trees);
            gex_free_matrix_data(gex);
            gex_free_morans_result(morans);
            mat_free(Sigma);
            mat_free(W);
            return 1;
        }
        gex_print_lrt_summary(lrt, gex, moran_fdr);
        {
            char lrt_path[4096];
            snprintf(lrt_path, sizeof(lrt_path), "%s.correlation.lrt.tsv", outprefix);
            if (gex_write_lrt_tsv(lrt_path, lrt, gex, moran_fdr) != 0) {
                fprintf(stderr, "ERROR: failed to write LRT correlation results to %s\n",
                        lrt_path);
                gex_free_trees(trees, n_trees);
                gex_free_matrix_data(gex);
                gex_free_morans_result(morans);
                gex_free_lrt_result(lrt);
                mat_free(Sigma);
                mat_free(W);
                return 1;
            }
            printf("Wrote LRT correlation results to %s\n", lrt_path);
        }
    }

    gex_filtered = gex_filter_genes_by_results(gex, morans, lrt, filter_mode,
                                               moran_fdr, moran_min_i);
    if (gex_filtered == NULL) {
        fprintf(stderr, "ERROR: failed to filter genes by selected test(s)\n");
        gex_free_trees(trees, n_trees);
        gex_free_matrix_data(gex);
        gex_free_morans_result(morans);
        gex_free_lrt_result(lrt);
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
        gex_free_lrt_result(lrt);
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
    gex_free_lrt_result(lrt);
    gex_free_pca(pca);
    if (Sigma != NULL)
        mat_free(Sigma);
    if (W != NULL)
        mat_free(W);

    return 0;
}
