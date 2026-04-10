
#include "gexbrownian.h"
#include "gexmatrix.h"
#include "gexmisc.h"
#include "gexmodel.h"
#include "gexparser.h"
#include "gexpca.h"
#include "gexphylofilter.h"

#include <phast/trees.h>
#include <phast/matrix.h>
#include <phast/misc.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


/* Print command line usage information to stderr. */
static void usage(const char *progname) {
    fprintf(stderr,
        "Usage: %s "
        "--trees <trees.nex> "
        "--outprefix <prefix> "
        "[--tree-total-time T] "
        "[--n-genes N] "
        "[--desired-tip-var V] "
        "[--use-n-trees N] "
        "[--dim K] "
        "[--sigma2-obs S] "
        "[--include-factorization ]"
        "[--identity-cov] "
        "[--verbose]\n",
        progname);
}

/* Main program entry point for gexLineage. */
int main(int argc, char *argv[]) {
    /* Data structures to store user inputs and other default parameters */
    const char *trees_file = NULL;  /* Path to input NEXUS file containing trees */
    const char *outprefix = NULL;   /* Prefix for all output files */
    double tree_total_time = -1.0;  /* If positive, rescale all trees uniformly to have this total height. */
    int n_genes = 100; /* Number of genes to simulate */
    double desired_tip_var = 5.0; /* Desired variance for tip nodes */
    int use_n_trees = -1;  /* -1: average covariance, 0: all trees, >0: first N trees */
    int k = 5; /* Number of latent factors to simulate */
    double sigma2_obs = 1.0; /* Variance of observation noise */
    int expr_only = 1; /* If nonzero, only output the simulated expression matrix. */
    int identity_cov = 0; /* If nonzero, use identity covariance (null model) instead of tree-based covariance for simulations. */
    int verbose = 0;    /* If nonzero, print additional progress messages during the run. */

    /* Data structures for calculations later */
    TreeNode **trees = NULL;    /* Array of tree pointers */
    Matrix **Sigmas = NULL; /* Phylogenetic covariance matrices, one per tree */
    Matrix *avg_Sigma = NULL; /* Average covariance used when --n-filter-trees=-1 */
    Matrix **use_Sigmas = NULL; /* Covariance matrices used for filtering */
    int n_trees = 0;    /* Number of input trees */
    GexMatrix *gex = scalloc(1, sizeof(GexMatrix));  /* Simulated expression matrix */
    char **gene_names = NULL; /* Gene names for the simulated expression matrix */
    int i;
    int n_cells;
    

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--trees") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            trees_file = argv[++i];
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
        else if (strcmp(argv[i], "--n-genes") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            n_genes = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--desired-tip-var") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            desired_tip_var = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--use-n-trees") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            use_n_trees = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--dim") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            k = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--sigma2-obs") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            sigma2_obs = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--include-factorization") == 0) {
            expr_only = 0;
        }
        else if (strcmp(argv[i], "--identity-cov") == 0) {
            identity_cov = 1;
        }
        else if (strcmp(argv[i], "--verbose") == 0) {
            verbose = 1;
        }
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;   /* Success since user just wants help */
        }
        else {
            fprintf(stderr, "ERROR: unknown argument: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    /* Check that all required inputs are specified */
    if (trees_file == NULL || outprefix == NULL) {
        usage(argv[0]);
        return 1;
    }

    /* Load the input trees */
    trees = read_nexus(trees_file, &n_trees);
    if (trees == NULL || n_trees < 1 || trees[0] == NULL) {
        fprintf(stderr, "ERROR: failed to load tree(s).\n");
        return 1;
    }
    printf("Loaded %d tree(s).\n", n_trees);

    if (use_n_trees < -1) {
        fprintf(stderr, "ERROR: --use-n-trees must be -1, 0, or a positive integer\n");
        return 1;
    }
    if (use_n_trees > n_trees) {
        fprintf(stderr, "ERROR: --use-n-trees (%d) cannot exceed the number of loaded trees (%d)\n",
                use_n_trees, n_trees);
        return 1;
    }

    /* Check that the input trees are ultrametric (required for cell lineage) */
    if (check_trees_ultrametric(trees, n_trees) != 0) {
        return 1;
    }

    if (verbose) {
        /* Print input/output summary for user verification */
        gex_print_io_summary(trees, n_trees, NULL);
    }

    /* Rescale the trees to a specified total height if requested */
    if (tree_total_time > 0.0) {
        uniform_rescale_trees(trees, n_trees, tree_total_time);
        printf("Rescaled tree(s) to total height %.6f.\n", tree_total_time);
    }

    /* Set the cell names in the gene expression matrix */
    List *leaf_names = tr_leaf_names(trees[0]);
    gex->cell_names = scalloc(lst_size(leaf_names), sizeof(char *));
    for (i = 0; i < lst_size(leaf_names); i++) {
        String *leaf_name = lst_get_ptr(leaf_names, i);
        gex->cell_names[i] = strdup(leaf_name->chars);
    }
    n_cells = lst_size(leaf_names);
    lst_free_strings(leaf_names);
    lst_free(leaf_names);

    /* Set the gene names in the gene expression matrix */
    gex->gene_names = scalloc(n_genes, sizeof(char *));
    gene_names = scalloc(n_genes, sizeof(char *));
    generate_names(gex->gene_names, n_genes, "gene");
    generate_names(gene_names, n_genes, "gene");   /* Keep an extra copy external to gex matrix */

    /* Decide which covariance matrix to use for simulations */
    if (!identity_cov) {
        /* Calculate the phylogenetic covariance matrix for each input tree. */
        Sigmas = scalloc(n_trees, sizeof(Matrix *));
        for (i = 0; i < n_trees; i++) {
            Sigmas[i] = covariance_from_tree(trees[i], gex->cell_names, n_cells);
            if (Sigmas[i] == NULL) {
                fprintf(stderr, "ERROR: failed to compute Brownian covariance matrix for tree %d.\n", i + 1);
                return 1;
            }
        }
        if (verbose) {
            printf("Computed phylogenetic covariance matrix for the first tree:\n");
            print_covariance_summary(Sigmas[0], gex->cell_names, n_cells);
        }

        /* Compute average covariance matrix over input trees if needed */
        if (use_n_trees == -1) {
            avg_Sigma = gex_average_tree_covariance(trees, n_trees,
                                                    gex->cell_names, n_cells);
            if (avg_Sigma == NULL) {
                fprintf(stderr, "ERROR: failed to compute average covariance for filtering.\n");
                return 1;
            }
            use_Sigmas = scalloc(1, sizeof(Matrix *));
            use_Sigmas[0] = avg_Sigma;
        }
        else {
            use_Sigmas = Sigmas;
            if (use_n_trees == 0)
                use_n_trees = n_trees;
        }

        if (use_n_trees == -1) {
            printf("Using the average covariance across all %d tree(s)...\n",
                    n_trees);
        } else if (use_n_trees == n_trees) {
            printf("Using all %d tree(s)...\n",
                    use_n_trees);
        } else {
            printf("Using the first %d tree(s)...\n",
                    use_n_trees);
        }

    } else {
        /* Use identity covariance for simulations instead of tree-based covariance if requested */
        use_Sigmas = scalloc(1, sizeof(Matrix *));
        use_Sigmas[0] = mat_new(n_cells, n_cells);
        mat_set_identity(use_Sigmas[0]);
        if (use_Sigmas[0] == NULL) {
            fprintf(stderr, "ERROR: failed to create identity covariance matrix for simulations.\n");
            return 1;
        }
        printf("Using identity covariance instead of phylogenetic covariance from trees.\n");
    }

    /* Run simulation */
    Vector *mu = vec_new(n_cells);   /* For now, assume zero mean */
    vec_zero(mu);
    gex->X = brownian_simulate(use_Sigmas, 
                                (use_n_trees == -1 ? 1 : use_n_trees), 
                                mu, 
                                (expr_only == 1 ? n_genes : k),
                                desired_tip_var);

    /* Write out the Brownian motion result */
    char expr_buf[4096];
    if (expr_only) {
        snprintf(expr_buf, sizeof(expr_buf), "%s.expr.tsv", outprefix);
        
    } else {
        snprintf(expr_buf, sizeof(expr_buf), "%s.Z.tsv", outprefix);

        /* Rename gene names to latent factors */
        generate_names(gex->gene_names, n_genes, "factor");
    }
    write_labeled_matrix_tsv(expr_buf, gex->X, gex->cell_names, gex->X->nrows,
                                        gex->gene_names, gex->X->ncols, "cell");


    if (!expr_only) {
        /* Use the Brownian result as the latent factors matrix Z (cells x factors) to 
        simulate L (factors x genes) and the reconstructed expression matrix X
        with noise for the input parameters. */
        Matrix *L = mat_new(k, n_genes);
        GexMatrix *gex_obs = scalloc(1, sizeof(GexMatrix));
        gex_obs->X = mat_new(n_cells, n_genes);
        gex_obs->cell_names = scalloc(n_cells, sizeof(char *));
        for (i = 0; i < n_cells; i++) {
            gex_obs->cell_names[i] = strdup(gex->cell_names[i]);
        }
        gex_obs->gene_names = scalloc(n_genes, sizeof(char *));
        for (i = 0; i < n_genes; i++) {
            gex_obs->gene_names[i] = strdup(gene_names[i]); /* Use extra copy in case gene names were replaced by factor names */
        }

        simulate_factorization_and_reconstruction(gex->X, gex->cell_names, n_cells, k, n_genes, sigma2_obs, L, gex_obs);

        /* Write out L */
        char l_buf[4096];
        snprintf(l_buf, sizeof(l_buf), "%s.L.tsv", outprefix);
        write_labeled_matrix_tsv(l_buf, L, gex->gene_names, k, gene_names, n_genes, "cell");

        /* Write out X */
        char x_buf[4096];
        snprintf(x_buf, sizeof(x_buf), "%s.X.tsv", outprefix);
        write_labeled_matrix_tsv(x_buf, gex_obs->X, gex_obs->cell_names, n_cells, gex_obs->gene_names, n_genes, "cell");

        /* Free memory */
        if (L != NULL)
            mat_free(L);
        if (gex_obs != NULL)
            gex_free_matrix_data(gex_obs);
    }

    printf("Wrote output to %s\n", outprefix);

    /* Free memory */
    vec_free(mu);
    gex_free_trees(trees, n_trees);
    gex_free_matrix_data(gex);
    if (Sigmas != NULL) {
        for (i = 0; i < n_trees; i++) {
            if (Sigmas[i] != NULL)
                mat_free(Sigmas[i]);
        }
        free(Sigmas);
    }
    if (use_Sigmas != NULL && use_Sigmas != Sigmas) {
        free(use_Sigmas);
    }
    if (avg_Sigma != NULL) {
        mat_free(avg_Sigma);
    }
    if (gene_names != NULL) {
        for (i = 0; i < n_genes; i++) {
            if (gene_names[i] != NULL) {
                free(gene_names[i]);
            }
        }
        free(gene_names);
    }

    return 0; /* Success */
}
