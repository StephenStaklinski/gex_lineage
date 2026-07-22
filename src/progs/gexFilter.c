
#include "brownian.h"
#include "external_libs.h"
#include "gexmatrix.h"
#include "misc.h"
#include "parser.h"
#include "phylofilter.h"

#include <phast/trees.h>
#include <phast/matrix.h>
#include <phast/misc.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Parse the filter mode from a string. Sets pointer to mode_out.
Returns 0 on success or -1 on failure. */
static int parse_filter_mode(const char *s, GexFilterMode *mode_out) {
    if (strcmp(s, "moran") == 0) {
        *mode_out = GEX_FILTER_MORAN;
        return 0;
    }
    if (strcmp(s, "lrt") == 0) {
        *mode_out = GEX_FILTER_LRT;
        return 0;
    }
    return -1;
}

static int parse_lrt_alt_mode(const char *s, GexLRTAltMode *mode_out) {
    if (strcmp(s, "full") == 0) {
        *mode_out = GEX_LRT_ALT_FULL;
        return 0;
    }
    if (strcmp(s, "lambda") == 0) {
        *mode_out = GEX_LRT_ALT_LAMBDA;
        return 0;
    }
    return -1;
}

/* Print command line usage information to stderr. */
static void usage(const char *progname) {
    fprintf(stderr,
        "Usage: %s --trees <trees.nex> --expr <matrix.tsv> --outprefix <prefix> "
        "[--filter-test lrt|moran] "
        "[--lrt-alt lambda|full] [--remove-ribo-mito-genes] "
        "[--no-preprocess] [--seed S]\n",
        progname);
}

/* Filter a gene-expression matrix for phylogenetic signal. */
int main(int argc, char *argv[]) {
    /* Data structures to store user inputs and other default parameters */
    const char *trees_file = NULL;  /* Path to input NEXUS file containing trees */
    const char *expr_file = NULL;   /* Path to input tab-delimited file containing expression matrix */
    const char *outprefix = NULL;   /* Prefix for all output files */
    GexFilterMode filter_mode = GEX_FILTER_LRT;   /* Test used to retain phylogenetically informative genes. */
    double max_q = 0.05;  /* False discovery rate for multiple testing correction */
    int n_perms = 1000; /* Number of permutations for monte-carlo based permutation tests */
    int preprocess = 1; /* If nonzero, normalize, log-transform, and center before filtering. */
    int remove_ribo_mito_genes = 0; /* If nonzero, remove ribosomal and mitochondrial genes before filtering. */
    GexLRTAltMode lrt_alt_mode = GEX_LRT_ALT_LAMBDA;   /* Which alternative model to use for the Brownian LRT */

    /* Data structures for calculations later */
    TreeNode **trees = NULL;    /* Array of tree pointers */
    GexMatrix *gex = NULL;  /* Original expression matrix */
    GexMatrix *gex_filtered = NULL; /* Filtered expression matrix */
    Matrix *filter_avg_Sigma = NULL; /* Average covariance used for filtering. */
    Matrix *filter_Sigmas[1] = {NULL}; /* One-element interface for the average covariance. */
    MoranResult *morans = NULL; /* Results from Moran's I calculation */
    GexLRTResult *lrt = NULL;   /* Results from Brownian LRT calculation */
    int n_trees = 0;    /* Number of input trees */
    int i;  /* Pre-allocated generic loop index variable */
    set_seed(-1); /* Random seed, for now */
    set_num_threads(1); /* Keep OpenMP and BLAS/LAPACK single-threaded for now. */

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
        else if (strcmp(argv[i], "--filter-test") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            if (parse_filter_mode(argv[++i], &filter_mode) != 0) {
                fprintf(stderr, "ERROR: --filter-test must be one of moran or lrt\n");
                return 1;
            }
        }
        else if (strcmp(argv[i], "--lrt-alt") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            if (parse_lrt_alt_mode(argv[++i], &lrt_alt_mode) != 0) {
                fprintf(stderr, "ERROR: --lrt-alt must be one of full, lambda\n");
                return 1;
            }
        }
        else if (strcmp(argv[i], "--no-preprocess") == 0) {
            preprocess = 0;
        }
        else if (strcmp(argv[i], "--remove-ribo-mito-genes") == 0) {
            remove_ribo_mito_genes = 1;
        }
        else if (strcmp(argv[i], "--seed") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            set_seed((unsigned int)atoi(argv[++i]));
        }
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;   /* Success since user just wants cli help */
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

    /* Load the input trees */
    trees = read_nexus(trees_file, &n_trees);
    if (trees == NULL || n_trees < 1 || trees[0] == NULL) {
        fprintf(stderr, "ERROR: failed to load tree(s).\n");
        return 1;
    }
    printf("Loaded %d tree(s).\n", n_trees);
    /* Check that the input trees are ultrametric (required for cell lineage) */
    if (check_trees_ultrametric(trees, n_trees) != 0) {
        return 1;
    }

    /* Load the input real expression matrix */
    gex = read_gex_matrix(expr_file);
    if (gex == NULL) {
        fprintf(stderr, "ERROR: failed to load expression matrix.\n");
        return 1;
    }
    printf("Loaded matrix with %d cell(s) and %d gene(s).\n", gex->X->nrows, gex->X->ncols);

    if (remove_ribo_mito_genes) {
        int n_removed = gex_remove_ribo_mito_genes(gex);
        if (n_removed < 0) {
            fprintf(stderr, "ERROR: ribosomal/mitochondrial gene filtering removed all genes.\n");
            return 1;
        }
        printf("Removed %d ribosomal/mitochondrial gene(s); matrix now has %d gene(s).\n",
               n_removed, gex->X->ncols);
    }

    /* Reconcile tree tips and expression cell names to the intersection of both sets
    if they do not perfectly match. */
    if (gex_reconcile_tree_and_expression(trees, n_trees, &gex) != 0) {
        fprintf(stderr, "ERROR: failed to reconcile tree tips and expression cell names.\n");
        return 1;
    }

    /* Use the same unit-height time scale for every filtering run. */
    uniform_rescale_trees(trees, n_trees, 1.0);
    printf("Rescaled tree(s) to total height 1.0.\n");

    /* Filtering always uses one covariance matrix averaged across all trees. */
    filter_avg_Sigma = gex_average_tree_covariance(trees, n_trees,
                                                   gex->cell_names, gex->X->nrows);
    if (filter_avg_Sigma == NULL) {
        fprintf(stderr, "ERROR: failed to compute the average tree covariance matrix.\n");
        return 1;
    }
    filter_Sigmas[0] = filter_avg_Sigma;

    /* Pre-process the gene expression data */
    if (preprocess) {
        printf("Pre-processing the gene expression data...\n");
        /* Library size normalization per-cell */
        mat_normalize_rows(gex->X);
        /* Scale by global factor to counts per 10k */
        double scale_factor = 10000.0;
        mat_scale(gex->X, scale_factor);
        /* Log-transform the data to stabilize variance and approximate Gaussian */
        mat_log1p(gex->X);
    }

    /* Transform data into the residuals */
    mat_center_cols(gex->X);


    /* Run the phylogenetic signal filter(s). */
    {
        printf("Applying the phylogenetic signal gene filter using the covariance averaged across all %d tree(s)...\n",
               n_trees);

        /* Pagel's lambda LRT is only for <=1000 cells; Full LRT is only for <=100 cells */
        if (filter_mode == GEX_FILTER_LRT && lrt_alt_mode == GEX_LRT_ALT_LAMBDA && gex->X->nrows > 1000) {
            filter_mode = GEX_FILTER_MORAN;
            printf("WARNING: Pagel's lambda Brownian LRT is only for <=1000 cells. Switching to Moran's I filter instead.\n");
        }
        if (filter_mode == GEX_FILTER_LRT && lrt_alt_mode == GEX_LRT_ALT_FULL && gex->X->nrows > 100) {
            filter_mode = GEX_FILTER_MORAN;
            printf("WARNING: Full Brownian LRT is only for <=100 cells. Switching to Moran's I filter instead.\n");
        }

        /* Run the phylogenetic autocorrelation filter tests if requested */
        if (filter_mode == GEX_FILTER_MORAN) {
            morans = gex_compute_morans_i(gex->X, filter_Sigmas, 1);

            char corr_path[4096];
            snprintf(corr_path, sizeof(corr_path), "%s.correlation.moran.tsv", outprefix);
            write_moran_tsv(corr_path, morans, gex, max_q);
        }

        /* Run the phylogenetic LRT filter tests if requested */
        if (filter_mode == GEX_FILTER_LRT) {
            lrt = gex_compute_brownian_lrt(gex->X, filter_Sigmas, 1,
                                           n_perms,lrt_alt_mode);

            char lrt_path[4096];
            if (lrt_alt_mode == GEX_LRT_ALT_FULL) {
                snprintf(lrt_path, sizeof(lrt_path), "%s.correlation.lrt.full.tsv", outprefix);
            } else if (lrt_alt_mode == GEX_LRT_ALT_LAMBDA) {
                snprintf(lrt_path, sizeof(lrt_path), "%s.correlation.lrt.lambda.tsv", outprefix);
            }

            write_lrt_tsv(lrt_path, lrt, gex, max_q);
        }

    }

    gex_filtered = gex_filter_genes(gex, morans, lrt, filter_mode, max_q);
    printf("Filtered matrix has %d cells and %d gene(s).\n",
           gex_filtered->X->nrows, gex_filtered->X->ncols);

    {
        char filtered_path[4096];
        snprintf(filtered_path, sizeof(filtered_path), "%s.filtered.tsv", outprefix);
        write_labeled_matrix_tsv(filtered_path, gex_filtered->X,
                                 gex_filtered->cell_names, gex_filtered->X->nrows,
                                 gex_filtered->gene_names, gex_filtered->X->ncols,
                                 "cell");
        printf("Wrote modeling-ready filtered expression matrix to %s\n", filtered_path);
    }

    /* Free memory */
    gex_free_trees(trees, n_trees);
    gex_free_matrix_data(gex);
    gex_free_matrix_data(gex_filtered);
    free_moran_result(morans);
    free_lrt_result(lrt);
    if (filter_avg_Sigma != NULL) {
        mat_free(filter_avg_Sigma);
    }

    return 0; /* Success */
}
