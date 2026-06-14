
#include "brownian.h"
#include "external_libs.h"
#include "gexmatrix.h"
#include "misc.h"
#include "model.h"
#include "latentflow.h"
#include "parser.h"
#include "pca.h"
#include "phylofilter.h"

#include <phast/trees.h>
#include <phast/matrix.h>
#include <phast/misc.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

static int parse_scale_invar_constraint(const char *s, GexScaleInvarConstraint *constraint_out) {
    if (strcmp(s, "sigma2s") == 0) {
        *constraint_out = GEX_SCALE_INVAR_SIGMA2S;
        return 0;
    }
    if (strcmp(s, "Lrows") == 0) {
        *constraint_out = GEX_SCALE_INVAR_LROWS;
        return 0;
    }
    if (strcmp(s, "none") == 0) {
        *constraint_out = GEX_SCALE_INVAR_NONE;
        return 0;
    }
    return -1;
}

static int parse_pca_method(const char *s, PcaMethod *method_out) {
    if (strcmp(s, "pca") == 0) {
        *method_out = PCA_METHOD_PCA;
        return 0;
    }
    if (strcmp(s, "phylopca") == 0) {
        *method_out = PCA_METHOD_PHYLOPCA;
        return 0;
    }
    if (strcmp(s, "maxphylopca") == 0) {
        *method_out = PCA_METHOD_MAX_PHYLOPCA;
        return 0;
    }
    if (strcmp(s, "none") == 0) {
        *method_out = PCA_METHOD_NONE;
        return 0;
    }
    return -1;
}

/* Print command line usage information to stderr. */
static void usage(const char *progname) {
    fprintf(stderr,
        "Usage: %s "
        "--trees <trees.nex> "
        "--expr <matrix.tsv> "
        "--outprefix <prefix> "
        "[--tree-total-time T] "
        "[--tree-index N] "
        "[--filter-covariance average|all] "
        "[--filter-test lrt|moran] "
        "[--lrt-alt lambda|full] "
        "[--scale-invar-constraint sigma2s|Lrows|none] "
        "[--no-post-hoc-identifiability] "
        "[--L-row-norm-interval N] "
        "[--L-l1-strength S] "
        "[--F-orthogonality-strength S] "
        "[--F-correlation-strength S] "
        "[--L-correlation-strength S] "
        "[--L-loading-overlap-strength S] "
        "[--final-absorbing-factor] "
        "[--final-absorbing-L-l2-strength S] "
        "[--max-iter N] "
        "[--dim K] "
        "[--pca-method maxphylopca|phylopca|pca|none] "
        "[--pca-var-threshold V] "
        "[--n-perms N] "
        "[--max-q Q] "
        "[--moran-min-i I] "
        "[--filter-only] "
        "[--no-write-latent-flow] "
        "[--no-filter] "
        "[--no-preprocess] "
        "[--remove-ribo-mito-genes] "
        "[--varimax] "
        "[--verbose] "
        "[--verbose-log] "
        "[--seed S] "
        "[--nthreads N]\n",
        progname);
}

/* Main program entry point for gexLineage. */
int main(int argc, char *argv[]) {
    /* Data structures to store user inputs and other default parameters */
    const char *trees_file = NULL;  /* Path to input NEXUS file containing trees */
    const char *expr_file = NULL;   /* Path to input tab-delimited file containing expression matrix */
    const char *outprefix = NULL;   /* Prefix for all output files */
    GexFilterMode filter_mode = GEX_FILTER_LRT;   /* Which test(s) to use for filtering genes before modeling */
    int n_perms = 1000; /* Number of permutations for monte-carlo based permutation tests */
    int filter_average_covariance = 1;  /* If nonzero, use the average covariance for phylogenetic filtering. */
    double max_q = 0.05;  /* False discovery rate for multiple testing correction */
    PcaMethod pca_method = PCA_METHOD_MAX_PHYLOPCA;  /* Method for performing PCA to initialize latent model fitting */
    double pca_var_threshold = 0.95;    /* Threshold of variance explained to retain PCA components up to */
    double tree_total_time = 1.0;  /* If positive, rescale all trees uniformly to have this total height. */
    int tree_index = 0;    /* If positive, keep only this 1-based tree from the input NEXUS. */
    double L_l1_strength = 0.1;  /* L1 regularization strength for loadings; 0 disables the penalty. */
    double F_orthogonality_strength = 0.0;  /* Strength of F-column orthogonality penalty; 0 disables it. */
    double F_correlation_strength = 0.0;  /* Strength of F-column correlation penalty; 0 disables it. */
    double L_correlation_strength = 0.0;  /* Strength of L-row correlation penalty; 0 disables it. */
    double L_loading_overlap_strength = 0.0;  /* Strength of absolute L-row loading-overlap penalty; 0 disables it. */
    int final_absorbing_factor = 0; /* If nonzero, use the final factor as dense absorbing background. */
    double L_absorbing_l2_strength = 1e-4; /* L2 strength for the final absorbing factor. */
    int max_iter = 100000;  /* Maximum number of optimization iterations for latent model fitting. */
    int k = 0;  /* Number of latent factors to fit; if 0, will be determined by pca_var_threshold */
    int filter_only = 0;    /* If nonzero, stop after writing filter outputs and exit successfully. */
    int write_latent_flow = 1; /* If nonzero, write latent factors for tips and reconstructed internal nodes. */
    int no_filter = 0;  /* If nonzero, skip the filter step and use all genes for modeling. */
    int preprocess = 1; /* If nonzero, preprocess the expression data before modeling. */
    int remove_ribo_mito_genes = 0; /* If nonzero, remove ribosomal and mitochondrial genes before modeling. */
    int varimax = 0;    /* If nonzero, rotate fitted factors with varimax before writing outputs. */
    int apply_post_hoc_identifiability = 1; /* If nonzero, apply post-hoc sign and permutation identifiability fixes. */
    int verbose = 0;    /* If nonzero, print additional progress messages during the run. */
    int verbose_log = 0;    /* If nonzero, write the full optimization log. */
    int nthreads = 1;   /* Number of OpenMP threads to use */
    GexLRTAltMode lrt_alt_mode = GEX_LRT_ALT_LAMBDA;   /* Which alternative model to use for the Brownian LRT */
    GexScaleInvarConstraint scale_invar_constraint = GEX_SCALE_INVAR_SIGMA2S; /* Which constraint to use for counteracting scale invariance */

    /* Data structures for calculations later */
    TreeNode **trees = NULL;    /* Array of tree pointers */
    GexMatrix *gex = NULL;  /* Original expression matrix */
    GexMatrix *gex_filtered = NULL; /* Filtered expression matrix */
    Matrix **Sigmas = NULL; /* Phylogenetic covariance matrices, one per tree */
    Matrix *filter_avg_Sigma = NULL; /* Average covariance used for filtering and/or phyloPCA */
    Matrix **filter_Sigmas = NULL; /* Covariance matrices used for filtering */
    int n_filter_sigmas = 0; /* Number of covariance matrices used for phylogenetic filtering */
    MoranResult *morans = NULL; /* Results from Moran's I calculation */
    GexLRTResult *lrt = NULL;   /* Results from Brownian LRT calculation */
    PCA *pca = NULL; /* PCA results */
    GexLatentBrownianModel *model = NULL;   /* Fitted latent Brownian gene expression model */
    int n_trees = 0;    /* Number of input trees */
    int i;  /* Pre-allocated generic loop index variable */
    set_seed(-1); /* Random seed, for now */

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
        else if (strcmp(argv[i], "--tree-index") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            tree_index = atoi(argv[++i]);
            if (tree_index <= 0) {
                fprintf(stderr, "ERROR: --tree-index must be a positive 1-based integer.\n");
                return 1;
            }
        }
        else if (strcmp(argv[i], "--filter-covariance") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            i++;
            if (strcmp(argv[i], "average") == 0) {
                filter_average_covariance = 1;
            }
            else if (strcmp(argv[i], "all") == 0) {
                filter_average_covariance = 0;
            }
            else {
                fprintf(stderr, "ERROR: --filter-covariance must be one of average or all\n");
                return 1;
            }
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
        else if (strcmp(argv[i], "--pca-method") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            if (parse_pca_method(argv[++i], &pca_method) != 0) {
                fprintf(stderr, "ERROR: --pca-method must be one of maxphylopca, phylopca, pca, or none\n");
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
        else if (strcmp(argv[i], "--scale-invar-constraint") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            if (parse_scale_invar_constraint(argv[++i], &scale_invar_constraint) != 0) {
                fprintf(stderr, "ERROR: --scale-invar-constraint must be one of sigma2s, Lrows, or none\n");
                return 1;
            }
        }
        else if (strcmp(argv[i], "--L-l1-strength") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            L_l1_strength = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--F-orthogonality-strength") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            F_orthogonality_strength = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--F-correlation-strength") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            F_correlation_strength = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--L-correlation-strength") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            L_correlation_strength = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--L-loading-overlap-strength") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            L_loading_overlap_strength = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--final-absorbing-factor") == 0) {
            final_absorbing_factor = 1;
        }
        else if (strcmp(argv[i], "--final-absorbing-L-l2-strength") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            L_absorbing_l2_strength = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--max-iter") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            max_iter = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--dim") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            k = atoi(argv[++i]);
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
        else if (strcmp(argv[i], "--n-perms") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            n_perms = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--max-q") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            max_q = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--filter-only") == 0) {
            filter_only = 1;
        }
        else if (strcmp(argv[i], "--no-write-latent-flow") == 0) {
            write_latent_flow = 0;
        }
        else if (strcmp(argv[i], "--no-filter") == 0) {
            no_filter = 1;
        }
        else if (strcmp(argv[i], "--no-preprocess") == 0) {
            preprocess = 0;
        }
        else if (strcmp(argv[i], "--remove-ribo-mito-genes") == 0) {
            remove_ribo_mito_genes = 1;
        }
        else if (strcmp(argv[i], "--varimax") == 0) {
            varimax = 1;
        }
        else if (strcmp(argv[i], "--no-post-hoc-identifiability") == 0) {
            apply_post_hoc_identifiability = 0;
        }
        else if (strcmp(argv[i], "--verbose") == 0) {
            verbose = 1;
        }
        else if (strcmp(argv[i], "--verbose-log") == 0) {
            verbose_log = 1;
        }
        else if (strcmp(argv[i], "--seed") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            set_seed((unsigned int)atoi(argv[++i]));
        }
        else if (strcmp(argv[i], "--nthreads") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            nthreads = atoi(argv[++i]);
            if (nthreads <= 0) {
                fprintf(stderr, "ERROR: thread count must be positive.\n");
                return 1;
            }
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
    if (filter_only && no_filter) {
        fprintf(stderr, "ERROR: --no-filter cannot be used together with --filter-only.\n");
        return 1;
    }
    if (pca_method == PCA_METHOD_NONE && k == 0) {
        fprintf(stderr, "ERROR: --dim must be specified when --pca-method is none.\n");
        return 1;
    }
    if (nthreads > 0) {
        if (nthreads > 1 && !has_thread_control()) {
            fprintf(stderr, "ERROR: this gexLineage build does not support OpenMP or BLAS thread control.\n");
            return 1;
        }
        set_num_threads(nthreads, verbose);
    }

    /* Load the input trees */
    trees = read_nexus(trees_file, &n_trees);
    if (trees == NULL || n_trees < 1 || trees[0] == NULL) {
        fprintf(stderr, "ERROR: failed to load tree(s).\n");
        return 1;
    }
    printf("Loaded %d tree(s).\n", n_trees);
    if (tree_index > 0) {
        int loaded_trees = n_trees;
        if (keep_one_tree(trees, &n_trees, tree_index) != 0) {
            fprintf(stderr, "ERROR: --tree-index %d is out of range for %d loaded tree(s).\n",
                    tree_index, loaded_trees);
            return 1;
        }
        printf("Keeping only tree %d from the input NEXUS.\n", tree_index);
    }

    if (L_l1_strength < 0.0) {
        fprintf(stderr, "ERROR: --L-l1-strength must be nonnegative (0 disables L1 regularization)\n");
        return 1;
    }
    if (F_orthogonality_strength < 0.0) {
        fprintf(stderr, "ERROR: --F-orthogonality-strength must be nonnegative (0 disables F orthogonality regularization)\n");
        return 1;
    }
    if (F_correlation_strength < 0.0) {
        fprintf(stderr, "ERROR: --F-correlation-strength must be nonnegative (0 disables F correlation regularization)\n");
        return 1;
    }
    if (L_correlation_strength < 0.0) {
        fprintf(stderr, "ERROR: --L-correlation-strength must be nonnegative (0 disables L correlation regularization)\n");
        return 1;
    }
    if (L_loading_overlap_strength < 0.0) {
        fprintf(stderr, "ERROR: --L-loading-overlap-strength must be nonnegative (0 disables L loading-overlap regularization)\n");
        return 1;
    }
    if (L_absorbing_l2_strength < 0.0) {
        fprintf(stderr, "ERROR: --final-absorbing-L-l2-strength must be nonnegative.\n");
        return 1;
    }
    if (max_iter <= 0) {
        fprintf(stderr, "ERROR: --max-iter must be positive.\n");
        return 1;
    }

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

    if (verbose) {
        /* Print input/output summary for user verification */
        gex_print_io_summary(trees, n_trees, gex);
    }

    /* Reconcile tree tips and expression cell names to the intersection of both sets
    if they do not perfectly match. */
    if (gex_reconcile_tree_and_expression(trees, n_trees, &gex) != 0) {
        fprintf(stderr, "ERROR: failed to reconcile tree tips and expression cell names.\n");
        return 1;
    }

    /* Rescale the trees to a specified total height if requested */
    if (tree_total_time > 0.0) {
        uniform_rescale_trees(trees, n_trees, tree_total_time);
        printf("Rescaled tree(s) to total height %.6f.\n", tree_total_time);
    }

    /* Calculate the phylogenetic covariance matrix for each input tree. */
    Sigmas = scalloc(n_trees, sizeof(Matrix *));
    for (i = 0; i < n_trees; i++) {
        Sigmas[i] = covariance_from_tree(trees[i], gex->cell_names, gex->X->nrows);
        if (Sigmas[i] == NULL) {
            fprintf(stderr, "ERROR: failed to compute Brownian covariance matrix for tree %d.\n", i + 1);
            return 1;
        }
    }
    if (verbose) {
        printf("Computed phylogenetic covariance matrix for the first tree:\n");
        print_covariance_summary(Sigmas[0], gex->cell_names, gex->X->nrows);
    }

    /* Compute average covariance matrix over input trees if needed */
    if (filter_average_covariance || pca_method == PCA_METHOD_PHYLOPCA || pca_method == PCA_METHOD_MAX_PHYLOPCA) {
        filter_avg_Sigma = gex_average_tree_covariance(trees, n_trees,
                                                       gex->cell_names, gex->X->nrows);
    }

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


    /* Select which covariance matrix or matrices to use for the phylogenetic signal filtering */
    if (filter_average_covariance) {
        filter_Sigmas = scalloc(1, sizeof(Matrix *));
        filter_Sigmas[0] = filter_avg_Sigma;
        n_filter_sigmas = 1;
    }
    else {
        filter_Sigmas = Sigmas;
        n_filter_sigmas = n_trees;
    }

    /* Run the phylogenetic signal filter(s). */
    if (!no_filter) {
        const char *tree_msg;
        int tree_count = n_trees;
        if (filter_average_covariance) {
            tree_msg = "the average covariance across all";
        } else {
            tree_msg = "all";
        }
        printf("Applying the phylogenetic signal gene filter(s) to the real input gene expression matrix data using %s %d tree(s)...\n",
            tree_msg, tree_count);

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
            morans = gex_compute_morans_i(gex->X, filter_Sigmas,
                                          n_filter_sigmas);

            char corr_path[4096];
            snprintf(corr_path, sizeof(corr_path), "%s.correlation.moran.tsv", outprefix);
            write_moran_tsv(corr_path, morans, gex, max_q);
        }

        /* Run the phylogenetic LRT filter tests if requested */
        if (filter_mode == GEX_FILTER_LRT) {
            lrt = gex_compute_brownian_lrt(gex->X, filter_Sigmas,
                                           n_filter_sigmas,
                                           n_perms,lrt_alt_mode);

            char lrt_path[4096];
            if (lrt_alt_mode == GEX_LRT_ALT_FULL) {
                snprintf(lrt_path, sizeof(lrt_path), "%s.correlation.lrt.full.tsv", outprefix);
            } else if (lrt_alt_mode == GEX_LRT_ALT_LAMBDA) {
                snprintf(lrt_path, sizeof(lrt_path), "%s.correlation.lrt.lambda.tsv", outprefix);
            }

            write_lrt_tsv(lrt_path, lrt, gex, max_q);
        }

        /* Stop early if only phylogenetic signal filtering is requested */
        if (filter_only) {
            return 0;
        }
    }


    if (no_filter) {
        /* Keep all genes */
        gex_filtered = gex;
        gex = NULL;
        printf("Skipping phylogenetic signal gene filtering and using all %d gene(s) for modeling.\n", gex_filtered->X->ncols);

    } else {
        /* Filter genes */
        gex_filtered = gex_filter_genes(gex, morans, lrt, filter_mode, max_q);
        printf("Filtered matrix has %d cells and %d gene(s).\n", gex_filtered->X->nrows, gex_filtered->X->ncols);
    }

    /* pPCA is only for <=1000 cells */
    if (pca_method == PCA_METHOD_PHYLOPCA &&
        (gex_filtered->X->nrows > 1000 || gex_filtered->X->ncols > 1000)) {
        pca_method = PCA_METHOD_PCA;
        printf("WARNING: pPCA is only for <=1000 cells and <=1000 genes. Switching to standard PCA instead.\n");
    }

    /* Run PCA on the filtered matrix and retain the smallest number of
    components needed to explain at least the requested variance. */
    if (pca_method == PCA_METHOD_PCA) {
        printf("Running PCA to initialize latent factors for the model...\n");
        pca = compute_pca(gex_filtered->X, k, pca_var_threshold);
    } else if (pca_method == PCA_METHOD_PHYLOPCA) {
        printf("Running pPCA to initialize latent factors for the model...\n");
        pca = compute_phylo_pca(gex_filtered->X, filter_avg_Sigma, k, pca_var_threshold);
    } else if (pca_method == PCA_METHOD_MAX_PHYLOPCA) {
        printf("Running maxPhyloPCA to initialize latent factors for the model...\n");
        pca = compute_max_phylo_pca(gex_filtered->X, filter_avg_Sigma, k, pca_var_threshold);
    }
    if (k == 0) {
        k = pca->K;
        printf("Retaining %d PCA component(s) to explain at least %.2f%% of variance.\n", k, 100.0 * pca_var_threshold);
    } else {
        printf("Retaining the top %d PCA component(s).\n", k);
    }
    write_pca_tsv(outprefix, pca, gex_filtered);

    /* Check for technical accuracy of PCA during testing */
    if (pca != NULL && pca->components != NULL) {
        char pca_gram_path[4096];
        char **pca_factor_names = scalloc(pca->K, sizeof(char *));
        Matrix *pca_components_t = mat_transpose(pca->components);
        Matrix *pca_gram = mat_new(pca->K, pca->K);

        generate_names(pca_factor_names, pca->K, "PC");
        mat_mult_lapack(pca_gram, pca->components, pca_components_t);

        snprintf(pca_gram_path, sizeof(pca_gram_path), "%s.pca.eigenvector_gram.tsv", outprefix);
        write_labeled_matrix_tsv(pca_gram_path, pca_gram,
                                 pca_factor_names, pca->K,
                                 pca_factor_names, pca->K,
                                 "PC");

        for (i = 0; i < pca->K; i++) {
            if (pca_factor_names[i] != NULL)
                free(pca_factor_names[i]);
        }
        free(pca_factor_names);
        mat_free(pca_components_t);
        mat_free(pca_gram);
    }

    /* Fit the latent Brownian model */
    printf("Fitting model to the filtered data with k=%d latent dimensions...\n", k);
    model = gex_fit_latent_brownian_model(gex_filtered, trees, n_trees,
                                          k, pca, scale_invar_constraint, L_l1_strength,
                                          final_absorbing_factor, L_absorbing_l2_strength,
                                          F_orthogonality_strength, F_correlation_strength,
                                          L_correlation_strength,
                                          L_loading_overlap_strength,
                                          apply_post_hoc_identifiability,
                                          outprefix, max_iter, verbose_log);

    if (varimax) {
        printf("Applying varimax rotation to fitted factors before writing outputs...\n");
        if (final_absorbing_factor)
            varimax_rotate_model_factors_prefix(model->L, model->F, k - 1, outprefix, 1000, 1e-6);
        else
            varimax_rotate_model_factors(model->L, model->F, outprefix, 1000, 1e-6);
        if (apply_post_hoc_identifiability)
            post_hoc_sign_identifiability(model->L, model->F);
        mat_mult_lapack(model->FL, model->F, model->L);
    }

    if (write_latent_flow) {
        printf("Writing latent flow table...\n");
        if (gex_write_latent_flow_outputs(outprefix, trees, 1, gex_filtered, model) != 0) {
            fprintf(stderr, "ERROR: failed to write latent flow outputs.\n");
            return 1;
        }
    }

    /* Write the fitted latent Brownian model parameters to files */
    char **factor_names = scalloc(k, sizeof(char *));
    generate_names(factor_names, k, "factor");
    double *sigma2_latent = scalloc(k, sizeof(double));
    for (i = 0; i < k; i++) {
        sigma2_latent[i] = exp(model->log_sigma2_latent[i]);
    }
    double sigma2_obs = exp(model->log_sigma2_obs);

    /* Reset the gex_filtered matrix to be the fit Z = F * L and then X ~N(Z, sigma2_obs) */
    mat_copy(gex_filtered->X, model->FL);
    double stddev_obs = sqrt(sigma2_obs);
    mat_add_gaussian_noise(gex_filtered->X, stddev_obs);

    write_model(outprefix, gex_filtered, model->L, model->F, 
                        gex_filtered->cell_names, gex_filtered->gene_names, factor_names, 
                        k, model->brownian_prior_objective, model->observation_objective,
                        sigma2_obs, sigma2_latent);

    printf("Wrote resulting model parameters to outprefix %s\n", outprefix);
    
    /* Write out the gram matrix of L rows to check linear independence */
    if (model != NULL && model->L != NULL) {
        char l_gram_path[4096];
        Matrix *Lt = mat_transpose(model->L);
        Matrix *l_gram = mat_new(k, k);

        mat_mult_lapack(l_gram, model->L, Lt);
        snprintf(l_gram_path, sizeof(l_gram_path), "%s.L.gram.tsv", outprefix);
        write_labeled_matrix_tsv(l_gram_path, l_gram,
                                 factor_names, k,
                                 factor_names, k,
                                 "factor");

        mat_free(Lt);
        mat_free(l_gram);
    }

    /* Compare the fitted factors to each other pairwise */
    Matrix *factor_corr = NULL;
    char factor_corr_path[4096];

    factor_corr = mat_factor_pearson_correlation(model->L, model->L, 1, 0);
    snprintf(factor_corr_path, sizeof(factor_corr_path),
                "%s.L.pearson_correlation.tsv", outprefix);
    write_labeled_matrix_tsv(factor_corr_path, factor_corr,
                                factor_names, k,
                                factor_names, k,
                                "PCA_factor");
    mat_free(factor_corr);

    /* Compare the PCA to each other pairwise (just checking for technical accuracy here) */
    if (pca != NULL) {
        factor_corr = mat_factor_pearson_correlation(pca->components, pca->components, 1, 0);
        snprintf(factor_corr_path, sizeof(factor_corr_path),
                    "%s.pca.pearson_correlation.tsv", outprefix);
        write_labeled_matrix_tsv(factor_corr_path, factor_corr,
                                    factor_names, pca->K,
                                    factor_names, pca->K,
                                    "PCA_factor");
        mat_free(factor_corr);
    }

    /* Compare the fitted factors to the initial PCA factors */
    if (pca != NULL) {
        Matrix *factor_corr = NULL;
        char factor_corr_path[4096];
        char **pca_factor_names = scalloc(pca->K, sizeof(char *));

        generate_names(pca_factor_names, pca->K, "PC");
        factor_corr = mat_factor_pearson_correlation(pca->components, model->L, 1, 0);

        snprintf(factor_corr_path, sizeof(factor_corr_path),
                    "%s.pca.L.pearson_correlation.tsv", outprefix);
        write_labeled_matrix_tsv(factor_corr_path, factor_corr,
                                    pca_factor_names, pca->K,
                                    factor_names, k,
                                    "PCA_factor");
        mat_free(factor_corr);

        for (i = 0; i < pca->K; i++) {
            if (pca_factor_names[i] != NULL)
                free(pca_factor_names[i]);
        }
        free(pca_factor_names);
    }

    /* Free memory */
    if (factor_names != NULL) {
        for (i = 0; i < k; i++) {
            if (factor_names[i] != NULL)
                free(factor_names[i]);
        }
        free(factor_names);
    }
    gex_free_trees(trees, n_trees);
    gex_free_matrix_data(gex);
    gex_free_matrix_data(gex_filtered);
    free_moran_result(morans);
    free_lrt_result(lrt);
    free_pca(pca);
    gex_free_latent_brownian_model(model);
    if (Sigmas != NULL) {
        for (i = 0; i < n_trees; i++) {
            if (Sigmas[i] != NULL)
                mat_free(Sigmas[i]);
        }
        free(Sigmas);
    }
    if (filter_Sigmas != NULL && filter_Sigmas != Sigmas) {
        free(filter_Sigmas);
    }
    if (filter_avg_Sigma != NULL) {
        mat_free(filter_avg_Sigma);
    }

    return 0; /* Success */
}
