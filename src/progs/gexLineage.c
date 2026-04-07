#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gex.h"
#include "gex_model.h"
#include "pca.h"
#include "brownian.h"

/* Select a subset of covariance matrices (trees) for latent model fitting
by sampling without replacement. */
static Matrix **gexlineage_select_model_sigmas(Matrix **Sigmas,
                                               int n_sigmas,
                                               int n_keep,
                                               unsigned int seed) {
    int i;
    Matrix **selected = NULL;
    int *indices = NULL;
    unsigned int rng_state = (seed == 0u ? 1u : seed);

    if (Sigmas == NULL || n_sigmas <= 0)
        return NULL;

    if (n_keep <= 0 || n_keep >= n_sigmas)
        n_keep = n_sigmas;

    selected = (Matrix **)calloc(n_keep, sizeof(Matrix *));
    if (selected == NULL)
        return NULL;

    if (n_keep == n_sigmas) {
        for (i = 0; i < n_sigmas; i++)
            selected[i] = Sigmas[i];
        return selected;
    }

    indices = (int *)malloc(n_sigmas * sizeof(int));
    if (indices == NULL) {
        free(selected);
        return NULL;
    }

    for (i = 0; i < n_sigmas; i++)
        indices[i] = i;

    for (i = 0; i < n_keep; i++) {
        int j;
        int tmp;

        rng_state = (rng_state * 1664525u) + 1013904223u;
        j = i + (int)(rng_state % (unsigned int)(n_sigmas - i));
        tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
        selected[i] = Sigmas[indices[i]];
    }

    free(indices);
    return selected;
}

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
    if (strcmp(s, "both") == 0) {
        *mode_out = GEX_FILTER_BOTH;
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
        "Usage: %s "
        "--trees <trees.nex> "
        "--expr <matrix.tsv> "
        "--outprefix <prefix> "
        "[--tree-total-time T] "
        "[--n-filter-trees N] "
        "[--n-model-trees N] "
        "[--filter-test lrt|moran|both] "
        "[--lrt-alt lambda|full] "
        "[--pca-var-threshold V] "
        "[--n-sims N] "
        "[--n-perms N] "
        "[--max-q Q] "
        "[--moran-min-i I] "
        "[--filter-only] "
        "[--no-filter] "
        "[--verbose] "
        "[--seed S]\n",
        progname);
}

/* Main program entry point for gexLineage. */
int main(int argc, char *argv[]) {
    /* Data structures to store user inputs and other default parameters */
    const char *trees_file = NULL;  /* Path to input NEXUS file containing trees */
    const char *expr_file = NULL;   /* Path to input tab-delimited file containing expression matrix */
    const char *outprefix = NULL;   /* Prefix for all output files */
    GexFilterMode filter_mode = GEX_FILTER_LRT;   /* Which test(s) to use for filtering genes before modeling */
    int n_sims = 100;   /* Number of simulations used for a pre-check of the filter step performance */
    int n_perms = 1000; /* Number of permutations for monte-carlo based permutation tests */
    int n_filter_trees = 1;  /* -1: average covariance, 0: all trees, >0: first N trees */
    int n_model_trees = 0;  /* Number of trees to use for latent model fitting; 0 means use all trees */
    double max_q = 0.05;  /* False discovery rate for multiple testing correction */
    double moran_min_i = 0.0;   /* Minimum Moran's I value for retention during filtering */
    double pca_var_threshold = 0.99;    /* Threshold of variance explained to retain PCA components up to */
    double tree_total_time = -1.0;  /* If positive, rescale all trees uniformly to have this total height. */
    int filter_only = 0;    /* If nonzero, stop after writing filter outputs and exit successfully. */
    int no_filter = 0;  /* If nonzero, skip the filter step and use all genes for modeling. */
    int verbose = 0;    /* If nonzero, print additional progress messages during the run. */
    unsigned int seed = 1u;   /* Random seed (positive) for all stochastic calculations */
    const double ultrametric_tol = 1e-3;   /* Tolerance for ultrametric tree checking */
    GexLRTAltMode lrt_alt_mode = GEX_LRT_ALT_LAMBDA;   /* Which alternative model to use for the Brownian LRT */

    /* Data structures for calculations later */
    TreeNode **trees = NULL;    /* Array of tree pointers */
    GexMatrix *gex = NULL;  /* Original expression matrix */
    GexMatrix *gex_filtered = NULL; /* Filtered expression matrix */
    Matrix **Sigmas = NULL; /* Phylogenetic covariance matrices, one per tree */
    Matrix *filter_avg_Sigma = NULL; /* Average covariance used when --n-filter-trees=-1 */
    Matrix **filter_Sigmas = NULL; /* Covariance matrices used for filtering */
    Matrix **model_Sigmas = NULL; /* Selected covariance matrices for latent model fitting */
    GexMoransResult *morans = NULL; /* Results from Moran's I calculation */
    GexLRTResult *lrt = NULL;   /* Results from Brownian LRT calculation */
    GexPCA *pca = NULL; /* PCA results */
    GexLatentBrownianModel *model = NULL;   /* Fitted latent Brownian gene expression model */
    int n_trees = 0;    /* Number of input trees */
    int i;  /* Pre-allocated generic loop index variable */
    int status = 1; /* Assume failure by default */
    

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--trees") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                goto cleanup;
            }
            trees_file = argv[++i];
        }
        else if (strcmp(argv[i], "--expr") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                goto cleanup;
            }
            expr_file = argv[++i];
        }
        else if (strcmp(argv[i], "--outprefix") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                goto cleanup;
            }
            outprefix = argv[++i];
        }
        else if (strcmp(argv[i], "--tree-total-time") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                goto cleanup;
            }
            tree_total_time = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--n-filter-trees") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                goto cleanup;
            }
            n_filter_trees = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--n-model-trees") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                goto cleanup;
            }
            n_model_trees = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--filter-test") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                goto cleanup;
            }
            if (parse_filter_mode(argv[++i], &filter_mode) != 0) {
                fprintf(stderr, "ERROR: --filter-test must be one of moran, lrt, both\n");
                goto cleanup;
            }
        }
        else if (strcmp(argv[i], "--pca-var-threshold") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                goto cleanup;
            }
            pca_var_threshold = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--lrt-alt") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                goto cleanup;
            }
            if (parse_lrt_alt_mode(argv[++i], &lrt_alt_mode) != 0) {
                fprintf(stderr, "ERROR: --lrt-alt must be one of full, lambda\n");
                goto cleanup;
            }
        }
        else if (strcmp(argv[i], "--n-sims") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                goto cleanup;
            }
            n_sims = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--n-perms") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                goto cleanup;
            }
            n_perms = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--max-q") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                goto cleanup;
            }
            max_q = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--moran-min-i") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                goto cleanup;
            }
            moran_min_i = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--seed") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                goto cleanup;
            }
            seed = (unsigned int)strtoul(argv[++i], NULL, 10);
        }
        else if (strcmp(argv[i], "--filter-only") == 0) {
            filter_only = 1;
        }
        else if (strcmp(argv[i], "--no-filter") == 0) {
            no_filter = 1;
        }
        else if (strcmp(argv[i], "--verbose") == 0) {
            verbose = 1;
        }
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            status = 0; /* Success since the user just wants command line help */
            goto cleanup;
        }
        else {
            fprintf(stderr, "ERROR: unknown argument: %s\n", argv[i]);
            usage(argv[0]);
            goto cleanup;
        }
    }

    /* Check that all required inputs are specified */
    if (trees_file == NULL || expr_file == NULL || outprefix == NULL) {
        usage(argv[0]);
        goto cleanup;
    }
    if (filter_only && no_filter) {
        fprintf(stderr, "ERROR: --filter-only and --no-filter cannot be used together.\n");
        goto cleanup;
    }

    /* Load the input trees */
    trees = gex_read_nexus(trees_file, &n_trees);
    if (trees == NULL || n_trees < 1 || trees[0] == NULL) {
        fprintf(stderr, "ERROR: failed to load tree(s).\n");
        goto cleanup;
    }
    printf("Loaded %d tree(s).\n", n_trees);

    if (n_filter_trees < -1) {
        fprintf(stderr, "ERROR: --n-filter-trees must be -1, 0, or a positive integer\n");
        goto cleanup;
    }
    if (n_filter_trees > n_trees) {
        fprintf(stderr, "ERROR: --n-filter-trees (%d) cannot exceed the number of loaded trees (%d)\n",
                n_filter_trees, n_trees);
        goto cleanup;
    }
    if (n_model_trees < 0) {
        fprintf(stderr, "ERROR: --n-model-trees must be nonnegative (0 means use all trees)\n");
        goto cleanup;
    }
    if (n_model_trees > n_trees) {
        fprintf(stderr, "ERROR: --n-model-trees (%d) cannot exceed the number of loaded trees (%d)\n",
                n_model_trees, n_trees);
        goto cleanup;
    }

    /* Check that the input trees are ultrametric (required for cell lineage) */
    if (gex_check_trees_ultrametric(trees, n_trees, ultrametric_tol) != 0) {
        goto cleanup;
    }

    /* Load the input expression matrix */
    gex = gex_read_labeled_matrix(expr_file);
    if (gex == NULL) {
        fprintf(stderr, "ERROR: failed to load expression matrix.\n");
        goto cleanup;
    }
    printf("Loaded matrix with %d cell(s) and %d gene(s).\n", gex->n_cells, gex->n_genes);

    if (verbose) {
        /* Print input/output summary for user verification */
        gex_print_io_summary(trees, n_trees, gex);
    }

    /* Reconcile tree tips and expression cell names to the intersection of both sets
    if they do not perfectly match. */
    if (gex_reconcile_tree_and_expression(trees, n_trees, &gex) != 0) {
        fprintf(stderr, "ERROR: failed to reconcile tree tips and expression cell names.\n");
        goto cleanup;
    }

    /* Rescale the trees to a specified total height if requested */
    if (tree_total_time > 0.0) {
        if (gex_rescale_trees_total_height(trees, n_trees, tree_total_time) != 0) {
            goto cleanup;
        }
        printf("Rescaled tree(s) to total height %.6f.\n", tree_total_time);
    }

    /* Calculate the phylogenetic covariance matrix for each input tree. */
    Sigmas = (Matrix **)calloc(n_trees, sizeof(Matrix *));
    if (Sigmas == NULL) {
        fprintf(stderr, "ERROR: failed to allocate Brownian covariance matrix list.\n");
        goto cleanup;
    }
    for (i = 0; i < n_trees; i++) {
        Sigmas[i] = covariance_from_tree(trees[i], gex->cell_names, gex->n_cells);
        if (Sigmas[i] == NULL) {
            fprintf(stderr, "ERROR: failed to compute Brownian covariance matrix for tree %d.\n", i + 1);
            goto cleanup;
        }
    }
    if (verbose) {
        printf("Computed phylogenetic covariance matrix for the first tree:\n");
        print_covariance_summary(Sigmas[0], gex->cell_names, gex->n_cells);
    }

    /* Compute average covariance matrix over input trees if needed */
    if (n_filter_trees == -1) {
        filter_avg_Sigma = gex_average_tree_covariance(trees, n_trees,
                                                       gex->cell_names, gex->n_cells);
        if (filter_avg_Sigma == NULL) {
            fprintf(stderr, "ERROR: failed to compute average covariance for filtering.\n");
            goto cleanup;
        }
        filter_Sigmas = (Matrix **)calloc(1, sizeof(Matrix *));
        if (filter_Sigmas == NULL) {
            fprintf(stderr, "ERROR: failed to allocate average covariance wrapper.\n");
            goto cleanup;
        }
        filter_Sigmas[0] = filter_avg_Sigma;
    }
    else {
        filter_Sigmas = Sigmas;
        if (n_filter_trees == 0)
            n_filter_trees = n_trees;
    }

    if (!no_filter) {
        /* Test the phylogenetic signal filter(s) with simulated data to understand
        the performance on the tree subset used for filtering. */
        if (n_filter_trees == -1) {
            printf("Running a simulation check of the phylogenetic signal gene filter(s) using the average covariance across all %d tree(s)...\n",
                   n_trees);
        } else if (n_filter_trees == n_trees) {
            printf("Running a simulation check of the phylogenetic signal gene filter(s) using all %d tree(s)...\n",
                   n_filter_trees);
        } else {
            printf("Running a simulation check of the phylogenetic signal gene filter(s) using the first %d tree(s)...\n",
                   n_filter_trees);
        }
        if (!brownian_run_simulation_check(gex->cell_names,
                                           gex->n_cells,
                                           n_sims,
                                           n_sims,
                                           filter_mode,
                                           lrt_alt_mode,
                                           n_perms,
                                           max_q,
                                           moran_min_i,
                                           filter_Sigmas,
                                           (n_filter_trees == -1 ? 1 : n_filter_trees),
                                           seed)) {
            if (verbose) {
                printf("WARNING: Simulation check of signal filter did NOT perfectly recover all positive/negative genes for the provided tree.\n");
            }
        } else {
            if (verbose) {
                printf("Simulation check of signal filter successfully recovered all positive/negative genes for the provided tree.\n");
            }
        }
    }

    if (!no_filter) {
        if (n_filter_trees == -1) {
            printf("Applying the phylogenetic signal gene filter(s) to the real input gene expression matrix data using the average covariance across all %d tree(s)...\n",
                   n_trees);
        } else if (n_filter_trees == n_trees) {
            printf("Applying the phylogenetic signal gene filter(s) to the real input gene expression matrix data using all %d tree(s)...\n", n_filter_trees);
        } else {
            printf("Applying the phylogenetic signal gene filter(s) to the real input gene expression matrix data using the first %d tree(s)...\n",
                    n_filter_trees);
        }
        /* Run the phylogenetic autocorrelation filter tests if requested */
        if (filter_mode == GEX_FILTER_MORAN || filter_mode == GEX_FILTER_BOTH) {
            morans = gex_compute_morans_i(gex, filter_Sigmas,
                                          (n_filter_trees == -1 ? 1 : n_filter_trees),
                                          n_perms, seed);
            if (morans == NULL) {
                fprintf(stderr, "ERROR: failed to compute Moran's I statistics.\n");
                goto cleanup;
            }
            if (verbose) {
                gex_print_morans_summary(morans, gex, max_q, moran_min_i);
            }

            {
                char corr_path[4096];
                snprintf(corr_path, sizeof(corr_path), "%s.correlation.moran.tsv", outprefix);
                if (gex_write_morans_tsv(corr_path, morans, gex, max_q, moran_min_i) != 0) {
                    fprintf(stderr, "ERROR: failed to write Moran correlation results to %s.\n",
                            corr_path);
                    goto cleanup;
                }
                if (verbose) {
                    printf("Wrote Moran correlation results to %s.\n", corr_path);
                }
            }
        }

        /* Run the phylogenetic LRT filter tests if requested */
        if (filter_mode == GEX_FILTER_LRT || filter_mode == GEX_FILTER_BOTH) {
            lrt = gex_compute_brownian_lrt(gex, filter_Sigmas,
                                           (n_filter_trees == -1 ? 1 : n_filter_trees),
                                           n_perms, seed, lrt_alt_mode);
            if (lrt == NULL) {
                fprintf(stderr, "ERROR: failed to compute Brownian LRT statistics.\n");
                goto cleanup;
            }
            if (verbose) {
                gex_print_lrt_summary(lrt, gex, max_q);
            }

            {
                char lrt_path[4096];
                snprintf(lrt_path, sizeof(lrt_path), "%s.correlation.lrt.tsv", outprefix);
                if (gex_write_lrt_tsv(lrt_path, lrt, gex, max_q) != 0) {
                    fprintf(stderr, "ERROR: failed to write LRT correlation results to %s\n",
                            lrt_path);
                    goto cleanup;
                }
                if (verbose) {
                    printf("Wrote LRT correlation results to %s.\n", lrt_path);
                }
            }
        }

        /* Stop early if only phylogenetic signal filtering is requested */
        if (filter_only) {
            printf("Done\n");
            status = 0;
            goto cleanup;
        }

        /* Filter genes based on the results of the correlation and/or LRT test(s) */
        gex_filtered = gex_filter_genes_by_results(gex, morans, lrt, filter_mode,
                                                   max_q, moran_min_i);
        if (gex_filtered == NULL) {
            fprintf(stderr, "ERROR: failed to filter genes by selected test(s).\n");
            goto cleanup;
        }
        printf("Filtered matrix has %d cells and %d gene(s).\n", gex_filtered->n_cells, gex_filtered->n_genes);
        printf("Running PCA on the filtered gene expression matrix to select the number of latent factor dimensions for the model...\n");
    } else {
        gex_filtered = gex;
        gex = NULL;
        printf("Skipping phylogenetic signal gene filtering and using all %d gene(s) for modeling.\n",
               gex_filtered->n_genes);
        printf("Running PCA on the input unfiltered gene expression matrix to select the number of latent factor dimensions for the model...\n");
    }

    /* Run PCA on the filtered matrix and retain the smallest number of
    components needed to explain at least the requested variance. */
    pca = gex_compute_pca(gex_filtered, pca_var_threshold);
    if (pca == NULL) {
        fprintf(stderr, "ERROR: PCA failed.\n");
        goto cleanup;
    }
    printf("Retained %d PCA component(s) to explain at least %.2f%% of variance.\n",
           pca->K, 100.0 * pca_var_threshold);
    if (verbose) {
        gex_print_pca_summary(pca);
    }

    /* Downsample the input set of covariance matrices (trees) for fitting the latent model if requested */
    if (n_model_trees > 0 && n_model_trees < n_trees) {
        model_Sigmas = gexlineage_select_model_sigmas(Sigmas, n_trees, n_model_trees, seed + 97u);
        if (model_Sigmas == NULL) {
            fprintf(stderr, "ERROR: failed to select covariance matrices for latent model fitting.\n");
            goto cleanup;
        }
        printf("Randomly downsampled (without replacement) %d tree(s) for latent model fitting.\n", n_model_trees);
    } else {
        model_Sigmas = Sigmas;
        n_model_trees = n_trees;
    }

    /* Fit the latent Brownian model */
    printf("Fitting model to the filtered data with k=%d latent dimensions...\n", pca->K);
    model = gex_fit_latent_brownian_model(gex_filtered, model_Sigmas, n_model_trees, pca, seed, outprefix);
    if (model == NULL) {
        fprintf(stderr, "ERROR: failed to fit latent Brownian gene expression model.\n");
        goto cleanup;
    }

    /* Write the fitted latent Brownian model results to files */
    if (gex_write_latent_brownian_model(outprefix, model, gex_filtered) != 0) {
        fprintf(stderr,
                "ERROR: failed to write latent Brownian model outputs with prefix %s.\n",
                outprefix);
        goto cleanup;
    }
    if (verbose) {
        printf("Wrote resulting model parameters to outprefix %s.\n", outprefix);
    }

    /* End of program */
    printf("Done.\n");
    status = 0; /* Success */
    goto cleanup;

    /* Free allocated memory */
    cleanup:
        gex_free_trees(trees, n_trees);
        gex_free_matrix_data(gex);
        gex_free_matrix_data(gex_filtered);
        gex_free_morans_result(morans);
        gex_free_lrt_result(lrt);
        gex_free_pca(pca);
        gex_free_latent_brownian_model(model);
        if (Sigmas != NULL) {
            for (i = 0; i < n_trees; i++) {
                if (Sigmas[i] != NULL)
                    mat_free(Sigmas[i]);
            }
            free(Sigmas);
        }
        /* Only free model_Sigmas if it is separately allocated */
        if (model_Sigmas != NULL && model_Sigmas != Sigmas) {
            free(model_Sigmas);
        }
        if (filter_Sigmas != NULL && filter_Sigmas != Sigmas) {
            free(filter_Sigmas);
        }
        if (filter_avg_Sigma != NULL) {
            mat_free(filter_avg_Sigma);
        }

        return status;
}
