#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gex.h"
#include "gex_model.h"
#include "pca.h"
#include "brownian.h"

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
        "[--filter-test lrt|moran|both] "
        "[--lrt-alt lambda|full] "
        "[--pca-var-threshold V] "
        "[--n-perms N] "
        "[--max-q Q] "
        "[--moran-min-i I] "
        "[--filter-only] "
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
    double max_q = 0.05;  /* False discovery rate for multiple testing correction */
    double moran_min_i = 0.0;   /* Minimum Moran's I value for retention during filtering */
    double pca_var_threshold = 0.99;    /* Threshold of variance explained to retain PCA components up to */
    double tree_total_time = -1.0;  /* If positive, rescale all trees uniformly to have this total height. */
    int filter_only = 0;    /* If nonzero, stop after writing filter outputs and exit successfully. */
    int verbose = 0;    /* If nonzero, print additional progress messages during the run. */
    unsigned int seed = 1u;   /* Random seed (positive) for all stochastic calculations */
    const double ultrametric_tol = 1e-3;   /* Tolerance for ultrametric tree checking */
    GexLRTAltMode lrt_alt_mode = GEX_LRT_ALT_LAMBDA;   /* Which alternative model to use for the Brownian LRT */

    /* Data structures for calculations later */
    TreeNode **trees = NULL;    /* Array of tree pointers */
    GexMatrix *gex = NULL;  /* Original expression matrix */
    GexMatrix *gex_filtered = NULL; /* Filtered expression matrix */
    Matrix *Sigma = NULL;   /* Phylogenetic covariance matrix */
    Matrix *W = NULL;   /* Phylogenetic covariance-based weight matrix */
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

    /* Load the input trees */
    trees = gex_read_nexus(trees_file, &n_trees);
    if (trees == NULL || n_trees < 1 || trees[0] == NULL) {
        fprintf(stderr, "ERROR: failed to load tree(s).\n");
        goto cleanup;
    }
    printf("Loaded %d tree(s).\n", n_trees);

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

    /* Use first tree only (for initial testing) for the steps here below. 
    TODO: Make these steps somewhat Bayesian by integrating over the set of all trees */

    /* Calculate the phylogenetic covariance matrix */
    Sigma = covariance_from_tree(trees[0], gex->cell_names, gex->n_cells);
    if (Sigma == NULL) {
        fprintf(stderr, "ERROR: failed to compute Brownian covariance matrix.\n");
        goto cleanup;
    }
    if (verbose) {
        print_covariance_summary(Sigma, gex->cell_names, gex->n_cells);
    }

    /* Calculate the weight matrix from the phylogenetic covariance matrix if
    needed for phylogenetic autocorrelation tests */
    if (filter_mode == GEX_FILTER_MORAN || filter_mode == GEX_FILTER_BOTH) {
        W = weight_matrix_from_covariance(Sigma);
        if (W == NULL) {
            fprintf(stderr, "ERROR: failed to compute Brownian weight matrix.\n");
            goto cleanup;
        }
        if (verbose) {
            print_weight_matrix_summary(W);
        }
    }
    
    /* Test the phylogenetic signal filter(s) with simulated data to understand
    the performance on the provided tree. */
    if (!brownian_run_simulation_check(trees[0],
                                       gex->cell_names,
                                       gex->n_cells,
                                       n_sims,
                                       n_sims,
                                       filter_mode,
                                       lrt_alt_mode,
                                       n_perms,
                                       max_q,
                                       moran_min_i,
                                       Sigma,
                                       W,
                                       seed)) {
        if (verbose) {
            printf("WARNING: Simulation check of signal filter did NOT perfectly recover all positive/negative genes for the provided tree.\n");
        }
    } else {
        if (verbose) {
            printf("Simulation check of signal filter successfully recovered all positive/negative genes for the provided tree.\n");
        }
    }

    /* Run the phylogenetic autocorrelation filter tests if requested */
    if (filter_mode == GEX_FILTER_MORAN || filter_mode == GEX_FILTER_BOTH) {
        morans = gex_compute_morans_i(gex, W, n_perms, seed);
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
        lrt = gex_compute_brownian_lrt(gex, Sigma, n_perms, seed, lrt_alt_mode);
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

    /* Fit the latent Brownian model */
    printf("Fitting model to the filtered data with k=%d latent dimensions...\n", pca->K);
    model = gex_fit_latent_brownian_model(gex_filtered, Sigma, pca, seed, outprefix);
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
        if (Sigma != NULL)
            mat_free(Sigma);
        if (W != NULL)
            mat_free(W);
        return status;
}
