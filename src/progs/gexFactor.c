
#include "external_libs.h"
#include "gexmatrix.h"
#include "misc.h"
#include "model.h"
#include "latentflow.h"
#include "parser.h"
#include "pca.h"

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
        "Usage: %s --trees <trees.nex> --expr <filtered.tsv> "
        "--outprefix <prefix> --dim K "
        "[--pca <pca.eigenvectors.tsv>] "
        "[--no-scale-constraint] [--L-l1-strength S] "
        "[--L-loading-overlap-strength S] "
        "[--no-post-hoc-identifiability] [--no-write-latent-flow] "
        "[--seed S]\n",
        progname);
}

/* Fit the latent Brownian factor model to modeling-ready expression data. */
int main(int argc, char *argv[]) {
    /* Data structures to store user inputs and other default parameters */
    const char *trees_file = NULL;  /* Path to input NEXUS file containing trees */
    const char *expr_file = NULL;   /* Path to input tab-delimited file containing expression matrix */
    const char *outprefix = NULL;   /* Prefix for all output files */
    const char *pca_file = NULL;    /* Optional saved PCA loadings used to initialize L. */
    double L_l1_strength = 0;  /* L1 regularization strength for loadings; 0 disables the penalty. */
    double L_loading_overlap_strength = 0.0;  /* Strength of absolute L-row loading-overlap penalty; 0 disables it. */
    int k = 0;  /* Number of latent factors to fit. */
    int write_latent_flow = 1; /* If nonzero, write latent factors for tips and reconstructed internal nodes. */
    int constrain_L_scale = 1; /* If nonzero, constrain each L row to unit norm. */
    int apply_post_hoc_identifiability = 1; /* If nonzero, apply post-hoc sign and permutation identifiability fixes. */

    /* Data structures for calculations later */
    TreeNode **trees = NULL;    /* Array of tree pointers */
    GexMatrix *gex_filtered = NULL; /* Expression matrix */
    PCA *pca = NULL; /* Optional loaded PCA initialization. */
    GexLatentBrownianModel *model = NULL;   /* Fitted latent Brownian gene expression model */
    char **factor_names = NULL; /* Names assigned to fitted latent factors */
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
        else if (strcmp(argv[i], "--pca") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            pca_file = argv[++i];
        }
        else if (strcmp(argv[i], "--L-l1-strength") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            L_l1_strength = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--L-loading-overlap-strength") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            L_loading_overlap_strength = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--dim") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            k = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--no-write-latent-flow") == 0) {
            write_latent_flow = 0;
        }
        else if (strcmp(argv[i], "--no-scale-constraint") == 0) {
            constrain_L_scale = 0;
        }
        else if (strcmp(argv[i], "--no-post-hoc-identifiability") == 0) {
            apply_post_hoc_identifiability = 0;
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
            return 0;
        }
        else {
            fprintf(stderr, "ERROR: unknown argument: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    /* Verify inputs are valid */
    if (trees_file == NULL || expr_file == NULL || outprefix == NULL) {
        usage(argv[0]);
        return 1;
    }
    if (k <= 0) {
        fprintf(stderr, "ERROR: --dim must be specified and positive.\n");
        return 1;
    }
    if (L_l1_strength < 0.0) {
        fprintf(stderr, "ERROR: --L-l1-strength must be nonnegative (0 disables L1 regularization)\n");
        return 1;
    }
    if (L_loading_overlap_strength < 0.0) {
        fprintf(stderr, "ERROR: --L-loading-overlap-strength must be nonnegative (0 disables L loading-overlap regularization)\n");
        return 1;
    }

    if (load_and_reconcile_tree_gex_inputs(trees_file, expr_file, -1,
                                           &trees, &n_trees,
                                           &gex_filtered) != 0)
        return 1;

    if (pca_file != NULL) {
        printf("Loading PCA initialization from %s...\n", pca_file);
        pca = read_pca_initialization_tsv(pca_file, gex_filtered, k);
        if (pca == NULL) {
            fprintf(stderr, "ERROR: failed to load PCA initialization.\n");
            return 1;
        }
    }

    /* Fit the latent Brownian model */
    printf("Fitting model to the filtered data with k=%d latent dimensions...\n", k);
    model = gex_fit_latent_brownian_model(gex_filtered, trees, n_trees,
                                          k, pca, constrain_L_scale,
                                          L_l1_strength,
                                          L_loading_overlap_strength,
                                          apply_post_hoc_identifiability,
                                          outprefix);
    if (model == NULL) {
        fprintf(stderr, "ERROR: failed to fit latent Brownian model.\n");
        return 1;
    }

    if (write_latent_flow) {
        printf("Writing latent flow table...\n");
        if (gex_write_latent_flow_outputs(outprefix, trees[0], gex_filtered, model) != 0) {
            fprintf(stderr, "ERROR: failed to write latent flow outputs.\n");
            return 1;
        }
    }

    /* Write the fitted latent Brownian model parameters to files */
    factor_names = scalloc(k, sizeof(char *));
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
    gex_free_matrix_data(gex_filtered);
    free_pca(pca);
    gex_free_latent_brownian_model(model);

    return 0;
}
