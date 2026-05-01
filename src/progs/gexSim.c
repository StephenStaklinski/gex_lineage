
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
        "[--L-l2-norm <csv>] "
        "[--desired-tip-var <csv> ] "
        "[--sigma2 <csv>] "
        "[--dim K] "
        "[--sigma2-obs S] "
        "[--include-factorization ]"
        "[--identity-cov] "
        "[--verbose]"
        "[--seed S]\n",
        progname);
}


/* Main program entry point for gexLineage. */
int main(int argc, char *argv[]) {
    /* Data structures to store user inputs and other default parameters */
    const char *trees_file = NULL;  /* Path to input NEXUS file containing trees */
    const char *outprefix = NULL;   /* Prefix for all output files */
    double tree_total_time = 1.0;  /* If positive, rescale all trees uniformly to have this total height. */
    int n_genes = 100; /* Number of genes to simulate */
    double desired_L_l2_norm = 1.0; /* Default desired L l2 norm for latent factors */
    int k = 5; /* Number of latent factors to simulate */
    double sigma2_obs = 1.0; /* Variance of observation noise */
    int expr_only = 1; /* If nonzero, only output the simulated expression matrix. */
    int identity_cov = 0; /* If nonzero, use identity covariance (null model) instead of tree-based covariance for simulations. */
    int verbose = 0;    /* If nonzero, print additional progress messages during the run. */

    /* Data structures for calculations later */
    TreeNode **trees = NULL;    /* Array of tree pointers */
    Matrix **Sigmas = NULL; /* Phylogenetic covariance matrices, one per tree */
    Matrix **use_Sigmas = NULL; /* Covariance matrices used for filtering */
    int n_trees = 0;    /* Number of input trees */
    int n_use_sigmas = 0; /* Number of covariance matrices used for simulation */
    Vector *mu;   /* Mean vector for simulations */
    Vector *L_row_norms; /* Row norms of L, for output */
    Vector *sigma2s = NULL;   /* Vector of Brownian variance parameters for simulations */
    Vector *input_L_l2_norms = NULL; /* Raw L l2 norms input from CLI */
    Vector *input_tip_vars = NULL; /* Raw desired tip variance input from CLI */
    Vector *input_sigma2s = NULL; /* Raw sigma2 input from CLI */
    GexMatrix *gex = scalloc(1, sizeof(GexMatrix));  /* Simulated expression matrix */
    int i;
    int n_cells;
    int sim_dim;
    set_seed(-1); /* Random seed, for now */
    

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
        else if (strcmp(argv[i], "--L-l2-norm") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            input_L_l2_norms = parse_csv_to_vec(argv[++i]);
        }
        else if (strcmp(argv[i], "--desired-tip-var") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            input_tip_vars = parse_csv_to_vec(argv[++i]);
        }
        else if (strcmp(argv[i], "--sigma2") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            input_sigma2s = parse_csv_to_vec(argv[++i]);
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
        else if (strcmp(argv[i], "--seed") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            set_seed((unsigned int)atoi(argv[++i]));
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
    if (input_L_l2_norms != NULL && input_tip_vars != NULL && input_sigma2s != NULL) {
        fprintf(stderr, "ERROR: specify only one of --L-l2-norm or --desired-tip-var or --sigma2\n");
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
    gex = gex_mat_new(lst_size(leaf_names), n_genes);
    for (i = 0; i < lst_size(leaf_names); i++) {
        String *leaf_name = lst_get_ptr(leaf_names, i);
        gex->cell_names[i] = strdup(leaf_name->chars);
    }
    n_cells = lst_size(leaf_names);
    lst_free_strings(leaf_names);
    lst_free(leaf_names);

    /* Set the gene names in the gene expression matrix */
    generate_names(gex->gene_names, n_genes, "gene");

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

        use_Sigmas = Sigmas;
        n_use_sigmas = n_trees;
        printf("Using all %d tree(s)...\n", n_use_sigmas);

    } else {
        /* Use identity covariance for simulations instead of tree-based covariance if requested */
        use_Sigmas = scalloc(1, sizeof(Matrix *));
        use_Sigmas[0] = mat_new(n_cells, n_cells);
        mat_set_identity(use_Sigmas[0]);
        if (use_Sigmas[0] == NULL) {
            fprintf(stderr, "ERROR: failed to create identity covariance matrix for simulations.\n");
            return 1;
        }
        n_use_sigmas = 1;
        printf("Using identity covariance instead of phylogenetic covariance from trees.\n");
    }

    /* For now, assume zero mean vector */
    mu = vec_new(n_cells);   
    vec_zero(mu);

    /* Determine the dimension of the simulation based on whether the Brownian simulation
    is for genes or latent factors */
    sim_dim = (expr_only ? n_genes : k);

    /* Default to a single L l2 norm if neither option was provided. */
    if (input_L_l2_norms == NULL && input_tip_vars == NULL && input_sigma2s == NULL) {
        input_L_l2_norms = vec_new(1);
        vec_set(input_L_l2_norms, 0, desired_L_l2_norm);
    }

    /* Otherwise read in what what provided for the Brownian sigma2s
    either directly from input or calculated from input desired tip variances */
    if (input_L_l2_norms != NULL) {
        /* All sigma2s set to 1.0 */
        sigma2s = vec_new(sim_dim);
        for (i = 0; i < sim_dim; i++)
            vec_set(sigma2s, i, 1.0);

        /* Use the provided L l2 norms */
        L_row_norms = expand_input_csv(input_L_l2_norms, sim_dim);
    }
    else if (input_tip_vars != NULL) {
        Vector *tip_vars = expand_input_csv(input_tip_vars, sim_dim);

        /* Do the calculations to get sigma2s for the desired tip variance(s) */
        double tree_height = mat_get(use_Sigmas[0], 0, 0);
        sigma2s = vec_new(sim_dim);
        for (i = 0; i < sim_dim; i++)
            vec_set(sigma2s, i, vec_get(tip_vars, i) / tree_height);

        /* Free memory */
        vec_free(input_tip_vars);
        vec_free(tip_vars);

        /* All row norms set to 1.0 */
        L_row_norms = vec_new(sim_dim);
        for (i = 0; i < sim_dim; i++)
            vec_set(L_row_norms, i, 1.0);
    }
    else {
        sigma2s = expand_input_csv(input_sigma2s, sim_dim);

        /* Free memory */
        vec_free(input_sigma2s);

        /* All row norms set to 1.0 */
        L_row_norms = vec_new(sim_dim);
        for (i = 0; i < sim_dim; i++)
            vec_set(L_row_norms, i, 1.0);
    }

    /* Run simulation */
    gex->X = brownian_simulate(use_Sigmas, 
                                n_use_sigmas,
                                mu, 
                                sim_dim,
                                sigma2s);

    /* Write out the Brownian motion result */
    char expr_buf[4096];
    if (expr_only) {
        snprintf(expr_buf, sizeof(expr_buf), "%s.expr.tsv", outprefix);
        write_labeled_matrix_tsv(expr_buf, gex->X, gex->cell_names, gex->X->nrows,
                                    gex->gene_names, gex->X->ncols, "cell");
    }


    if (!expr_only) {
        /* Use the Brownian result as the latent factors matrix F (cells x factors) to 
        simulate L (factors x genes) and the reconstructed expression matrix X
        with noise for the input parameters. */
        Matrix *L = mat_new(k, n_genes);
        GexMatrix *gex_obs = gex_mat_new(n_cells, n_genes);
        copy_string_array_inplace(gex_obs->cell_names, gex->cell_names, n_cells);
        copy_string_array_inplace(gex_obs->gene_names, gex->gene_names, n_genes);
        generate_names(gex->gene_names, sim_dim, "factor"); /* gex becomes F now */
        

        simulate_factorization_and_reconstruction(gex->X, gex->cell_names, n_cells, k, n_genes, 
                                                    sigma2_obs, L_row_norms, L, gex_obs);

        /* Deterministic transformation to prevent permutation invariance */
        double *log_sigma2_latent = scalloc(k, sizeof(double));
        for (i = 0; i < k; i++)
            log_sigma2_latent[i] = log(vec_get(sigma2s, i));

        if (input_L_l2_norms != NULL) {
            reorder_factors_by_row_norm(L, gex->X);
        }
        else {
            reorder_factors_by_sigma2_latent(L, gex->X, log_sigma2_latent);

            /* Reset sigma2s to the re-ordered values */
            for (i = 0; i < k; i++)
                vec_set(sigma2s, i, exp(log_sigma2_latent[i]));
        }

        /* Deterministic transformation to prevent sign invariance */
        post_hoc_sign_identifiability(L, gex->X);

        double brownian_negll = 0.0;
        if (!identity_cov) {
            brownian_negll = gex_brownian_prior_from_trees(gex->X,
                                                           log_sigma2_latent,
                                                           trees,
                                                           n_use_sigmas,
                                                           gex->cell_names,
                                                           NULL,
                                                           NULL);
        }

        Matrix *FL = mat_new(n_cells, n_genes);
        mat_mult_lapack(FL, gex->X, L);
        double observation_negll = gaussian_observation_term(FL, gex->X, L, log(sigma2_obs), gex_obs->X, NULL, NULL, NULL, NULL);

        /* Write out results */
        write_model(outprefix, gex_obs, L, gex->X, gex_obs->cell_names, gex_obs->gene_names,
                        gex->gene_names, k, brownian_negll, observation_negll, sigma2_obs, sigma2s->data);

        /* Free memory */
        if (log_sigma2_latent != NULL)
            free(log_sigma2_latent);
        if (L != NULL)
            mat_free(L);
        if (gex_obs != NULL)
            gex_free_matrix_data(gex_obs);
        if (FL != NULL)
            mat_free(FL);
    }

    printf("Wrote output to %s\n", outprefix);

    /* Free memory */
    vec_free(input_L_l2_norms);
    vec_free(mu);
    vec_free(sigma2s);
    gex_free_trees(trees, n_trees);
    gex_free_matrix_data(gex);
    if (Sigmas != NULL) {
        for (i = 0; i < n_trees; i++) {
            if (Sigmas[i] != NULL)
                mat_free(Sigmas[i]);
        }
        free(Sigmas);
    }
    if (identity_cov && use_Sigmas != NULL && use_Sigmas[0] != NULL) {
        mat_free(use_Sigmas[0]);
    }
    if (use_Sigmas != NULL && use_Sigmas != Sigmas) {
        free(use_Sigmas);
    }

    return 0; /* Success */
}
