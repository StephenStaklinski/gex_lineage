#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "brownian.h"
#include "gex.h"

/* Parse a comma-separated list of double values into an array 
and count the number of values */
static int parse_double_list(const char *spec,
                                     double **values_io,
                                     int *n_values_io) {
    char *copy = NULL;
    char *tok = NULL;
    char *saveptr = NULL;
    int n = 0;

    if (spec == NULL || values_io == NULL || n_values_io == NULL)
        return -1;

    copy = strdup(spec);
    if (copy == NULL)
        return -1;

    /* Split on commas and parse each value */
    for (tok = strtok_r(copy, ",", &saveptr); tok != NULL; tok = strtok_r(NULL, ",", &saveptr)) {
        double *tmp = NULL;
        char *endptr = NULL;
        double val = strtod(tok, &endptr);
        if (endptr == tok || *endptr != '\0') {
            free(copy);
            return -1;
        }
        tmp = (double *)realloc(*values_io, (size_t)(*n_values_io + 1) * sizeof(double));
        if (tmp == NULL) {
            free(copy);
            return -1;
        }
        *values_io = tmp;
        (*values_io)[*n_values_io] = val;
        (*n_values_io)++;
        n++;
    }

    free(copy);
    return (n > 0 ? 0 : 1); /* Return 0 if successful, 1 if no values parsed */
}


static void usage(const char *progname) {
    fprintf(stderr,
            "Usage: %s "
            "--trees <trees.nex> "
            "--outprefix <prefix> "
            "--k <int> "
            "--n-genes <int> "
            "--sigma2-obs <float> "
            "--sigma2-latent <comma-list-floats> "
            "[--use-n-trees <int>] "
            "[--seed <int>]\n",
            progname);
}

int main(int argc, char *argv[]) {
    const char *trees_file = NULL;
    const char *outprefix = NULL;
    TreeNode **trees = NULL;
    char **cell_names = NULL;
    char *expr_path = NULL;
    Matrix **Sigmas = NULL;
    Matrix *Sigma = NULL;
    Matrix *avg_Sigmas[1] = {NULL};
    Matrix **sim_Sigmas = NULL;
    GexMatrix *per_sim_Z = NULL; /* Temporarily stores the simulated latent factors for each individual tree */
    GexMatrix *Z = NULL; /* Stores the overall simulated latent factors from Brownian motion */
    GexMatrix *L = NULL;
    GexMatrix *gex = NULL;
    double *sigma2_latent_raw = NULL;
    double *sigma2_latent = NULL;
    int n_trees = 0;
    int n_cells = 0;
    int k = 0;
    int n_genes = 0;
    int use_n_trees = -1; /* Match gexLineage: -1 average covariance, 0 all trees, >0 first N trees */
    int selected_n_trees = 0;
    int n_sigma_latent_raw = 0;
    int n_sim_sigmas = 0;
    int i;
    int status = 1;
    double sigma2_obs = 0.0;
    unsigned int seed = 1u;
    const double ultrametric_tol = 1e-3;

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--trees") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                goto cleanup;
            }
            trees_file = argv[++i];
        }
        else if (strcmp(argv[i], "--outprefix") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                goto cleanup;
            }
            outprefix = argv[++i];
        }
        else if (strcmp(argv[i], "--k") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                goto cleanup;
            }
            k = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--n-genes") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                goto cleanup;
            }
            n_genes = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--sigma2-obs") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                goto cleanup;
            }
            sigma2_obs = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--use-n-trees") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                goto cleanup;
            }
            use_n_trees = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--sigma2-latent") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                goto cleanup;
            }
            if (parse_double_list(argv[++i], &sigma2_latent_raw, &n_sigma_latent_raw) != 0) {
                fprintf(stderr, "ERROR: failed to parse --sigma2-latent values.\n");
                goto cleanup;
            }
        }
        else if (strcmp(argv[i], "--seed") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                goto cleanup;
            }
            seed = (unsigned int)strtoul(argv[++i], NULL, 10);
        }
        else {
            usage(argv[0]);
            goto cleanup;
        }
    }

    if (trees_file == NULL || outprefix == NULL || k <= 0 || 
        n_genes <= 0 || sigma2_obs < 0.0 || n_sigma_latent_raw <= 0) {
        usage(argv[0]);
        goto cleanup;
    }

    /* Read in trees to use for the simulation */
    trees = gex_read_nexus(trees_file, &n_trees);
    if (trees == NULL || n_trees <= 0) {
        fprintf(stderr, "ERROR: failed to load trees from %s\n", trees_file);
        goto cleanup;
    }
    if (gex_check_trees_ultrametric(trees, n_trees, ultrametric_tol) != 0) {
        fprintf(stderr, "ERROR: simulator requires ultrametric trees.\n");
        goto cleanup;
    }

    /* Get cell names and number of cells from the first tree 
    (assuming all trees have the same tips from the nexus file) */
    List *leaf_list = tr_leaf_names(trees[0]);
    if (leaf_list == NULL || lst_size(leaf_list) <= 0) {
        fprintf(stderr, "ERROR: failed to get leaf names from first tree.\n");
        goto cleanup;
    }
    n_cells = lst_size(leaf_list);

    /* Convert leaf list to cell names array*/
    cell_names = (char **)calloc(n_cells, sizeof(char *));
    if (cell_names == NULL) {
        fprintf(stderr, "ERROR: failed to allocate cell names.\n");
        goto cleanup;
    }
    for (i = 0; i < n_cells; i++) {
        String *s = lst_get_ptr(leaf_list, i);
        if (s == NULL) {
            fprintf(stderr, "ERROR: failed to get leaf name %d.\n", i);
            goto cleanup;
        }
        cell_names[i] = strdup(s->chars);
        if (cell_names[i] == NULL) {
            fprintf(stderr, "ERROR: failed to allocate cell name.\n");
            goto cleanup;
        }
    }
    lst_free_strings(leaf_list);
    lst_free(leaf_list);

    if (use_n_trees < -1) {
        fprintf(stderr, "ERROR: --use-n-trees must be -1, 0, or a positive integer\n");
        goto cleanup;
    }
    if (use_n_trees > n_trees) {
        fprintf(stderr, "ERROR: --use-n-trees (%d) cannot exceed the number of loaded trees (%d)\n",
                use_n_trees, n_trees);
        goto cleanup;
    }
    selected_n_trees = (use_n_trees == 0 ? n_trees : use_n_trees);
    /* Allocate and initialize the latent variance parameters */
    sigma2_latent = (double *)calloc((size_t)k, sizeof(double));
    if (sigma2_latent == NULL)
        goto cleanup;
    if (n_sigma_latent_raw == 1) {
        /* Use one variance value for all latent dimensions. */
        for (i = 0; i < k; i++)
            sigma2_latent[i] = sigma2_latent_raw[0];
    }
    else if (n_sigma_latent_raw == k) {
        /* Copy each provided variance for each latent dimension */
        for (i = 0; i < k; i++)
            sigma2_latent[i] = sigma2_latent_raw[i];
    }
    else {
        fprintf(stderr, "ERROR: --sigma2-latent must provide either 1 value or exactly k=%d values.\n", k);
        goto cleanup;
    }

    Sigmas = (Matrix **)calloc(n_trees, sizeof(Matrix *));
    if (Sigmas == NULL) {
        fprintf(stderr, "ERROR: failed to allocate Brownian covariance matrix list.\n");
        goto cleanup;
    }
    for (i = 0; i < n_trees; i++) {
        Sigmas[i] = covariance_from_tree(trees[i], cell_names, n_cells);
        if (Sigmas[i] == NULL) {
            fprintf(stderr, "ERROR: failed to compute Brownian covariance matrix for tree %d.\n", i + 1);
            goto cleanup;
        }
    }

    if (use_n_trees == -1) {
        Sigma = gex_average_tree_covariance(trees, n_trees, cell_names, n_cells);
        if (Sigma == NULL) {
            fprintf(stderr, "ERROR: failed to compute the average Brownian covariance matrix.\n");
            goto cleanup;
        }
        avg_Sigmas[0] = Sigma;
        sim_Sigmas = avg_Sigmas;
        n_sim_sigmas = 1;
    } else {
        sim_Sigmas = Sigmas;
        n_sim_sigmas = (use_n_trees == 0 ? n_trees : use_n_trees);
    }

    for (i = 0; i < n_sim_sigmas; i++) {
        /* Simulate the latent factors matrix Z from Brownian motion given
        the input latent factors sigma2 values */
        per_sim_Z = brownian_simulate_expression_from_covariance(sim_Sigmas[i],
                                                        cell_names,
                                                        n_cells,
                                                        k,
                                                        sigma2_latent,
                                                        n_sigma_latent_raw,
                                                        seed + (unsigned int)(104729u * i));

        if (per_sim_Z == NULL) {
            fprintf(stderr, "ERROR: failed to simulate latent factors from Brownian covariance for tree %d.\n", i + 1);
            goto cleanup;
        }
        
        if (i == 0) {
            /* Take the first simulation as is */
            Z = per_sim_Z;
            per_sim_Z = NULL;
        } else {
            /* Add in place for subsequent simulations */
            if (add_matrix_in_place(Z->X, per_sim_Z->X) != 0) {
                fprintf(stderr, "ERROR: failed to accumulate simulated latent factor matrices\n");
                goto cleanup;
            }
            gex_free_matrix_data(per_sim_Z);
            per_sim_Z = NULL;
        }
    }

    /* Scale the latent factors matrix Z to complete the expectation over the simulated matrices */
    if (Z == NULL || Z->X == NULL ||
        scale_matrix_in_place(Z->X, 1.0 / (double)n_sim_sigmas) != 0) {
        fprintf(stderr, "ERROR: failed to finalize simulated latent factor matrix Z.\n");
        goto cleanup;
    }

    /* Use the expected latent factors matrix Z to simulate L and the expression matrix X
    for the input parameters. */
    if (gex_simulate_from_latent_factors(Z, cell_names, n_cells, k, n_genes, sigma2_obs, seed + 7919u, &L, &gex) != 0) {
        fprintf(stderr, "ERROR: failed to simulate expression matrix from latent factors.\n");
        goto cleanup;
    }

    /* Write the simulated data to output files */
    if (gex_write_model(outprefix, gex, L->X, Z->X, gex->cell_names, gex->gene_names, k, sigma2_obs, sigma2_latent) != 0) {
        fprintf(stderr, "ERROR: failed to write simulation outputs.\n");
        goto cleanup;
    }

    /* Log simulation run settings and progress to the terminal */
    printf("Simulated latent Brownian expression for %d cells, %d genes, k=%d.\n",
           n_cells, n_genes, k);
    if (use_n_trees == -1)
        printf("Simulation mode: average covariance across all %d tree(s).\n", n_trees);
    else if (selected_n_trees == n_trees)
        printf("Simulation mode: expectation over all %d tree(s).\n", n_trees);
    else
        printf("Simulation mode: expectation over the first %d tree(s).\n", selected_n_trees);

    printf("Wrote simulated data files with prefix %s.*\n", outprefix);

    printf("Done.\n");
    status = 0; /* Success */

cleanup:
    if (expr_path != NULL)
        free(expr_path);
    if (sigma2_latent_raw != NULL)
        free(sigma2_latent_raw);
    if (sigma2_latent != NULL)
        free(sigma2_latent);
    if (per_sim_Z != NULL)
        gex_free_matrix_data(per_sim_Z);
    if (Z != NULL)
        gex_free_matrix_data(Z);
    if (L != NULL)
        gex_free_matrix_data(L);
    if (gex != NULL)
        gex_free_matrix_data(gex);
    if (Sigma != NULL)
        mat_free(Sigma);
    if (Sigmas != NULL) {
        for (i = 0; i < n_trees; i++)
            mat_free(Sigmas[i]);
        free(Sigmas);
    }
    if (cell_names != NULL) {
        for (i = 0; i < n_cells; i++)
            free(cell_names[i]);
        free(cell_names);
    }
    if (trees != NULL)
        gex_free_trees(trees, n_trees);
    return status;
}
