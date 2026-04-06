#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "brownian.h"
#include "gex.h"
#include "gex_sim.h"

static char *gexsimprog_strdup(const char *s) {
    size_t n;
    char *out;

    if (s == NULL)
        return NULL;
    n = strlen(s);
    out = (char *)malloc(n + 1);
    if (out == NULL)
        return NULL;
    memcpy(out, s, n + 1);
    return out;
}

static int parse_sigma_latent_values(const char *spec,
                                     double **values_io,
                                     int *n_values_io) {
    char *copy = NULL;
    char *tok = NULL;
    char *saveptr = NULL;
    int n = 0;

    if (spec == NULL || values_io == NULL || n_values_io == NULL)
        return -1;

    copy = gexsimprog_strdup(spec);
    if (copy == NULL)
        return -1;

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
    return (n > 0 ? 0 : -1);
}

static void usage(const char *progname) {
    fprintf(stderr,
            "Usage: %s "
            "--trees <trees.nex> "
            "--outprefix <prefix> "
            "--k <int> "
            "--n-genes <int> "
            "[--sigma-obs <float>] "
            "[--sigma-latent <comma-list>] "
            "[--sigma-latent <float> ...] "
            "[--seed <int>]\n",
            progname);
}

int main(int argc, char *argv[]) {
    const char *trees_file = NULL;
    const char *outprefix = NULL;
    TreeNode **trees = NULL;
    char **cell_names = NULL;
    char *expr_path = NULL;
    Matrix *Sigma = NULL;
    GexSimulationTruth *truth = NULL;
    GexMatrix *gex = NULL;
    double *sigma2_latent_raw = NULL;
    double *sigma2_latent = NULL;
    int n_trees = 0;
    int n_cells = 0;
    int k = 0;
    int n_genes = 0;
    int n_sigma_latent_raw = 0;
    int i;
    int status = 1;
    double sigma2_obs = 0.05;
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
        else if (strcmp(argv[i], "--sigma-obs") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                goto cleanup;
            }
            sigma2_obs = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--sigma-latent") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                goto cleanup;
            }
            if (parse_sigma_latent_values(argv[++i], &sigma2_latent_raw, &n_sigma_latent_raw) != 0) {
                fprintf(stderr, "ERROR: failed to parse --sigma-latent values.\n");
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

    if (trees_file == NULL || outprefix == NULL || k <= 0 || n_genes <= 0 || sigma2_obs < 0.0) {
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

    if (gex_get_shared_leaf_names(trees, n_trees, &cell_names, &n_cells) != 0 || n_cells <= 0) {
        fprintf(stderr, "ERROR: failed to derive shared leaf names across the input trees.\n");
        goto cleanup;
    }

    /* Allocate and initialize the latent variance parameters */
    sigma2_latent = (double *)calloc(k, sizeof(double));
    if (sigma2_latent == NULL)
        goto cleanup;
    if (n_sigma_latent_raw == 0) {
        for (i = 0; i < k; i++)
            sigma2_latent[i] = 1.0;
    }
    else if (n_sigma_latent_raw == 1) {
        for (i = 0; i < k; i++)
            sigma2_latent[i] = sigma2_latent_raw[0];
    }
    else if (n_sigma_latent_raw == k) {
        for (i = 0; i < k; i++)
            sigma2_latent[i] = sigma2_latent_raw[i];
    }
    else {
        fprintf(stderr, "ERROR: --sigma-latent must provide either 1 value or exactly k=%d values.\n", k);
        goto cleanup;
    }

    /* Compute the average covariance matrix across the input trees to use
    for the simulation */
    Sigma = gex_average_tree_covariance(trees, n_trees, cell_names, n_cells);
    if (Sigma == NULL) {
        fprintf(stderr, "ERROR: failed to compute the average Brownian covariance matrix.\n");
        goto cleanup;
    }

    /* Simulate the latent expression and observed expression matrix */
    truth = gex_simulate_latent_brownian_expression(Sigma, cell_names, n_cells, k, n_genes,
                                                    sigma2_latent, sigma2_obs, seed, &gex);
    if (truth == NULL || gex == NULL) {
        fprintf(stderr, "ERROR: failed to simulate latent Brownian expression.\n");
        goto cleanup;
    }

    expr_path = (char *)malloc(strlen(outprefix) + 16u);
    if (expr_path == NULL)
        goto cleanup;
    snprintf(expr_path, strlen(outprefix) + 16u, "%s.expr.tsv", outprefix);

    if (gex_write_labeled_matrix_tsv(expr_path, gex->X, gex->cell_names, gex->n_cells,
                                     gex->gene_names, gex->n_genes, "cell") != 0) {
        fprintf(stderr, "ERROR: failed to write simulated expression matrix.\n");
        goto cleanup;
    }

    if (gex_write_simulation_truth(outprefix, truth, gex->cell_names, gex->gene_names) != 0) {
        fprintf(stderr, "ERROR: failed to write simulation truth outputs.\n");
        goto cleanup;
    }

    printf("Simulated latent Brownian expression for %d cells, %d genes, k=%d.\n",
           n_cells, n_genes, k);
    printf("Wrote expression matrix to %s\n", expr_path);
    printf("Wrote truth files with prefix %s.truth.*\n", outprefix);
    status = 0;

cleanup:
    if (expr_path != NULL)
        free(expr_path);
    if (sigma2_latent_raw != NULL)
        free(sigma2_latent_raw);
    if (sigma2_latent != NULL)
        free(sigma2_latent);
    if (truth != NULL)
        gex_free_simulation_truth(truth);
    if (gex != NULL)
        gex_free_matrix_data(gex);
    if (Sigma != NULL)
        mat_free(Sigma);
    if (cell_names != NULL) {
        for (i = 0; i < n_cells; i++)
            free(cell_names[i]);
        free(cell_names);
    }
    if (trees != NULL)
        gex_free_trees(trees, n_trees);
    return status;
}
