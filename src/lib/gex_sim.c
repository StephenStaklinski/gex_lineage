#include "gex_sim.h"

#include "brownian.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <phast/eigen.h>
#include <phast/lists.h>
#include <phast/stringsplus.h>

static unsigned int gexsim_rand_u32(unsigned int *state) {
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static double gexsim_uniform_open(unsigned int *state) {
    return ((double)gexsim_rand_u32(state) + 1.0) / 4294967297.0;
}

static double gexsim_rand_normal(unsigned int *state) {
    double u1 = gexsim_uniform_open(state);
    double u2 = gexsim_uniform_open(state);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

static void gexsim_free_string_array(char **names, int n) {
    int i;

    if (names == NULL)
        return;
    for (i = 0; i < n; i++)
        free(names[i]);
    free(names);
}

static int gexsim_name_in_list(const char *name, List *names) {
    int i;

    if (name == NULL || names == NULL)
        return 0;
    for (i = 0; i < lst_size(names); i++) {
        String *s = lst_get_ptr(names, i);
        if (s != NULL && strcmp(name, s->chars) == 0)
            return 1;
    }
    return 0;
}

static char **gexsim_make_factor_names(int k) {
    int i;
    char **names = NULL;

    if (k <= 0)
        return NULL;
    names = (char **)calloc(k, sizeof(char *));
    if (names == NULL)
        return NULL;

    for (i = 0; i < k; i++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "LF%d", i + 1);
        names[i] = strdup(buf);
        if (names[i] == NULL) {
            gexsim_free_string_array(names, i);
            return NULL;
        }
    }

    return names;
}

static char **gexsim_make_gene_names(int n_genes) {
    int j;
    char **names = NULL;

    if (n_genes <= 0)
        return NULL;
    names = (char **)calloc(n_genes, sizeof(char *));
    if (names == NULL)
        return NULL;

    for (j = 0; j < n_genes; j++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "sim_gene_%04d", j + 1);
        names[j] = strdup(buf);
        if (names[j] == NULL) {
            gexsim_free_string_array(names, j);
            return NULL;
        }
    }

    return names;
}

int gex_get_shared_leaf_names(TreeNode **trees,
                              int n_trees,
                              char ***names_out,
                              int *n_names_out) {
    int i, j;
    List *first = NULL;
    List **name_lists = NULL;
    char **names = NULL;
    int names_capacity = 0;
    int n_keep = 0;
    int status = -1;

    if (names_out == NULL || n_names_out == NULL || trees == NULL || n_trees <= 0)
        return -1;

    *names_out = NULL;
    *n_names_out = 0;

    name_lists = (List **)calloc(n_trees, sizeof(List *));
    if (name_lists == NULL)
        return -1;

    for (i = 0; i < n_trees; i++) {
        if (trees[i] == NULL)
            goto cleanup;
        name_lists[i] = tr_leaf_names(trees[i]);
        if (name_lists[i] == NULL)
            goto cleanup;
    }

    first = name_lists[0];
    if (first == NULL || lst_size(first) <= 0)
        goto cleanup;

    names_capacity = lst_size(first);
    names = (char **)calloc(names_capacity, sizeof(char *));
    if (names == NULL)
        goto cleanup;

    for (i = 0; i < lst_size(first); i++) {
        String *s = lst_get_ptr(first, i);
        int present_in_all = 1;
        if (s == NULL)
            continue;
        for (j = 1; j < n_trees; j++) {
            if (!gexsim_name_in_list(s->chars, name_lists[j])) {
                present_in_all = 0;
                break;
            }
        }
        if (present_in_all) {
            names[n_keep] = strdup(s->chars);
            if (names[n_keep] == NULL)
                goto cleanup;
            n_keep++;
        }
    }

    if (n_keep <= 0)
        goto cleanup;

    *names_out = names;
    *n_names_out = n_keep;
    names = NULL;
    status = 0;

cleanup:
    if (name_lists != NULL) {
        for (i = 0; i < n_trees; i++) {
            if (name_lists[i] != NULL) {
                lst_free_strings(name_lists[i]);
                lst_free(name_lists[i]);
            }
        }
        free(name_lists);
    }
    if (names != NULL)
        gexsim_free_string_array(names, names_capacity);
    return status;
}

/* Simulate latent Brownian expression under the provided parameters and covariance matrix. */
GexSimulationTruth *gex_simulate_latent_brownian_expression(Matrix *Sigma,
                                                            char **cell_names,
                                                            int n_cells,
                                                            int k,
                                                            int n_genes,
                                                            const double *sigma2_latent,
                                                            double sigma2_obs,
                                                            unsigned int seed,
                                                            GexMatrix **gex_out) {
    int i, j, d;    /* Loop indices */
    GexSimulationTruth *truth = NULL; /* Simulation truth outputs */
    GexMatrix *gex = NULL;  /* Simulation output gene expression object */
    Matrix *Sigma_reg = NULL;   /* Regularized covariance matrix */
    Matrix *chol = NULL;    /* Cholesky factor */
    Matrix *signal = NULL;  /* Gene expression matrix */
    double *std_normals = NULL; /* Standard normal random variables */
    char **gene_names = NULL;   /* Gene names */
    unsigned int rng_state = (seed == 0u ? 1u : seed);  /* Random number generator state */
    double max_diag = 0.0;  /* Maximum diagonal element of the covariance matrix */
    double jitter;  /* Jitter for regularization */

    if (gex_out == NULL || Sigma == NULL || cell_names == NULL || n_cells <= 0 ||
        k <= 0 || n_genes <= 0 || sigma2_latent == NULL || sigma2_obs < 0.0)
        return NULL;
    if (Sigma->nrows != n_cells || Sigma->ncols != n_cells)
        return NULL;

    *gex_out = NULL;

    /* Allocate objects in memory */
    truth = (GexSimulationTruth *)calloc(1, sizeof(GexSimulationTruth));
    gex = (GexMatrix *)calloc(1, sizeof(GexMatrix));
    Sigma_reg = mat_create_copy(Sigma);
    chol = mat_new(n_cells, n_cells);
    signal = mat_new(n_cells, n_genes);
    gene_names = gexsim_make_gene_names(n_genes);
    if (truth == NULL || gex == NULL || Sigma_reg == NULL || chol == NULL ||
        signal == NULL || gene_names == NULL)
        goto cleanup;

    /* Apply relative jitter to the diagonal of the covariance matrix to regularize it 
    for numerical stability. */
    for (i = 0; i < n_cells; i++) {
        double diag = mat_get(Sigma, i, i);
        if (diag > max_diag)
            max_diag = diag;
    }
    jitter = (max_diag > 0.0 ? 1e-8 * max_diag : 1e-8);
    for (i = 0; i < n_cells; i++)
        mat_set(Sigma_reg, i, i, mat_get(Sigma_reg, i, i) + jitter);

    /* Compute the Cholesky factor of the regularized covariance matrix. */
    if (mat_cholesky(chol, Sigma_reg) != 0)
        goto cleanup;

    /* Initialize the simulation truth outputs */
    truth->n_cells = n_cells;
    truth->n_genes = n_genes;
    truth->k = k;
    truth->Z = mat_new(n_cells, k);
    truth->L = mat_new(k, n_genes);
    truth->sigma2_latent = (double *)calloc(k, sizeof(double));
    truth->sigma2_obs = sigma2_obs;
    if (truth->Z == NULL || truth->L == NULL || truth->sigma2_latent == NULL)
        goto cleanup;

    /* Set the latent variance parameters */
    for (d = 0; d < k; d++)
        truth->sigma2_latent[d] = sigma2_latent[d];

    /* Generate each latent dimension as a Gaussian random vector with
    covariance sigma2_latent[d] * Sigma_reg, using the Cholesky factor
    of the regularized phylogenetic covariance matrix. This is meant to 
    simulate a factorized Brownian motion model. */
    for (d = 0; d < k; d++) {
        double scale = sqrt(sigma2_latent[d] > 0.0 ? sigma2_latent[d] : 0.0);
        std_normals = (double *)calloc(n_cells, sizeof(double));
        if (std_normals == NULL)
            goto cleanup;
        for (j = 0; j < n_cells; j++)
            std_normals[j] = gexsim_rand_normal(&rng_state);
        for (i = 0; i < n_cells; i++) {
            double sum = 0.0;
            for (j = 0; j <= i; j++)
                sum += mat_get(chol, i, j) * std_normals[j];
            mat_set(truth->Z, i, d, scale * sum);
        }
        free(std_normals);
        std_normals = NULL;
    }

    /* Draw gene loadings L ~ N(0,1) and rescale each row to have norm
    sqrt(n_genes / k), ensuring each latent dimension contributes
    equal expected magnitude to the noiseless gene expression data. */
    for (d = 0; d < k; d++) {
        double row_ss = 0.0;
        double target_norm = sqrt((double)n_genes / (double)k);
        for (j = 0; j < n_genes; j++) {
            double val = gexsim_rand_normal(&rng_state);
            mat_set(truth->L, d, j, val);
            row_ss += val * val;
        }
        if (row_ss > 0.0) {
            double row_scale = target_norm / sqrt(row_ss);
            for (j = 0; j < n_genes; j++)
                mat_set(truth->L, d, j, row_scale * mat_get(truth->L, d, j));
        }
    }

    /* Compute the noiseless expression matrix from the 
    simulated Z and L matrix factorization. */
    mat_mult(signal, truth->Z, truth->L);

    /* Initialize the simulated gene expression matrix from the noiseless expression matrix */
    gex->n_cells = n_cells;
    gex->n_genes = n_genes;
    gex->X = mat_new(n_cells, n_genes);
    gex->cell_names = (char **)calloc(n_cells, sizeof(char *));
    gex->gene_names = (char **)calloc(n_genes, sizeof(char *));
    if (gex->X == NULL || gex->cell_names == NULL || gex->gene_names == NULL)
        goto cleanup;

    for (i = 0; i < n_cells; i++) {
        gex->cell_names[i] = strdup(cell_names[i]);
        if (gex->cell_names[i] == NULL)
            goto cleanup;
    }
    for (j = 0; j < n_genes; j++) {
        gex->gene_names[j] = gene_names[j];
        gene_names[j] = NULL;
    }

    /* Add noise to the noiseless expression matrix based on 
    the sigma2_obs parameter input. */
    for (i = 0; i < n_cells; i++) {
        for (j = 0; j < n_genes; j++) {
            double val = mat_get(signal, i, j);
            if (sigma2_obs > 0.0)
                val += sqrt(sigma2_obs) * gexsim_rand_normal(&rng_state);
            mat_set(gex->X, i, j, val);
        }
    }

    *gex_out = gex;
    gex = NULL;
    mat_free(Sigma_reg);
    mat_free(chol);
    mat_free(signal);
    gexsim_free_string_array(gene_names, n_genes);
    return truth;

cleanup:
    if (gene_names != NULL)
        gexsim_free_string_array(gene_names, n_genes);
    if (std_normals != NULL)
        free(std_normals);
    if (signal != NULL)
        mat_free(signal);
    if (chol != NULL)
        mat_free(chol);
    if (Sigma_reg != NULL)
        mat_free(Sigma_reg);
    if (gex != NULL)
        gex_free_matrix_data(gex);
    if (truth != NULL)
        gex_free_simulation_truth(truth);
    return NULL;
}

int gex_write_labeled_matrix_tsv(const char *filename,
                                 Matrix *X,
                                 char **row_names,
                                 int n_rows,
                                 char **col_names,
                                 int n_cols,
                                 const char *corner_label) {
    int i, j;
    FILE *out = NULL;

    if (filename == NULL || X == NULL || row_names == NULL || col_names == NULL ||
        n_rows <= 0 || n_cols <= 0 || X->nrows != n_rows || X->ncols != n_cols)
        return -1;

    out = fopen(filename, "w");
    if (out == NULL)
        return -1;

    fprintf(out, "%s", corner_label == NULL ? "id" : corner_label);
    for (j = 0; j < n_cols; j++)
        fprintf(out, "\t%s", col_names[j]);
    fprintf(out, "\n");

    for (i = 0; i < n_rows; i++) {
        fprintf(out, "%s", row_names[i]);
        for (j = 0; j < n_cols; j++)
            fprintf(out, "\t%.17g", mat_get(X, i, j));
        fprintf(out, "\n");
    }

    fclose(out);
    return 0;
}

int gex_write_simulation_truth(const char *outprefix,
                               GexSimulationTruth *truth,
                               char **cell_names,
                               char **gene_names) {
    char summary_path[4096];
    char z_path[4096];
    char l_path[4096];
    char **factor_names = NULL;
    FILE *summary_out = NULL;
    int j;
    int status = -1;

    if (outprefix == NULL || truth == NULL || cell_names == NULL || gene_names == NULL)
        return -1;

    factor_names = gexsim_make_factor_names(truth->k);
    if (factor_names == NULL)
        return -1;

    snprintf(summary_path, sizeof(summary_path), "%s.truth.summary.tsv", outprefix);
    snprintf(z_path, sizeof(z_path), "%s.truth.Z.tsv", outprefix);
    snprintf(l_path, sizeof(l_path), "%s.truth.L.tsv", outprefix);

    summary_out = fopen(summary_path, "w");
    if (summary_out == NULL)
        goto cleanup;
    fprintf(summary_out, "parameter\tvalue\n");
    fprintf(summary_out, "n_cells\t%d\n", truth->n_cells);
    fprintf(summary_out, "n_genes\t%d\n", truth->n_genes);
    fprintf(summary_out, "k\t%d\n", truth->k);
    fprintf(summary_out, "sigma_obs\t%.17g\n", truth->sigma2_obs);
    for (j = 0; j < truth->k; j++)
        fprintf(summary_out, "sigma_latent_LF%d\t%.17g\n", j + 1, truth->sigma2_latent[j]);
    fclose(summary_out);
    summary_out = NULL;

    if (gex_write_labeled_matrix_tsv(z_path, truth->Z, cell_names, truth->n_cells,
                                     factor_names, truth->k, "cell") != 0)
        goto cleanup;
    if (gex_write_labeled_matrix_tsv(l_path, truth->L, factor_names, truth->k,
                                     gene_names, truth->n_genes, "factor") != 0)
        goto cleanup;

    status = 0;

cleanup:
    if (summary_out != NULL)
        fclose(summary_out);
    gexsim_free_string_array(factor_names, truth->k);
    return status;
}

void gex_free_simulation_truth(GexSimulationTruth *truth) {
    if (truth == NULL)
        return;
    if (truth->Z != NULL)
        mat_free(truth->Z);
    if (truth->L != NULL)
        mat_free(truth->L);
    if (truth->sigma2_latent != NULL)
        free(truth->sigma2_latent);
    free(truth);
}
