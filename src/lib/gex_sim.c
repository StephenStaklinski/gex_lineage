#include "gex_sim.h"

#include "brownian.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <phast/eigen.h>
#include <phast/lists.h>
#include <phast/stringsplus.h>

static char *gexsim_strdup(const char *s) {
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

static void gexsim_center_columns(Matrix *src, Matrix *dest) {
    int i, j;

    for (j = 0; j < src->ncols; j++) {
        double mean = 0.0;
        for (i = 0; i < src->nrows; i++)
            mean += mat_get(src, i, j);
        mean /= (double)src->nrows;
        for (i = 0; i < src->nrows; i++)
            mat_set(dest, i, j, mat_get(src, i, j) - mean);
    }
}

static Matrix *gexsim_compute_cell_covariance(Matrix *X) {
    int i, j, g;
    Matrix *Xc = NULL;
    Matrix *cov = NULL;
    double denom;

    if (X == NULL || X->nrows <= 0 || X->ncols <= 0)
        return NULL;

    Xc = mat_new(X->nrows, X->ncols);
    cov = mat_new(X->nrows, X->nrows);
    if (Xc == NULL || cov == NULL) {
        if (Xc != NULL) mat_free(Xc);
        if (cov != NULL) mat_free(cov);
        return NULL;
    }

    gexsim_center_columns(X, Xc);
    denom = (X->ncols > 1 ? (double)X->ncols : 1.0);
    for (i = 0; i < X->nrows; i++) {
        for (j = 0; j < X->nrows; j++) {
            double sum = 0.0;
            for (g = 0; g < X->ncols; g++)
                sum += mat_get(Xc, i, g) * mat_get(Xc, j, g);
            mat_set(cov, i, j, sum / denom);
        }
    }

    mat_free(Xc);
    return cov;
}

static Matrix *gexsim_compute_gene_covariance(Matrix *X) {
    int i, g1, g2;
    Matrix *Xc = NULL;
    Matrix *cov = NULL;
    double denom;

    if (X == NULL || X->nrows <= 0 || X->ncols <= 0)
        return NULL;

    Xc = mat_new(X->nrows, X->ncols);
    cov = mat_new(X->ncols, X->ncols);
    if (Xc == NULL || cov == NULL) {
        if (Xc != NULL) mat_free(Xc);
        if (cov != NULL) mat_free(cov);
        return NULL;
    }

    gexsim_center_columns(X, Xc);
    denom = (X->nrows > 1 ? (double)X->nrows : 1.0);
    for (g1 = 0; g1 < X->ncols; g1++) {
        for (g2 = 0; g2 < X->ncols; g2++) {
            double sum = 0.0;
            for (i = 0; i < X->nrows; i++)
                sum += mat_get(Xc, i, g1) * mat_get(Xc, i, g2);
            mat_set(cov, g1, g2, sum / denom);
        }
    }

    mat_free(Xc);
    return cov;
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
        names[i] = gexsim_strdup(buf);
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
        names[j] = gexsim_strdup(buf);
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
            names[n_keep] = gexsim_strdup(s->chars);
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

GexSimulationTruth *gex_simulate_latent_brownian_expression(Matrix *Sigma,
                                                            char **cell_names,
                                                            int n_cells,
                                                            int k,
                                                            int n_genes,
                                                            const double *sigma2_latent,
                                                            double sigma2_obs,
                                                            unsigned int seed,
                                                            GexMatrix **gex_out) {
    int i, j, d;
    GexSimulationTruth *truth = NULL;
    GexMatrix *gex = NULL;
    Matrix *Sigma_reg = NULL;
    Matrix *chol = NULL;
    Matrix *signal = NULL;
    double *std_normals = NULL;
    char **gene_names = NULL;
    unsigned int rng_state = (seed == 0u ? 1u : seed);
    double max_diag = 0.0;
    double jitter;

    if (gex_out == NULL || Sigma == NULL || cell_names == NULL || n_cells <= 0 ||
        k <= 0 || n_genes <= 0 || sigma2_latent == NULL || sigma2_obs < 0.0)
        return NULL;
    if (Sigma->nrows != n_cells || Sigma->ncols != n_cells)
        return NULL;

    *gex_out = NULL;

    truth = (GexSimulationTruth *)calloc(1, sizeof(GexSimulationTruth));
    gex = (GexMatrix *)calloc(1, sizeof(GexMatrix));
    Sigma_reg = mat_create_copy(Sigma);
    chol = mat_new(n_cells, n_cells);
    signal = mat_new(n_cells, n_genes);
    gene_names = gexsim_make_gene_names(n_genes);
    if (truth == NULL || gex == NULL || Sigma_reg == NULL || chol == NULL ||
        signal == NULL || gene_names == NULL)
        goto cleanup;

    for (i = 0; i < n_cells; i++) {
        double diag = mat_get(Sigma, i, i);
        if (diag > max_diag)
            max_diag = diag;
    }
    jitter = (max_diag > 0.0 ? 1e-8 * max_diag : 1e-8);
    for (i = 0; i < n_cells; i++)
        mat_set(Sigma_reg, i, i, mat_get(Sigma_reg, i, i) + jitter);
    if (mat_cholesky(chol, Sigma_reg) != 0)
        goto cleanup;

    truth->n_cells = n_cells;
    truth->n_genes = n_genes;
    truth->k = k;
    truth->Z = mat_new(n_cells, k);
    truth->L = mat_new(k, n_genes);
    truth->sigma2_latent = (double *)calloc(k, sizeof(double));
    truth->sigma2_obs = sigma2_obs;
    if (truth->Z == NULL || truth->L == NULL || truth->sigma2_latent == NULL)
        goto cleanup;

    for (d = 0; d < k; d++)
        truth->sigma2_latent[d] = sigma2_latent[d];

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

    mat_mult(signal, truth->Z, truth->L);

    gex->n_cells = n_cells;
    gex->n_genes = n_genes;
    gex->X = mat_new(n_cells, n_genes);
    gex->cell_names = (char **)calloc(n_cells, sizeof(char *));
    gex->gene_names = (char **)calloc(n_genes, sizeof(char *));
    if (gex->X == NULL || gex->cell_names == NULL || gex->gene_names == NULL)
        goto cleanup;

    for (i = 0; i < n_cells; i++) {
        gex->cell_names[i] = gexsim_strdup(cell_names[i]);
        if (gex->cell_names[i] == NULL)
            goto cleanup;
    }
    for (j = 0; j < n_genes; j++) {
        gex->gene_names[j] = gene_names[j];
        gene_names[j] = NULL;
    }

    for (i = 0; i < n_cells; i++) {
        for (j = 0; j < n_genes; j++) {
            double val = mat_get(signal, i, j);
            if (sigma2_obs > 0.0)
                val += sqrt(sigma2_obs) * gexsim_rand_normal(&rng_state);
            mat_set(gex->X, i, j, val);
        }
    }

    truth->latent_cov = gexsim_compute_cell_covariance(truth->Z);
    truth->gene_cov = gexsim_compute_gene_covariance(signal);
    if (truth->latent_cov == NULL || truth->gene_cov == NULL)
        goto cleanup;

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
    char latent_cov_path[4096];
    char gene_cov_path[4096];
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
    snprintf(latent_cov_path, sizeof(latent_cov_path), "%s.truth.latent_cov.tsv", outprefix);
    snprintf(gene_cov_path, sizeof(gene_cov_path), "%s.truth.gene_cov.tsv", outprefix);

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
    if (gex_write_labeled_matrix_tsv(latent_cov_path, truth->latent_cov, cell_names,
                                     truth->n_cells, cell_names, truth->n_cells,
                                     "cell") != 0)
        goto cleanup;
    if (gex_write_labeled_matrix_tsv(gene_cov_path, truth->gene_cov, gene_names,
                                     truth->n_genes, gene_names, truth->n_genes,
                                     "gene") != 0)
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
    if (truth->latent_cov != NULL)
        mat_free(truth->latent_cov);
    if (truth->gene_cov != NULL)
        mat_free(truth->gene_cov);
    if (truth->sigma2_latent != NULL)
        free(truth->sigma2_latent);
    free(truth);
}
