#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <phast/eigen.h>

#include "gex.h"

typedef struct {
    int n_cells;
    int n_genes;
    int k;
    double sigma_obs;
    double *sigma2_latent;
} GexEvalSummary;

typedef struct {
    double value;
} DescValue;

static int cmp_desc_value(const void *a, const void *b) {
    const DescValue *da = (const DescValue *)a;
    const DescValue *db = (const DescValue *)b;

    if (da->value > db->value) return -1;
    if (da->value < db->value) return 1;
    return 0;
}

static char *gexeval_strdup(const char *s) {
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

static int gexeval_find_name(char **names, int n, const char *target) {
    int i;

    if (names == NULL || target == NULL)
        return -1;
    for (i = 0; i < n; i++) {
        if (names[i] != NULL && strcmp(names[i], target) == 0)
            return i;
    }
    return -1;
}

static GexMatrix *gexeval_subset_rows_by_names(GexMatrix *src,
                                               char **target_names,
                                               int n_target) {
    int i, j;
    GexMatrix *out = NULL;

    if (src == NULL || target_names == NULL || n_target <= 0)
        return NULL;
    out = (GexMatrix *)calloc(1, sizeof(GexMatrix));
    if (out == NULL)
        return NULL;
    out->n_cells = n_target;
    out->n_genes = src->n_genes;
    out->X = mat_new(n_target, src->n_genes);
    out->cell_names = (char **)calloc(n_target, sizeof(char *));
    out->gene_names = (char **)calloc(src->n_genes, sizeof(char *));
    if (out->X == NULL || out->cell_names == NULL || out->gene_names == NULL) {
        gex_free_matrix_data(out);
        return NULL;
    }

    for (j = 0; j < src->n_genes; j++) {
        out->gene_names[j] = gexeval_strdup(src->gene_names[j]);
        if (out->gene_names[j] == NULL) {
            gex_free_matrix_data(out);
            return NULL;
        }
    }

    for (i = 0; i < n_target; i++) {
        int src_idx = gexeval_find_name(src->cell_names, src->n_cells, target_names[i]);
        if (src_idx < 0) {
            gex_free_matrix_data(out);
            return NULL;
        }
        out->cell_names[i] = gexeval_strdup(target_names[i]);
        if (out->cell_names[i] == NULL) {
            gex_free_matrix_data(out);
            return NULL;
        }
        for (j = 0; j < src->n_genes; j++)
            mat_set(out->X, i, j, mat_get(src->X, src_idx, j));
    }

    return out;
}

static GexMatrix *gexeval_subset_cols_by_names(GexMatrix *src,
                                               char **target_names,
                                               int n_target) {
    int i, j;
    GexMatrix *out = NULL;

    if (src == NULL || target_names == NULL || n_target <= 0)
        return NULL;
    out = (GexMatrix *)calloc(1, sizeof(GexMatrix));
    if (out == NULL)
        return NULL;
    out->n_cells = src->n_cells;
    out->n_genes = n_target;
    out->X = mat_new(src->n_cells, n_target);
    out->cell_names = (char **)calloc(src->n_cells, sizeof(char *));
    out->gene_names = (char **)calloc(n_target, sizeof(char *));
    if (out->X == NULL || out->cell_names == NULL || out->gene_names == NULL) {
        gex_free_matrix_data(out);
        return NULL;
    }

    for (i = 0; i < src->n_cells; i++) {
        out->cell_names[i] = gexeval_strdup(src->cell_names[i]);
        if (out->cell_names[i] == NULL) {
            gex_free_matrix_data(out);
            return NULL;
        }
    }

    for (j = 0; j < n_target; j++) {
        int src_idx = gexeval_find_name(src->gene_names, src->n_genes, target_names[j]);
        if (src_idx < 0) {
            gex_free_matrix_data(out);
            return NULL;
        }
        out->gene_names[j] = gexeval_strdup(target_names[j]);
        if (out->gene_names[j] == NULL) {
            gex_free_matrix_data(out);
            return NULL;
        }
        for (i = 0; i < src->n_cells; i++)
            mat_set(out->X, i, j, mat_get(src->X, i, src_idx));
    }

    return out;
}

static int gexeval_collect_common_names(char **names_ref,
                                        int n_ref,
                                        char **names_other,
                                        int n_other,
                                        char ***common_out,
                                        int *n_common_out) {
    int i;
    int n_common = 0;
    char **common = NULL;

    if (common_out == NULL || n_common_out == NULL)
        return -1;
    *common_out = NULL;
    *n_common_out = 0;

    common = (char **)calloc(n_ref, sizeof(char *));
    if (common == NULL)
        return -1;

    for (i = 0; i < n_ref; i++) {
        if (gexeval_find_name(names_other, n_other, names_ref[i]) >= 0) {
            common[n_common] = gexeval_strdup(names_ref[i]);
            if (common[n_common] == NULL) {
                int j;
                for (j = 0; j < n_common; j++)
                    free(common[j]);
                free(common);
                return -1;
            }
            n_common++;
        }
    }

    if (n_common <= 0) {
        free(common);
        return -1;
    }

    *common_out = common;
    *n_common_out = n_common;
    return 0;
}

static void gexeval_free_names(char **names, int n) {
    int i;

    if (names == NULL)
        return;
    for (i = 0; i < n; i++)
        free(names[i]);
    free(names);
}

static Matrix *gexeval_center_columns(Matrix *X) {
    int i, j;
    Matrix *Xc = NULL;

    if (X == NULL)
        return NULL;
    Xc = mat_new(X->nrows, X->ncols);
    if (Xc == NULL)
        return NULL;

    for (j = 0; j < X->ncols; j++) {
        double mean = 0.0;
        for (i = 0; i < X->nrows; i++)
            mean += mat_get(X, i, j);
        mean /= (double)X->nrows;
        for (i = 0; i < X->nrows; i++)
            mat_set(Xc, i, j, mat_get(X, i, j) - mean);
    }

    return Xc;
}

static Matrix *gexeval_signal_matrix(Matrix *Z, Matrix *L) {
    Matrix *signal = NULL;

    if (Z == NULL || L == NULL || Z->ncols != L->nrows)
        return NULL;
    signal = mat_new(Z->nrows, L->ncols);
    if (signal == NULL)
        return NULL;
    mat_mult(signal, Z, L);
    return signal;
}

static Matrix *gexeval_compute_cell_covariance(Matrix *Z) {
    int i, j, d;
    Matrix *Zc = NULL;
    Matrix *cov = NULL;
    double denom;

    if (Z == NULL)
        return NULL;
    Zc = gexeval_center_columns(Z);
    cov = mat_new(Z->nrows, Z->nrows);
    if (Zc == NULL || cov == NULL) {
        if (Zc != NULL) mat_free(Zc);
        if (cov != NULL) mat_free(cov);
        return NULL;
    }

    denom = (Z->ncols > 1 ? (double)Z->ncols : 1.0);
    for (i = 0; i < Z->nrows; i++) {
        for (j = 0; j < Z->nrows; j++) {
            double sum = 0.0;
            for (d = 0; d < Z->ncols; d++)
                sum += mat_get(Zc, i, d) * mat_get(Zc, j, d);
            mat_set(cov, i, j, sum / denom);
        }
    }

    mat_free(Zc);
    return cov;
}

static Matrix *gexeval_compute_gene_covariance(Matrix *signal) {
    int i, g1, g2;
    Matrix *Xc = NULL;
    Matrix *cov = NULL;
    double denom;

    if (signal == NULL)
        return NULL;
    Xc = gexeval_center_columns(signal);
    cov = mat_new(signal->ncols, signal->ncols);
    if (Xc == NULL || cov == NULL) {
        if (Xc != NULL) mat_free(Xc);
        if (cov != NULL) mat_free(cov);
        return NULL;
    }

    denom = (signal->nrows > 1 ? (double)signal->nrows : 1.0);
    for (g1 = 0; g1 < signal->ncols; g1++) {
        for (g2 = 0; g2 < signal->ncols; g2++) {
            double sum = 0.0;
            for (i = 0; i < signal->nrows; i++)
                sum += mat_get(Xc, i, g1) * mat_get(Xc, i, g2);
            mat_set(cov, g1, g2, sum / denom);
        }
    }

    mat_free(Xc);
    return cov;
}

static double gexeval_matrix_relative_error(Matrix *truth, Matrix *fit) {
    int i, j;
    double num = 0.0;
    double den = 0.0;

    if (truth == NULL || fit == NULL || truth->nrows != fit->nrows || truth->ncols != fit->ncols)
        return HUGE_VAL;
    for (i = 0; i < truth->nrows; i++) {
        for (j = 0; j < truth->ncols; j++) {
            double a = mat_get(truth, i, j);
            double b = mat_get(fit, i, j);
            num += (a - b) * (a - b);
            den += a * a;
        }
    }
    if (den <= 0.0)
        return HUGE_VAL;
    return sqrt(num / den);
}

static double gexeval_matrix_correlation(Matrix *A, Matrix *B) {
    int i, j;
    int n = 0;
    double mean_a = 0.0;
    double mean_b = 0.0;
    double num = 0.0;
    double den_a = 0.0;
    double den_b = 0.0;

    if (A == NULL || B == NULL || A->nrows != B->nrows || A->ncols != B->ncols)
        return -2.0;

    n = A->nrows * A->ncols;
    if (n <= 1)
        return -2.0;

    for (i = 0; i < A->nrows; i++) {
        for (j = 0; j < A->ncols; j++) {
            mean_a += mat_get(A, i, j);
            mean_b += mat_get(B, i, j);
        }
    }
    mean_a /= (double)n;
    mean_b /= (double)n;

    for (i = 0; i < A->nrows; i++) {
        for (j = 0; j < A->ncols; j++) {
            double da = mat_get(A, i, j) - mean_a;
            double db = mat_get(B, i, j) - mean_b;
            num += da * db;
            den_a += da * da;
            den_b += db * db;
        }
    }

    if (den_a <= 0.0 || den_b <= 0.0)
        return -2.0;
    return num / sqrt(den_a * den_b);
}

static Matrix *gexeval_orthonormal_basis(Matrix *X, int *rank_out) {
    int i, j, qcol;
    Matrix *Xc = NULL;
    Matrix *Q = NULL;
    int rank = 0;

    if (rank_out == NULL || X == NULL)
        return NULL;
    *rank_out = 0;

    Xc = gexeval_center_columns(X);
    Q = mat_new(X->nrows, X->ncols);
    if (Xc == NULL || Q == NULL) {
        if (Xc != NULL) mat_free(Xc);
        if (Q != NULL) mat_free(Q);
        return NULL;
    }
    mat_zero(Q);

    for (j = 0; j < Xc->ncols; j++) {
        double norm2 = 0.0;
        for (i = 0; i < Xc->nrows; i++)
            mat_set(Q, i, rank, mat_get(Xc, i, j));

        for (qcol = 0; qcol < rank; qcol++) {
            double dot = 0.0;
            for (i = 0; i < Xc->nrows; i++)
                dot += mat_get(Q, i, rank) * mat_get(Q, i, qcol);
            for (i = 0; i < Xc->nrows; i++)
                mat_set(Q, i, rank, mat_get(Q, i, rank) - dot * mat_get(Q, i, qcol));
        }

        for (i = 0; i < Xc->nrows; i++) {
            double v = mat_get(Q, i, rank);
            norm2 += v * v;
        }
        if (norm2 > 1e-10) {
            double inv_norm = 1.0 / sqrt(norm2);
            for (i = 0; i < Xc->nrows; i++)
                mat_set(Q, i, rank, inv_norm * mat_get(Q, i, rank));
            rank++;
        }
    }

    mat_free(Xc);
    if (rank == 0) {
        mat_free(Q);
        return NULL;
    }
    mat_resize(Q, X->nrows, rank);
    *rank_out = rank;
    return Q;
}

static double gexeval_latent_subspace_similarity(Matrix *Z_true, Matrix *Z_fit) {
    int r_true = 0;
    int r_fit = 0;
    int i, j, k;
    Matrix *Q_true = NULL;
    Matrix *Q_fit = NULL;
    Matrix *cross = NULL;
    Matrix *gram = NULL;
    Matrix *gram_copy = NULL;
    Vector *evals = NULL;
    Matrix *evecs = NULL;
    double overlap = -1.0;
    int denom_rank;

    if (Z_true == NULL || Z_fit == NULL || Z_true->nrows != Z_fit->nrows)
        return -1.0;

    Q_true = gexeval_orthonormal_basis(Z_true, &r_true);
    Q_fit = gexeval_orthonormal_basis(Z_fit, &r_fit);
    if (Q_true == NULL || Q_fit == NULL)
        goto cleanup;

    cross = mat_new(r_true, r_fit);
    gram = mat_new(r_fit, r_fit);
    gram_copy = mat_new(r_fit, r_fit);
    evals = vec_new(r_fit);
    evecs = mat_new(r_fit, r_fit);
    if (cross == NULL || gram == NULL || gram_copy == NULL || evals == NULL || evecs == NULL)
        goto cleanup;

    for (i = 0; i < r_true; i++) {
        for (j = 0; j < r_fit; j++) {
            double sum = 0.0;
            for (k = 0; k < Z_true->nrows; k++)
                sum += mat_get(Q_true, k, i) * mat_get(Q_fit, k, j);
            mat_set(cross, i, j, sum);
        }
    }

    for (i = 0; i < r_fit; i++) {
        for (j = 0; j < r_fit; j++) {
            double sum = 0.0;
            for (k = 0; k < r_true; k++)
                sum += mat_get(cross, k, i) * mat_get(cross, k, j);
            mat_set(gram, i, j, sum);
            mat_set(gram_copy, i, j, sum);
        }
    }

    if (mat_diagonalize_sym(gram_copy, evals, evecs) != 0)
        goto cleanup;

    denom_rank = (r_true < r_fit ? r_true : r_fit);
    if (denom_rank <= 0)
        goto cleanup;

    overlap = 0.0;
    for (i = 0; i < r_fit; i++) {
        double lambda = vec_get(evals, i);
        if (lambda < 0.0)
            lambda = 0.0;
        if (lambda > 1.0)
            lambda = 1.0;
        overlap += lambda;
    }
    overlap /= (double)denom_rank;

cleanup:
    if (Q_true != NULL) mat_free(Q_true);
    if (Q_fit != NULL) mat_free(Q_fit);
    if (cross != NULL) mat_free(cross);
    if (gram != NULL) mat_free(gram);
    if (gram_copy != NULL) mat_free(gram_copy);
    if (evals != NULL) vec_free(evals);
    if (evecs != NULL) mat_free(evecs);
    return overlap;
}

static double *gexeval_factor_contributions(Matrix *L,
                                            const double *sigma2_latent,
                                            int k) {
    int d, j;
    double *vals = NULL;

    if (L == NULL || sigma2_latent == NULL || L->nrows != k)
        return NULL;
    vals = (double *)calloc(k, sizeof(double));
    if (vals == NULL)
        return NULL;

    for (d = 0; d < k; d++) {
        double row_ss = 0.0;
        for (j = 0; j < L->ncols; j++) {
            double v = mat_get(L, d, j);
            row_ss += v * v;
        }
        vals[d] = sigma2_latent[d] * row_ss;
    }
    return vals;
}

static void gexeval_sort_desc(double *vals, int n) {
    int i;
    DescValue *tmp = NULL;

    if (vals == NULL || n <= 0)
        return;
    tmp = (DescValue *)calloc(n, sizeof(DescValue));
    if (tmp == NULL)
        return;
    for (i = 0; i < n; i++)
        tmp[i].value = vals[i];
    qsort(tmp, n, sizeof(DescValue), cmp_desc_value);
    for (i = 0; i < n; i++)
        vals[i] = tmp[i].value;
    free(tmp);
}

static double gexeval_vector_correlation(const double *a, const double *b, int n) {
    int i;
    double mean_a = 0.0;
    double mean_b = 0.0;
    double num = 0.0;
    double den_a = 0.0;
    double den_b = 0.0;

    if (a == NULL || b == NULL || n <= 1)
        return -2.0;

    for (i = 0; i < n; i++) {
        mean_a += a[i];
        mean_b += b[i];
    }
    mean_a /= (double)n;
    mean_b /= (double)n;

    for (i = 0; i < n; i++) {
        double da = a[i] - mean_a;
        double db = b[i] - mean_b;
        num += da * db;
        den_a += da * da;
        den_b += db * db;
    }
    if (den_a <= 0.0 || den_b <= 0.0)
        return -2.0;
    return num / sqrt(den_a * den_b);
}

static double gexeval_vector_relative_error(const double *a, const double *b, int n) {
    int i;
    double num = 0.0;
    double den = 0.0;

    if (a == NULL || b == NULL || n <= 0)
        return HUGE_VAL;
    for (i = 0; i < n; i++) {
        num += (a[i] - b[i]) * (a[i] - b[i]);
        den += a[i] * a[i];
    }
    if (den <= 0.0)
        return HUGE_VAL;
    return sqrt(num / den);
}

static GexEvalSummary *gexeval_read_summary(const char *filename) {
    char line[4096];
    FILE *in = NULL;
    GexEvalSummary *summary = NULL;

    if (filename == NULL)
        return NULL;
    in = fopen(filename, "r");
    if (in == NULL)
        return NULL;

    summary = (GexEvalSummary *)calloc(1, sizeof(GexEvalSummary));
    if (summary == NULL) {
        fclose(in);
        return NULL;
    }

    while (fgets(line, sizeof(line), in) != NULL) {
        char *tab = strchr(line, '\t');
        char *key = line;
        char *value = NULL;
        if (tab == NULL)
            continue;
        *tab = '\0';
        value = tab + 1;
        value[strcspn(value, "\r\n")] = '\0';
        if (strcmp(key, "parameter") == 0)
            continue;
        if (strcmp(key, "n_cells") == 0)
            summary->n_cells = atoi(value);
        else if (strcmp(key, "n_genes") == 0)
            summary->n_genes = atoi(value);
        else if (strcmp(key, "k") == 0) {
            int old_k = summary->k;
            summary->k = atoi(value);
            if (summary->k > 0 && summary->k != old_k) {
                double *tmp = (double *)realloc(summary->sigma2_latent,
                                                (size_t)summary->k * sizeof(double));
                if (tmp == NULL) {
                    fclose(in);
                    free(summary->sigma2_latent);
                    free(summary);
                    return NULL;
                }
                summary->sigma2_latent = tmp;
                memset(summary->sigma2_latent, 0, (size_t)summary->k * sizeof(double));
            }
        }
        else if (strcmp(key, "sigma_obs") == 0)
            summary->sigma_obs = atof(value);
        else if (strncmp(key, "sigma_latent_LF", 15) == 0) {
            int idx = atoi(key + 15) - 1;
            if (idx >= 0 && idx < summary->k && summary->sigma2_latent != NULL)
                summary->sigma2_latent[idx] = atof(value);
        }
    }

    fclose(in);
    return summary;
}

static void gexeval_free_summary(GexEvalSummary *summary) {
    if (summary == NULL)
        return;
    if (summary->sigma2_latent != NULL)
        free(summary->sigma2_latent);
    free(summary);
}

static void usage(const char *progname) {
    fprintf(stderr,
            "Usage: %s "
            "--truth-prefix <prefix> "
            "--fit-prefix <prefix> "
            "--outprefix <prefix>\n",
            progname);
}

int main(int argc, char *argv[]) {
    const char *truth_prefix = NULL;
    const char *fit_prefix = NULL;
    const char *outprefix = NULL;
    char truth_summary_path[4096];
    char truth_z_path[4096];
    char truth_l_path[4096];
    char fit_summary_path[4096];
    char fit_z_path[4096];
    char fit_l_path[4096];
    char eval_summary_path[4096];
    GexEvalSummary *truth_summary = NULL;
    GexEvalSummary *fit_summary = NULL;
    GexMatrix *truth_Z = NULL;
    GexMatrix *truth_L = NULL;
    GexMatrix *fit_Z = NULL;
    GexMatrix *fit_L = NULL;
    GexMatrix *truth_Z_aligned = NULL;
    GexMatrix *fit_Z_aligned = NULL;
    GexMatrix *truth_L_common = NULL;
    GexMatrix *fit_L_common = NULL;
    Matrix *truth_latent_cov = NULL;
    Matrix *fit_latent_cov = NULL;
    Matrix *truth_gene_cov = NULL;
    Matrix *fit_gene_cov = NULL;
    Matrix *truth_signal = NULL;
    Matrix *fit_signal = NULL;
    double latent_subspace_similarity;
    double cell_cov_corr;
    double cell_cov_rel_error;
    double gene_cov_corr;
    double gene_cov_rel_error;
    double sigma_obs_rel_error;
    double variance_trend_corr = -2.0;
    double variance_trend_rel_error = HUGE_VAL;
    double *truth_contrib = NULL;
    double *fit_contrib = NULL;
    char **common_cells = NULL;
    char **common_genes = NULL;
    int n_common_cells = 0;
    int n_common_genes = 0;
    int i;
    int status = 1;
    FILE *out = NULL;

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--truth-prefix") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            truth_prefix = argv[++i];
        }
        else if (strcmp(argv[i], "--fit-prefix") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            fit_prefix = argv[++i];
        }
        else if (strcmp(argv[i], "--outprefix") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            outprefix = argv[++i];
        }
        else {
            usage(argv[0]);
            return 1;
        }
    }

    if (truth_prefix == NULL || fit_prefix == NULL || outprefix == NULL) {
        usage(argv[0]);
        return 1;
    }

    /* Use relative paths for the comparison based on the input prefixes */
    snprintf(truth_summary_path, sizeof(truth_summary_path), "%s.truth.summary.tsv", truth_prefix);
    snprintf(truth_z_path, sizeof(truth_z_path), "%s.truth.Z.tsv", truth_prefix);
    snprintf(truth_l_path, sizeof(truth_l_path), "%s.truth.L.tsv", truth_prefix);
    snprintf(fit_summary_path, sizeof(fit_summary_path), "%s.model.summary.tsv", fit_prefix);
    snprintf(fit_z_path, sizeof(fit_z_path), "%s.model.Z.tsv", fit_prefix);
    snprintf(fit_l_path, sizeof(fit_l_path), "%s.model.L.tsv", fit_prefix);
    snprintf(eval_summary_path, sizeof(eval_summary_path), "%s.eval.summary.tsv", outprefix);

    truth_summary = gexeval_read_summary(truth_summary_path);
    fit_summary = gexeval_read_summary(fit_summary_path);
    truth_Z = gex_read_labeled_matrix(truth_z_path);
    truth_L = gex_read_labeled_matrix(truth_l_path);
    fit_Z = gex_read_labeled_matrix(fit_z_path);
    fit_L = gex_read_labeled_matrix(fit_l_path);
    if (truth_summary == NULL || fit_summary == NULL || truth_Z == NULL || truth_L == NULL ||
        fit_Z == NULL || fit_L == NULL) {
        fprintf(stderr, "ERROR: failed to read required truth/model files.\n");
        goto cleanup;
    }

    /* Derive the set of common cells and genes between the truth and fitted outputs 
    so that we are not comparing model fits on the subset of genes simulated. */
    if (gexeval_collect_common_names(fit_Z->cell_names, fit_Z->n_cells,
                                     truth_Z->cell_names, truth_Z->n_cells,
                                     &common_cells, &n_common_cells) != 0 ||
        gexeval_collect_common_names(fit_L->gene_names, fit_L->n_genes,
                                     truth_L->gene_names, truth_L->n_genes,
                                     &common_genes, &n_common_genes) != 0) {
        fprintf(stderr, "ERROR: failed to derive shared cells/genes between truth and fitted outputs.\n");
        goto cleanup;
    }

    truth_Z_aligned = gexeval_subset_rows_by_names(truth_Z, common_cells, n_common_cells);
    fit_Z_aligned = gexeval_subset_rows_by_names(fit_Z, common_cells, n_common_cells);
    truth_L_common = gexeval_subset_cols_by_names(truth_L, common_genes, n_common_genes);
    fit_L_common = gexeval_subset_cols_by_names(fit_L, common_genes, n_common_genes);
    if (truth_Z_aligned == NULL || fit_Z_aligned == NULL || truth_L_common == NULL || fit_L_common == NULL) {
        fprintf(stderr, "ERROR: failed to align truth and fitted matrices on shared names.\n");
        goto cleanup;
    }

    truth_signal = gexeval_signal_matrix(truth_Z_aligned->X, truth_L_common->X);
    fit_signal = gexeval_signal_matrix(fit_Z_aligned->X, fit_L_common->X);
    truth_latent_cov = gexeval_compute_cell_covariance(truth_Z_aligned->X);
    fit_latent_cov = gexeval_compute_cell_covariance(fit_Z_aligned->X);
    truth_gene_cov = gexeval_compute_gene_covariance(truth_signal);
    fit_gene_cov = gexeval_compute_gene_covariance(fit_signal);
    if (truth_signal == NULL || fit_signal == NULL || truth_latent_cov == NULL ||
        fit_latent_cov == NULL || truth_gene_cov == NULL || fit_gene_cov == NULL) {
        fprintf(stderr, "ERROR: failed to derive fitted covariance summaries.\n");
        goto cleanup;
    }

    latent_subspace_similarity = gexeval_latent_subspace_similarity(truth_Z_aligned->X, fit_Z_aligned->X);
    cell_cov_corr = gexeval_matrix_correlation(truth_latent_cov, fit_latent_cov);
    cell_cov_rel_error = gexeval_matrix_relative_error(truth_latent_cov, fit_latent_cov);
    gene_cov_corr = gexeval_matrix_correlation(truth_gene_cov, fit_gene_cov);
    gene_cov_rel_error = gexeval_matrix_relative_error(truth_gene_cov, fit_gene_cov);
    sigma_obs_rel_error = fabs(fit_summary->sigma_obs - truth_summary->sigma_obs) /
                          (fabs(truth_summary->sigma_obs) > 1e-12 ? fabs(truth_summary->sigma_obs) : 1.0);

    truth_contrib = gexeval_factor_contributions(truth_L->X, truth_summary->sigma2_latent, truth_summary->k);
    fit_contrib = gexeval_factor_contributions(fit_L->X, fit_summary->sigma2_latent, fit_summary->k);
    if (truth_contrib != NULL && fit_contrib != NULL) {
        int n_compare = (truth_summary->k < fit_summary->k ? truth_summary->k : fit_summary->k);
        gexeval_sort_desc(truth_contrib, truth_summary->k);
        gexeval_sort_desc(fit_contrib, fit_summary->k);
        variance_trend_corr = gexeval_vector_correlation(truth_contrib, fit_contrib, n_compare);
        variance_trend_rel_error = gexeval_vector_relative_error(truth_contrib, fit_contrib, n_compare);
    }

    out = fopen(eval_summary_path, "w");
    if (out == NULL) {
        fprintf(stderr, "ERROR: failed to open evaluator output file.\n");
        goto cleanup;
    }

    fprintf(out, "metric\tvalue\n");
    fprintf(out, "k_true\t%d\n", truth_summary->k);
    fprintf(out, "k_fit\t%d\n", fit_summary->k);
    fprintf(out, "k_abs_diff\t%d\n", abs(fit_summary->k - truth_summary->k));
    fprintf(out, "latent_subspace_similarity\t%.17g\n", latent_subspace_similarity);
    fprintf(out, "cell_cov_correlation\t%.17g\n", cell_cov_corr);
    fprintf(out, "cell_cov_relative_error\t%.17g\n", cell_cov_rel_error);
    fprintf(out, "gene_cov_correlation\t%.17g\n", gene_cov_corr);
    fprintf(out, "gene_cov_relative_error\t%.17g\n", gene_cov_rel_error);
    fprintf(out, "sigma_obs_true\t%.17g\n", truth_summary->sigma_obs);
    fprintf(out, "sigma_obs_fit\t%.17g\n", fit_summary->sigma_obs);
    fprintf(out, "sigma_obs_relative_error\t%.17g\n", sigma_obs_rel_error);
    fprintf(out, "latent_variance_trend_correlation\t%.17g\n", variance_trend_corr);
    fprintf(out, "latent_variance_trend_relative_error\t%.17g\n", variance_trend_rel_error);
    fclose(out);
    out = NULL;

    printf("Evaluated simulated recovery.\n");
    printf("latent_subspace_similarity=%.4f\n", latent_subspace_similarity);
    printf("cell_cov_correlation=%.4f gene_cov_correlation=%.4f\n",
           cell_cov_corr, gene_cov_corr);
    printf("Wrote evaluation summary to %s\n", eval_summary_path);
    status = 0;

cleanup:
    if (out != NULL)
        fclose(out);
    if (truth_contrib != NULL)
        free(truth_contrib);
    if (fit_contrib != NULL)
        free(fit_contrib);
    if (common_cells != NULL)
        gexeval_free_names(common_cells, n_common_cells);
    if (common_genes != NULL)
        gexeval_free_names(common_genes, n_common_genes);
    if (truth_signal != NULL)
        mat_free(truth_signal);
    if (fit_signal != NULL)
        mat_free(fit_signal);
    if (truth_latent_cov != NULL)
        mat_free(truth_latent_cov);
    if (fit_latent_cov != NULL)
        mat_free(fit_latent_cov);
    if (truth_gene_cov != NULL)
        mat_free(truth_gene_cov);
    if (fit_gene_cov != NULL)
        mat_free(fit_gene_cov);
    if (truth_summary != NULL)
        gexeval_free_summary(truth_summary);
    if (fit_summary != NULL)
        gexeval_free_summary(fit_summary);
    if (truth_Z != NULL)
        gex_free_matrix_data(truth_Z);
    if (truth_L != NULL)
        gex_free_matrix_data(truth_L);
    if (fit_Z != NULL)
        gex_free_matrix_data(fit_Z);
    if (fit_L != NULL)
        gex_free_matrix_data(fit_L);
    if (truth_Z_aligned != NULL)
        gex_free_matrix_data(truth_Z_aligned);
    if (fit_Z_aligned != NULL)
        gex_free_matrix_data(fit_Z_aligned);
    if (truth_L_common != NULL)
        gex_free_matrix_data(truth_L_common);
    if (fit_L_common != NULL)
        gex_free_matrix_data(fit_L_common);
    return status;
}
