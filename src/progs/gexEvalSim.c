#include "gexmatrix.h"
#include "gexparser.h"

#include <phast/matrix.h>
#include <phast/misc.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *progname) {
    fprintf(stderr,
            "Usage: %s "
            "--sim-prefix <prefix> "
            "--fit-prefix <prefix> "
            "--outprefix <prefix>\n",
            progname);
}

/* Absolute Pearson correlation between row r1 of A and row r2 of B.
   Used to compare rows of L across genes. */
static double vec_abs_pearson_correlation(Vector *A, Vector *B) {
    int j;
    int n = A->size;
    double mean_a = 0.0;
    double mean_b = 0.0;
    double num = 0.0;
    double den_a = 0.0;
    double den_b = 0.0;

    /* Compute row means */
    for (j = 0; j < n; j++) {
        mean_a += vec_get(A, j);
        mean_b += vec_get(B, j);
    }
    mean_a /= (double)n;
    mean_b /= (double)n;

    /* Compute Pearson correlation numerator and denominator */
    for (j = 0; j < n; j++) {
        double da = vec_get(A, j) - mean_a;
        double db = vec_get(B, j) - mean_b;
        num += da * db;
        den_a += da * da;
        den_b += db * db;
    }
    num /= sqrt(den_a * den_b);

    return fabs(num);
}

/* Mean Pearson correlation between cols of F_true and F_fit. */
static double F_col_mean_correlation(Matrix *F_true, Matrix *F_fit) {
    int i;
    int n = F_true->ncols;
    double score = 0.0;

    for (i = 0; i < n; i++) {
        score += vec_abs_pearson_correlation(mat_get_col(F_true, i), mat_get_col(F_fit, i));
    }

    return score / (double)n;
}

/* Mean Pearson correlation between rows of L_true and L_fit. */
static double L_row_mean_correlation(Matrix *L_true, Matrix *L_fit) {
    int i;
    int n = L_true->nrows;
    double score = 0.0;

    for (i = 0; i < n; i++) {
        score += vec_abs_pearson_correlation(mat_get_row(L_true, i), mat_get_row(L_fit, i));
    }

    return score / (double)n;
}

int main(int argc, char *argv[]) {
    const char *sim_prefix = NULL;
    const char *fit_prefix = NULL;
    const char *outprefix = NULL;

    char sim_f_path[4096];
    char sim_l_path[4096];
    char sim_x_path[4096];
    char fit_f_path[4096];
    char fit_l_path[4096];
    char fit_x_path[4096];
    char eval_summary_path[4096];

    GexMatrix *sim_F = NULL;
    Matrix *sim_Fc = NULL;
    Matrix *sim_cell_cov = NULL;
    GexMatrix *sim_L = NULL;
    Matrix *sim_Z = NULL;
    GexMatrix *sim_X = NULL;
    GexMatrix *fit_F = NULL;
    Matrix *fit_Fc = NULL;
    Matrix *fit_cell_cov = NULL;
    GexMatrix *fit_L = NULL;
    Matrix *fit_Z = NULL;
    GexMatrix *fit_X = NULL;

    double z_rmse = -1.0;
    double x_rmse = -1.0;
    double cell_cov_corr = -2.0;
    double gene_cov_corr = -2.0;
    double F_col_mean_abs_corr = -1.0;
    double L_row_mean_abs_corr = -1.0;

    FILE *out = NULL;
    int i;

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--sim-prefix") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            sim_prefix = argv[++i];
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

    if (sim_prefix == NULL || fit_prefix == NULL || outprefix == NULL) {
        usage(argv[0]);
        return 1;
    }

    snprintf(sim_f_path, sizeof(sim_f_path), "%s.F.tsv", sim_prefix);
    snprintf(sim_l_path, sizeof(sim_l_path), "%s.L.tsv", sim_prefix);
    snprintf(sim_x_path, sizeof(sim_x_path), "%s.X.tsv", sim_prefix);
    snprintf(fit_f_path, sizeof(fit_f_path), "%s.F.tsv", fit_prefix);
    snprintf(fit_l_path, sizeof(fit_l_path), "%s.L.tsv", fit_prefix);
    snprintf(fit_x_path, sizeof(fit_x_path), "%s.X.tsv", fit_prefix);
    snprintf(eval_summary_path, sizeof(eval_summary_path), "%s.summary.tsv", outprefix);

    sim_F = read_gex_matrix(sim_f_path);
    sim_L = read_gex_matrix(sim_l_path);
    sim_X = read_gex_matrix(sim_x_path);
    fit_F = read_gex_matrix(fit_f_path);
    fit_L = read_gex_matrix(fit_l_path);
    fit_X = read_gex_matrix(fit_x_path);

    /* RMSE of Z = F * L */
    sim_Z = mat_new(sim_F->X->nrows, sim_L->X->ncols);
    fit_Z = mat_new(fit_F->X->nrows, fit_L->X->ncols);
    mat_mult(sim_Z, sim_F->X, sim_L->X);
    mat_mult(fit_Z, fit_F->X, fit_L->X);
    z_rmse = mat_rmse(sim_Z, fit_Z);

    /* RMSE of X */
    x_rmse = mat_rmse(sim_X->X, fit_X->X);

    /* Cell-cell covariance induced by F */
    sim_Fc = mat_create_copy(sim_F->X);
    mat_center_rows(sim_Fc);
    sim_cell_cov = mat_centered_row_cov(sim_Fc);

    fit_Fc = mat_create_copy(fit_F->X);
    mat_center_rows(fit_Fc);
    fit_cell_cov = mat_centered_row_cov(fit_Fc);

    cell_cov_corr = mat_pearson_correlation(sim_cell_cov, fit_cell_cov);

    /* Gene-gene covariance induced by L */
    sim_cell_cov = mat_create_copy(sim_L->X);
    mat_center_cols(sim_cell_cov);
    sim_cell_cov = mat_centered_col_cov(sim_cell_cov);

    fit_cell_cov = mat_create_copy(fit_L->X);
    mat_center_cols(fit_cell_cov);
    fit_cell_cov = mat_centered_col_cov(fit_cell_cov);

    gene_cov_corr = mat_pearson_correlation(sim_cell_cov, fit_cell_cov);

    /* Compare learned factors in F */
    F_col_mean_abs_corr = F_col_mean_correlation(sim_F->X, fit_F->X);

    /* Compare learned factors in L */
    L_row_mean_abs_corr = L_row_mean_correlation(sim_L->X, fit_L->X);

    out = fopen(eval_summary_path, "w");
    fprintf(out, "metric\tvalue\n");
    fprintf(out, "z_rmse\t%.17g\n", z_rmse);
    fprintf(out, "x_rmse\t%.17g\n", x_rmse);
    fprintf(out, "cell_cov_correlation\t%.17g\n", cell_cov_corr);
    fprintf(out, "gene_cov_correlation\t%.17g\n", gene_cov_corr);
    fprintf(out, "F_col_mean_abs_corr\t%.17g\n", F_col_mean_abs_corr);
    fprintf(out, "L_row_mean_abs_corr\t%.17g\n", L_row_mean_abs_corr);
    fclose(out);
    out = NULL;

    if (sim_Z != NULL)
        mat_free(sim_Z);
    if (fit_Z != NULL)
        mat_free(fit_Z);
    if (sim_cell_cov != NULL)
        mat_free(sim_cell_cov);
    if (fit_cell_cov != NULL)
        mat_free(fit_cell_cov);
    if (sim_F != NULL)
        gex_free_matrix_data(sim_F);
    if (sim_L != NULL)
        gex_free_matrix_data(sim_L);
    if (fit_F != NULL)
        gex_free_matrix_data(fit_F);
    if (fit_L != NULL)
        gex_free_matrix_data(fit_L);
    if (sim_X != NULL)
        gex_free_matrix_data(sim_X);
    if (fit_X != NULL)
        gex_free_matrix_data(fit_X);

    return 0;
}