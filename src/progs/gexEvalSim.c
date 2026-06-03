#include "gexmatrix.h"
#include "parser.h"

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
    Matrix *F_factor_corr = NULL;
    Matrix *L_factor_corr = NULL;

    double z_rmse = -1.0;
    double x_rmse = -1.0;
    double cell_cov_corr = -2.0;
    double gene_cov_corr = -2.0;
    double F_col_mean_abs_corr = -1.0;
    double L_row_mean_abs_corr = -1.0;
    double F_col_mean_corr = -2.0;
    double L_row_mean_corr = -2.0;

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
    F_factor_corr = mat_factor_pearson_correlation(sim_F->X, fit_F->X, 0, 1);
    F_col_mean_abs_corr = mat_diag_mean(F_factor_corr);
    mat_free(F_factor_corr);
    F_factor_corr = mat_factor_pearson_correlation(sim_F->X, fit_F->X, 0, 0);
    F_col_mean_corr = mat_diag_mean(F_factor_corr);

    /* Compare learned factors in L */
    L_factor_corr = mat_factor_pearson_correlation(sim_L->X, fit_L->X, 1, 1);
    L_row_mean_abs_corr = mat_diag_mean(L_factor_corr);
    mat_free(L_factor_corr);
    L_factor_corr = mat_factor_pearson_correlation(sim_L->X, fit_L->X, 1, 0);
    L_row_mean_corr = mat_diag_mean(L_factor_corr);

    out = fopen(eval_summary_path, "w");
    fprintf(out, "metric\tvalue\n");
    fprintf(out, "z_rmse\t%.17g\n", z_rmse);
    fprintf(out, "x_rmse\t%.17g\n", x_rmse);
    fprintf(out, "cell_cov_correlation\t%.17g\n", cell_cov_corr);
    fprintf(out, "gene_cov_correlation\t%.17g\n", gene_cov_corr);
    fprintf(out, "F_col_mean_abs_corr\t%.17g\n", F_col_mean_abs_corr);
    fprintf(out, "L_row_mean_abs_corr\t%.17g\n", L_row_mean_abs_corr);
    fprintf(out, "F_col_mean_corr\t%.17g\n", F_col_mean_corr);
    fprintf(out, "L_row_mean_corr\t%.17g\n", L_row_mean_corr);
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
    if (F_factor_corr != NULL)
        mat_free(F_factor_corr);
    if (L_factor_corr != NULL)
        mat_free(L_factor_corr);
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
