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
static double gexeval_row_abs_correlation(Matrix *A, int r1, Matrix *B, int r2) {
    int j;
    int n;
    double mean_a = 0.0;
    double mean_b = 0.0;
    double num = 0.0;
    double den_a = 0.0;
    double den_b = 0.0;

    if (A == NULL || B == NULL || A->ncols != B->ncols ||
        r1 < 0 || r1 >= A->nrows || r2 < 0 || r2 >= B->nrows)
        return -2.0;

    n = A->ncols;
    if (n <= 1)
        return -2.0;

    for (j = 0; j < n; j++) {
        mean_a += mat_get(A, r1, j);
        mean_b += mat_get(B, r2, j);
    }
    mean_a /= (double)n;
    mean_b /= (double)n;

    for (j = 0; j < n; j++) {
        double da = mat_get(A, r1, j) - mean_a;
        double db = mat_get(B, r2, j) - mean_b;
        num += da * db;
        den_a += da * da;
        den_b += db * db;
    }

    if (den_a <= 0.0 || den_b <= 0.0)
        return -2.0;

    num /= sqrt(den_a * den_b);
    if (num < 0.0)
        num = -num;

    return num;
}

/* Greedy matching score for rows of L.
Matches each true factor to one fitted factor using max absolute row correlation.
Returns mean matched abs correlation in [0,1]. */
static double greedy_L_match_score(Matrix *L_true, Matrix *L_fit) {
    int i, j, k;
    int n_match;
    int *used_true = NULL;
    int *used_fit = NULL;
    double score = -1.0;

    if (L_true == NULL || L_fit == NULL || L_true->ncols != L_fit->ncols)
        return -1.0;

    n_match = (L_true->nrows < L_fit->nrows ? L_true->nrows : L_fit->nrows);
    if (n_match <= 0)
        return -1.0;

    used_true = scalloc(L_true->nrows, sizeof(int));
    used_fit = scalloc(L_fit->nrows, sizeof(int));
    if (used_true == NULL || used_fit == NULL) {
        if (used_true != NULL) free(used_true);
        if (used_fit != NULL) free(used_fit);
        return -1.0;
    }

    score = 0.0;

    for (i = 0; i < n_match; i++) {
        int best_true = -1;
        int best_fit = -1;
        double best_corr = -1.0;

        for (j = 0; j < L_true->nrows; j++) {
            if (used_true[j])
                continue;
            for (k = 0; k < L_fit->nrows; k++) {
                double corr;
                if (used_fit[k])
                    continue;
                corr = gexeval_row_abs_correlation(L_true, j, L_fit, k);
                if (corr > best_corr) {
                    best_corr = corr;
                    best_true = j;
                    best_fit = k;
                }
            }
        }

        if (best_true < 0 || best_fit < 0 || best_corr < 0.0) {
            free(used_true);
            free(used_fit);
            return -1.0;
        }

        used_true[best_true] = 1;
        used_fit[best_fit] = 1;
        score += best_corr;
    }

    free(used_true);
    free(used_fit);

    return score / (double)n_match;
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
    GexMatrix *fit_F = NULL;
    Matrix *fit_Fc = NULL;
    Matrix *fit_cell_cov = NULL;
    GexMatrix *fit_L = NULL;
    Matrix *fit_Z = NULL;

    double z_rmse = -1.0;
    double cell_cov_corr = -2.0;
    double L_factor_match_score = -1.0;

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
    fit_F = read_gex_matrix(fit_f_path);
    fit_L = read_gex_matrix(fit_l_path);

    /* RMSE of Z = F * L */
    sim_Z = mat_new(sim_F->X->nrows, sim_L->X->ncols);
    fit_Z = mat_new(fit_F->X->nrows, fit_L->X->ncols);
    mat_mult(sim_Z, sim_F->X, sim_L->X);
    mat_mult(fit_Z, fit_F->X, fit_L->X);
    z_rmse = mat_rmse(sim_Z, fit_Z);

    /* Cell-cell covariance induced by F */
    sim_Fc = mat_create_copy(sim_F->X);
    mat_center_cols(sim_Fc);
    sim_cell_cov = mat_centered_row_cov(sim_Fc);

    fit_Fc = mat_create_copy(fit_F->X);
    mat_center_cols(fit_Fc);
    fit_cell_cov = mat_centered_row_cov(fit_Fc);

    cell_cov_corr = mat_pearson_correlation(sim_cell_cov, fit_cell_cov);

    /* Compare learned factors in L */
    L_factor_match_score = greedy_L_match_score(sim_L->X, fit_L->X);

    out = fopen(eval_summary_path, "w");
    fprintf(out, "metric\tvalue\n");
    fprintf(out, "z_rmse\t%.17g\n", z_rmse);
    fprintf(out, "cell_cov_correlation\t%.17g\n", cell_cov_corr);
    fprintf(out, "L_factor_match_score\t%.17g\n", L_factor_match_score);
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

    return 0;
}