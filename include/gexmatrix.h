#ifndef GEXMATRIX_H
#define GEXMATRIX_H

#include <phast/matrix.h>
#include <phast/vector.h>

typedef struct {
    Matrix *X;  /* cells (nrows) x genes (ncols) */
    char **cell_names;
    char **gene_names;
} GexMatrix;

GexMatrix *gex_mat_new(int n_cells, int n_genes);

void gex_free_matrix_data(GexMatrix *gex);

GexMatrix *gex_mat_copy(GexMatrix *gex);

void mat_set_col(Matrix *res, int i, Vector *sim_vec);

void mat_add_mat(Matrix *dest, Matrix *src);

void mat_normalize_rows(Matrix *X);

void mat_log1p(Matrix *X);

void mat_center_cols(Matrix *X);

void mat_center_rows(Matrix *X);

Matrix *mat_centered_col_cov(Matrix *Xc);

Matrix *mat_centered_row_cov(Matrix *Xc);

void mat_standardize_cols(Matrix *X);

void mat_col_shuffle(Matrix *X);

double mat_sum_entries(Matrix *X);

double mat_sum_squared_entries(Matrix *X);

Matrix *mat_mult_elementwise(Matrix *dest, Matrix *A, Matrix *B);

void mat_square_elementwise(Matrix *X);

void mat_sqrt_elementwise(Matrix *X);

void mat_div_elementwise(Matrix *dest, Matrix *A, Matrix *B);

void mat_add_diag(Matrix *X, double val);

double *mat_row_l2_norms(Matrix *L);

Vector *mat_get_diag(Matrix *X);

Vector *mat_row_sums(Matrix *X);

Vector *mat_col_sums(Matrix *X);

double vec_sum_squared_entries(Vector *v);

void write_labeled_matrix_tsv(const char *filename,
                                 Matrix *X,
                                 char **row_names,
                                 int n_rows,
                                 char **col_names,
                                 int n_cols,
                                 const char *corner_label);

double mat_frobenius_norm(Matrix *M);

double mat_rmse(Matrix *A, Matrix *B);

double mat_pearson_correlation(Matrix *A, Matrix *B);

Matrix *mat_factor_pearson_correlation(Matrix *A, Matrix *B, int compare_rows, int absolute);

double mat_logdet_chol(Matrix *L);

double mat_logdet(Matrix *Sigma);

void mat_add_gaussian_noise(Matrix *X, double sigma2);

void mat_mult_lapack(Matrix *prod, Matrix *m1, Matrix *m2);

void mat_forward_subst_lapack(Matrix *L, Vector *z, Vector *y);

void mat_backward_subst_lapack(Matrix *L, Vector *z, Vector *y);

void mat_svd_lapack(Matrix *X, Matrix **U_out, Vector **S_out, Matrix **VT_out);

#endif
