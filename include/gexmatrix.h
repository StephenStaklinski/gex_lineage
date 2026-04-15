#ifndef GEXMATRIX_H
#define GEXMATRIX_H

#include <phast/matrix.h>
#include <phast/vector.h>

typedef struct {
    Matrix *X;  /* cells (nrows) x genes (ncols) */
    char **cell_names;
    char **gene_names;
} GexMatrix;

void gex_free_matrix_data(GexMatrix *gex);

void mat_set_col(Matrix *res, int i, Vector *sim_vec);

void mat_add_mat(Matrix *dest, Matrix *src);

void mat_normalize_rows(Matrix *X);

void mat_log1p(Matrix *X);

void mat_center_cols(Matrix *X);

Matrix *mat_centered_cov(Matrix *Xc);

void mat_standardize_cols(Matrix *X);

void mat_col_shuffle(Matrix *X);

double mat_sum_entries(Matrix *X);

double mat_sum_squared_entries(Matrix *X);

Matrix *mat_mult_elementwise(Matrix *dest, Matrix *A, Matrix *B);

void mat_square_elementwise(Matrix *X);

void mat_sqrt_elementwise(Matrix *X);

void mat_div_elementwise(Matrix *dest, Matrix *A, Matrix *B);

void mat_add_diag(Matrix *X, double val);

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

double mat_logdet_chol(Matrix *L);

double mat_logdet(Matrix *Sigma);

#endif
