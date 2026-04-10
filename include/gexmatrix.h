#ifndef GEXMATRIX_H
#define GEXMATRIX_H

#include <phast/matrix.h>

typedef struct {
    Matrix *X;  /* cells (nrows) x genes (ncols) */
    char **cell_names;
    char **gene_names;
} GexMatrix;

void gex_free_matrix_data(GexMatrix *gex);

void mat_normalize_rows(Matrix *X);

void mat_log1p(Matrix *X);

void mat_center_cols(Matrix *X);

int gex_write_labeled_matrix_tsv(const char *filename,
                                 Matrix *X,
                                 char **row_names,
                                 int n_rows,
                                 char **col_names,
                                 int n_cols,
                                 const char *corner_label);

#endif
