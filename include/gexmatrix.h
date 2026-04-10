#ifndef GEXMATRIX_H
#define GEXMATRIX_H

#include <phast/matrix.h>

typedef struct {
    Matrix *X;  /* cells (nrows) x genes (ncols) */
    char **cell_names;
    char **gene_names;
} GexMatrix;

void normalize_by_row_sums(Matrix *X);

void log1p_transform(Matrix *X);

void center_matrix_inplace(Matrix *X);

#endif
