#include "gexmatrix.h"

#include <stdlib.h>
#include <math.h>

#include <phast/matrix.h>


/* Normalize the entries in a row by the row sum in-place.
Returns 0 on success, -1 on failure. */
void normalize_by_row_sums(Matrix *X) {
    int i, j;
    
    for (i = 0; i < X->nrows; i++) {
        double row_sum = 0.0;
        /* Accumulate row sum */
        for (j = 0; j < X->ncols; j++) {
            row_sum += mat_get(X, i, j);
        } 
        if (row_sum < 1e-12)
            continue;   /* Skip normalization if the row sum is negligibly small */
        /* Apply the normalization to elements of the row */
        for (j = 0; j < X->ncols; j++) {
            double val = mat_get(X, i, j);
            val /= row_sum;
            mat_set(X, i, j, val);
        }
    }
}

/* Transform a matrix using the log1p function (log(1+x))) element-wise
in-place. */
void log1p_transform(Matrix *X) {
    int i, j;

    for (i = 0; i < X->nrows; i++) {
        for (j = 0; j < X->ncols; j++) {
            double val = mat_get(X, i, j);
            val = log1p(val);
            mat_set(X, i, j, val);
        }
    }
}

/* Center the columns of a matrix by subtracting the mean of each column
to get the residuals in-place. */
void center_matrix_inplace(Matrix *X) {
    int i, j;

    for (j = 0; j < X->ncols; j++) {

        /* Get the mean of the column */
        double mean = 0.0;
        for (i = 0; i < X->nrows; i++)
            mean += mat_get(X, i, j);
        mean /= X->nrows;

        /* Center the column */
        for (i = 0; i < X->nrows; i++) {
            double val = mat_get(X, i, j);
            val -= mean;
            mat_set(X, i, j, val);
        }
    }
}
