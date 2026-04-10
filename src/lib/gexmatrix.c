#include "gexmatrix.h"

#include <phast/matrix.h>

#include <errno.h>
#include <math.h>
#include <stdlib.h>


/* Normalize the entries in a row by the row sum in-place.
Returns 0 on success, -1 on failure. */
void mat_normalize_rows(Matrix *X) {
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
void mat_log1p(Matrix *X) {
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
void mat_center_cols(Matrix *X) {
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
    if (out == NULL) {
        fprintf(stderr, "ERROR: failed to open %s for writing: %s\n",
                filename, strerror(errno));
        return -1;
    }

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

void gex_free_matrix_data(GexMatrix *gex) {
    int i;

    if (gex == NULL) return;

    if (gex->cell_names != NULL) {
        for (i = 0; i < gex->X->nrows; i++)
            free(gex->cell_names[i]);
        free(gex->cell_names);
    }

    if (gex->gene_names != NULL) {
        for (i = 0; i < gex->X->ncols; i++)
            free(gex->gene_names[i]);
        free(gex->gene_names);
    }

    if (gex->X != NULL)
        mat_free(gex->X);

    free(gex);
}

