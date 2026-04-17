#include "gexmatrix.h"

#include "gexmisc.h"

#include <phast/matrix.h>
#include <phast/misc.h>

#include <errno.h>
#include <math.h>
#include <stdlib.h>


GexMatrix *gex_mat_new(int n_cells, int n_genes) {
    GexMatrix *gex = scalloc(1, sizeof(GexMatrix));
    gex->X = mat_new(n_cells, n_genes);
    gex->cell_names = scalloc(n_cells, sizeof(char *));
    gex->gene_names = scalloc(n_genes, sizeof(char *));
    return gex;
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

GexMatrix *gex_mat_copy(GexMatrix *gex) {
    if (gex == NULL)
        return NULL;

    GexMatrix *copy = scalloc(1, sizeof(GexMatrix));
    copy->X = mat_create_copy(gex->X);
    copy->cell_names = copy_string_array(gex->cell_names, gex->X->nrows);
    copy->gene_names = copy_string_array(gex->gene_names, gex->X->ncols);

    return copy;
}

/* Set the i-th column of matrix res to the values in vector sim_vec 
in-place. */
void mat_set_col(Matrix *res, int i, Vector *sim_vec) {
    int j;
    for (j = 0; j < res->nrows; j++) {
        mat_set(res, j, i, vec_get(sim_vec, j));
    }
}

/* Add one matrix to another in place element-wise. */
void mat_add_mat(Matrix *dest, Matrix *src) {
    int i, j;
    double sum;
    for (i = 0; i < dest->nrows; i++) {
        for (j = 0; j < dest->ncols; j++) {
            sum = mat_get(dest, i, j) + mat_get(src, i, j);
            mat_set(dest, i, j, sum);
        }
    }
}

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

/* Compute the covariance matrix of a centered matrix. */
Matrix *mat_centered_cov(Matrix *Xc) {
    int i, j, k;
    int n = Xc->nrows;
    int p = Xc->ncols;
    Matrix *Cov = NULL;

    Cov = mat_new(p, p);

    for (j = 0; j < p; j++) {
        for (k = j; k < p; k++) {
            double sum = 0.0;
            for (i = 0; i < n; i++) {
                sum += mat_get(Xc, i, j) * mat_get(Xc, i, k);
            }
            sum /= (double)(n - 1);
            mat_set(Cov, j, k, sum);
            mat_set(Cov, k, j, sum);
        }
    }
    return Cov;
}

/* Standardize the columns of a matrix by subtracting the mean and 
dividing by the standard deviation in-place */
void mat_standardize_cols(Matrix *X) {
    int i, j;

    for (j = 0; j < X->ncols; j++) {
        double mean = 0.0;
        double var = 0.0;
        double sd;

        /* Get the mean of the column */
        for (i = 0; i < X->nrows; i++)
            mean += mat_get(X, i, j);
        mean /= X->nrows;

        /* Get the variance of the column */
        for (i = 0; i < X->nrows; i++) {
            double diff = mat_get(X, i, j) - mean;
            var += diff * diff;
        }
        var /= X->nrows;

        sd = sqrt(var);
        if (sd < 1e-12)
            continue;

        /* Standardize the column */
        for (i = 0; i < X->nrows; i++) {
            double val = mat_get(X, i, j);
            val -= mean;
            val /= sd;
            mat_set(X, i, j, val);
        }
    }
}

/* Shuffle all columns of a matrix in-place using Fisher–Yates */
void mat_col_shuffle(Matrix *X) {
    int i, j;
    double tmp;

    for (j = 0; j < X->ncols; j++) {
        for (i = X->nrows - 1; i > 0; i--) {
            int k = rand() % (i + 1);  /* 0 ≤ k ≤ i */
            tmp = mat_get(X, i, j);
            mat_set(X, i, j, mat_get(X, k, j));
            mat_set(X, k, j, tmp);
        }
    }
}

/* Sum all entries of a Matrix */
double mat_sum_entries(Matrix *X) {
    int i, j;
    double sum = 0.0;

    /* Calculate the sum of all entries */
    for (i = 0; i < X->nrows; i++) {
        for (j = 0; j < X->ncols; j++) {
            sum += mat_get(X, i, j);
        }
    }

    return sum;
}

/* Sum all squared entries of a Matrix */
double mat_sum_squared_entries(Matrix *X) {
    int i, j;
    double sum = 0.0;

    /* Calculate the sum of all squared entries */
    for (i = 0; i < X->nrows; i++) {
        for (j = 0; j < X->ncols; j++) {
            double val = mat_get(X, i, j);
            sum += val * val;
        }
    }

    return sum;
}

/* Multiply two matrices element-wise (NOT true matrix multiplication)
and store the result in a third matrix */
Matrix *mat_mult_elementwise(Matrix *dest, Matrix *A, Matrix *B) {
    int i, j;

    if (dest == NULL || A == NULL || B == NULL)
        return NULL;

    if (A->nrows != B->nrows || A->ncols != B->ncols)
        return NULL;

    for (i = 0; i < A->nrows; i++) {
        for (j = 0; j < A->ncols; j++) {
            double val = mat_get(A, i, j) * mat_get(B, i, j);
            mat_set(dest, i, j, val);
        }
    }

    return dest;
}

/* Square the entries of a matrix element-wise in-place. */
void mat_square_elementwise(Matrix *X) {
    int i, j;

    for (i = 0; i < X->nrows; i++) {
        for (j = 0; j < X->ncols; j++) {
            double val = mat_get(X, i, j);
            val *= val;
            mat_set(X, i, j, val);
        }
    }
}

/* Take the square root of the entries of a matrix element-wise 
in-place. */
void mat_sqrt_elementwise(Matrix *X) {
    int i, j;

    for (i = 0; i < X->nrows; i++) {
        for (j = 0; j < X->ncols; j++) {
            double val = mat_get(X, i, j);
            val = sqrt(val);
            mat_set(X, i, j, val);
        }
    }
}

/* Divide two matrices element-wise (NOT true matrix multiplication)
and store the result in a third matrix */
void mat_div_elementwise(Matrix *dest, Matrix *A, Matrix *B) {
    int i, j;

    if (dest == NULL || A == NULL || B == NULL)
        return;

    if (A->nrows != B->nrows || A->ncols != B->ncols)
        return;

    for (i = 0; i < A->nrows; i++) {
        for (j = 0; j < A->ncols; j++) {
            double val = mat_get(A, i, j) / mat_get(B, i, j);
            mat_set(dest, i, j, val);
        }
    }
}

/* Add a scalar value to the diagonal entries of a matrix */
void mat_add_diag(Matrix *X, double val) {
    int i, n;

    n = X->nrows < X->ncols ? X->nrows : X->ncols;
    for (i = 0; i < n; i++)
        mat_set(X, i, i, mat_get(X, i, i) + val);
}

/* Return a vector containing the diagonal entries of a matrix */
Vector *mat_get_diag(Matrix *X) {
    int i, n;
    Vector *diag;

    n = X->nrows < X->ncols ? X->nrows : X->ncols;
    diag = vec_new(n);

    for (i = 0; i < n; i++)
        vec_set(diag, i, mat_get(X, i, i));

    return diag;
}

/* Return a vector containing the row sums of a matrix */
Vector *mat_row_sums(Matrix *X) {
    int i, j;
    Vector *row_sums;

    row_sums = vec_new(X->nrows);
    for (i = 0; i < X->nrows; i++) {
        double sum = 0.0;
        for (j = 0; j < X->ncols; j++)
            sum += mat_get(X, i, j);
        vec_set(row_sums, i, sum);
    }
    return row_sums;
}

/* Return a vector containing the column sums of a matrix */
Vector *mat_col_sums(Matrix *X) {
    int i, j;
    Vector *col_sums;

    col_sums = vec_new(X->ncols);
    for (j = 0; j < X->ncols; j++) {
        double sum = 0.0;
        for (i = 0; i < X->nrows; i++)
            sum += mat_get(X, i, j);
        vec_set(col_sums, j, sum);
    }
    return col_sums;
}

/* Return the sum of the squares of the entries in a vector */
double vec_sum_squared_entries(Vector *v) {
    int i;
    double sum = 0.0;

    for (i = 0; i < v->size; i++) {
        double val = vec_get(v, i);
        sum += val * val;
    }

    return sum;
}

void write_labeled_matrix_tsv(const char *filename,
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
        return;

    out = fopen(filename, "w");
    if (out == NULL) {
        fprintf(stderr, "ERROR: failed to open %s for writing: %s\n",
                filename, strerror(errno));
        return;
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
    out = NULL;
}

/* Compute the Frobenius norm of a matrix.
This is equivalent to the Euclidean (l2) norm of all entries
treated as a single vector. */
double mat_frobenius_norm(Matrix *M) {
    int i, j;
    double ss = 0.0;
    double val;

    for (i = 0; i < M->nrows; i++) {
        for (j = 0; j < M->ncols; j++) {
            val = mat_get(M, i, j);
            ss += val * val;
        }
    }

    return sqrt(ss);
}

/* Compute the log-determinant of a matrix given its Cholesky factor L. */
double mat_logdet_chol(Matrix *L) {
    int j;
    double logdet_sigma = 0.0;
    for (j = 0; j < L->nrows; j++) {
        double diag = mat_get(L, j, j);
        logdet_sigma += log(diag);
    }
    logdet_sigma *= 2.0;

    return logdet_sigma;
}

/* Compute the log-determinant of a matrix*/
double mat_logdet(Matrix *Sigma) {

    Matrix *L = mat_new(Sigma->nrows, Sigma->nrows);
    mat_cholesky(L, Sigma);

    double logdet_sigma =  mat_logdet_chol(L);

    mat_free(L);

    return logdet_sigma;
}
