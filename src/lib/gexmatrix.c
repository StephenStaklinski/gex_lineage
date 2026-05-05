#include "gexmatrix.h"

#include "gexmisc.h"
#include "gex_external_libs.h"

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

/* Center the rows of a matrix by subtracting the mean of each row
to get the residuals in-place. */
void mat_center_rows(Matrix *X) {
    int i, j;

    for (i = 0; i < X->nrows; i++) {

        /* Get the mean of the row */
        double mean = 0.0;
        for (j = 0; j < X->ncols; j++)
            mean += mat_get(X, i, j);
        mean /= X->ncols;

        /* Center the row */
        for (j = 0; j < X->ncols; j++) {
            double val = mat_get(X, i, j);
            val -= mean;
            mat_set(X, i, j, val);
        }
    }
}

/* Compute the col-col covariance matrix of a centered matrix. */
Matrix *mat_centered_col_cov(Matrix *Xc) {
    int n = Xc->nrows;
    int p = Xc->ncols;
    double denom = (double)(n - 1);
    Matrix *Xct = mat_transpose(Xc);
    Matrix *Cov = mat_new(p, p);

    mat_mult_lapack(Cov, Xct, Xc);
    mat_scale(Cov, 1.0 / denom);

    /* Free memory */
    if (Xct != NULL)
        mat_free(Xct);
    
    return Cov;
}

/* Compute the row-row covariance matrix of a centered matrix. */
Matrix *mat_centered_row_cov(Matrix *Xc) {
    int n = Xc->nrows;
    int p = Xc->ncols;
    double denom = (double)(p - 1);
    Matrix *Xct = mat_transpose(Xc);
    Matrix *Cov = mat_new(n, n);

    mat_mult_lapack(Cov, Xc, Xct);
    mat_scale(Cov, 1.0 / denom);

    /* Free memory */
    if (Xct != NULL)
        mat_free(Xct);

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
        double *row = X->data[i];
        for (j = 0; j < X->ncols; j++) {
            double val = row[j];
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

/* Return an array containing the row L2 norms of a matrix */
double *mat_row_l2_norms(Matrix *L) {
    int i, j;
    double *row_norms = malloc(L->nrows * sizeof(double));

    for (i = 0; i < L->nrows; i++) {
        row_norms[i] = 0.0;
        for (j = 0; j < L->ncols; j++) {
            double val = mat_get(L, i, j);
            row_norms[i] += val * val;
        }
        row_norms[i] = sqrt(row_norms[i]);
    }

    return row_norms;
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

void write_top_loading_genes_tsv(const char *filename,
                                 Matrix *loadings,
                                 char **factor_names,
                                 int n_factors,
                                 char **gene_names,
                                 int n_genes,
                                 int n_top,
                                 const char *factor_label) {
    int factor, rank, j;
    FILE *out = NULL;
    int *used = NULL;

    if (filename == NULL || loadings == NULL || factor_names == NULL || gene_names == NULL ||
        n_factors <= 0 || n_genes <= 0 ||
        loadings->nrows != n_factors || loadings->ncols != n_genes)
        return;

    if (n_top > n_genes)
        n_top = n_genes;

    out = fopen(filename, "w");
    if (out == NULL) {
        fprintf(stderr, "ERROR: failed to open %s for writing: %s\n",
                filename, strerror(errno));
        return;
    }

    fprintf(out, "%s", factor_label == NULL ? "factor" : factor_label);
    for (rank = 0; rank < n_top; rank++)
        fprintf(out, "\tgene_up_%d", rank + 1);
    for (rank = 0; rank < n_top; rank++)
        fprintf(out, "\tgene_down_%d", rank + 1);
    fprintf(out, "\n");

    used = scalloc(n_genes, sizeof(int));
    for (factor = 0; factor < n_factors; factor++) {
        fprintf(out, "%s", factor_names[factor]);
        for (j = 0; j < n_genes; j++)
            used[j] = 0;

        for (rank = 0; rank < n_top; rank++) {
            int best_j = -1;
            double best_loading = 0.0;

            for (j = 0; j < n_genes; j++) {
                double loading = mat_get(loadings, factor, j);
                if (used[j])
                    continue;
                if (best_j < 0 || loading > best_loading) {
                    best_j = j;
                    best_loading = loading;
                }
            }

            if (best_j >= 0) {
                used[best_j] = 1;
                fprintf(out, "\t%s", gene_names[best_j]);
            } else {
                fprintf(out, "\t");
            }
        }
        for (j = 0; j < n_genes; j++)
            used[j] = 0;

        for (rank = 0; rank < n_top; rank++) {
            int best_j = -1;
            double best_loading = 0.0;

            for (j = 0; j < n_genes; j++) {
                double loading = mat_get(loadings, factor, j);
                if (used[j])
                    continue;
                if (best_j < 0 || loading < best_loading) {
                    best_j = j;
                    best_loading = loading;
                }
            }

            if (best_j >= 0) {
                used[best_j] = 1;
                fprintf(out, "\t%s", gene_names[best_j]);
            } else {
                fprintf(out, "\t");
            }
        }
        fprintf(out, "\n");
    }

    free(used);
    fclose(out);
}

/* Compute the Frobenius norm of a matrix.
This is equivalent to the Euclidean (l2) norm of all entries
treated as a single vector. */
double mat_frobenius_norm(Matrix *M) {
    int i, j;
    double ss = 0.0;

    for (i = 0; i < M->nrows; i++) {
        double *row = M->data[i];
        for (j = 0; j < M->ncols; j++) {
            double val = row[j];
            ss += val * val;
        }
    }

    return sqrt(ss);
}

double mat_diag_mean(Matrix *X) {
    int i;
    double sum = 0.0;

    if (X == NULL || X->nrows != X->ncols || X->nrows == 0)
        return NAN;

    for (i = 0; i < X->nrows; i++)
        sum += mat_get(X, i, i);

    return sum / (double)X->nrows;
}

/* RMSE between two matrices */
double mat_rmse(Matrix *A, Matrix *B) {
    int i, j;
    int n = A->nrows;
    int p = A->ncols;
    double ss = 0.0;
    double denom = (double)n * p;

    if (A == NULL || B == NULL || A->nrows != B->nrows || A->ncols != B->ncols)
        return -1.0;

    for (i = 0; i < n; i++) {
        for (j = 0; j < p; j++) {
            double d = mat_get(A, i, j) - mat_get(B, i, j);
            ss += d * d;
        }
    }

    return sqrt(ss / denom);
}

/* Pearson correlation between all entries of two same-shaped matrices,
treating them as flattened vectors */
double mat_pearson_correlation(Matrix *A, Matrix *B) {
    int i, j;
    int n = A->nrows;
    int p = A->ncols;
    double denom = (double)n * p;
    double mean_a = 0.0;
    double mean_b = 0.0;
    double num = 0.0;
    double den_a = 0.0;
    double den_b = 0.0;

    if (A == NULL || B == NULL || A->nrows != B->nrows || A->ncols != B->ncols)
        return -2.0;

    /* Compute means */
    for (i = 0; i < n; i++) {
        for (j = 0; j < p; j++) {
            mean_a += mat_get(A, i, j);
            mean_b += mat_get(B, i, j);
        }
    }
    mean_a /= denom;
    mean_b /= denom;

    /* Compute numerator and denominator of Pearson */
    for (i = 0; i < n; i++) {
        for (j = 0; j < p; j++) {
            double da = mat_get(A, i, j) - mean_a;
            double db = mat_get(B, i, j) - mean_b;
            num += da * db;
            den_a += da * da;
            den_b += db * db;
        }
    }

    return num / sqrt(den_a * den_b);
}

static double pearson_arrays(const double *a, const double *b, int n) {
    int i;
    double mean_a = 0.0;
    double mean_b = 0.0;
    double num = 0.0;
    double den_a = 0.0;
    double den_b = 0.0;

    if (a == NULL || b == NULL || n <= 0)
        return NAN;

    for (i = 0; i < n; i++) {
        mean_a += a[i];
        mean_b += b[i];
    }
    mean_a /= (double)n;
    mean_b /= (double)n;

    for (i = 0; i < n; i++) {
        double da = a[i] - mean_a;
        double db = b[i] - mean_b;
        num += da * db;
        den_a += da * da;
        den_b += db * db;
    }

    if (den_a <= 0.0 || den_b <= 0.0)
        return NAN;

    return num / sqrt(den_a * den_b);
}

Matrix *mat_factor_pearson_correlation(Matrix *A, Matrix *B, int compare_rows, int absolute) {
    int i, j, k;
    int n_factors_a, n_factors_b, n_values;
    Matrix *corr = NULL;
    double *a = NULL;
    double *b = NULL;

    if (A == NULL || B == NULL)
        return NULL;

    if (compare_rows) {
        if (A->ncols != B->ncols)
            return NULL;
        n_factors_a = A->nrows;
        n_factors_b = B->nrows;
        n_values = A->ncols;
    } else {
        if (A->nrows != B->nrows)
            return NULL;
        n_factors_a = A->ncols;
        n_factors_b = B->ncols;
        n_values = A->nrows;
    }

    corr = mat_new(n_factors_a, n_factors_b);
    a = scalloc(n_values, sizeof(double));
    b = scalloc(n_values, sizeof(double));

    for (i = 0; i < n_factors_a; i++) {
        for (j = 0; j < n_factors_b; j++) {
            for (k = 0; k < n_values; k++) {
                if (compare_rows) {
                    a[k] = mat_get(A, i, k);
                    b[k] = mat_get(B, j, k);
                } else {
                    a[k] = mat_get(A, k, i);
                    b[k] = mat_get(B, k, j);
                }
            }
            double r = pearson_arrays(a, b, n_values);
            if (absolute && !isnan(r))
                r = fabs(r);
            mat_set(corr, i, j, r);
        }
    }

    free(a);
    free(b);

    return corr;
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

/* Add gaussian noise element-wise to a matrix */
void mat_add_gaussian_noise(Matrix *X, double stddev) {
    int i, j;
    double val;
    double noisy_val;
    for (i = 0; i < X->nrows; i++) {
        for (j = 0; j < X->ncols; j++) {
            val = mat_get(X, i, j);
            noisy_val = norm_draw(val, stddev);
            mat_set(X, i, j, noisy_val);
        }
    }
}

/* Multiply two matrices using LAPACK */
void mat_mult_lapack(Matrix *prod, Matrix *m1, Matrix *m2) {
    #ifdef SKIP_LAPACK
    die("ERROR: BLAS/LAPACK required for matrix multiplication.\n");
    #else
    LAPACK_INT m, n, k;
    LAPACK_INT lda, ldb, ldc;
    LAPACK_DOUBLE alpha = 1.0, beta = 0.0;
    LAPACK_DOUBLE *A = NULL, *B = NULL, *C = NULL;
    char transa = 'N', transb = 'N';

    if (!(m1->ncols == m2->nrows &&
            prod->nrows == m1->nrows && prod->ncols == m2->ncols))
        die("ERROR mat_mult: bad matrix dimensions\n");

    m = (LAPACK_INT)prod->nrows;
    n = (LAPACK_INT)prod->ncols;
    k = (LAPACK_INT)m1->ncols;

    lda = m;
    ldb = k;
    ldc = m;

    A = smalloc((size_t)m * (size_t)k * sizeof(*A));
    B = smalloc((size_t)k * (size_t)n * sizeof(*B));
    C = smalloc((size_t)m * (size_t)n * sizeof(*C));

    mat_to_lapack(m1, A);
    mat_to_lapack(m2, B);

    dgemm_(&transa, &transb,
            &m, &n, &k,
            &alpha,
            A, &lda,
            B, &ldb,
            &beta,
            C, &ldc);

    mat_from_lapack(prod, C);

    sfree(A);
    sfree(B);
    sfree(C);
    #endif
}

/* Solve Ly = z for y, where L is lower-triangular, using LAPACK dtrtrs */
void mat_forward_subst_lapack(Matrix *L, Vector *z, Vector *y) {
    #ifdef SKIP_LAPACK
    die("ERROR: LAPACK required for triangular solve.\n");
    #else
    int i;
    LAPACK_INT n, nrhs, lda, ldb, info;
    LAPACK_DOUBLE *a = NULL, *b = NULL;
    char uplo = 'L';
    char trans = 'N';
    char diag = 'N';

    if (L->nrows != L->ncols || L->nrows != z->size || z->size != y->size)
        die("ERROR in mat_forward_subst_lapack: bad dimensions.\n");

    n = (LAPACK_INT)L->nrows;
    nrhs = 1;
    lda = n;
    ldb = n;

    a = smalloc((size_t)n * (size_t)n * sizeof(*a));
    b = smalloc((size_t)n * sizeof(*b));

    mat_to_lapack(L, a);
    for (i = 0; i < n; i++)
        b[i] = (LAPACK_DOUBLE)z->data[i];

    dtrtrs_(&uplo, &trans, &diag, &n, &nrhs, a, &lda, b, &ldb, &info);

    if (info != 0) {
        sfree(a);
        sfree(b);
        if (info > 0)
        die("ERROR in mat_forward_subst_lapack: triangular matrix is singular.\n");
        else
        die("ERROR in mat_forward_subst_lapack: LAPACK dtrtrs illegal argument.\n");
    }

    for (i = 0; i < n; i++)
        y->data[i] = (double)b[i];

    sfree(a);
    sfree(b);
    #endif
}

/* Solve L^T y = z for y, where L is lower-triangular, using LAPACK dtrtrs */
void mat_backward_subst_lapack(Matrix *L, Vector *z, Vector *y) {
    #ifdef SKIP_LAPACK
    die("ERROR: LAPACK required for triangular solve.\n");
    #else
    int i;
    LAPACK_INT n, nrhs, lda, ldb, info;
    LAPACK_DOUBLE *a = NULL, *b = NULL;
    char uplo = 'L';
    char trans = 'T';
    char diag = 'N';

    if (L->nrows != L->ncols || L->nrows != z->size || z->size != y->size)
        die("ERROR in mat_backward_subst_lapack: bad dimensions.\n");

    n = (LAPACK_INT)L->nrows;
    nrhs = 1;
    lda = n;
    ldb = n;

    a = smalloc((size_t)n * (size_t)n * sizeof(*a));
    b = smalloc((size_t)n * sizeof(*b));

    mat_to_lapack(L, a);
    for (i = 0; i < n; i++)
        b[i] = (LAPACK_DOUBLE)z->data[i];

    dtrtrs_(&uplo, &trans, &diag, &n, &nrhs, a, &lda, b, &ldb, &info);

    if (info != 0) {
        sfree(a);
        sfree(b);
        if (info > 0)
        die("ERROR in mat_backward_subst_lapack: triangular matrix is singular.\n");
        else
        die("ERROR in mat_backward_subst_lapack: LAPACK dtrtrs illegal argument.\n");
    }

    for (i = 0; i < n; i++)
        y->data[i] = (double)b[i];

    sfree(a);
    sfree(b);
    #endif
}

void mat_svd_lapack(Matrix *X, Matrix **U_out, Vector **S_out, Matrix **VT_out) {
#ifdef SKIP_LAPACK
    die("ERROR: LAPACK required for SVD.\n");
#else
    int m0, n0, r, i;
    LAPACK_INT m, n, lda, ldu, ldvt, lwork, info;
    LAPACK_DOUBLE *a = NULL;
    LAPACK_DOUBLE *s = NULL;
    LAPACK_DOUBLE *u = NULL;
    LAPACK_DOUBLE *vt = NULL;
    LAPACK_DOUBLE *work = NULL;
    LAPACK_INT *iwork = NULL;
    LAPACK_DOUBLE wkopt;
    char jobz = 'S';

    Matrix *U = NULL;
    Matrix *VT = NULL;
    Vector *S = NULL;

    if (X == NULL || S_out == NULL)
        die("ERROR in mat_svd_lapack: X and S_out must be non-NULL.\n");

    m0 = X->nrows;
    n0 = X->ncols;
    r = (m0 < n0 ? m0 : n0);

    m = (LAPACK_INT)m0;
    n = (LAPACK_INT)n0;
    lda = m;
    ldu = m;
    ldvt = (LAPACK_INT)r;

    a = smalloc((size_t)m * (size_t)n * sizeof(*a));
    s = smalloc((size_t)r * sizeof(*s));
    u = smalloc((size_t)ldu * (size_t)r * sizeof(*u));
    vt = smalloc((size_t)ldvt * (size_t)n * sizeof(*vt));
    iwork = smalloc((size_t)(8 * r) * sizeof(*iwork));

    mat_to_lapack(X, a);

    /* workspace query */
    lwork = -1;
    dgesdd_(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt,
            &wkopt, &lwork, iwork, &info);
    if (info != 0)
        die("ERROR in mat_svd_lapack: LAPACK dgesdd failed.\n");

    lwork = (LAPACK_INT)wkopt;
    work = smalloc((size_t)lwork * sizeof(*work));

    /* actual SVD */
    dgesdd_(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt,
            work, &lwork, iwork, &info);
    if (info != 0)
        die("ERROR in mat_svd_lapack: LAPACK dgesdd failed.\n");

    S = vec_new(r);
    for (i = 0; i < r; i++)
        S->data[i] = (double)s[i];

    if (U_out != NULL) {
        int row, col;
        U = mat_new(m0, r);
        for (col = 0; col < r; col++) {
            for (row = 0; row < m0; row++) {
                mat_set(U, row, col, u[row + (size_t)col * ldu]);
            }
        }
        *U_out = U;
    }

    if (VT_out != NULL) {
        int row, col;
        VT = mat_new(r, n0);
        for (col = 0; col < n0; col++) {
            for (row = 0; row < r; row++) {
                mat_set(VT, row, col, vt[row + (size_t)col * ldvt]);
            }
        }
        *VT_out = VT;
    }

    *S_out = S;

    sfree(a);
    sfree(s);
    sfree(u);
    sfree(vt);
    sfree(work);
    sfree(iwork);
#endif
}
