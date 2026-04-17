
#include "gexmatrix.h"

#include "gexparser.h"

#include <phast/eigen.h>
#include <phast/matrix.h>
#include <phast/misc.h>
#include <phast/vector.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int n_cells;
    int n_genes;
    int k;
    double sigma_obs;
    double *sigma2_latent;
} GexEvalSummary;

typedef struct {
    double value;
} DescValue;

static int cmp_desc_value(const void *a, const void *b) {
    const DescValue *da = (const DescValue *)a;
    const DescValue *db = (const DescValue *)b;

    if (da->value > db->value) return -1;
    if (da->value < db->value) return 1;
    return 0;
}

static int gexeval_find_name(char **names, int n, const char *target) {
    int i;

    if (names == NULL || target == NULL)
        return -1;
    for (i = 0; i < n; i++) {
        if (names[i] != NULL && strcmp(names[i], target) == 0)
            return i;
    }
    return -1;
}

/* Subset rows of a GexMatrix by names to keep only those names
input. */
static GexMatrix *gexeval_subset_rows_by_names(GexMatrix *src,
                                               char **target_names,
                                               int n_target) {
    int i, j;
    GexMatrix *out = NULL;

    if (src == NULL || target_names == NULL || n_target <= 0)
        return NULL;
    out = scalloc(1, sizeof(GexMatrix));
    out->X = mat_new(n_target, src->X->ncols);
    out->cell_names = scalloc(n_target, sizeof(char *));
    out->gene_names = scalloc(src->X->ncols, sizeof(char *));

    for (j = 0; j < src->X->ncols; j++) {
        out->gene_names[j] = strdup(src->gene_names[j]);
        if (out->gene_names[j] == NULL) {
            gex_free_matrix_data(out);
            return NULL;
        }
    }

    for (i = 0; i < n_target; i++) {
        int src_idx = gexeval_find_name(src->cell_names, src->X->nrows, target_names[i]);
        if (src_idx < 0) {
            gex_free_matrix_data(out);
            return NULL;
        }
        out->cell_names[i] = strdup(target_names[i]);
        if (out->cell_names[i] == NULL) {
            gex_free_matrix_data(out);
            return NULL;
        }
        for (j = 0; j < src->X->ncols; j++)
            mat_set(out->X, i, j, mat_get(src->X, src_idx, j));
    }

    return out;
}

static GexMatrix *gexeval_subset_cols_by_names(GexMatrix *src,
                                               char **target_names,
                                               int n_target) {
    int i, j;
    GexMatrix *out = NULL;

    if (src == NULL || target_names == NULL || n_target <= 0)
        return NULL;
    out = scalloc(1, sizeof(GexMatrix));
    out->X = mat_new(src->X->nrows, n_target);
    out->cell_names = scalloc(src->X->nrows, sizeof(char *));
    out->gene_names = scalloc(n_target, sizeof(char *));

    for (i = 0; i < src->X->nrows; i++) {
        out->cell_names[i] = strdup(src->cell_names[i]);
        if (out->cell_names[i] == NULL) {
            gex_free_matrix_data(out);
            return NULL;
        }
    }

    for (j = 0; j < n_target; j++) {
        int src_idx = gexeval_find_name(src->gene_names, src->X->ncols, target_names[j]);
        if (src_idx < 0) {
            gex_free_matrix_data(out);
            return NULL;
        }
        out->gene_names[j] = strdup(target_names[j]);
        if (out->gene_names[j] == NULL) {
            gex_free_matrix_data(out);
            return NULL;
        }
        for (i = 0; i < src->X->nrows; i++)
            mat_set(out->X, i, j, mat_get(src->X, i, src_idx));
    }

    return out;
}

static int gexeval_collect_common_names(char **names_ref,
                                        int n_ref,
                                        char **names_other,
                                        int n_other,
                                        char ***common_out,
                                        int *n_common_out) {
    int i;
    int n_common = 0;
    char **common = NULL;

    if (common_out == NULL || n_common_out == NULL)
        return -1;
    *common_out = NULL;
    *n_common_out = 0;

    common = scalloc(n_ref, sizeof(char *));

    for (i = 0; i < n_ref; i++) {
        if (gexeval_find_name(names_other, n_other, names_ref[i]) >= 0) {
            common[n_common] = strdup(names_ref[i]);
            if (common[n_common] == NULL) {
                int j;
                for (j = 0; j < n_common; j++)
                    free(common[j]);
                free(common);
                return -1;
            }
            n_common++;
        }
    }

    if (n_common <= 0) {
        free(common);
        return -1;
    }

    *common_out = common;
    *n_common_out = n_common;
    return 0;
}

static void gexeval_free_names(char **names, int n) {
    int i;

    if (names == NULL)
        return;
    for (i = 0; i < n; i++)
        free(names[i]);
    free(names);
}

static Matrix *gexeval_center_columns(Matrix *X) {
    int i, j;
    Matrix *Xc = NULL;

    if (X == NULL)
        return NULL;
    Xc = mat_new(X->nrows, X->ncols);
    if (Xc == NULL)
        return NULL;

    for (j = 0; j < X->ncols; j++) {
        double mean = 0.0;
        for (i = 0; i < X->nrows; i++)
            mean += mat_get(X, i, j);
        mean /= (double)X->nrows;
        for (i = 0; i < X->nrows; i++)
            mat_set(Xc, i, j, mat_get(X, i, j) - mean);
    }

    return Xc;
}

/* Reconstruct the gex matrix X = Z * L */
static Matrix *gexeval_reconstruct_gex_matrix(Matrix *Z, Matrix *L) {
    Matrix *signal = NULL;

    if (Z == NULL || L == NULL || Z->ncols != L->nrows)
        return NULL;
    signal = mat_new(Z->nrows, L->ncols);
    if (signal == NULL)
        return NULL;
    mat_mult(signal, Z, L);
    return signal;
}

/* Compute the covariance matrix of the cells from their latent factor vectors in
the rows of Z */
static Matrix *gexeval_compute_cell_covariance(Matrix *Z) {
    int i, j, d;
    Matrix *Zc = NULL;
    Matrix *cov = NULL;
    double denom;

    if (Z == NULL)
        return NULL;
    Zc = gexeval_center_columns(Z);
    cov = mat_new(Z->nrows, Z->nrows);
    if (Zc == NULL || cov == NULL) {
        if (Zc != NULL) mat_free(Zc);
        if (cov != NULL) mat_free(cov);
        return NULL;
    }

    denom = (Z->ncols > 1 ? (double)Z->ncols : 1.0);
    for (i = 0; i < Z->nrows; i++) {
        for (j = 0; j < Z->nrows; j++) {
            double sum = 0.0;
            for (d = 0; d < Z->ncols; d++)
                sum += mat_get(Zc, i, d) * mat_get(Zc, j, d);
            mat_set(cov, i, j, sum / denom);
        }
    }

    mat_free(Zc);
    return cov;
}

/* Compute the covariance matrix of genes in the gene expression matrix 
from the columns of X */
static Matrix *gexeval_compute_gene_covariance(Matrix *signal) {
    int i, g1, g2;
    Matrix *Xc = NULL;
    Matrix *cov = NULL;
    double denom;

    if (signal == NULL)
        return NULL;
    Xc = gexeval_center_columns(signal);
    cov = mat_new(signal->ncols, signal->ncols);
    if (Xc == NULL || cov == NULL) {
        if (Xc != NULL) mat_free(Xc);
        if (cov != NULL) mat_free(cov);
        return NULL;
    }

    denom = (signal->nrows > 1 ? (double)signal->nrows : 1.0);
    for (g1 = 0; g1 < signal->ncols; g1++) {
        for (g2 = 0; g2 < signal->ncols; g2++) {
            double sum = 0.0;
            for (i = 0; i < signal->nrows; i++)
                sum += mat_get(Xc, i, g1) * mat_get(Xc, i, g2);
            mat_set(cov, g1, g2, sum / denom);
        }
    }

    mat_free(Xc);
    return cov;
}

/* Compute the Pearson correlation between all entries of matrices A and B.
 * Treats both matrices as flattened vectors and returns correlation in [-1,1].
 * Returns -2.0 if inputs are invalid or variance is zero.
 */
static double gexeval_matrix_correlation(Matrix *A, Matrix *B) {
    int i, j;
    int n = 0;              /* Total number of entries */
    double mean_a = 0.0;    /* Mean of all entries in A */
    double mean_b = 0.0;    /* Mean of all entries in B */
    double num = 0.0;       /* Numerator: covariance */
    double den_a = 0.0;     /* Variance term for A */
    double den_b = 0.0;     /* Variance term for B */

    /* Require same shape and valid inputs */
    if (A == NULL || B == NULL || A->nrows != B->nrows || A->ncols != B->ncols)
        return -2.0;

    n = A->nrows * A->ncols;
    if (n <= 1)
        return -2.0;

    /* Compute means of both matrices */
    for (i = 0; i < A->nrows; i++) {
        for (j = 0; j < A->ncols; j++) {
            mean_a += mat_get(A, i, j);
            mean_b += mat_get(B, i, j);
        }
    }
    mean_a /= (double)n;
    mean_b /= (double)n;

    /* Compute covariance (num) and variances (den_a, den_b) */
    for (i = 0; i < A->nrows; i++) {
        for (j = 0; j < A->ncols; j++) {
            double da = mat_get(A, i, j) - mean_a;  /* Centered value from A */
            double db = mat_get(B, i, j) - mean_b;  /* Centered value from B */
            num += da * db;     /* Accumulate covariance */
            den_a += da * da;   /* Accumulate variance of A */
            den_b += db * db;   /* Accumulate variance of B */
        }
    }

    /* Avoid division by zero if one matrix has no variance */
    if (den_a <= 0.0 || den_b <= 0.0)
        return -2.0;

    /* Return Pearson correlation */
    return num / sqrt(den_a * den_b);
}

/* Compute relative Frobenius error between matrices A and B:
 *   ||A - B||_F / ||A||_F
 * Returns -1.0 on invalid input.
 */
static double gexeval_relative_frobenius_error(Matrix *A, Matrix *B) {
    int i, j;
    double ss_diff = 0.0;
    double ss_ref = 0.0;

    if (A == NULL || B == NULL || A->nrows != B->nrows || A->ncols != B->ncols)
        return -1.0;

    for (i = 0; i < A->nrows; i++) {
        for (j = 0; j < A->ncols; j++) {
            double a = mat_get(A, i, j);
            double b = mat_get(B, i, j);
            double d = a - b;
            ss_diff += d * d;
            ss_ref += a * a;
        }
    }

    if (ss_ref <= 0.0)
        return -1.0;

    return sqrt(ss_diff) / sqrt(ss_ref);
}

static Matrix *gexeval_orthonormal_basis(Matrix *X, int *rank_out) {
    int i, j, qcol;
    Matrix *Xc = NULL;
    Matrix *Q = NULL;
    int rank = 0;

    if (rank_out == NULL || X == NULL)
        return NULL;
    *rank_out = 0;

    Xc = gexeval_center_columns(X);
    Q = mat_new(X->nrows, X->ncols);
    if (Xc == NULL || Q == NULL) {
        if (Xc != NULL) mat_free(Xc);
        if (Q != NULL) mat_free(Q);
        return NULL;
    }
    mat_zero(Q);

    for (j = 0; j < Xc->ncols; j++) {
        double norm2 = 0.0;
        for (i = 0; i < Xc->nrows; i++)
            mat_set(Q, i, rank, mat_get(Xc, i, j));

        for (qcol = 0; qcol < rank; qcol++) {
            double dot = 0.0;
            for (i = 0; i < Xc->nrows; i++)
                dot += mat_get(Q, i, rank) * mat_get(Q, i, qcol);
            for (i = 0; i < Xc->nrows; i++)
                mat_set(Q, i, rank, mat_get(Q, i, rank) - dot * mat_get(Q, i, qcol));
        }

        for (i = 0; i < Xc->nrows; i++) {
            double v = mat_get(Q, i, rank);
            norm2 += v * v;
        }
        if (norm2 > 1e-10) {
            double inv_norm = 1.0 / sqrt(norm2);
            for (i = 0; i < Xc->nrows; i++)
                mat_set(Q, i, rank, inv_norm * mat_get(Q, i, rank));
            rank++;
        }
    }

    mat_free(Xc);
    if (rank == 0) {
        mat_free(Q);
        return NULL;
    }
    mat_resize(Q, X->nrows, rank);
    *rank_out = rank;
    return Q;
}

/* Compute similarity between the column spaces of Z_true and Z_fit.
 * Returns a value in [0,1]: 1 = identical subspaces, 0 = orthogonal.
 * This is invariant to rotations or scaling of the latent factors.
 */
static double gexeval_latent_subspace_similarity(Matrix *Z_true, Matrix *Z_fit) {
    int r_true = 0;
    int r_fit = 0;
    int i, j, k;
    Matrix *Q_true = NULL;   /* Orthonormal basis for col(Z_true) */
    Matrix *Q_fit = NULL;    /* Orthonormal basis for col(Z_fit) */
    Matrix *cross = NULL;    /* Q_true^T Q_fit: pairwise basis overlaps */
    Matrix *gram = NULL;     /* cross^T cross */
    Matrix *gram_copy = NULL;
    Vector *evals = NULL;    /* Eigenvalues of gram */
    Matrix *evecs = NULL;
    double overlap = -1.0;
    int denom_rank;

    /* Require same ambient space (same number of rows) */
    if (Z_true == NULL || Z_fit == NULL || Z_true->nrows != Z_fit->nrows)
        return -1.0;

    /* Convert both matrices to orthonormal bases of their column spaces */
    Q_true = gexeval_orthonormal_basis(Z_true, &r_true);
    Q_fit = gexeval_orthonormal_basis(Z_fit, &r_fit);
    if (Q_true == NULL || Q_fit == NULL)
        return 1;

    /* Allocate working matrices */
    cross = mat_new(r_true, r_fit);
    gram = mat_new(r_fit, r_fit);
    gram_copy = mat_new(r_fit, r_fit);
    evals = vec_new(r_fit);
    evecs = mat_new(r_fit, r_fit);

    /* cross[i,j] = dot product between basis vectors of the two subspaces */
    for (i = 0; i < r_true; i++) {
        for (j = 0; j < r_fit; j++) {
            double sum = 0.0;
            for (k = 0; k < Z_true->nrows; k++)
                sum += mat_get(Q_true, k, i) * mat_get(Q_fit, k, j);
            mat_set(cross, i, j, sum);
        }
    }

    /* gram = cross^T cross measures how much Q_fit lies in Q_true */
    for (i = 0; i < r_fit; i++) {
        for (j = 0; j < r_fit; j++) {
            double sum = 0.0;
            for (k = 0; k < r_true; k++)
                sum += mat_get(cross, k, i) * mat_get(cross, k, j);
            mat_set(gram, i, j, sum);
            mat_set(gram_copy, i, j, sum);
        }
    }

    /* Eigenvalues = squared cosines of principal angles between subspaces */
    if (mat_diagonalize_sym(gram_copy, evals, evecs) != 0)
        return 1;

    /* Normalize by smaller subspace dimension */
    denom_rank = (r_true < r_fit ? r_true : r_fit);
    if (denom_rank <= 0)
        return 1;

    /* Average squared cosine overlap (clamp for numerical stability) */
    overlap = 0.0;
    for (i = 0; i < r_fit; i++) {
        double lambda = vec_get(evals, i);
        if (lambda < 0.0) lambda = 0.0;
        if (lambda > 1.0) lambda = 1.0;
        overlap += lambda;
    }
    overlap /= (double)denom_rank;

    /* Free memory */
    if (Q_true != NULL) mat_free(Q_true);
    if (Q_fit != NULL) mat_free(Q_fit);
    if (cross != NULL) mat_free(cross);
    if (gram != NULL) mat_free(gram);
    if (gram_copy != NULL) mat_free(gram_copy);
    if (evals != NULL) vec_free(evals);
    if (evecs != NULL) mat_free(evecs);

    return overlap;
}


/* Compute per-latent-factor contribution magnitudes.
 * For each latent dimension d, this returns:
 *   sigma2_latent[d] * ||L[d, :]||^2
 * i.e., the variance of factor d scaled by the squared norm of its loadings.
 */
static double *gexeval_factor_contributions(Matrix *L,
                                            const double *sigma2_latent,
                                            int k) {
    int d, j;
    double *vals = NULL;

    /* L should have k rows (one per latent factor) */
    if (L == NULL || sigma2_latent == NULL || L->nrows != k)
        return NULL;

    vals = scalloc(k, sizeof(double));
    if (vals == NULL)
        return NULL;

    for (d = 0; d < k; d++) {
        double row_ss = 0.0;

        /* Compute squared norm of loadings for factor d */
        for (j = 0; j < L->ncols; j++) {
            double v = mat_get(L, d, j);
            row_ss += v * v;
        }

        /* Scale by latent variance for factor d */
        vals[d] = sigma2_latent[d] * row_ss;
    }

    return vals;
}

static void gexeval_sort_desc(double *vals, int n) {
    int i;
    DescValue *tmp = NULL;

    if (vals == NULL || n <= 0)
        return;
    tmp = scalloc(n, sizeof(DescValue));
    for (i = 0; i < n; i++)
        tmp[i].value = vals[i];
    qsort(tmp, n, sizeof(DescValue), cmp_desc_value);
    for (i = 0; i < n; i++)
        vals[i] = tmp[i].value;
    free(tmp);
}

/* Normalize a nonnegative vector to sum to 1.
 * Returns a newly allocated probability vector, or NULL on failure.
 */
static double *gexeval_normalize_nonnegative_vector(const double *x, int n) {
    int i;
    double sum = 0.0;
    double *p = NULL;

    if (x == NULL || n <= 0)
        return NULL;

    p = scalloc(n, sizeof(double));

    for (i = 0; i < n; i++) {
        double v = x[i];
        if (v < 0.0)
            v = 0.0;
        p[i] = v;
        sum += v;
    }

    if (sum <= 0.0) {
        free(p);
        return NULL;
    }

    for (i = 0; i < n; i++)
        p[i] /= sum;

    return p;
}

/* Compute L1 distance between two vectors of equal length.
 * Returns value in [0,2] for probability vectors, or -1.0 on failure.
 */
static double gexeval_vector_l1_distance(const double *a, const double *b, int n) {
    int i;
    double s = 0.0;

    if (a == NULL || b == NULL || n <= 0)
        return -1.0;

    for (i = 0; i < n; i++) {
        double d = a[i] - b[i];
        if (d < 0.0)
            d = -d;
        s += d;
    }

    return s;
}

static double gexeval_vector_correlation(const double *a, const double *b, int n) {
    int i;
    double mean_a = 0.0;
    double mean_b = 0.0;
    double num = 0.0;
    double den_a = 0.0;
    double den_b = 0.0;

    if (a == NULL || b == NULL || n <= 1)
        return -2.0;

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
        return -2.0;
    return num / sqrt(den_a * den_b);
}

/* Compute Pearson correlation between vectors a and b.
 * Returns value in [-1,1], or -2.0 on invalid input.
 */
static double gexeval_vector_correlation_basic(const double *a, const double *b, int n) {
    return gexeval_vector_correlation(a, b, n);
}

/* Compute absolute Pearson correlation between columns c1 and c2
 * of matrices A and B. The matrices must have the same number of rows.
 * Returns value in [0,1], or -2.0 on failure.
 */
static double gexeval_column_abs_correlation(Matrix *A, int c1, Matrix *B, int c2) {
    int i;
    int n;
    double mean_a = 0.0;
    double mean_b = 0.0;
    double num = 0.0;
    double den_a = 0.0;
    double den_b = 0.0;

    if (A == NULL || B == NULL || A->nrows != B->nrows ||
        c1 < 0 || c1 >= A->ncols || c2 < 0 || c2 >= B->ncols)
        return -2.0;

    n = A->nrows;
    if (n <= 1)
        return -2.0;

    for (i = 0; i < n; i++) {
        mean_a += mat_get(A, i, c1);
        mean_b += mat_get(B, i, c2);
    }
    mean_a /= (double)n;
    mean_b /= (double)n;

    for (i = 0; i < n; i++) {
        double da = mat_get(A, i, c1) - mean_a;
        double db = mat_get(B, i, c2) - mean_b;
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

/* Match latent factors by greedily pairing columns of Z_true and Z_fit
 * using maximum absolute Pearson correlation.
 *
 * This is a factor-level metric, unlike subspace overlap. It asks whether
 * individual latent factors are recovered up to sign flips.
 *
 * Returns the mean matched absolute correlation in [0,1], or -1.0 on failure.
 * Compares up to min(ncol(Z_true), ncol(Z_fit)) factors.
 */
static double gexeval_greedy_factor_match_score(Matrix *Z_true, Matrix *Z_fit) {
    int i, j;
    int n_match;
    int *used_true = NULL;
    int *used_fit = NULL;
    double score = -1.0;

    if (Z_true == NULL || Z_fit == NULL || Z_true->nrows != Z_fit->nrows)
        return -1.0;

    n_match = (Z_true->ncols < Z_fit->ncols ? Z_true->ncols : Z_fit->ncols);
    if (n_match <= 0)
        return -1.0;

    used_true = scalloc(Z_true->ncols, sizeof(int));
    used_fit = scalloc(Z_fit->ncols, sizeof(int));

    score = 0.0;
    for (i = 0; i < n_match; i++) {
        int best_true = -1;
        int best_fit = -1;
        double best_corr = -1.0;

        for (j = 0; j < Z_true->ncols; j++) {
            int k;
            if (used_true[j])
                continue;
            for (k = 0; k < Z_fit->ncols; k++) {
                double corr;
                if (used_fit[k])
                    continue;
                corr = gexeval_column_abs_correlation(Z_true, j, Z_fit, k);
                if (corr > best_corr) {
                    best_corr = corr;
                    best_true = j;
                    best_fit = k;
                }
            }
        }

        if (best_true < 0 || best_fit < 0 || best_corr < 0.0) {
            score = -1.0;
            return 1;
        }

        used_true[best_true] = 1;
        used_fit[best_fit] = 1;
        score += best_corr;
    }

    score /= (double)n_match;

    /* Free memory */
    if (used_true != NULL)
        free(used_true);
    if (used_fit != NULL)
        free(used_fit);

    return score;
}

/* Compare normalized factor contribution profiles between sim and fit.
 *
 * The input vectors should be per-factor contribution magnitudes such as
 *   sigma2_latent[d] * ||L[d,:]||^2
 *
 * Steps:
 *   1. sort both vectors in descending order
 *   2. keep the top min(n_true, n_fit) entries
 *   3. normalize each truncated vector to sum to 1
 *   4. return L1 distance between the two normalized profiles
 *
 * Returns 0 for identical profiles, larger values for more mismatch.
 * Maximum is 2 for probability vectors. Returns -1.0 on failure.
 */
static double gexeval_normalized_contribution_l1(const double *sim_contrib,
                                                 int n_true,
                                                 const double *fit_contrib,
                                                 int n_fit) {
    int i;
    int n_compare;
    double *sim_sorted = NULL;
    double *fit_sorted = NULL;
    double *sim_p = NULL;
    double *fit_p = NULL;
    double out = -1.0;

    if (sim_contrib == NULL || fit_contrib == NULL || n_true <= 0 || n_fit <= 0)
        return -1.0;

    n_compare = (n_true < n_fit ? n_true : n_fit);
    if (n_compare <= 0)
        return -1.0;

    sim_sorted = scalloc(n_true, sizeof(double));
    fit_sorted = scalloc(n_fit, sizeof(double));

    for (i = 0; i < n_true; i++)
        sim_sorted[i] = sim_contrib[i];
    for (i = 0; i < n_fit; i++)
        fit_sorted[i] = fit_contrib[i];

    gexeval_sort_desc(sim_sorted, n_true);
    gexeval_sort_desc(fit_sorted, n_fit);

    sim_p = gexeval_normalize_nonnegative_vector(sim_sorted, n_compare);
    fit_p = gexeval_normalize_nonnegative_vector(fit_sorted, n_compare);
    if (sim_p == NULL || fit_p == NULL)
        return 1;

    out = gexeval_vector_l1_distance(sim_p, fit_p, n_compare);

    /* Free memory */
    if (sim_sorted != NULL)
        free(sim_sorted);
    if (fit_sorted != NULL)
        free(fit_sorted);
    if (sim_p != NULL)
        free(sim_p);
    if (fit_p != NULL)
        free(fit_p);

    return out;
}

static GexEvalSummary *gexeval_read_summary(const char *filename) {
    char line[4096];
    FILE *in = NULL;
    GexEvalSummary *summary = NULL;

    if (filename == NULL)
        return NULL;
    in = fopen(filename, "r");
    if (in == NULL)
        return NULL;

    summary = scalloc(1, sizeof(GexEvalSummary));

    while (fgets(line, sizeof(line), in) != NULL) {
        char *tab = strchr(line, '\t');
        char *key = line;
        char *value = NULL;
        if (tab == NULL)
            continue;
        *tab = '\0';
        value = tab + 1;
        value[strcspn(value, "\r\n")] = '\0';
        if (strcmp(key, "parameter") == 0)
            continue;
        if (strcmp(key, "n_cells") == 0)
            summary->n_cells = atoi(value);
        else if (strcmp(key, "n_genes") == 0)
            summary->n_genes = atoi(value);
        else if (strcmp(key, "k") == 0) {
            int old_k = summary->k;
            summary->k = atoi(value);
            if (summary->k > 0 && summary->k != old_k) {
                double *tmp = srealloc(summary->sigma2_latent,
                                                (size_t)summary->k * sizeof(double));
                summary->sigma2_latent = tmp;
                memset(summary->sigma2_latent, 0, (size_t)summary->k * sizeof(double));
            }
        }
        else if (strcmp(key, "sigma_obs") == 0)
            summary->sigma_obs = atof(value);
        else if (strncmp(key, "sigma_latent_LF", 15) == 0) {
            int idx = atoi(key + 15) - 1;
            if (idx >= 0 && idx < summary->k && summary->sigma2_latent != NULL)
                summary->sigma2_latent[idx] = atof(value);
        }
    }

    fclose(in);
    return summary;
}

static void gexeval_free_summary(GexEvalSummary *summary) {
    if (summary == NULL)
        return;
    if (summary->sigma2_latent != NULL)
        free(summary->sigma2_latent);
    free(summary);
}

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
    char sim_summary_path[4096];
    char sim_f_path[4096];
    char sim_l_path[4096];
    char fit_summary_path[4096];
    char fit_f_path[4096];
    char fit_l_path[4096];
    char eval_summary_path[4096];
    GexEvalSummary *sim_summary = NULL;
    GexEvalSummary *fit_summary = NULL;
    GexMatrix *sim_F = NULL;
    GexMatrix *sim_L = NULL;
    GexMatrix *fit_F = NULL;
    GexMatrix *fit_L = NULL;
    GexMatrix *sim_F_aligned = NULL;
    GexMatrix *fit_F_aligned = NULL;
    GexMatrix *sim_L_common = NULL;
    GexMatrix *fit_L_common = NULL;
    Matrix *sim_latent_cov = NULL;
    Matrix *fit_latent_cov = NULL;
    Matrix *sim_gene_cov = NULL;
    Matrix *fit_gene_cov = NULL;
    Matrix *sim_signal = NULL;
    Matrix *fit_signal = NULL;
    double latent_subspace_similarity;
    double latent_factor_match_score = -1.0;
    double cell_cov_corr;
    double gene_cov_corr;
    double signal_relative_frobenius_error = -1.0;
    double cell_cov_relative_frobenius_error = -1.0;
    double gene_cov_relative_frobenius_error = -1.0;
    double variance_trend_corr = -2.0;
    double normalized_contribution_l1 = -1.0;
    double *sim_contrib = NULL;
    double *fit_contrib = NULL;
    char **common_cells = NULL;
    char **common_genes = NULL;
    int n_common_cells = 0;
    int n_common_genes = 0;
    int i;
    FILE *out = NULL;

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

    /* Use relative paths for the comparison based on the input prefixes */
    snprintf(sim_summary_path, sizeof(sim_summary_path), "%s.summary.tsv", sim_prefix);
    snprintf(sim_f_path, sizeof(sim_f_path), "%s.F.tsv", sim_prefix);
    snprintf(sim_l_path, sizeof(sim_l_path), "%s.L.tsv", sim_prefix);
    snprintf(fit_summary_path, sizeof(fit_summary_path), "%s.summary.tsv", fit_prefix);
    snprintf(fit_f_path, sizeof(fit_f_path), "%s.F.tsv", fit_prefix);
    snprintf(fit_l_path, sizeof(fit_l_path), "%s.L.tsv", fit_prefix);
    snprintf(eval_summary_path, sizeof(eval_summary_path), "%s.summary.tsv", outprefix);

    /* Read in the simulated and fit parameters */
    sim_summary = gexeval_read_summary(sim_summary_path);
    fit_summary = gexeval_read_summary(fit_summary_path);
    sim_F = read_gex_matrix(sim_f_path);
    sim_L = read_gex_matrix(sim_l_path);
    fit_F = read_gex_matrix(fit_f_path);
    fit_L = read_gex_matrix(fit_l_path);
    if (sim_summary == NULL || fit_summary == NULL || sim_F == NULL || sim_L == NULL ||
        fit_F == NULL || fit_L == NULL) {
        fprintf(stderr, "ERROR: failed to read required sim/model files.\n");
        return 1;
    }

    /* Derive the set of common cells and genes between the sim and fitted outputs 
    so that we are not comparing model fits on the subset of genes simulated. */
    if (gexeval_collect_common_names(fit_F->cell_names, fit_F->X->nrows,
                                     sim_F->cell_names, sim_F->X->nrows,
                                     &common_cells, &n_common_cells) != 0 ||
        gexeval_collect_common_names(fit_L->gene_names, fit_L->X->ncols,
                                     sim_L->gene_names, sim_L->X->ncols,
                                     &common_genes, &n_common_genes) != 0) {
        fprintf(stderr, "ERROR: failed to derive shared cells/genes between sim and fitted outputs.\n");
        return 1;
    }

    /* Subset Z and L to the common cells and genes */
    sim_F_aligned = gexeval_subset_rows_by_names(sim_F, common_cells, n_common_cells);
    fit_F_aligned = gexeval_subset_rows_by_names(fit_F, common_cells, n_common_cells);
    sim_L_common = gexeval_subset_cols_by_names(sim_L, common_genes, n_common_genes);
    fit_L_common = gexeval_subset_cols_by_names(fit_L, common_genes, n_common_genes);
    if (sim_F_aligned == NULL || fit_F_aligned == NULL || sim_L_common == NULL || fit_L_common == NULL) {
        fprintf(stderr, "ERROR: failed to align sim and fitted matrices on shared names.\n");
        return 1;
    }

    /* Compute the reconstructed gex matrix X from Z and L for both simulated and fitted models */
    sim_signal = gexeval_reconstruct_gex_matrix(sim_F_aligned->X, sim_L_common->X);
    fit_signal = gexeval_reconstruct_gex_matrix(fit_F_aligned->X, fit_L_common->X);

    /* Compute covariance between cells implied by their latent factor vectors */
    sim_latent_cov = gexeval_compute_cell_covariance(sim_F_aligned->X);
    fit_latent_cov = gexeval_compute_cell_covariance(fit_F_aligned->X);

    /* Compute covariance between genes in the reconstructed gene expression matrix */
    sim_gene_cov = gexeval_compute_gene_covariance(sim_signal);
    fit_gene_cov = gexeval_compute_gene_covariance(fit_signal);
    if (sim_signal == NULL || fit_signal == NULL || sim_latent_cov == NULL ||
        fit_latent_cov == NULL || sim_gene_cov == NULL || fit_gene_cov == NULL) {
        fprintf(stderr, "ERROR: failed to derive fitted covariance summaries.\n");
        return 1;
    }

    /* Compute the similarity between the latent subspaces of the sim and fitted models */
    latent_subspace_similarity = gexeval_latent_subspace_similarity(sim_F_aligned->X, fit_F_aligned->X);

    /* Compute factor-level recovery up to permutation and sign flip. */
    latent_factor_match_score = gexeval_greedy_factor_match_score(sim_F_aligned->X, fit_F_aligned->X);

    /* Compute pearson correlations between the flattened cell covariance matrices */
    cell_cov_corr = gexeval_matrix_correlation(sim_latent_cov, fit_latent_cov);

    /* Compute pearson correlations between the flattened gene covariance matrices */
    gene_cov_corr = gexeval_matrix_correlation(sim_gene_cov, fit_gene_cov);

    /* Compute scale-sensitive matrix reconstruction and covariance errors. */
    signal_relative_frobenius_error = gexeval_relative_frobenius_error(sim_signal, fit_signal);
    cell_cov_relative_frobenius_error = gexeval_relative_frobenius_error(sim_latent_cov, fit_latent_cov);
    gene_cov_relative_frobenius_error = gexeval_relative_frobenius_error(sim_gene_cov, fit_gene_cov);

    /* Compute how much each factor contributes to the total variance in the fit model
    to see if any factor dominates in reconstruction from the matrix factorization components
    Z and L. This is a scale-invariant way to compare the latent sigmas between simulated and fitted models */
    sim_contrib = gexeval_factor_contributions(sim_L->X, sim_summary->sigma2_latent, sim_summary->k);
    fit_contrib = gexeval_factor_contributions(fit_L->X, fit_summary->sigma2_latent, fit_summary->k);
    if (sim_contrib != NULL && fit_contrib != NULL) {
        int n_compare = (sim_summary->k < fit_summary->k ? sim_summary->k : fit_summary->k);
        gexeval_sort_desc(sim_contrib, sim_summary->k);
        gexeval_sort_desc(fit_contrib, fit_summary->k);
        variance_trend_corr = gexeval_vector_correlation_basic(sim_contrib, fit_contrib, n_compare);
        normalized_contribution_l1 = gexeval_normalized_contribution_l1(sim_contrib, sim_summary->k,
                                                                        fit_contrib, fit_summary->k);
    }

    out = fopen(eval_summary_path, "w");
    if (out == NULL) {
        fprintf(stderr, "ERROR: failed to open evaluator output file.\n");
        return 1;
    }

    fprintf(out, "metric\tvalue\n");
    fprintf(out, "k_true\t%d\n", sim_summary->k);
    fprintf(out, "k_fit\t%d\n", fit_summary->k);
    fprintf(out, "latent_subspace_similarity\t%.17g\n", latent_subspace_similarity);
    fprintf(out, "latent_factor_match_score\t%.17g\n", latent_factor_match_score);
    fprintf(out, "cell_cov_correlation\t%.17g\n", cell_cov_corr);
    fprintf(out, "gene_cov_correlation\t%.17g\n", gene_cov_corr);
    fprintf(out, "signal_relative_frobenius_error\t%.17g\n", signal_relative_frobenius_error);
    fprintf(out, "cell_cov_relative_frobenius_error\t%.17g\n", cell_cov_relative_frobenius_error);
    fprintf(out, "gene_cov_relative_frobenius_error\t%.17g\n", gene_cov_relative_frobenius_error);
    fprintf(out, "sigma_obs_true\t%.17g\n", sim_summary->sigma_obs);
    fprintf(out, "sigma_obs_fit\t%.17g\n", fit_summary->sigma_obs);
    fprintf(out, "latent_variance_trend_correlation\t%.17g\n", variance_trend_corr);
    fprintf(out, "normalized_contribution_l1\t%.17g\n", normalized_contribution_l1);
    fclose(out);
    out = NULL;

    printf("Evaluation summary written to: %s\n", eval_summary_path);

    /* Free memory */
    if (out != NULL)
        fclose(out);
    if (sim_contrib != NULL)
        free(sim_contrib);
    if (fit_contrib != NULL)
        free(fit_contrib);
    if (common_cells != NULL)
        gexeval_free_names(common_cells, n_common_cells);
    if (common_genes != NULL)
        gexeval_free_names(common_genes, n_common_genes);
    if (sim_signal != NULL)
        mat_free(sim_signal);
    if (fit_signal != NULL)
        mat_free(fit_signal);
    if (sim_latent_cov != NULL)
        mat_free(sim_latent_cov);
    if (fit_latent_cov != NULL)
        mat_free(fit_latent_cov);
    if (sim_gene_cov != NULL)
        mat_free(sim_gene_cov);
    if (fit_gene_cov != NULL)
        mat_free(fit_gene_cov);
    if (sim_summary != NULL)
        gexeval_free_summary(sim_summary);
    if (fit_summary != NULL)
        gexeval_free_summary(fit_summary);
    if (sim_F != NULL)
        gex_free_matrix_data(sim_F);
    if (sim_L != NULL)
        gex_free_matrix_data(sim_L);
    if (fit_F != NULL)
        gex_free_matrix_data(fit_F);
    if (fit_L != NULL)
        gex_free_matrix_data(fit_L);
    if (sim_F_aligned != NULL)
        gex_free_matrix_data(sim_F_aligned);
    if (fit_F_aligned != NULL)
        gex_free_matrix_data(fit_F_aligned);
    if (sim_L_common != NULL)
        gex_free_matrix_data(sim_L_common);
    if (fit_L_common != NULL)
        gex_free_matrix_data(fit_L_common);

    return 0; /* Success */
}
