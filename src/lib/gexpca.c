#include "gexpca.h"

#include "gexmatrix.h"

#include <phast/matrix.h>
#include <phast/misc.h>
#include <phast/eigen.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>


typedef struct {
    double val;
    int idx;
} EigPair;

/* Compare two eigenvalue-index pairs in descending order of eigenvalues */
static int cmp_eigpair_desc(const void *a, const void *b) {
    const EigPair *ea = (const EigPair *)a;
    const EigPair *eb = (const EigPair *)b;

    if (ea->val < eb->val) return 1;
    if (ea->val > eb->val) return -1;
    return 0;
}

/* Compute the number of PCA components needed to explain a certain proportion of the total variance. */
static int pca_components_for_variance_threshold_internal(double *var_explained,
                                                              int K,
                                                              double threshold) {
    int i;
    double cumulative = 0.0;

    if (K <= 0)
        return 0;
    if (threshold <= 0.0)
        return 1;
    if (threshold >= 1.0)
        threshold = 1.0;

    /* Iterate through components that were sorted in descending order of eigenvalues (variance explained)*/
    for (i = 0; i < K; i++) {
        cumulative += var_explained[i];
        /* Return the number of components once the threshold is reached or crossed */
        if (cumulative >= threshold)
            return i + 1;
    }

    return K;
}

static PCA *pca_eigen(Matrix *Cov) {
    int i, j;
    int p = Cov->nrows;
    Vector *eigvals = NULL;
    Matrix *eigvecs = NULL;
    EigPair *pairs = NULL;
    PCA *out = NULL;
    double total_var = 0.0;

    /* Allocate memory for eigenvectors and eigenvalues */
    eigvals = vec_new(p);
    eigvecs = mat_new(p, p);

    /* Perform eigendecomposition of the symmetric covariance matrix */
    if (mat_diagonalize_sym(Cov, eigvals, eigvecs) != 0) {
        fprintf(stderr, "ERROR: symmetric eigendecomposition failed in PCA\n");
        return NULL;
    }

    /* Create an array of eigenvalue/index pairs to sort the eigenvalues in descending order while keeping track of their original indices */
    pairs = smalloc(p * sizeof(EigPair));

    /* Populate the eigenvalue/index pairs */
    for (i = 0; i < p; i++) {
        double val = vec_get(eigvals, i);
        if (val < 0.0 || fabs(val) < 1e-12)
            val = 0.0;
        pairs[i].val = val;
        pairs[i].idx = i;
    }

    /* Sort the eigenvalue/index pairs in descending order */
    qsort(pairs, p, sizeof(EigPair), cmp_eigpair_desc);

    /* Compute total variance as the sum of eigenvalues for computing variance explained */
    for (i = 0; i < p; i++)
        total_var += pairs[i].val;

    /* Allocate memory for the output PCA result */
    out = scalloc(1, sizeof(PCA));
    out->components = mat_new(p, p);
    out->var_explained = scalloc(p, sizeof(double));
    out->K = p;

    /* Fill the output PCA result with eigenvectors and variance explained in sorted order */
    for (i = 0; i < p; i++) {
        int idx = pairs[i].idx;
        double lambda = pairs[i].val;

        /* Normalize variance explained by this component */
        out->var_explained[i] = lambda / total_var;

        for (j = 0; j < p; j++) {
            /* eigenvectors are columns of eigvecs */
            mat_set(out->components, i, j, mat_get(eigvecs, j, idx));
        }
    }

    /* Free memory */
    if (pairs != NULL)
        free(pairs);
    if (eigvals != NULL)
        vec_free(eigvals);
    if (eigvecs != NULL)
        mat_free(eigvecs);

    return out;
}

static void filter_pca_components(PCA *out, int k, double variance_threshold) {
    int i, j;
    int keep_K = k;
    Matrix *new_components = NULL;
    double *new_var = NULL;

    /* If k was not specified, determine how many components to keep based on the variance threshold */
    if (k == 0) {
        keep_K = pca_components_for_variance_threshold_internal(out->var_explained,
                                                                out->K,
                                                                variance_threshold);
    }

    /* Allocate memory for the reduced PCA components and variance explained */
    new_components = mat_new(keep_K, out->components->ncols);
    new_var = scalloc(keep_K, sizeof(double));

    /* Fill the reduced PCA result with the top components and their variance explained */
    for (i = 0; i < keep_K; i++) {
        new_var[i] = out->var_explained[i];
        for (j = 0; j < out->components->ncols; j++)
            mat_set(new_components, i, j, mat_get(out->components, i, j));
    }

    /* Replace the original PCA components and variance explained with the reduced versions */
    mat_free(out->components);
    free(out->var_explained);
    out->components = new_components;
    out->var_explained = new_var;
    out->K = keep_K;
}

/* Compute PCA for a matrix. */
PCA *compute_pca(Matrix *X, int k, double variance_threshold) {
    Matrix *Xc = mat_create_copy(X);
    Matrix *Cov = NULL;
    PCA *out = NULL;

    /* Compute the covariance matrix of the centered data */
    mat_center_cols(Xc);
    Cov = mat_centered_col_cov(Xc);

    /* Free memory */
    if (Xc != NULL)
        mat_free(Xc);

    /* Compute the PCA */
    out = pca_eigen(Cov);

    /* Free memory */
    if (Cov != NULL)
        mat_free(Cov);

    /* Filter the PCA components based on the specified number of components or variance threshold */
    filter_pca_components(out, k, variance_threshold);

    return out;
}

/* Covariance matrix after GLS centering to regress
out known phylogenetic covariance */
static Matrix *mat_cov_gls(Matrix *X, Matrix *C) {
    int i, j;
    int n = X->nrows;
    int p = X->ncols;
    Matrix *L = NULL;
    Matrix *W = NULL;
    Matrix *Wt = NULL;
    Matrix *Cov = NULL;
    Vector *ones = NULL;
    Vector *tmp = NULL;
    Vector *u = NULL;
    Vector *rhs = NULL;
    Vector *sol = NULL;
    double *a = NULL;
    double denom;

    L = mat_new(n, n);
    mat_cholesky(L, C);

    ones = vec_new(n);
    tmp  = vec_new(n);
    u    = vec_new(n);
    vec_set_all(ones, 1.0);
    mat_forward_subst(L, ones, tmp);
    mat_backward_subst(L, tmp, u);

    denom = vec_sum(u);

    /* Compute GLS column means */
    a = scalloc(p, sizeof(double));
    for (i = 0; i < n; i++) {
        double w = u->data[i];
        double *Xi = X->data[i];
        for (j = 0; j < p; j++)
            a[j] += w * Xi[j];
    }

    for (j = 0; j < p; j++)
        a[j] /= denom;

    /* Build whitened centered matrix */
    W   = mat_new(n, p);
    rhs = vec_new(n);
    sol = vec_new(n);

    for (j = 0; j < p; j++) {
        double aj = a[j];
        for (i = 0; i < n; i++)
            rhs->data[i] = X->data[i][j] - aj;
        mat_forward_subst(L, rhs, sol);
        for (i = 0; i < n; i++)
            W->data[i][j] = sol->data[i];
    }

    /* Cov = W^T W / (n - 1) */
    Wt  = mat_transpose(W);
    Cov = mat_new(p, p);
    mat_mult_lapack(Cov, Wt, W);
    mat_scale(Cov, 1.0 / (double)(n - 1));

    /* Free memory */
    if (L   != NULL) mat_free(L);
    if (W   != NULL) mat_free(W);
    if (Wt  != NULL) mat_free(Wt);
    if (ones != NULL) vec_free(ones);
    if (tmp  != NULL) vec_free(tmp);
    if (u    != NULL) vec_free(u);
    if (rhs  != NULL) vec_free(rhs);
    if (sol  != NULL) vec_free(sol);
    if (a != NULL) free(a);

    return Cov;
}

/* Compute Revell 2009 phylogenetic PCA to obtain evolutionarily
independent components of variation among traits where the phylogenetic
correlation between scores on each axis will be zero.
X is n x p (rows = taxa, columns = traits).
C is n x n phylogenetic covariance among rows of X.
*/
PCA *compute_phylo_pca(Matrix *X, Matrix *C, int k, double variance_threshold) {
    Matrix *Cov = NULL;
    PCA *out = NULL;

    /* Revell GLS-covariance */
    Cov = mat_cov_gls(X, C);

    /* Compute the PCA */
    out = pca_eigen(Cov);

    /* Free memory */
    if (Cov != NULL)
        mat_free(Cov);

    /* Filter the PCA components based on the specified number of components or variance threshold */
    filter_pca_components(out, k, variance_threshold);

    return out;
}

/* Print a summary of the PCA results */
void print_pca_summary(PCA *pca) {
    int i;
    double cumulative = 0.0;

    if (pca == NULL) {
        printf("PCA result is NULL\n");
        return;
    }

    /* Report the variance explained by each component */
    printf("\n");
    printf("PCA variance explained:\n");
    for (i = 0; i < pca->K; i++) {
        cumulative += pca->var_explained[i];
        printf("  PC%d: %.6f (%.2f%%), cumulative: %.6f (%.2f%%)\n",
               i + 1,
               pca->var_explained[i],
               100.0 * pca->var_explained[i],
               cumulative,
               100.0 * cumulative);
    }
    printf("\n");
}

void free_pca(PCA *pca) {
    if (pca == NULL) return;

    if (pca->components != NULL)
        mat_free(pca->components);
    if (pca->var_explained != NULL)
        free(pca->var_explained);

    free(pca);
}
