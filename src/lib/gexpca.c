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

/* Compute PCA for a gene expression matrix. 
Return a pointer to the result or NULL on failure. */
PCA *compute_pca(Matrix *X, int k, double variance_threshold) {
    int i, j;   /* Loop indices */
    int p = X->ncols;;  /* Number of genes */
    int keep_K = k; /* Number of components to keep */
    Matrix *Xc = mat_create_copy(X); /* Centered gene expression matrix */
    Matrix *Cov = NULL; /* Covariance matrix of the centered data */
    Matrix *eigvecs = NULL; /* Matrix of eigenvectors (columns) from eigendecomposition of covariance matrix */
    Vector *eigvals = NULL; /* Vector of eigenvalues from eigendecomposition of covariance matrix */
    EigPair *pairs = NULL;   /* Array of eigenvalue/index pairs for sorting eigenvalues in descending order */
    PCA *out = NULL; /* Output PCA result */
    double total_var = 0.0; /* Total variance (sum of eigenvalues) for computing variance explained */
    Matrix *new_components = NULL; /* Reduced components matrix */
    double *new_var = NULL;    /* Reduced variance explained array */

    /* Center the gene expression matrix by subtracting the mean of each column */
    mat_center_cols(Xc);

    /* Compute the covariance matrix of the centered data */
    Cov = mat_centered_cov(Xc);

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
        if (val < 0.0 && fabs(val) < 1e-12)
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

    if (total_var <= 0.0) {
        fprintf(stderr, "WARNING: total PCA variance is non-positive; reporting zeros\n");
        for (i = 0; i < p; i++) {
            out->var_explained[i] = 0.0;
            for (j = 0; j < p; j++)
                mat_set(out->components, i, j, mat_get(eigvecs, j, pairs[i].idx));
        }
    }
    else {
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
    }

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

    /* Free memory */
    if (pairs != NULL)
        free(pairs);
    if (Xc != NULL)
        mat_free(Xc);
    if (Cov != NULL)
        mat_free(Cov);
    if (eigvals != NULL)
        vec_free(eigvals);
    if (eigvecs != NULL)
        mat_free(eigvecs);

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
