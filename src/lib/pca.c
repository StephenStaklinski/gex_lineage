
#include "pca.h"

#include <phast/eigen.h>

/* -------------------- PCA -------------------- */

typedef struct {
    double val;
    int idx;
} GexEigPair;

static int gex_cmp_eigpair_desc(const void *a, const void *b) {
    const GexEigPair *ea = (const GexEigPair *)a;
    const GexEigPair *eb = (const GexEigPair *)b;

    if (ea->val < eb->val) return 1;
    if (ea->val > eb->val) return -1;
    return 0;
}

static Matrix *gex_center_matrix(Matrix *X) {
    int i, j;
    int n = X->nrows;
    int p = X->ncols;
    double *means = NULL;
    Matrix *Xc = NULL;

    means = (double *)calloc(p, sizeof(double));
    if (means == NULL) {
        fprintf(stderr, "ERROR: out of memory allocating PCA means\n");
        return NULL;
    }

    Xc = mat_new(n, p);
    if (Xc == NULL) {
        fprintf(stderr, "ERROR: out of memory allocating centered matrix\n");
        free(means);
        return NULL;
    }

    for (j = 0; j < p; j++) {
        for (i = 0; i < n; i++)
            means[j] += mat_get(X, i, j);
        means[j] /= (double)n;
    }

    for (i = 0; i < n; i++) {
        for (j = 0; j < p; j++) {
            mat_set(Xc, i, j, mat_get(X, i, j) - means[j]);
        }
    }

    free(means);
    return Xc;
}

static Matrix *gex_compute_covariance(Matrix *Xc) {
    int i, j, k;
    int n = Xc->nrows;
    int p = Xc->ncols;
    Matrix *Cov = NULL;

    Cov = mat_new(p, p);
    if (Cov == NULL) {
        fprintf(stderr, "ERROR: out of memory allocating covariance matrix\n");
        return NULL;
    }

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

static int gex_pca_components_for_variance_threshold_internal(double *var_explained,
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

    for (i = 0; i < K; i++) {
        cumulative += var_explained[i];
        if (cumulative >= threshold)
            return i + 1;
    }

    return K;
}

GexPCA *gex_compute_pca(GexMatrix *gex, double variance_threshold) {
    int i, j;
    int p;
    int keep_K;
    Matrix *Xc = NULL;
    Matrix *Cov = NULL;
    Matrix *eigvecs = NULL;
    Vector *eigvals = NULL;
    GexEigPair *pairs = NULL;
    GexPCA *out = NULL;
    double total_var = 0.0;

    if (gex == NULL || gex->X == NULL) {
        fprintf(stderr, "ERROR: gex_compute_pca received NULL matrix\n");
        return NULL;
    }

    if (gex->n_cells < 2) {
        fprintf(stderr, "ERROR: need at least 2 rows to compute PCA\n");
        return NULL;
    }

    p = gex->n_genes;

    Xc = gex_center_matrix(gex->X);
    if (Xc == NULL)
        return NULL;

    Cov = gex_compute_covariance(Xc);
    if (Cov == NULL) {
        mat_free(Xc);
        return NULL;
    }

    eigvals = vec_new(p);
    eigvecs = mat_new(p, p);
    if (eigvals == NULL || eigvecs == NULL) {
        fprintf(stderr, "ERROR: out of memory allocating eigensystem objects\n");
        if (Xc != NULL) mat_free(Xc);
        if (Cov != NULL) mat_free(Cov);
        if (eigvals != NULL) vec_free(eigvals);
        if (eigvecs != NULL) mat_free(eigvecs);
        return NULL;
    }

    if (mat_diagonalize_sym(Cov, eigvals, eigvecs) != 0) {
        fprintf(stderr, "ERROR: symmetric eigendecomposition failed in PCA\n");
        mat_free(Xc);
        mat_free(Cov);
        vec_free(eigvals);
        mat_free(eigvecs);
        return NULL;
    }

    pairs = (GexEigPair *)malloc(p * sizeof(GexEigPair));
    if (pairs == NULL) {
        fprintf(stderr, "ERROR: out of memory allocating eigenvalue sort buffer\n");
        mat_free(Xc);
        mat_free(Cov);
        vec_free(eigvals);
        mat_free(eigvecs);
        return NULL;
    }

    for (i = 0; i < p; i++) {
        double val = vec_get(eigvals, i);
        if (val < 0.0 && fabs(val) < 1e-12)
            val = 0.0;
        pairs[i].val = val;
        pairs[i].idx = i;
    }

    qsort(pairs, p, sizeof(GexEigPair), gex_cmp_eigpair_desc);

    for (i = 0; i < p; i++)
        total_var += pairs[i].val;

    out = (GexPCA *)calloc(1, sizeof(GexPCA));
    if (out == NULL) {
        fprintf(stderr, "ERROR: out of memory allocating PCA result\n");
        free(pairs);
        mat_free(Xc);
        mat_free(Cov);
        vec_free(eigvals);
        mat_free(eigvecs);
        return NULL;
    }

    out->components = mat_new(p, p);
    out->var_explained = (double *)calloc(p, sizeof(double));
    out->K = p;

    if (out->components == NULL || out->var_explained == NULL) {
        fprintf(stderr, "ERROR: out of memory allocating PCA outputs\n");
        gex_free_pca(out);
        free(pairs);
        mat_free(Xc);
        mat_free(Cov);
        vec_free(eigvals);
        mat_free(eigvecs);
        return NULL;
    }

    if (total_var <= 0.0) {
        fprintf(stderr, "WARNING: total PCA variance is non-positive; reporting zeros\n");
        for (i = 0; i < p; i++) {
            out->var_explained[i] = 0.0;
            for (j = 0; j < p; j++)
                mat_set(out->components, i, j, mat_get(eigvecs, j, pairs[i].idx));
        }
    }
    else {
        for (i = 0; i < p; i++) {
            int idx = pairs[i].idx;
            double lambda = pairs[i].val;

            out->var_explained[i] = lambda / total_var;

            for (j = 0; j < p; j++) {
                /* eigenvectors are columns of eigvecs */
                mat_set(out->components, i, j, mat_get(eigvecs, j, idx));
            }
        }
    }

    free(pairs);
    mat_free(Xc);
    mat_free(Cov);
    vec_free(eigvals);
    mat_free(eigvecs);

    Matrix *new_components;
    double *new_var;
    keep_K = gex_pca_components_for_variance_threshold_internal(out->var_explained,
                                                                out->K,
                                                                variance_threshold);
    if (keep_K <= 0 || keep_K > out->K) {
        gex_free_pca(out);
        return NULL;
    }

    new_components = mat_new(keep_K, out->components->ncols);
    new_var = (double *)calloc(keep_K, sizeof(double));
    if (new_components == NULL || new_var == NULL) {
        if (new_components != NULL)
            mat_free(new_components);
        free(new_var);
        gex_free_pca(out);
        return NULL;
    }

    for (i = 0; i < keep_K; i++) {
        new_var[i] = out->var_explained[i];
        for (j = 0; j < out->components->ncols; j++)
            mat_set(new_components, i, j, mat_get(out->components, i, j));
    }

    mat_free(out->components);
    free(out->var_explained);
    out->components = new_components;
    out->var_explained = new_var;
    out->K = keep_K;

    return out;
}

void gex_print_pca_summary(GexPCA *pca) {
    int i;
    double cumulative = 0.0;

    if (pca == NULL) {
        printf("PCA result is NULL\n");
        return;
    }

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
}

void gex_free_pca(GexPCA *pca) {
    if (pca == NULL) return;

    if (pca->components != NULL)
        mat_free(pca->components);
    if (pca->var_explained != NULL)
        free(pca->var_explained);

    free(pca);
}
