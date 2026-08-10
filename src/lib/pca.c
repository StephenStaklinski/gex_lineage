#include "pca.h"
#include "parser.h"

#include "external_libs.h"
#include "gexmatrix.h"
#include "misc.h"

#include <phast/matrix.h>
#include <phast/misc.h>
#include <phast/eigen.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <string.h>


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

/* Compute the number of PCA components needed to explain a target cumulative variance. */
static int pca_components_for_cumulative_variance_internal(double *var_explained,
                                                           int K,
                                                           double target) {
    int i;
    double cumulative = 0.0;

    if (K <= 0)
        return 0;
    if (target <= 0.0)
        return 1;
    if (target >= 1.0)
        target = 1.0;

    /* Iterate through components that were sorted in descending order of eigenvalues (variance explained)*/
    for (i = 0; i < K; i++) {
        cumulative += var_explained[i];
        /* Return the number of components once the target is reached or crossed. */
        if (cumulative >= target)
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
    out->eigenvalues = scalloc(p, sizeof(double));
    out->var_explained = scalloc(p, sizeof(double));
    out->K = p;

    /* Fill the output PCA result with eigenvectors and variance explained in sorted order */
    for (i = 0; i < p; i++) {
        int idx = pairs[i].idx;
        double lambda = pairs[i].val;

        /* Normalize variance explained by this component */
        out->eigenvalues[i] = lambda;
        out->var_explained[i] = total_var > 0.0 ? lambda / total_var : 0.0;

        for (j = 0; j < p; j++) {
            /* eigenvectors are columns of eigvecs */
            mat_set(out->components, i, j, mat_get(eigvecs, j, idx));
        }
    }

    /* Free memory */
    free(pairs);
    if (eigvals != NULL)
        vec_free(eigvals);
    if (eigvecs != NULL)
        mat_free(eigvecs);

    return out;
}

/* Compute only the leading k principal components. LAPACK's selected
symmetric eigensolver works on the smaller Gram matrix, avoiding a full SVD. */
static PCA *pca_lapack(Matrix *Xc, int k) {
    LAPACK_DOUBLE *gram = NULL;
    LAPACK_DOUBLE *eigvals = NULL;
    LAPACK_DOUBLE *eigvecs = NULL;
    LAPACK_DOUBLE *work = NULL;
    LAPACK_INT *iwork = NULL;
    LAPACK_INT *isuppz = NULL;
    PCA *out = NULL;
    Matrix *gram_matrix = NULL;
    int i, j, component;
    int n_samples = Xc->nrows;
    int n_features = Xc->ncols;
    int gram_dim = n_features <= n_samples ? n_features : n_samples;
    int requested = k < gram_dim ? k : gram_dim;
    double total_var;
    double max_eigenvalue = 0.0;
    double eigenvalue_tol;
    LAPACK_INT n = (LAPACK_INT)gram_dim;
    LAPACK_INT lda = n;
    LAPACK_INT il = n - (LAPACK_INT)requested + 1;
    LAPACK_INT iu = n;
    LAPACK_INT found = 0;
    LAPACK_INT ldz = n;
    LAPACK_INT lwork = -1;
    LAPACK_INT liwork = -1;
    LAPACK_INT info = 0;
    LAPACK_INT iwkopt;
    LAPACK_DOUBLE wkopt;
    LAPACK_DOUBLE vl = 0.0;
    LAPACK_DOUBLE vu = 0.0;
    LAPACK_DOUBLE abstol = 0.0;
    char jobz = 'V';
    char range = 'I';
    char uplo = 'U';

    if (Xc == NULL || requested <= 0 || n_samples <= 1)
        return NULL;

    gram_matrix = mat_new(gram_dim, gram_dim);
    if (n_features <= n_samples)
        mat_mult_lapack_transpose(gram_matrix, Xc, 1, Xc, 0);
    else
        mat_mult_lapack_transpose(gram_matrix, Xc, 0, Xc, 1);

    gram = smalloc((size_t)gram_dim * (size_t)gram_dim * sizeof(*gram));
    eigvals = smalloc((size_t)gram_dim * sizeof(*eigvals));
    eigvecs = smalloc((size_t)gram_dim * (size_t)requested * sizeof(*eigvecs));
    isuppz = smalloc((size_t)(2 * requested) * sizeof(*isuppz));

    for (j = 0; j < gram_dim; j++)
        for (i = 0; i < gram_dim; i++)
            gram[(size_t)j * (size_t)gram_dim + (size_t)i] =
                mat_get(gram_matrix, i, j);

    dsyevr_(&jobz, &range, &uplo, &n, gram, &lda, &vl, &vu, &il, &iu,
            &abstol, &found, eigvals, eigvecs, &ldz, isuppz,
            &wkopt, &lwork, &iwkopt, &liwork, &info);
    if (info != 0) {
        fprintf(stderr, "ERROR: LAPACK dsyevr workspace query failed in PCA (info=%d)\n",
                (int)info);
        goto cleanup;
    }

    lwork = (LAPACK_INT)wkopt;
    liwork = iwkopt;
    work = smalloc((size_t)lwork * sizeof(*work));
    iwork = smalloc((size_t)liwork * sizeof(*iwork));

    /* dsyevr overwrites the Gram matrix, so restore it after the query. */
    for (j = 0; j < gram_dim; j++)
        for (i = 0; i < gram_dim; i++)
            gram[(size_t)j * (size_t)gram_dim + (size_t)i] =
                mat_get(gram_matrix, i, j);

    dsyevr_(&jobz, &range, &uplo, &n, gram, &lda, &vl, &vu, &il, &iu,
            &abstol, &found, eigvals, eigvecs, &ldz, isuppz,
            work, &lwork, iwork, &liwork, &info);
    if (info != 0 || found <= 0) {
        fprintf(stderr, "ERROR: selected eigendecomposition failed in PCA (info=%d)\n",
                (int)info);
        goto cleanup;
    }

    max_eigenvalue = eigvals[found - 1];
    eigenvalue_tol = fmax(1.0, max_eigenvalue) *
                     (double)(n_samples > n_features ? n_samples : n_features) *
                     DBL_EPSILON;
    total_var = mat_sum_squared_entries(Xc) / (double)(n_samples - 1);

    out = scalloc(1, sizeof(PCA));
    out->components = mat_new((int)found, n_features);
    out->eigenvalues = scalloc((int)found, sizeof(double));
    out->var_explained = scalloc((int)found, sizeof(double));

    /* dsyevr returns selected eigenpairs in ascending order. */
    for (component = 0; component < (int)found; component++) {
        int source = (int)found - 1 - component;
        double gram_eigenvalue = fmax(0.0, eigvals[source]);
        double lambda = gram_eigenvalue / (double)(n_samples - 1);

        if (gram_eigenvalue <= eigenvalue_tol)
            break;

        out->eigenvalues[component] = lambda;
        out->var_explained[component] =
            total_var > 0.0 ? lambda / total_var : 0.0;

        if (n_features <= n_samples) {
            for (j = 0; j < n_features; j++)
                mat_set(out->components, component, j,
                        eigvecs[(size_t)source * (size_t)gram_dim + (size_t)j]);
        }
        else {
            double singular_value = sqrt(gram_eigenvalue);
            for (j = 0; j < n_features; j++) {
                double loading = 0.0;
                for (i = 0; i < n_samples; i++) {
                    double left_loading =
                        eigvecs[(size_t)source * (size_t)gram_dim + (size_t)i];
                    loading += mat_get(Xc, i, j) * left_loading;
                }
                mat_set(out->components, component, j,
                        loading / singular_value);
            }
        }
    }
    out->K = component;

cleanup:
    if (gram_matrix != NULL)
        mat_free(gram_matrix);
    free(gram);
    free(eigvals);
    free(eigvecs);
    free(isuppz);
    free(work);
    free(iwork);
    return out;
}

static void filter_pca_components(PCA *out, int k) {
    int i, j;
    int keep_K;
    Matrix *new_components = NULL;
    double *new_eigenvalues = NULL;
    double *new_var = NULL;

    if (out == NULL || k <= 0)
        return;
    keep_K = k < out->K ? k : out->K;

    /* Allocate memory for the reduced PCA components and variance explained */
    new_components = mat_new(keep_K, out->components->ncols);
    new_eigenvalues = scalloc(keep_K, sizeof(double));
    new_var = scalloc(keep_K, sizeof(double));

    /* Fill the reduced PCA result with the top components and their variance explained */
    for (i = 0; i < keep_K; i++) {
        new_eigenvalues[i] = out->eigenvalues[i];
        new_var[i] = out->var_explained[i];
        for (j = 0; j < out->components->ncols; j++)
            mat_set(new_components, i, j, mat_get(out->components, i, j));
    }

    /* Replace the original PCA components and variance explained with the reduced versions */
    mat_free(out->components);
    free(out->eigenvalues);
    free(out->var_explained);
    out->components = new_components;
    out->eigenvalues = new_eigenvalues;
    out->var_explained = new_var;
    out->K = keep_K;
}

/* Compute PCA for a matrix. */
PCA *compute_pca(Matrix *X, int k) {
    Matrix *Xc = mat_create_copy(X);
    PCA *out = NULL;

    /* Compute the covariance matrix of the centered data */
    mat_center_cols(Xc);

    /* Compute only the requested leading components with LAPACK. */
    out = pca_lapack(Xc, k);

    /* Free memory */
    if (Xc != NULL) 
        mat_free(Xc);

    return out;
}

/* Compute PCA after projecting expression onto the top phylogenetic
covariance eigenvectors, so the retained axes emphasize maximum
phylogenetic signal. */
PCA *compute_max_phylo_pca(Matrix *X, Matrix *C, int k) {
    Matrix *VtY = NULL;
    Matrix *Yhat = NULL;
    PCA *phylo_pca = NULL;
    PCA *out = NULL;
    int phylo_k;
    const double phylo_projection_target = 0.999;

    /* Compute top eigenvectors of the phylogenetic covariance. */
    phylo_pca = pca_eigen(C);
    if (phylo_pca == NULL)
        return NULL;

    phylo_k = pca_components_for_cumulative_variance_internal(phylo_pca->var_explained,
                                                              phylo_pca->K,
                                                              phylo_projection_target);
    if (k > phylo_k)
        phylo_k = k;
    filter_pca_components(phylo_pca, phylo_k);

    /* Project Y onto fitted values from the phylogenetic eigenvectors:
    Y_hat = V_phy V_phy^T Y.  phylo_pca->components stores V_phy^T. */
    VtY = mat_new(phylo_pca->K, X->ncols);
    Yhat = mat_new(X->nrows, X->ncols);
    mat_mult_lapack(VtY, phylo_pca->components, X);
    mat_mult_lapack_transpose(Yhat, phylo_pca->components, 1, VtY, 0);

    /* Run PCA on the fitted values and use the standard PCA rule to
    keep the requested model rank. */
    out = compute_pca(Yhat, k);

    /* Free memory */
    if (VtY != NULL)
        mat_free(VtY);
    if (Yhat != NULL)
        mat_free(Yhat);
    if (phylo_pca != NULL)
        free_pca(phylo_pca);

    return out;
}

/* Read a gene-by-PC loading table written by write_pca_tsv and reorder its
   genes to match the expression matrix. Only the first k components are used. */
PCA *read_pca_initialization_tsv(const char *filename, GexMatrix *gex, int k) {
    GexMatrix *table = NULL;
    PCA *out = NULL;
    int i, j, d;

    if (filename == NULL || gex == NULL || gex->X == NULL || k <= 0)
        return NULL;

    table = read_gex_matrix(filename);
    if (table == NULL)
        return NULL;

    if (table->X->ncols < k) {
        fprintf(stderr,
                "ERROR: PCA initialization has %d component(s), but --dim requires %d: %s\n",
                table->X->ncols, k, filename);
        gex_free_matrix_data(table);
        return NULL;
    }
    if (table->X->nrows != gex->X->ncols) {
        fprintf(stderr,
                "ERROR: PCA initialization has %d gene(s), but expression has %d: %s\n",
                table->X->nrows, gex->X->ncols, filename);
        gex_free_matrix_data(table);
        return NULL;
    }

    out = scalloc(1, sizeof(PCA));
    out->K = k;
    out->components = mat_new(k, gex->X->ncols);
    out->eigenvalues = scalloc(k, sizeof(double));
    out->var_explained = scalloc(k, sizeof(double));

    for (j = 0; j < gex->X->ncols; j++) {
        int matched_row = -1;

        for (i = 0; i < table->X->nrows; i++) {
            if (strcmp(gex->gene_names[j], table->cell_names[i]) == 0) {
                if (matched_row >= 0) {
                    fprintf(stderr,
                            "ERROR: duplicate gene '%s' in PCA initialization: %s\n",
                            gex->gene_names[j], filename);
                    gex_free_matrix_data(table);
                    free_pca(out);
                    return NULL;
                }
                matched_row = i;
            }
        }

        if (matched_row < 0) {
            fprintf(stderr,
                    "ERROR: expression gene '%s' is missing from PCA initialization: %s\n",
                    gex->gene_names[j], filename);
            gex_free_matrix_data(table);
            free_pca(out);
            return NULL;
        }

        for (d = 0; d < k; d++)
            mat_set(out->components, d, j, mat_get(table->X, matched_row, d));
    }

    gex_free_matrix_data(table);
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

void write_pca_tsv(const char *outprefix, PCA *pca, GexMatrix *gex) {
    char eval_filename[1024];
    char evec_filename[1024];
    char top_genes_filename[1024];
    FILE *eval_file = NULL;
    FILE *evec_file = NULL;
    double cumulative = 0.0;
    int i, j;

    if (pca == NULL || gex == NULL || gex->X == NULL)
        return;

    snprintf(eval_filename, sizeof(eval_filename), "%s.pca.eigenvalues.tsv", outprefix);
    eval_file = fopen(eval_filename, "w");
    if (eval_file == NULL) {
        fprintf(stderr, "ERROR: failed to open file %s for writing PCA eigenvalues\n", eval_filename);
        return;
    }

    fprintf(eval_file, "PC\tEigenvalue\tVarianceExplained\tCumulativeVarianceExplained\n");
    for (i = 0; i < pca->K; i++) {
        cumulative += pca->var_explained[i];
        fprintf(eval_file, "PC%d\t%.10g\t%.10g\t%.10g\n",
                i + 1,
                pca->eigenvalues[i],
                pca->var_explained[i],
                cumulative);
    }
    fclose(eval_file);

    snprintf(evec_filename, sizeof(evec_filename), "%s.pca.eigenvectors.tsv", outprefix);
    evec_file = fopen(evec_filename, "w");
    if (evec_file == NULL) {
        fprintf(stderr, "ERROR: failed to open file %s for writing PCA eigenvectors\n", evec_filename);
        return;
    }

    fprintf(evec_file, "Gene");
    for (i = 0; i < pca->K; i++)
        fprintf(evec_file, "\tPC%d", i + 1);
    fprintf(evec_file, "\n");

    for (j = 0; j < gex->X->ncols; j++) {
        fprintf(evec_file, "%s", gex->gene_names[j]);
        for (i = 0; i < pca->K; i++)
            fprintf(evec_file, "\t%.10g", mat_get(pca->components, i, j));
        fprintf(evec_file, "\n");
    }

    fclose(evec_file);

    snprintf(top_genes_filename, sizeof(top_genes_filename), "%s.pca.top_genes.tsv", outprefix);
    char **pc_names = scalloc(pca->K, sizeof(char *));
    for (i = 0; i < pca->K; i++) {
        pc_names[i] = smalloc(64 * sizeof(char));
        snprintf(pc_names[i], 64, "PC%d", i + 1);
    }
    write_top_loading_genes_tsv(top_genes_filename, pca->components, pc_names, pca->K,
                                gex->gene_names, gex->X->ncols, 10, "PC");
    for (i = 0; i < pca->K; i++)
        free(pc_names[i]);
    free(pc_names);
}

void write_pca_gram(const char *outprefix, PCA *pca) {
    char filename[4096];
    char **pc_names;
    Matrix *components_t;
    Matrix *gram;
    int i;

    pc_names = scalloc(pca->K, sizeof(char *));
    generate_names(pc_names, pca->K, "PC");
    components_t = mat_transpose(pca->components);
    gram = mat_new(pca->K, pca->K);
    mat_mult_lapack(gram, pca->components, components_t);

    snprintf(filename, sizeof(filename),
             "%s.pca.eigenvector_gram.tsv", outprefix);
    write_labeled_matrix_tsv(filename, gram,
                             pc_names, pca->K,
                             pc_names, pca->K,
                             "PC");

    for (i = 0; i < pca->K; i++)
        free(pc_names[i]);
    free(pc_names);
    mat_free(components_t);
    mat_free(gram);
}

void free_pca(PCA *pca) {
    if (pca == NULL) return;

    if (pca->components != NULL)
        mat_free(pca->components);
    free(pca->eigenvalues);
    free(pca->var_explained);

    free(pca);
}
