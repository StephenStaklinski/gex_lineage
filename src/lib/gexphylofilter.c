#include "gexphylofilter.h"

#include "gexmisc.h"

#include <phast/trees.h>
#include <phast/matrix.h>
#include <phast/vector.h>
#include <phast/misc.h>
#include <phast/numerical_opt.h>

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>


typedef struct {
    double pval;
    int idx;
} GexPvalPair;

static int gex_cmp_pval_asc(const void *a, const void *b) {
    const GexPvalPair *pa = (const GexPvalPair *)a;
    const GexPvalPair *pb = (const GexPvalPair *)b;

    if (pa->pval < pb->pval) return -1;
    if (pa->pval > pb->pval) return 1;
    return 0;
}

/* Adjust p-values for multiple testing using the Benjamini-Hochberg procedure.
Fills the qvals array with the adjusted p-values. */
static void gex_bh_adjust(double *pvals, double *qvals, int n) {
    GexPvalPair *pairs;
    int i;
    double running;

    /* Initialize the pairs array */
    pairs = smalloc(n * sizeof(GexPvalPair));

    /* Fill the pairs array with p-values and their original indices */
    for (i = 0; i < n; i++) {
        pairs[i].pval = pvals[i];
        pairs[i].idx = i;
    }

    /* Sort the pairs array in ascending order of p-values */
    qsort(pairs, n, sizeof(GexPvalPair), gex_cmp_pval_asc);

    /* Adjust p-values using the Benjamini-Hochberg procedure.
    For sorted p-values p_(i), compute (n / i) * p_(i),
   then apply a reverse cumulative minimum to ensure
   the adjusted p-values are non-decreasing. */
    running = 1.0;
    for (i = n - 1; i >= 0; i--) {
        double rank = (double)(i + 1);
        double val = pairs[i].pval * (double)n / rank;
        if (val > 1.0) 
            val = 1.0;
        if (val < running) 
            running = val;
        qvals[pairs[i].idx] = running;
    }

    free(pairs);
}

static int gex_count_kept_genes(MoranResult *res, double max_q, double min_i) {
    int j;
    int nkeep = 0;

    for (j = 0; j < res->n_genes; j++) {
        if (res->qvals[j] <= max_q && res->morans_i[j] > min_i)
            nkeep++;
    }

    return nkeep;
}

static int gex_count_kept_lrt_genes(GexLRTResult *res, double max_q) {
    int j;
    int nkeep = 0;

    for (j = 0; j < res->n_genes; j++) {
        if (res->qvals[j] <= max_q && res->lrt_stat[j] > 0.0)
            nkeep++;
    }

    return nkeep;
}

static double gex_loglik_centered_gaussian_identity(int n, double sigma2) {
    return -0.5 * ((double)n * (log(2.0 * M_PI * sigma2) + 1.0));
}

/* Compute the log-likelihood of y under N(mu 1, sigma2 * Sigma),
   profiling out mu and sigma2, using the Cholesky factor L of Sigma. */
static double gex_loglik_centered_gaussian_chol(double *y,
                                                Matrix *L,
                                                double logdet_sigma) {
    int i, n;
    Vector *ones = NULL;
    Vector *yvec = NULL;
    Vector *tmp = NULL;
    Vector *Sinv1 = NULL;
    Vector *Sinvy = NULL;
    Vector *resid = NULL;
    Vector *Sinv_resid = NULL;
    double ones_Sinv_ones = 0.0;
    double ones_Sinv_y = 0.0;
    double quad = 0.0;
    double muhat;
    double sigma2;
    double ll;

    n = L->nrows;

    if (n != L->ncols)
        return NAN;

    ones = vec_new(n);
    yvec = vec_new(n);
    tmp = vec_new(n);
    Sinv1 = vec_new(n);
    Sinvy = vec_new(n);
    resid = vec_new(n);
    Sinv_resid = vec_new(n);

    for (i = 0; i < n; i++) {
        vec_set(ones, i, 1.0);
        vec_set(yvec, i, y[i]);
    }

    /* Solve Sigma * Sinv1 = 1 via L tmp = 1, then L^T Sinv1 = tmp */
    mat_forward_subst(L, ones, tmp);
    mat_backward_subst(L, tmp, Sinv1);

    /* Solve Sigma * Sinvy = y via L tmp = y, then L^T Sinvy = tmp */
    mat_forward_subst(L, yvec, tmp);
    mat_backward_subst(L, tmp, Sinvy);

    for (i = 0; i < n; i++) {
        ones_Sinv_ones += vec_get(Sinv1, i);
        ones_Sinv_y += vec_get(Sinvy, i);
    }

    if (ones_Sinv_ones <= 0.0)
        return NAN;

    /* GLS estimate of the mean */
    muhat = ones_Sinv_y / ones_Sinv_ones;

    for (i = 0; i < n; i++)
        vec_set(resid, i, y[i] - muhat);

    /* Solve Sigma * Sinv_resid = resid via L tmp = resid, then L^T Sinv_resid = tmp */
    mat_forward_subst(L, resid, tmp);
    mat_backward_subst(L, tmp, Sinv_resid);

    /* quad = resid^T Sigma^{-1} resid */
    for (i = 0; i < n; i++)
        quad += vec_get(resid, i) * vec_get(Sinv_resid, i);

    sigma2 = quad / (double)n;
    if (sigma2 < 1e-12)
        sigma2 = 1e-12;

    ll = -0.5 * ((double)n * log(2.0 * M_PI * sigma2) +
                 logdet_sigma +
                 (double)n);

    vec_free(ones);
    vec_free(yvec);
    vec_free(tmp);
    vec_free(Sinv1);
    vec_free(Sinvy);
    vec_free(resid);
    vec_free(Sinv_resid);
    return ll;
}

typedef struct {
    double *y;
    Matrix *Sigma;
    Matrix *Sigma_lambda;
    Matrix *L;
    double diag_mean;
    double jitter;
} GexPagelsLambdaOptData;

/* Compute the log-likelihood under a Pagel's lambda style covariance model.
The diagonal entries are held fixed at the common tip variance and the
off-diagonal entries are scaled by lambda. */
static double gex_loglik_pagels_lambda_cov(double *y,
                                           Matrix *Sigma,
                                           double diag_mean,
                                           double jitter,
                                           double lambda,
                                           Matrix *Sigma_lambda,
                                           Matrix *L) {
    int i, j, n;    /* Loop indices */
    double logdet_sigma = 0.0;  /* Log determinant of the lambda-transformed covariance matrix */

    /* Get the number of rows in the covariance matrix */
    n = Sigma->nrows;
    if (lambda < 0.0 || lambda > 1.0)
        return -HUGE_VAL;

    /* Build the lambda-transformed covariance matrix.
    For ultrametric trees, all diagonals should match, so keeping a common
    diagonal value makes lambda = 0 correspond to the null model up to scale. */
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i == j)
                mat_set(Sigma_lambda, i, j, diag_mean + jitter);
            else
                mat_set(Sigma_lambda, i, j, lambda * mat_get(Sigma, i, j));
        }
    }

    /* Compute the Cholesky decomposition of the lambda-transformed covariance matrix */
    if (mat_cholesky(L, Sigma_lambda) != 0)
        return -HUGE_VAL;

    /* Compute the log determinant of the lambda-transformed covariance matrix */
    for (i = 0; i < n; i++) {
        double diag = mat_get(L, i, i);
        if (diag <= 0.0)
            return -HUGE_VAL;
        logdet_sigma += 2.0 * log(diag);
    }

    return gex_loglik_centered_gaussian_chol(y, L, logdet_sigma);
}

/* Objective function for fitting Pagel's lambda by one-dimensional numerical optimization. */
static double gex_pagels_lambda_negloglik(double lambda, void *data) {
    GexPagelsLambdaOptData *d = (GexPagelsLambdaOptData *)data;
    double ll = gex_loglik_pagels_lambda_cov(d->y,
                                             d->Sigma,
                                             d->diag_mean,
                                             d->jitter,
                                             lambda,
                                             d->Sigma_lambda,
                                             d->L);

    if (!isfinite(ll))
        return HUGE_VAL;
    return -ll;
}

/* Fit Pagel's lambda using PHAST's bounded one-dimensional optimizer.
Returns the optimized log-likelihood and sets lambda_hat to the MLE. */
static double gex_fit_pagels_lambda_loglik(double *y,
                                           Matrix *Sigma,
                                           double jitter,
                                           Matrix *Sigma_lambda,
                                           Matrix *L,
                                           double *lambda_hat) {
    int i, status;  /* Loop index and optimization status */
    double diag_mean = 0.0; /* Mean of the diagonal elements of the covariance matrix */
    double lambda;  /* Parameter for Pagel's lambda */
    double fx;  /* Objective function value */
    double fx0; /* Objective function value at lambda = 0 (null model) */
    double fx1; /* Objective function value at lambda = 1 (full model) */
    double best_lambda; /* Best lambda value found among boundary and interior evaluations */
    double best_fx; /* Best objective function value found among boundary and interior evaluations */
    GexPagelsLambdaOptData data;    /* Data structure to pass to the objective function for optimization */

    /* Compute the mean of the diagonal elements of the covariance matrix */
    for (i = 0; i < Sigma->nrows; i++)
        diag_mean += mat_get(Sigma, i, i);
    diag_mean /= (double)Sigma->nrows;

    /* Set up the data structure for optimization */
    data.y = y;
    data.Sigma = Sigma;
    data.Sigma_lambda = Sigma_lambda;
    data.L = L;
    data.diag_mean = diag_mean;
    data.jitter = jitter;

    /* Explicitly evaluate both boundaries because lambda is allowed to sit on
    the edge of the parameter space and the mixture null depends on that case. */
    fx0 = gex_pagels_lambda_negloglik(0.0, &data);
    fx1 = gex_pagels_lambda_negloglik(1.0, &data);
    if (fx0 <= fx1) {
        best_fx = fx0;
        best_lambda = 0.0;
    }
    else {
        best_fx = fx1;
        best_lambda = 1.0;
    }

    /* Evaluate the objective function starting at an interior point */
    lambda = 0.5;
    fx = gex_pagels_lambda_negloglik(lambda, &data);

    /* Optimize the objective function using Newton's method */
    status = opt_newton_1d(gex_pagels_lambda_negloglik,
                           &lambda,
                           &data,
                           &fx,
                           6,
                           0.0,
                           1.0,
                           NULL,
                           NULL,
                           NULL);

    /* If the optimizer converges to a finite value that is better than the boundary values, update the best solution */
    if (isfinite(fx) && fx < best_fx) {
        best_fx = fx;
        best_lambda = lambda;
    }

    /* If the optimizer does not fully converge, still retain the best valid
    value found among the boundary and interior evaluations. */
    if (status != 0 && !isfinite(best_fx))
        return -HUGE_VAL;

    *lambda_hat = best_lambda;
    return -best_fx;
}

/* Calculate the weight matrix from a phylogenetic covariance matrix.
This weight matrix approach is based on the PATH method by Schiffman et al. 2024 
Nature Genetics (PMID: 39317739) and is calculated as the element-wise inverse pairwise distance matrix.
The weight W_ij = 1/(d_ij + eps) where d_ij is the pairwise distance between tips i and j which can be
calculated from the covariance matrix as d_ij = Sigma_ii + Sigma_jj - 2*Sigma_ij and eps is a small 
constant to avoid division by zero. The weight matrix is then normalized to sum to 1.
Returns a pointer to the allocated weight matrix or NULL on failure. */
Matrix *weight_matrix_from_covariance(Matrix *Sigma) {
    int i, j;
    int n;
    double max_dist = 0.0;
    double eps;
    double total = 0.0;
    Matrix *W = NULL;

    if (Sigma == NULL || Sigma->nrows != Sigma->ncols || Sigma->nrows <= 0) {
        fprintf(stderr, "ERROR: weight_matrix_from_covariance got invalid input\n");
        return NULL;
    }

    /* Allocate the weight matrix with the same dimensions as the covariance matrix*/
    n = Sigma->nrows;  
    W = mat_new(n, n);

    /* Calculate the maximum pairwise distance from the covariance matrix to use for setting eps
    and simultaneously fill the weight matrix with initial pairwise distance values */
    for (i = 0; i < n; i++) {
        mat_set(W, i, i, 0.0);  /* Set diagonal elements (comparing each tip to itself) to zero pairwise distance */
        for (j = i + 1; j < n; j++) {
            double dij = mat_get(Sigma, i, i) + mat_get(Sigma, j, j) -
                         (2.0 * mat_get(Sigma, i, j));

            if (dij < 0.0) {
                fprintf(stderr, "ERROR: covariance implied negative distance\n");
                mat_free(W);
                return NULL;
            }

            /* Handle numerical precision issues */
            if (dij < 0.0 && fabs(dij) < 1e-12)
                dij = 0.0;
            
            /* Update the maximum distance for setting relative eps */
            if (dij > max_dist)
                max_dist = dij;

            /* Store the pairwise distance in the weight matrix temporarily for now, will convert to weights after setting eps */
            mat_set(W, i, j, dij);
            mat_set(W, j, i, dij);
        }
    }

    /* Set the epsilon value as a relative tolerance based on the maximum distance */
    eps = (max_dist > 0.0 ? 1e-8 * max_dist : 1e-8);

    /* Fill the weight matrix */
    for (i = 0; i < n; i++) {
        mat_set(W, i, i, 0.0);
        for (j = i + 1; j < n; j++) {
            double dij = mat_get(W, i, j);
            double wij = 1.0 / (dij + eps);
            mat_set(W, i, j, wij);
            mat_set(W, j, i, wij);
            total += 2.0 * wij;
        }
    }

    if (total <= 0.0) {
        fprintf(stderr, "ERROR: weight matrix normalization failed\n");
        mat_free(W);
        return NULL;
    }

    /* Normalize the weight matrix to sum to 1 */
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++)
            mat_set(W, i, j, mat_get(W, i, j) / total);
    }

    return W;
}

/* Print a summary of the covariance-based weight matrix. */
void print_weight_matrix_summary(Matrix *W) {
    int i, j;

    if (W == NULL) {
        fprintf(stderr, "ERROR: cannot summarize NULL weight matrix\n");
        return;
    }

    printf("\n");
    printf("First few entries of covariance-based weight matrix:\n");
    for (i = 0; i < W->nrows && i < 10; i++) {
        for (j = 0; j < W->ncols && j < 10; j++)
            printf(" %g", mat_get(W, i, j));
        printf("\n");
    }
    printf("\n");
}


/* Compute Moran's I for each gene in the expression matrix.
Moran's I is a measure of spatial autocorrelation, which is
calculated here as the expectation over trees of Z^T * W_t * Z,
where z is the column-wise standardized gene expression matrix and
W_t is derived internally from the Brownian covariance matrix Sigma_t.
Returns a pointer to the result structure. */
MoranResult *gex_compute_morans_i(Matrix *X,
                                      Matrix **Sigmas,
                                      int n_sigmas,
                                      int n_perm) {
    int j, k, t;
    int n_cells = X->nrows;
    int n_genes = X->ncols;
    Matrix *per_W = NULL;   /* Weight matrix temp for each covariance matrix */
    Matrix *W = mat_new(n_cells, n_cells);   /* Expected weight matrix across trees */
    Matrix *B = mat_new(n_cells, n_genes);   /* Intermediate matrix for E[W] * Z */
    Matrix *Z = mat_new(n_cells, n_genes);   /* Standardized gene expression matrix */
    Matrix *Zt = mat_new(n_genes, n_cells);  /* Transpose of Z for computing Z^T * B */
    Matrix *corr = mat_new(n_genes, n_genes);    /* Moran's I correlation matrix */
    MoranResult *res = scalloc(1, sizeof(MoranResult));    /* Result structure for Moran's I computation */
    Matrix *permZ = mat_new(n_cells, n_genes);   /* Temp matrix to hold permuted columns */
    Vector *gene_perm_counts = vec_new(n_genes); /* Array to count permutations for each gene */

    /* Get the E[W] (expected weight matrix) from the covariance matrices */
    mat_zero(W);
    for (t = 0; t < n_sigmas; t++) {
        per_W = weight_matrix_from_covariance(Sigmas[t]);
        mat_add_mat(W, per_W);
        mat_free(per_W);
        per_W = NULL;
    }
    mat_scale(W, 1.0 / (double)n_sigmas);

    /* Normalize the gene expression matrix */
    mat_copy(Z, X);
    mat_standardize_cols(Z);

    /* Compute B = E[W * Z] as B = E[W] * Z */
    mat_mult(B, W, Z);

    /* Compute the Moran's I correlation matrix from Z^T x B */
    mat_trans(Zt, Z);
    mat_mult(corr, Zt, B);

    /* Setup result structure */
    res->n_genes = n_genes;
    res->morans_i = scalloc(n_genes, sizeof(double));
    for (j = 0; j < n_genes; j++) {
        res->morans_i[j] = mat_get(corr, j, j);
    }
    res->pvals = scalloc(n_genes, sizeof(double));
    res->qvals = scalloc(n_genes, sizeof(double));

    /* Run permutation tests to get Monte Carlo p-values */
    vec_zero(gene_perm_counts);

    for (k = 0; k < n_perm; k++) {
        mat_copy(permZ, Z);
        mat_col_shuffle(permZ);
        mat_mult(B, W, permZ);
        mat_trans(Zt, permZ);
        mat_mult(corr, Zt, B);

        double perm_i;
        double curr_count;
        for (j = 0; j < n_genes; j++) {
            perm_i = mat_get(corr, j, j);
            if (perm_i >= res->morans_i[j]) {
                curr_count = vec_get(gene_perm_counts, j);
                vec_set(gene_perm_counts, j, curr_count + 1.0);
            }
        }
    }

    for (j = 0; j < n_genes; j++) {
        res->pvals[j] = vec_get(gene_perm_counts, j) / (double)n_perm;
    }

    /* Adjust p-values for multiple testing and count significant genes */
    gex_bh_adjust(res->pvals, res->qvals, n_genes);
    res->n_significant = gex_count_kept_genes(res, 0.05, 0.0);

    /* Free memory */
    if (permZ != NULL)
        mat_free(permZ);
    if (gene_perm_counts != NULL)
        vec_free(gene_perm_counts);
    if (Z != NULL)
        mat_free(Z);
    if (Zt != NULL)
        mat_free(Zt);
    if (per_W != NULL)
        mat_free(per_W);
    if (W != NULL)
        mat_free(W);
    if (B != NULL)
        mat_free(B);
    if (corr != NULL)
        mat_free(corr);
    
    return res;
}

/* Write Moran's I results to a TSV file */
void write_moran_tsv(const char *filename,
                         MoranResult *res,
                         GexMatrix *gex,
                         double max_q,
                         double min_i) {
    int i;
    FILE *out;

    out = fopen(filename, "w");
    fprintf(out, "gene\tmorans_I\tp_value\tq_value\tkeep\n");
    for (i = 0; i < res->n_genes; i++) {
        int keep = (res->qvals[i] <= max_q && res->morans_i[i] > min_i);
        fprintf(out, "%s\t%.17g\t%.17g\t%.17g\t%s\n",
                gex->gene_names[i],
                res->morans_i[i],
                res->pvals[i],
                res->qvals[i],
                (keep ? "True" : "False"));
    }
    fclose(out);
}

/* Compute the Brownian LRT for a given expression matrix and tree-set of phylogenetic covariance matrices.
   The LRT compares:

     Null model:   y ~ N(mu, sigma^2 * I)
                   (no phylogenetic structure; identity covariance used)

     Alternative:  y ~ N(mu, sigma^2 * Sigma_t)
                   (Brownian motion on each tree t; the alternative log-likelihood
                   is averaged across the supplied tree set)

   For each gene, the function computes the log-likelihood under both models
   and forms the likelihood ratio statistic LRT = 2 * (logLik_alt - logLik_null).
   P-values are obtained either from Monte Carlo simulation under the null
   for the full Brownian alternative or from the standard 50:50 mixture of
   chi-square with 1 dof and a point mass at zero for the Pagel's lambda
   alternative. Returns a pointer to the result structure or NULL on failure.
*/
GexLRTResult *gex_compute_brownian_lrt(GexMatrix *gex,
                                       Matrix **Sigmas,
                                       int n_sigmas,
                                       int n_mc,
                                       unsigned int seed,
                                       GexLRTAltMode alt_mode) {
    int i, j, t;   /* Loop indices */
    int n;  /* Number of cells */
    Matrix **Sigma_regs = NULL;   /* Regularized covariance matrices */
    Matrix **Ls = NULL;   /* Cholesky factors of regularized covariance matrices */
    Matrix **Sigma_lambdas = NULL;   /* Lambda-transformed covariance matrices */
    GexLRTResult *res = NULL;   /* Result structure for the LRT computation */
    double *logdet_sigmas = NULL;   /* Log determinants of the regularized covariance matrices */
    double *jitters = NULL;   /* Per-tree diagonal jitters for numerical stability */
    double *y = NULL;   /* Vector for storing the expression values */
    double *y_sim = NULL;   /* Vector for storing simulated expression values */
    unsigned int rng_state; /* Random number generator state */

    if (gex == NULL || gex->X == NULL || Sigmas == NULL || n_sigmas <= 0 ||
        alt_mode < GEX_LRT_ALT_FULL || alt_mode > GEX_LRT_ALT_LAMBDA ||
        (alt_mode == GEX_LRT_ALT_FULL && n_mc <= 0)) {
        fprintf(stderr, "ERROR: gex_compute_brownian_lrt got invalid input\n");
        return NULL;
    }

    n = gex->X->nrows;
    Sigma_regs = scalloc(n_sigmas, sizeof(Matrix *));
    Ls = scalloc(n_sigmas, sizeof(Matrix *));
    logdet_sigmas = scalloc(n_sigmas, sizeof(double));
    jitters = scalloc(n_sigmas, sizeof(double));
    if (alt_mode == GEX_LRT_ALT_LAMBDA)
        Sigma_lambdas = scalloc(n_sigmas, sizeof(Matrix *));

    for (t = 0; t < n_sigmas; t++) {
        double max_diag = 0.0;

        if (Sigmas[t] == NULL || Sigmas[t]->nrows != n || Sigmas[t]->ncols != n) {
            fprintf(stderr, "ERROR: covariance matrix dimensions do not match number of cells for tree %d\n",
                    t + 1);
            return NULL;
        }

        Sigma_regs[t] = mat_new(n, n);
        Ls[t] = mat_new(n, n);
        if (alt_mode == GEX_LRT_ALT_LAMBDA)
            Sigma_lambdas[t] = mat_new(n, n);

        for (i = 0; i < n; i++) {
            double d = mat_get(Sigmas[t], i, i);
            if (d > max_diag)
                max_diag = d;
        }
        jitters[t] = (max_diag > 0.0 ? 1e-8 * max_diag : 1e-8);

        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++)
                mat_set(Sigma_regs[t], i, j, mat_get(Sigmas[t], i, j));
            mat_set(Sigma_regs[t], i, i, mat_get(Sigma_regs[t], i, i) + jitters[t]);
        }

        /* Compute the Chloesky factor and log determinant of the covariance matrix once
        and store the results */
        if (mat_cholesky(Ls[t], Sigma_regs[t]) != 0) {
            fprintf(stderr, "ERROR: failed to initialize Brownian covariance matrices for LRT tree %d\n",
                    t + 1);
            return NULL;
        }
        for (i = 0; i < n; i++) {
            double diag = mat_get(Ls[t], i, i);
            if (diag <= 0.0) {
                fprintf(stderr, "ERROR: invalid Cholesky factor for LRT tree %d\n", t + 1);
                return NULL;
            }
            logdet_sigmas[t] += 2.0 * log(diag);
        }
    }

    /* Initialize result structure and temporary vectors for the LRT computation */
    res = scalloc(1, sizeof(GexLRTResult));
    y = smalloc(n * sizeof(double));
    if (alt_mode == GEX_LRT_ALT_FULL)
        y_sim = smalloc(n * sizeof(double));
    res->ll_null = scalloc(gex->X->ncols, sizeof(double));
    res->ll_alt = scalloc(gex->X->ncols, sizeof(double));
    if (alt_mode == GEX_LRT_ALT_LAMBDA)
        res->lambda_hat = scalloc(gex->X->ncols, sizeof(double));
    res->lrt_stat = scalloc(gex->X->ncols, sizeof(double));
    res->pvals = scalloc(gex->X->ncols, sizeof(double));
    res->qvals = scalloc(gex->X->ncols, sizeof(double));
    res->alt_mode = alt_mode;
    res->n_genes = gex->X->ncols;

    /* Run the LRT on each gene */
    rng_state = (seed == 0u ? 1u : seed);
    for (j = 0; j < gex->X->ncols; j++) {
        double ll_null; /* Log-likelihood under the null model */
        double ll_alt;  /* Log-likelihood under the alternative model */
        double *ll_alts; /* Log-likelihoods under the alternative model for each tree */
        double mu0; /* Mean under the null model (estimated from the data) */
        double sigma20; /* Variance under the null model (estimated from the data) */

        /* Extract gene expression data for the current gene across all cells */
        for (i = 0; i < n; i++)
            y[i] = mat_get(gex->X, i, j);

        /* Calculate the mean and variance of the gene expression data */
        calculate_mean_variance(y, n, &mu0, &sigma20);

        /* Compute the log-likelihood under the null model, which is the same
        independent of the alternative model */
        ll_null = gex_loglik_centered_gaussian_identity(n, sigma20);
        res->ll_null[j] = ll_null;

        /* Compute the log-likelihood under the alternative model */
        ll_alts = smalloc(n_sigmas * sizeof(double));
        ll_alt = 0.0;
        if (alt_mode == GEX_LRT_ALT_FULL) {
            for (t = 0; t < n_sigmas; t++) {
                ll_alts[t] = gex_loglik_centered_gaussian_chol(y,
                                                            Ls[t],
                                                            logdet_sigmas[t]);
            }
            if (n_sigmas > 1) {
                ll_alt = logsumexp(ll_alts, n_sigmas) - log((double)n_sigmas);
            } else {
                ll_alt = ll_alts[0];
            }
        }
        else {
            double lambda_hat_sum = 0.0;
            for (t = 0; t < n_sigmas; t++) {
                double lambda_hat = 0.0;
                ll_alts[t] = gex_fit_pagels_lambda_loglik(y,
                                                    Sigmas[t],
                                                    jitters[t],
                                                    Sigma_lambdas[t],
                                                    Ls[t],
                                                    &lambda_hat);
                lambda_hat_sum += lambda_hat;
            }
            if (n_sigmas > 1) {
                ll_alt = logsumexp(ll_alts, n_sigmas) - log((double)n_sigmas);
                res->lambda_hat[j] = lambda_hat_sum / (double)n_sigmas;
            } else {
                ll_alt = ll_alts[0];
                res->lambda_hat[j] = lambda_hat_sum;
            }
        }
        res->ll_alt[j] = ll_alt;

        /* Compute the LRT statistic */
        res->lrt_stat[j] = 2.0 * (ll_alt - ll_null);

        if (alt_mode == GEX_LRT_ALT_FULL) {
            /* Estimate p-values by simulating data under the null model to
            use a monte carlo estimate. */
            int ge_count = 0;
            int rep;
            for (rep = 0; rep < n_mc; rep++) {
                double ll0_sim;
                double ll1_sim;
                double stat_sim;
                for (i = 0; i < n; i++)
                    /* Draw simulated data independently from the null model N(μ0, σ20) */
                    y_sim[i] = mu0 + sqrt(sigma20) * rand_normal(&rng_state);

                /* Re-calculate the mean and variance of the simulated gene expression data.
                This is necessary because we only simulate finite samples, so we are not guaranteed
                to have the generating distribution mean and variance parameters exactly. */
                calculate_mean_variance(y_sim, n, &mu0, &sigma20);

                /* Compute the log-likelihood under the null model */
                ll0_sim = gex_loglik_centered_gaussian_identity(n, sigma20);

                /* Compute the expected log-likelihood under the alternative model */
                ll1_sim = 0.0;
                for (t = 0; t < n_sigmas; t++) {
                    ll1_sim += gex_loglik_centered_gaussian_chol(y_sim,
                                                                Ls[t],
                                                                logdet_sigmas[t]);
                }
                ll1_sim /= (double)n_sigmas;

                /* Compute the LRT statistic for the simulated data */
                stat_sim = 2.0 * (ll1_sim - ll0_sim);
                if (stat_sim >= res->lrt_stat[j])
                    ge_count++;
            }
            res->pvals[j] = ((double)ge_count + 1.0) / ((double)n_mc + 1.0);
        }
        else {
            /* Under the boundary null lambda = 0, use the standard 50:50
            mixture of a point mass at zero and chi-square with 1 dof. */
            res->pvals[j] = half_chisq_cdf(res->lrt_stat[j], 1.0, FALSE);
        }
    }

    gex_bh_adjust(res->pvals, res->qvals, res->n_genes);    /* Adjust p-values for multiple testing */
    res->n_significant = gex_count_kept_lrt_genes(res, 0.05);   /* Number of significant genes to keep*/

    /* Free memory */
    free(y);
    free(y_sim);
    if (Sigma_regs != NULL) {
        for (t = 0; t < n_sigmas; t++) {
            if (Sigma_regs[t] != NULL)
                mat_free(Sigma_regs[t]);
        }
        free(Sigma_regs);
    }
    if (Ls != NULL) {
        for (t = 0; t < n_sigmas; t++) {
            if (Ls[t] != NULL)
                mat_free(Ls[t]);
        }
        free(Ls);
    }
    if (Sigma_lambdas != NULL) {
        for (t = 0; t < n_sigmas; t++) {
            if (Sigma_lambdas[t] != NULL)
                mat_free(Sigma_lambdas[t]);
        }
        free(Sigma_lambdas);
    }
    free(logdet_sigmas);
    free(jitters);

    return res;
}

void write_lrt_tsv(const char *filename,
                      GexLRTResult *res,
                      GexMatrix *gex,
                      double max_q) {
    int i;
    FILE *out;

    out = fopen(filename, "w");

    if (res->alt_mode == GEX_LRT_ALT_LAMBDA && res->lambda_hat != NULL)
        fprintf(out, "gene\tll_null\tll_alt\tlambda_hat\tlrt_stat\tp_value\tq_value\tkeep\n");
    else
        fprintf(out, "gene\tll_null\tll_alt\tlrt_stat\tp_value\tq_value\tkeep\n");

    for (i = 0; i < res->n_genes; i++) {
        int keep = (res->qvals[i] <= max_q && res->lrt_stat[i] > 0.0);
        if (res->alt_mode == GEX_LRT_ALT_LAMBDA && res->lambda_hat != NULL) {
            fprintf(out, "%s\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%s\n",
                    gex->gene_names[i],
                    res->ll_null[i],
                    res->ll_alt[i],
                    res->lambda_hat[i],
                    res->lrt_stat[i],
                    res->pvals[i],
                    res->qvals[i],
                    (keep ? "True" : "False"));
        }
        else {
            fprintf(out, "%s\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%s\n",
                    gex->gene_names[i],
                    res->ll_null[i],
                    res->ll_alt[i],
                    res->lrt_stat[i],
                    res->pvals[i],
                    res->qvals[i],
                    (keep ? "True" : "False"));
        }
    }
    fclose(out);
}

void free_moran_result(MoranResult *res) {
    if (res == NULL)
        return;
    if (res->morans_i != NULL)
        free(res->morans_i);
    if (res->pvals != NULL)
        free(res->pvals);
    if (res->qvals != NULL)
        free(res->qvals);

    free(res);
}

void gex_free_lrt_result(GexLRTResult *res) {
    if (res == NULL)
        return;

    if (res->lrt_stat != NULL)
        free(res->lrt_stat);
    if (res->pvals != NULL)
        free(res->pvals);
    if (res->qvals != NULL)
        free(res->qvals);
    if (res->ll_null != NULL)
        free(res->ll_null);
    if (res->ll_alt != NULL)
        free(res->ll_alt);
    if (res->lambda_hat != NULL)
        free(res->lambda_hat);
    free(res);
}

/* Determine if a gene should be kept based on the specified 
filter mode and thresholds. */
static int gex_keep_gene(MoranResult *morans,
                         GexLRTResult *lrt,
                         int gene_idx,
                         GexFilterMode mode,
                         double max_q,
                         double min_i) {
    int keep_moran = 0; /* Flag indicating if the gene passes the Moran's I filter */
    int keep_lrt = 0;   /* Flag indicating if the gene passes the LRT filter */

    /* Apply the filters based on if the provided objects are not NULL */
    if (morans != NULL)
        keep_moran = (morans->qvals[gene_idx] <= max_q &&
                      morans->morans_i[gene_idx] > min_i);
    if (lrt != NULL)
        keep_lrt = (lrt->qvals[gene_idx] <= max_q &&
                    lrt->lrt_stat[gene_idx] > 0.0);

    /* Return the appropriate filter result based on the filter mode */
    if (mode == GEX_FILTER_MORAN)
        return keep_moran;
    if (mode == GEX_FILTER_LRT)
        return keep_lrt;
    /* Return the intersection if both tests are run */
    return (keep_moran && keep_lrt);
}

/* Filter genes based on LRT and Moran's I results 
to keep only those passing the filter(s) with the 
given significance and signal strength thresholds.
Returns a pointer to a new GexMatrix containing only 
the filtered genes. */
GexMatrix *gex_filter_genes_by_results(GexMatrix *gex,
                                       MoranResult *morans,
                                       GexLRTResult *lrt,
                                       GexFilterMode mode,
                                       double max_q,
                                       double min_i) {
    int i, j;   /* Loop indices */
    int out_j = 0;  /* Index for the output matrix */
    int nkeep = 0;  /* Number of genes retained */
    GexMatrix *out = NULL;  /* Output matrix */

    if (gex == NULL)
        return NULL;
    if ((mode == GEX_FILTER_MORAN || mode == GEX_FILTER_BOTH) &&
        (morans == NULL || gex->X->ncols != morans->n_genes))
        return NULL;
    if ((mode == GEX_FILTER_LRT || mode == GEX_FILTER_BOTH) &&
        (lrt == NULL || gex->X->ncols != lrt->n_genes))
        return NULL;

    /* Count how many genes pass the filter(s) to determine the size of the output matrix */
    for (j = 0; j < gex->X->ncols; j++) {
        if (gex_keep_gene(morans, lrt, j, mode, max_q, min_i))
            nkeep++;
    }
    if (nkeep <= 0) {
        fprintf(stderr, "ERROR: no genes passed the filter(s)\n");
        return NULL;
    }

    /* Initialize the output matrix */
    out = scalloc(1, sizeof(GexMatrix));
    out->X = mat_new(gex->X->nrows, nkeep);
    out->cell_names = smalloc(gex->X->nrows * sizeof(char *));
    out->gene_names = smalloc(nkeep * sizeof(char *));

    /* Copy cell names */
    for (i = 0; i < out->X->nrows; i++) {
        out->cell_names[i] = strdup(gex->cell_names[i]);
        if (out->cell_names[i] == NULL) {
            gex_free_matrix_data(out);
            return NULL;
        }
    }

    /* Fill the output matrix with passing genes  */
    for (j = 0; j < gex->X->ncols; j++) {
        if (gex_keep_gene(morans, lrt, j, mode, max_q, min_i)) {
            /* Copy gene name that passed the filter(s) */
            out->gene_names[out_j] = strdup(gex->gene_names[j]);
            if (out->gene_names[out_j] == NULL) {
                gex_free_matrix_data(out);
                return NULL;
            }
            /* Copy expression values for the passing gene */
            for (i = 0; i < gex->X->nrows; i++)
                mat_set(out->X, i, out_j, mat_get(gex->X, i, j));
            out_j++;
        }
    }

    return out;
}
