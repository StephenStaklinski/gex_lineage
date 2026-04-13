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

/* Calculate the inverse pairwise distance weight matrix from a phylogenetic covariance matrix.
This weight matrix approach is based on the PATH method by Schiffman et al. 2024 
Nature Genetics (PMID: 39317739). */
Matrix *weight_matrix_from_covariance(Matrix *Sigma) {
    int i, j;
    int n;
    double wij;
    Matrix *W = NULL;

    /* Allocate the weight matrix with the same dimensions as the covariance matrix*/
    n = Sigma->nrows;  
    W = mat_new(n, n);

    for (i = 0; i < n; i++) {
        mat_set(W, i, i, 0.0);  /* Set diagonal elements to zero pairwise distance */
        for (j = i + 1; j < n; j++) {
            double dij = mat_get(Sigma, i, i) + mat_get(Sigma, j, j) -
                         (2.0 * mat_get(Sigma, i, j));

            /* Set the weight as the inverse pairwise distance */
            wij = 1.0 / dij;
            mat_set(W, i, j, wij);
            mat_set(W, j, i, wij);
        }
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

/* Compute Moran's I-based phylogenetic autocorrelation for each 
gene in the expression matrix. This function is written to match 
the xcor function from the R package PATH by Schiffman et al. 
2024 Nature Genetics (PMID: 39317739). 

Note: There were some inifficiencies in the PATH implementation
for which some of them I have fixed here. I left commented code for the exact
match to the original code. There are still inefficiencies left behind
to fix in the future if a fully optimized version is needed.
*/
MoranResult *gex_compute_morans_i(Matrix *X,
                                      Matrix **Sigmas,
                                      int n_sigmas) {
    int j, t;
    int n = X->nrows;
    int n_genes = X->ncols;
    double w;
    Matrix *per_W = NULL;   /* Weight matrix temp for each covariance matrix */
    Matrix *W = mat_new(n, n);   /* Expected weight matrix across trees */

    /* Get the E[W] (expected weight matrix) from the covariance matrices */
    mat_zero(W);
    for (t = 0; t < n_sigmas; t++) {
        per_W = weight_matrix_from_covariance(Sigmas[t]);
        mat_add_mat(W, per_W);
        mat_free(per_W);
        per_W = NULL;
    }
    mat_scale(W, 1.0 / (double)n_sigmas);
    /* Normalize all entries to sum to 1 */
    w = mat_sum_entries(W);
    mat_scale(W, 1.0 / w);
    w = 1.0;  /* W was already normalized to sum to 1 */

    /* Free memory */
    if (per_W != NULL)
        mat_free(per_W);

    /* Center genes */
    Matrix *d0 = mat_new(n, n_genes);
    mat_copy(d0, X);
    mat_center_cols(d0);

    /* Compute the covariance matrix of genes */
    Matrix *d0t = mat_new(n_genes, n);
    mat_trans(d0t, d0);
    Matrix *d1 = mat_new(n_genes, n_genes);
    mat_mult(d1, d0t, d0);
    mat_scale(d1, 1.0 / (double)n);

    Vector *d1_diag = mat_get_diag(d1);
    Matrix *d1_2 = mat_new(n_genes, n_genes);
    vec_outer_prod(d1_2, d1_diag, d1_diag);
    Matrix *d1_2_sqrt = mat_new(n_genes, n_genes);
    mat_copy(d1_2_sqrt, d1_2);
    mat_sqrt_elementwise(d1_2_sqrt);

    /* Free memory */
    if (d1 != NULL)
        mat_free(d1);
    if (d1_diag != NULL)
        vec_free(d1_diag);
    if (d1_2 != NULL)
        mat_free(d1_2);

    /* Unused code from PATH, retained but commented out here */
    // /* Compute the matrix of cross-products of squared centered gene values. */
    // Matrix *d0squared = mat_new(n, n_genes);
    // mat_copy(d0squared, d0);
    // mat_square_elementwise(d0squared);
    // Matrix *d0squaredt = mat_new(n_genes, n);
    // mat_trans(d0squaredt, d0squared);
    // Matrix *d2 = mat_new(n_genes, n_genes);
    // mat_mult(d2, d0squaredt, d0squared);
    // mat_scale(d2, 1.0 / (double)n);

    // /* Free memory */
    // if (d0squared != NULL)
    //     mat_free(d0squared);
    // if (d0squaredt != NULL)
    //     mat_free(d0squaredt);
    // if (d2 != NULL)
    //     mat_free(d2);

    /* Unnecessary S3 calculation from PATH, retained but commented out here.
    Simpler calculation below for S4 which is equivalent to S3 here since W 
    is symmetric */
    // /* Sum of all elements in W^T * W */
    // Matrix *Wt = mat_new(n, n);
    // mat_trans(Wt, W);
    // Matrix *WtW = mat_new(n, n);
    // mat_mult_elementwise(WtW, Wt, W);
    // double S3 = mat_sum_entries(WtW);

    // /* Free memory */
    // if (Wt != NULL)
    //     mat_free(Wt);
    // if (WtW != NULL)
    //     mat_free(WtW);

    /* Sum of squared W elements */
    double S4 = mat_sum_squared_entries(W);
    double S3 = S4; /* Same as S4 since W is symmetric, original calculation code commented out above */

    /* Sum of all elements in W*W */
    Matrix *WW = mat_new(n, n);
    mat_mult(WW, W, W);
    double S5 = mat_sum_entries(WW);

    /* Free memory */
    if (WW != NULL)
        mat_free(WW);

    /* Sum of rowSum and colSum from W */
    Vector *WrowSums = mat_row_sums(W);
    double sum_WrowSums_squared = vec_sum_squared_entries(WrowSums);
    // Vector *WcolSums = mat_col_sums(W);
    // double sum_WcolSums_squared = vec_sum_squared_entries(WcolSums);
    // double S6 = sum_WrowSums_squared + sum_WcolSums_squared;
    double S6 = 2.0 * sum_WrowSums_squared; /* Same as sum_WcolSums_squared since W is symmetric, original calculation code commented out above */

    /* Free memory */
    if (WrowSums != NULL)
        vec_free(WrowSums);
    // if (WcolSums != NULL)
    //     vec_free(WcolSums);

    double S1 = S3 + S4;
    double S2 = 2.0 * S5 + S6;

    /* Calculate the Moran's I correlation matrix */
    Matrix *B = mat_new(n, n_genes);   /* Intermediate matrix for E[W] * Z */
    mat_mult(B, W, d0);
    Matrix *Iraw = mat_new(n_genes, n_genes);
    mat_mult(Iraw, d0t, B);
    Matrix *I = mat_new(n_genes, n_genes);
    mat_div_elementwise(I, Iraw, d1_2_sqrt);

    /* Free memory */
    if (d0t != NULL)
        mat_free(d0t);
    if (B != NULL)
        mat_free(B);

    /* Get the expected statistic under the null */
    double E_I2 = -(1.0 / (double)(n - 1));

    /* Get the variance terms for each gene */
    double A = 2.0 * (w * w - S2 + S1)
             + (2.0 * S3 - 2.0 * S5) * (n - 3)
             + S3 * (n - 2) * (n - 3);

    double Bterm = 6.0 * (w * w - S2 + S1)
                 + (4.0 * S1 - 2.0 * S2) * (n - 3)
                 + S1 * (n - 2) * (n - 3);

    double Cterm = (w * w - S2 + S1)
                 + (2.0 * S4 - S6) * (n - 3)
                 + S4 * (n - 2) * (n - 3);

    double denom = (double)(n - 1) * (n - 2) * (n - 3) * (w * w);

    double *Vjs = smalloc(n_genes * sizeof(double));

    for (j = 0; j < n_genes; j++) {
        int i;
        double ss2 = 0.0;
        double ss4 = 0.0;
        double d1j, d2j;

        for (i = 0; i < n; i++) {
            double z = mat_get(d0, i, j);
            double z2 = z * z;
            ss2 += z2;
            ss4 += z2 * z2;
        }

        d1j = ss2 / (double)n;
        d2j = ss4 / (double)n;

        if (d1j <= 0.0 || denom <= 0.0) {
            Vjs[j] = 0.0;
            continue;
        }

        Vjs[j] = (n * A
                  - (d2j / (d1j * d1j)) * Bterm
                  + n * Cterm) / denom
               - 1.0 / ((double)(n - 1) * (n - 1));

        if (!isfinite(Vjs[j]) || Vjs[j] <= 0.0)
            Vjs[j] = 0.0;
    }

    /* Setup result structure */
    MoranResult *res = scalloc(1, sizeof(MoranResult));
    res->n_genes = n_genes;
    res->morans_i = scalloc(n_genes, sizeof(double));
    res->pvals = scalloc(n_genes, sizeof(double));
    res->qvals = scalloc(n_genes, sizeof(double));
    res->zscores = scalloc(n_genes, sizeof(double));

    /* Copy the diagonal statistic and compute analytical p-values
    from z-scores */
    for (j = 0; j < n_genes; j++) {
        res->morans_i[j] = mat_get(I, j, j);
        res->zscores[j] = (res->morans_i[j] - E_I2) / sqrt(Vjs[j]);
        res->pvals[j] = erfc(fabs(res->zscores[j]) / sqrt(2.0));
    }

    /* Adjust p-values for multiple testing and count significant genes */
    gex_bh_adjust(res->pvals, res->qvals, n_genes);

    /* Free memory */
    if (W != NULL)
        mat_free(W);
    if (d0 != NULL)
        mat_free(d0);
    if (I != NULL)
        mat_free(I);
    if (Vjs != NULL)
        free(Vjs);

    return res;
}

/* Write Moran's I results to a TSV file */
void write_moran_tsv(const char *filename,
                         MoranResult *res,
                         GexMatrix *gex,
                         double max_q) {
    int i;
    FILE *out;

    out = fopen(filename, "w");
    fprintf(out, "gene\tphy_cor\tz_score\tp_value\tpadj\tkeep\n");
    for (i = 0; i < res->n_genes; i++) {
        int keep = (res->qvals[i] <= max_q);
        fprintf(out, "%s\t%.17g\t%.17g\t%.17g\t%.17g\t%s\n",
                gex->gene_names[i],
                res->morans_i[i],
                res->zscores[i],
                res->pvals[i],
                res->qvals[i],
                (keep ? "True" : "False"));
    }
    fclose(out);
}


/* Return the log-likelihood of a centered Gaussian with identity covariance */
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
    
    if (!isfinite(ll))
        ll = -HUGE_VAL;

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
} GexPagelsLambdaOptData;

/* Calculate the lambda adjusted covariance matrix to fit the model with */
static double gex_pagels_lambda_negloglik(double lambda, void *data) {
    GexPagelsLambdaOptData *d = (GexPagelsLambdaOptData *)data;

    int i, j;    /* Loop indices */
    int n = d->Sigma->nrows;  /* Number of cells */
    double logdet_sigma = 0.0;  /* Log determinant of the lambda-transformed covariance matrix */
    double eps = 1e-12;

    /* Check that lambda is in the valid range [0, 1] */
    if (lambda < 0.0 || lambda > 1.0)
        return -HUGE_VAL;
    
    mat_copy(d->Sigma_lambda, d->Sigma);  /* Start with the original covariance matrix */

    /* Diagonal elements are the same, adjusted for numerical stability */
    for (i = 0; i < n; i++)
        mat_set(d->Sigma_lambda, i, i, mat_get(d->Sigma, i, i) + eps);
    
    double lambda_scaled;
    for (i = 0; i < n; i++) {
        for (j = i + 1; j < n; j++) {
            lambda_scaled = lambda * mat_get(d->Sigma, i, j);  /* Off-diagonal elements are scaled by lambda */
            mat_set(d->Sigma_lambda, i, j, lambda_scaled);
            mat_set(d->Sigma_lambda, j, i, lambda_scaled);
        }
    }

    /* Compute the Cholesky decomposition of the lambda-transformed covariance matrix */
    if (mat_cholesky(d->L, d->Sigma_lambda) != 0)
        return -HUGE_VAL;

    /* Compute the log determinant of the lambda-transformed covariance matrix */
    for (i = 0; i < n; i++) {
        double diag = mat_get(d->L, i, i);
        if (diag <= 0.0)
            return -HUGE_VAL;
        logdet_sigma += log(diag);
    }
    logdet_sigma *= 2.0;

    return -gex_loglik_centered_gaussian_chol(d->y, d->L, logdet_sigma);
}

/* Fit Pagel's lambda using PHAST's bounded one-dimensional optimizer.
Returns the optimized log-likelihood and sets lambda_hat to the MLE. */
static double gex_fit_pagels_lambda_loglik(double *y,
                                           Matrix *Sigma,
                                           double *lambda_hat) {
    int status;  /* Loop index and optimization status */
    double lambda;  /* Parameter for Pagel's lambda */
    double fx;  /* Objective function value */
    double fx0; /* Objective function value at lambda = 0 (null model) */
    double fx1; /* Objective function value at lambda = 1 (full model) */
    double best_lambda; /* Best lambda value found among boundary and interior evaluations */
    double best_fx; /* Best objective function value found among boundary and interior evaluations */
    GexPagelsLambdaOptData data;    /* Data structure to pass to the objective function for optimization */

    /* Set up the data structure for optimization */
    data.y = y;
    data.Sigma = Sigma;
    data.Sigma_lambda = mat_new(Sigma->nrows, Sigma->ncols);  /* Lambda-transformed covariance matrix for optimization */
    data.L = mat_new(Sigma->nrows, Sigma->ncols);  /* Cholesky factor for optimization */

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

/* Compute the Brownian LRT for a given expression matrix and tree-set of phylogenetic covariance matrices.
   The LRT compares:

     Null model:   y ~ N(mu, sigma^2 * I)
                   (no phylogenetic structure; identity covariance used)

     Alternative:  y ~ N(mu, sigma^2 * Sigma_t)
                   (Brownian motion on each tree t; the alternative log-likelihood
                   is averaged across the supplied tree set; Additional alternative 
                   with Pagel's lambda transformation of the covariance matrices 
                   is also supported)

   For each gene, the function computes the log-likelihood under both models
   and forms the likelihood ratio statistic LRT = 2 * (logLik_alt - logLik_null).
   P-values are obtained either from Monte Carlo simulation under the null
   for the full Brownian alternative or from the standard 50:50 mixture of
   chi-square with 1 dof and a point mass at zero for the Pagel's lambda
   alternative.
*/
GexLRTResult *gex_compute_brownian_lrt(Matrix *X,
                                       Matrix **Sigmas,
                                       int n_sigmas,
                                       int n_perm,
                                       unsigned int seed,
                                       GexLRTAltMode alt_mode) {
    int i, j, t;   /* Loop indices */
    int n = X->nrows;  /* Number of cells */
    int n_genes = X->ncols;  /* Number of genes */
    double eps = 1e-12;  /* Jitter to add to covariance matrix diagonal elements for numerical stability */
    Matrix **Sigma_regs = NULL;   /* Regularized covariance matrices */
    Matrix **Ls = NULL;   /* Cholesky factors of regularized covariance matrices */
    GexLRTResult *res = NULL;   /* Result structure for the LRT computation */
    double *logdet_sigmas = NULL;   /* Log determinants of the regularized covariance matrices */
    double *y = NULL;   /* Vector for storing the expression values */
    unsigned int rng_state; /* Random number generator state */

    Sigma_regs = scalloc(n_sigmas, sizeof(Matrix *));
    Ls = scalloc(n_sigmas, sizeof(Matrix *));
    logdet_sigmas = scalloc(n_sigmas, sizeof(double));

    /* Pre-compute reused terms from the covariance matrices */
    for (t = 0; t < n_sigmas; t++) {
        Sigma_regs[t] = mat_new(n, n);
        mat_copy(Sigma_regs[t], Sigmas[t]);
        mat_add_diag(Sigma_regs[t], eps);   /* Add jitter to the diagonal for numerical stability */

        /* Compute the Cholesky factor of the covariance matrix */
        Ls[t] = mat_new(n, n);
        if (mat_cholesky(Ls[t], Sigma_regs[t]) != 0) {
            fprintf(stderr, "ERROR: failed cholesky decomposition for LRT tree %d\n", t + 1);
            return NULL;
        }

        /* Get the log determinant of the covariance matrix from the Cholesky diagonal elements */
        for (i = 0; i < n; i++) {
            double diag = mat_get(Ls[t], i, i);
            logdet_sigmas[t] += log(diag);
        }
        logdet_sigmas[t] *= 2.0;
    }

    /* Initialize LRT result object */
    res = scalloc(1, sizeof(GexLRTResult));
    y = smalloc(n * sizeof(double));
    res->ll_null = scalloc(n_genes, sizeof(double));
    res->ll_alt = scalloc(n_genes, sizeof(double));
    if (alt_mode == GEX_LRT_ALT_LAMBDA)
        res->lambda_hat = scalloc(n_genes, sizeof(double));
    res->lrt_stat = scalloc(n_genes, sizeof(double));
    res->pvals = scalloc(n_genes, sizeof(double));
    res->qvals = scalloc(n_genes, sizeof(double));
    res->alt_mode = alt_mode;
    res->n_genes = n_genes;

    /* Center the data */
    Matrix *X_centered = mat_new(n, n_genes);
    mat_copy(X_centered, X);
    mat_center_cols(X_centered);

    /* Run the LRT on each gene */
    rng_state = (seed == 0u ? 1u : seed);
    double ll_null; /* Log-likelihood under the null model */
    double ll_alt;  /* Log-likelihood under the alternative model */
    double *ll_alts = smalloc(n_sigmas * sizeof(double)); /* Log-likelihoods under the alternative model for each tree */
    double sigma20; /* Variance under the null model (estimated from the data) */

    for (j = 0; j < n_genes; j++) {

        /* Extract gene expression data for the current gene across all cells */
        for (i = 0; i < n; i++)
            y[i] = mat_get(X_centered, i, j);

        /* Calculate the variance of the gene expression data */
        sigma20 = 0.0;
        for (i = 0; i < n; i++)            
            sigma20 += y[i] * y[i];
        sigma20 /= (double)n;

        /* Compute the log-likelihood under the null model */
        ll_null = gex_loglik_centered_gaussian_identity(n, sigma20);
        res->ll_null[j] = ll_null;

        /* Compute the log-likelihood under the alternative model */
        ll_alt = 0.0;
        if (alt_mode == GEX_LRT_ALT_FULL) {
            for (t = 0; t < n_sigmas; t++) {
                ll_alts[t] = gex_loglik_centered_gaussian_chol(y, Ls[t], logdet_sigmas[t]);
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
            double *y_sim = smalloc(n * sizeof(double));
            int ge_count = 0;
            int rep;
            for (rep = 0; rep < n_perm; rep++) {
                for (i = 0; i < n; i++)
                    /* Draw simulated data independently from the null model N(μ0, σ20) */
                    y_sim[i] = 0 + sqrt(sigma20) * rand_normal(&rng_state);

                /* Compute the log-likelihood under the null model */
                ll_null = gex_loglik_centered_gaussian_identity(n, sigma20);

                /* Compute the expected log-likelihood under the alternative model */
                for (t = 0; t < n_sigmas; t++) {
                    ll_alts[t] = gex_loglik_centered_gaussian_chol(y_sim, Ls[t], logdet_sigmas[t]);
                }
                if (n_sigmas > 1) {
                    ll_alt = logsumexp(ll_alts, n_sigmas) - log((double)n_sigmas);
                } else {
                    ll_alt = ll_alts[0];
                }

                /* Compute the LRT statistic for the simulated data */
                double stat_sim = 2.0 * (ll_alt - ll_null);
                if (stat_sim >= res->lrt_stat[j])
                    ge_count++;
            }
            res->pvals[j] = ((double)ge_count) / ((double)n_perm);

            /* Free memory */
            free(y_sim);
        }
        else {
            /* Under the boundary null lambda = 0, use the standard 50:50
            mixture of a point mass at zero and chi-square with 1 dof. */
            res->pvals[j] = half_chisq_cdf(res->lrt_stat[j], 1.0, FALSE);
        }
    }

    gex_bh_adjust(res->pvals, res->qvals, res->n_genes);    /* Adjust p-values for multiple testing */

    /* Free memory */
    if (y != NULL)
        free(y);
    if (ll_alts != NULL)
        free(ll_alts);
    if (X_centered != NULL)
        mat_free(X_centered);
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
    if (logdet_sigmas != NULL)
        free(logdet_sigmas);

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
    if (res->zscores != NULL)
        free(res->zscores);
    if (res->pvals != NULL)
        free(res->pvals);
    if (res->qvals != NULL)
        free(res->qvals);

    free(res);
}

void free_lrt_result(GexLRTResult *res) {
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
                         double max_q) {
    int keep_moran = 0; /* Flag indicating if the gene passes the Moran's I filter */
    int keep_lrt = 0;   /* Flag indicating if the gene passes the LRT filter */

    /* Apply the filters based on if the provided objects are not NULL */
    if (morans != NULL)
        keep_moran = (morans->qvals[gene_idx] <= max_q);
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
given significance threshold. */
GexMatrix *gex_filter_genes(GexMatrix *gex,
                            MoranResult *morans,
                            GexLRTResult *lrt,
                            GexFilterMode mode,
                            double max_q) {
    int i, j;   /* Loop indices */
    int out_j = 0;  /* Index for the output matrix */
    int nkeep = 0;  /* Number of genes retained */
    GexMatrix *out = NULL;  /* Output matrix */

    /* Count how many genes pass the filter(s) to determine the size of the output matrix */
    for (j = 0; j < gex->X->ncols; j++) {
        if (gex_keep_gene(morans, lrt, j, mode, max_q))
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
        if (gex_keep_gene(morans, lrt, j, mode, max_q)) {
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
