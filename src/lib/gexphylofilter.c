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

/* Calculate the inverse pairwise distance weight matrix from a phylogenetic 
covariance matrix. This weight matrix approach is based on the PATH method 
by Schiffman et al. 2024 Nature Genetics (PMID: 39317739). */
void weight_matrix_from_covariance(Matrix *W, Matrix *Sigma) {
    int i, j;
    int n = Sigma->nrows; 
    double Sii;
    double dij;
    double wij;

    for (i = 0; i < n; i++) {
        mat_set(W, i, i, 0.0);  /* Set diagonal elements to zero pairwise distance */
        Sii = mat_get(Sigma, i, i);
        for (j = i + 1; j < n; j++) {
            dij = Sii + mat_get(Sigma, j, j) - (2.0 * mat_get(Sigma, i, j));
            dij = fmax(dij, 1e-12); /* Avoid numerical issues with near-zero distances */
            wij = 1.0 / dij; /* Set the weight as the inverse pairwise distance */
            mat_set(W, i, j, wij);
            mat_set(W, j, i, wij);
        }
    }
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
gene in the expression matrix. This function is written to functionally 
match the xcor function from the R package PATH by Schiffman et al. 
2024 Nature Genetics (PMID: 39317739). 

Note: There were some inefficiencies in the PATH implementation
that I fixed, so the code is not an exact match.
*/
MoranResult *gex_compute_morans_i(Matrix *X,
                                      Matrix **Sigmas,
                                      int n_sigmas) {
    int j, t;
    int n = X->nrows;
    int n_genes = X->ncols;
    double w;
    Matrix *perW = mat_new(n, n);  /* Per-tree weight matrix */
    Matrix *W = mat_new(n, n);   /* Expected weight matrix across trees */
    mat_zero(W);

    /* Get the E[W] (expected weight matrix) from the covariance matrices */
    for (t = 0; t < n_sigmas; t++) {
        weight_matrix_from_covariance(perW, Sigmas[t]);
        mat_add_mat(W, perW);
    }
    mat_scale(W, 1.0 / (double)n_sigmas);
    /* Normalize all entries to sum to 1, which is necessary for the 
    way calculations are setup below. */
    w = mat_sum_entries(W);
    mat_scale(W, 1.0 / w);

    /* Free memory */
    if (perW != NULL)
        mat_free(perW);

    /* Center genes */
    Matrix *d0 = mat_new(n, n_genes);
    mat_copy(d0, X);
    mat_center_cols(d0);

    /* Precompute B = W * d0 */
    Matrix *B = mat_new(n, n_genes);
    mat_mult_lapack(B, W, d0);

    /* Row sums */
    Vector *WrowSums = mat_row_sums(W);
    double row_ss = vec_sum_squared_entries(WrowSums);

    /* Free memory */
    if (WrowSums != NULL)
        vec_free(WrowSums);

    double S4 = mat_sum_squared_entries(W);
    double S1 = 2.0 * S4;
    double S5 = row_ss;      /* assuming symmetric W */
    double S6 = 2.0 * row_ss;
    double S2 = 2.0 * S5 + S6;

    /* Get the expected statistic under the null */
    double E_I2 = -(1.0 / (double)(n - 1));

    /* Get the variance terms for each gene */
    double A = 2.0 * (1.0 - S2 + S1)
             + (2.0 * S4 - 2.0 * S5) * (n - 3)
             + S4 * (n - 2) * (n - 3);

    double Bterm = 6.0 * (1.0 - S2 + S1)
                 + (4.0 * S1 - 2.0 * S2) * (n - 3)
                 + S1 * (n - 2) * (n - 3);

    double Cterm = (1.0 - S2 + S1)
                 + (2.0 * S4 - S6) * (n - 3)
                 + S4 * (n - 2) * (n - 3);

    double denom = (double)(n - 1) * (n - 2) * (n - 3);

    double *Vjs = smalloc(n_genes * sizeof(double));

    /* Setup result structure */
    MoranResult *res = scalloc(1, sizeof(MoranResult));
    res->n_genes = n_genes;
    res->morans_i = scalloc(n_genes, sizeof(double));
    res->pvals = scalloc(n_genes, sizeof(double));
    res->qvals = scalloc(n_genes, sizeof(double));
    res->zscores = scalloc(n_genes, sizeof(double));

    /* Compute only the diagonal Moran statistic and its z-score per gene. */
    for (j = 0; j < n_genes; j++) {
        double numerator = 0.0;
        double ss2 = 0.0;
        double ss4 = 0.0;
        double d1j, d2j;

        for (t = 0; t < n; t++) {
            double zi = d0->data[t][j];
            double z2 = zi * zi;
            double bi = B->data[t][j];
            numerator += zi * bi;
            ss2 += z2;
            ss4 += z2 * z2;
        }

        d1j = ss2 / (double)n;
        d2j = ss4 / (double)n;

        /* Handle degenerate cases safely */
        if (d1j <= 0.0 || denom <= 0.0) {
            Vjs[j] = 0.0;
            res->morans_i[j] = 0.0;
            res->zscores[j] = 0.0;
            res->pvals[j] = 1.0;
            continue;
        }

        Vjs[j] = (n * A
                - (d2j / (d1j * d1j)) * Bterm
                + n * Cterm) / denom
            - 1.0 / ((double)(n - 1) * (n - 1));

        res->morans_i[j] = numerator / d1j;

        if (!isfinite(Vjs[j]) || Vjs[j] <= 0.0 || !isfinite(res->morans_i[j])) {
            Vjs[j] = 0.0;
            res->zscores[j] = 0.0;
            res->pvals[j] = 1.0;
        }
        else {
            res->zscores[j] = (res->morans_i[j] - E_I2) / sqrt(Vjs[j]);
            res->pvals[j] = erfc(fabs(res->zscores[j]) / sqrt(2.0));
        }
    }

    /* Adjust p-values for multiple testing and count significant genes */
    gex_bh_adjust(res->pvals, res->qvals, n_genes);

    /* Free memory */
    if (W != NULL)
        mat_free(W);
    if (d0 != NULL)
        mat_free(d0);
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

/* Compute the profiled log-likelihood of y under y ~ N(mu * 1, sigma2 * Sigma),
where mu and sigma2 are estimated by MLE, using the Cholesky factor L of Sigma. */
static double gex_loglik_centered_gaussian_chol(Vector *yvec,
                                                Matrix *L,
                                                double logdet_sigma,
                                                Vector *ones,
                                                Vector *tmp,
                                                Vector *Sinv1,
                                                Vector *Sinvy) {
    int i;
    int n = L->nrows;

    /* GLS estimate of the mean */
    mat_forward_subst_lapack(L, ones, tmp);
    mat_backward_subst_lapack(L, tmp, Sinv1);
    mat_forward_subst_lapack(L, yvec, tmp);
    mat_backward_subst_lapack(L, tmp, Sinvy);
    double sum_Sinv1 = 0.0;
    double sum_Sinvy = 0.0;
    for (i = 0; i < n; i++) {
        sum_Sinv1 += Sinv1->data[i];
        sum_Sinvy += Sinvy->data[i];
    }
    double muhat = sum_Sinvy / sum_Sinv1;

    double quad = 0.0;
    double *ydata = yvec->data;
    double *s1 = Sinv1->data;
    double *sy = Sinvy->data;
    for (i = 0; i < n; i++) {
        double ri = ydata[i] - muhat;
        double sinv_ri = sy[i] - muhat * s1[i];
        quad += ri * sinv_ri;
    }

    double sigma2 = quad / (double)n;

    /* Compute the log-likelihood */
    return -0.5 * ((double)n * log(2.0 * M_PI * sigma2) +
                 logdet_sigma +
                 (double)n);
}

typedef struct {
    Vector *y;
    Matrix *Sigma;
    Matrix *Sigma_lambda;
    Matrix *L;
    Vector *ones;
    Vector *tmp;
    Vector *Sinv1;
    Vector *Sinvy;
} GexPagelsLambdaOptData;

/* Calculate the lambda adjusted covariance matrix to fit the model with */
static double gex_pagels_lambda_negloglik(double lambda, void *data) {
    GexPagelsLambdaOptData *d = (GexPagelsLambdaOptData *)data;
    int i, j;
    int n = d->Sigma->nrows;

    if (lambda < 0.0 || lambda > 1.0)
        return -HUGE_VAL;
    
    double lambda_scaled;
    for (i = 0; i < n; i++) {
        double *Sigma_row = d->Sigma->data[i];
        double *Sigma_lambda_row = d->Sigma_lambda->data[i];
        for (j = i + 1; j < n; j++) {
            lambda_scaled = lambda * Sigma_row[j];  /* Off-diagonal elements are scaled by lambda */
            Sigma_lambda_row[j] = lambda_scaled;
            d->Sigma_lambda->data[j][i] = lambda_scaled;
        }
    }

    /* Compute the Cholesky decomposition of the lambda-transformed covariance matrix */
    mat_cholesky(d->L, d->Sigma_lambda);
    double log_det = mat_logdet_chol(d->L);

    return -gex_loglik_centered_gaussian_chol(d->y, d->L, log_det, d->ones, d->tmp, d->Sinv1, d->Sinvy);
}

/* Fit Pagel's lambda using PHAST's bounded one-dimensional optimizer.
Returns the optimized log-likelihood and sets lambda_hat to the MLE. */
static double gex_fit_pagels_lambda_loglik(GexPagelsLambdaOptData *data,
                                           double *lambda_hat) {
    double fx;  /* Objective function value */
    double fx0; /* Objective function value at lambda = 0 (null model) */
    double fx1; /* Objective function value at lambda = 1 (full model) */
    double best_lambda; /* Best lambda value found among boundary and interior evaluations */
    double best_fx; /* Best objective function value found among boundary and interior evaluations */

    /* Explicitly evaluate both boundaries */
    fx0 = gex_pagels_lambda_negloglik(0.0, data);
    fx1 = gex_pagels_lambda_negloglik(1.0, data);
    if (fx0 <= fx1) {
        best_fx = fx0;
        best_lambda = 0.0;
    } else {
        best_fx = fx1;
        best_lambda = 1.0;
    }

    /* Only run Brent if the midpoint is lower than both boundaries,
    which is required for a valid bracket */
    double fx_mid = gex_pagels_lambda_negloglik(0.5, data);
    if (fx_mid < fx0 && fx_mid < fx1) {
        double tol = 1e-4;
        double xmin = 0.5;
        fx = opt_brent(0.0, 0.5, 1.0,
                       gex_pagels_lambda_negloglik,
                       tol,
                       &xmin,
                       data,
                       NULL);
        if (isfinite(fx) && fx < best_fx) {
            best_fx = fx;
            best_lambda = xmin;
        }
    }

    if (!isfinite(best_fx))
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
                                       GexLRTAltMode alt_mode) {
    int i, j, t;   /* Loop indices */
    int n = X->nrows;  /* Number of cells */
    int n_genes = X->ncols;  /* Number of genes */
    const double sigma_jitter = 1e-12;  /* Jitter to add to covariance matrix diagonal elements for numerical stability */
    Matrix **Ls = scalloc(n_sigmas, sizeof(Matrix *));   /* Cholesky factors of regularized covariance matrices */
    GexLRTResult *res = NULL;   /* Result structure for the LRT computation */
    double *logdet_sigmas = scalloc(n_sigmas, sizeof(double));   /* Log determinants of the regularized covariance matrices */
    Vector *y = vec_new(n);   /* Vector for storing the expression values */
    unsigned int rng_state; /* Random number generator state */
    Vector *y_sim = vec_new(n);  /* Vector for simulating data under the null model for p-value estimation */

    /* Pre-compute reused terms from the covariance matrices */
    for (t = 0; t < n_sigmas; t++) {
        /* Add jitter to the diagonal for numerical stability */
        mat_add_diag(Sigmas[t], sigma_jitter);

        /* Compute the Cholesky factor of the covariance matrix */
        Ls[t] = mat_new(n, n);
        mat_cholesky(Ls[t], Sigmas[t]);

        /* Get the log determinant of the covariance matrix from the Cholesky factor */
        logdet_sigmas[t] = mat_logdet_chol(Ls[t]);
    }

    /* Initialize LRT result object */
    res = scalloc(1, sizeof(GexLRTResult));
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
    double ll_null; /* Log-likelihood under the null model */
    double ll_alt;  /* Log-likelihood under the alternative model */
    double *ll_alts = smalloc(n_sigmas * sizeof(double)); /* Log-likelihoods under the alternative model for each tree */
    double sigma20; /* Variance under the null model (estimated from the data) */
    Vector *ones = vec_new(n);  /* Vector of ones for mean estimation in the alternative model */
    vec_set_all(ones, 1.0);
    Vector *tmp = vec_new(n);  /* Temporary vector for computations in the alternative model */
    Vector *Sinv1 = vec_new(n);  /* Temporary vector for computations in the alternative model */
    Vector *Sinvy = vec_new(n);  /* Temporary vector for computations in the alternative model */

    GexPagelsLambdaOptData data;    /* Data structure to pass to the objective function for optimization */

    /* Set up the data structure for optimization */
    data.y = y;
    data.Sigma_lambda = mat_new(n, n);  /* Lambda-transformed covariance matrix for optimization */
    data.L = mat_new(n, n);  /* Cholesky factor for optimization */
    data.ones = ones;
    data.tmp = tmp;
    data.Sinv1 = Sinv1;
    data.Sinvy = Sinvy;

    for (j = 0; j < n_genes; j++) {

        /* Extract gene expression data for the current gene across all cells */
        for (i = 0; i < n; i++)
            vec_set(y, i, mat_get(X_centered, i, j));

        /* Calculate the variance of the gene expression data */
        sigma20 = 0.0;
        for (i = 0; i < n; i++)            
            sigma20 += vec_get(y, i) * vec_get(y, i);
        sigma20 /= (double)n;

        /* Compute the log-likelihood under the null model */
        ll_null = gex_loglik_centered_gaussian_identity(n, sigma20);
        res->ll_null[j] = ll_null;

        /* Compute the log-likelihood under the alternative model */
        ll_alt = 0.0;
        if (alt_mode == GEX_LRT_ALT_FULL) {
            for (t = 0; t < n_sigmas; t++) {
                ll_alts[t] = gex_loglik_centered_gaussian_chol(y, Ls[t], logdet_sigmas[t], ones, tmp, Sinv1, Sinvy);
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
                data.Sigma = Sigmas[t];
                /* Diagonal never changes, so we set it here */
                for (i = 0; i < n; i++)
                    data.Sigma_lambda->data[i][i] = data.Sigma->data[i][i];
                double lambda_hat = 0.0;
                ll_alts[t] = gex_fit_pagels_lambda_loglik(&data,
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
            double sqrt_sigma20 = sqrt(sigma20);
            double sigma20_sim;
            for (rep = 0; rep < n_perm; rep++) {
                for (i = 0; i < n; i++) {
                    /* Draw simulated data independently from the null model N(μ0, σ20) */
                    rng_state = (unsigned int)random();
                    vec_set(y_sim, i, 0 + sqrt_sigma20 * rand_normal(&rng_state));
                }
                
                /* Recalculate the variance of the simulated data */
                sigma20_sim = 0.0;
                for (i = 0; i < n; i++)
                    sigma20_sim += vec_get(y_sim, i) * vec_get(y_sim, i);
                sigma20_sim /= (double)n;

                /* Compute the log-likelihood under the null model */
                ll_null = gex_loglik_centered_gaussian_identity(n, sigma20_sim);

                /* Compute the expected log-likelihood under the alternative model */
                for (t = 0; t < n_sigmas; t++) {
                    ll_alts[t] = gex_loglik_centered_gaussian_chol(y_sim, Ls[t], logdet_sigmas[t], ones, tmp, Sinv1, Sinvy);
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
        }
        else {
            /* Under the boundary null lambda = 0, use the standard 50:50
            mixture of a point mass at zero and chi-square with 1 dof. */
            res->pvals[j] = half_chisq_cdf(res->lrt_stat[j], 1.0, FALSE);
        }
    }

    gex_bh_adjust(res->pvals, res->qvals, res->n_genes);    /* Adjust p-values for multiple testing */

    for (t = 0; t < n_sigmas; t++) {
        /* Remove jitter from the diagonal to restore the original matrices */
        mat_add_diag(Sigmas[t], -sigma_jitter);
    }

    /* Free memory */
    if (y != NULL)
        vec_free(y);
    if (y_sim != NULL)
        vec_free(y_sim);
    if (ll_alts != NULL)
        free(ll_alts);
    if (X_centered != NULL)
        mat_free(X_centered);
    if (Ls != NULL) {
        for (t = 0; t < n_sigmas; t++) {
            if (Ls[t] != NULL)
                mat_free(Ls[t]);
        }
        free(Ls);
    }
    if (logdet_sigmas != NULL)
        free(logdet_sigmas);
    if (ones != NULL)
        vec_free(ones);
    if (tmp != NULL)
        vec_free(tmp);
    if (Sinv1 != NULL)
        vec_free(Sinv1);
    if (Sinvy != NULL)
        vec_free(Sinvy);
    if (data.Sigma_lambda != NULL)
        mat_free(data.Sigma_lambda);
    if (data.L != NULL)
        mat_free(data.L);

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
    }

    /* Fill the output matrix with passing genes  */
    for (j = 0; j < gex->X->ncols; j++) {
        if (gex_keep_gene(morans, lrt, j, mode, max_q)) {
            /* Copy gene name that passed the filter(s) */
            out->gene_names[out_j] = strdup(gex->gene_names[j]);

            /* Copy expression values for the passing gene */
            for (i = 0; i < gex->X->nrows; i++)
                mat_set(out->X, i, out_j, mat_get(gex->X, i, j));
            out_j++;
        }
    }

    return out;
}
