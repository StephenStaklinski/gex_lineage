#include "gex_model.h"

#include "pca.h"
#include <adam_scheduler.h>
#include <variational.h>

#include <math.h>
#include <phast/misc.h>
#include <stdlib.h>
#include <string.h>


/* Compute Gaussian observation negative log-likelihood and gradients for
X ~ N(ZL, sigma2_obs). Returns the full contribution to the objective,
including the Gaussian normalization constant, fills residual matrix,
and computes gradients w.r.t. Z and log(sigma2_obs). */
static double gaussian_observation_term(const GexLatentBrownianModel *model,
                                        Matrix *Xc,
                                        double sigma2_obs,
                                        Matrix *resid,
                                        Matrix *grad_Z,
                                        double *grad_log_sigma_obs) {
    int i, j, d;
    int n = Xc->nrows;
    int p = Xc->ncols;
    int k = model->Z->ncols;
    double obj = 0.0;

    /* Initialize gradient accumulator for log(sigma2_obs) */
    if (grad_log_sigma_obs != NULL)
        *grad_log_sigma_obs = 0.0;

    /* Gaussian observation model X_ij ~ N((ZL)_ij, sigma2_obs).
       Compute residuals r_ij = X_ij - (ZL)_ij and accumulate the
       negative log-likelihood and its gradient w.r.t. log(sigma2_obs). */
    for (i = 0; i < n; i++) {
        for (j = 0; j < p; j++) {
            double pred = 0.0;
            double r;

            /* Compute predicted value (ZL)_ij */
            for (d = 0; d < k; d++)
                pred += mat_get(model->Z, i, d) * mat_get(model->L, d, j);

            /* Residual is observed minus predicted */
            r = mat_get(Xc, i, j) - pred;

            /* Store residual for later use */
            mat_set(resid, i, j, r);

            /* Quadratic term of Gaussian negative log-likelihood: (1/2σ²) r^2 */
            obj += 0.5 * r * r / sigma2_obs;

            /* Gradient w.r.t. log(sigma2_obs) from quadratic term */
            if (grad_log_sigma_obs != NULL)
                *grad_log_sigma_obs += -0.5 * r * r / sigma2_obs;
        }
    }

    /* Log-determinant term of Gaussian likelihood: (np/2) log(σ²) */
    double n_entries = (double)(n * p);
    obj += 0.5 * n_entries * log(sigma2_obs);

    /* Gaussian normalization constant: (np/2) log(2π). This is not
    dependent on parameters so it does not matter for optimization but is
    included here to obtain full likelihood values. */
    obj += 0.5 * n_entries * log(2.0 * M_PI);

    /* Gradient w.r.t. log(sigma2_obs) from log(σ²) term */
    if (grad_log_sigma_obs != NULL)
        *grad_log_sigma_obs += 0.5 * n_entries;

    /* Compute gradient w.r.t. Z: dL/dZ = -(1/σ²) R L^T */
    if (grad_Z != NULL) {
        for (i = 0; i < n; i++) {
            for (d = 0; d < k; d++) {
                double gz = 0.0;

                /* Accumulate gradient contribution across features */
                for (j = 0; j < p; j++)
                    gz += -mat_get(resid, i, j) * mat_get(model->L, d, j) / sigma2_obs;

                mat_set(grad_Z, i, d, gz);
            }
        }
    }

    return obj;
}

/* Compute the full MVN log density for one vector z, including
normalization constants. Optionally stores intermediate values for reuse.*/
static double log_mvn_vec(const double *z,
                            Matrix *Sigma_inv,
                            double logdet_sigma,
                            int n,
                            double sigma2,
                            double *sigma_inv_z,
                            double *quad_out) {
    int i, ii;
    double quad = 0.0;

    if (sigma2 <= 0.0)
        return -HUGE_VAL;

    for (i = 0; i < n; i++) {
        double val = 0.0;
        for (ii = 0; ii < n; ii++)
            val += mat_get(Sigma_inv, i, ii) * z[ii];

        if (sigma_inv_z != NULL)
            sigma_inv_z[i] = val;

        quad += z[i] * val;
    }

    if (quad_out != NULL)
        *quad_out = quad;

    return -0.5 * (double)n * log(2.0 * M_PI)
           -0.5 * (double)n * log(sigma2)
           -0.5 * logdet_sigma
           -0.5 * quad / sigma2;
}

/* Compute log(sum_i exp(x[i])) in a numerically stable way using the
log-sum-exp trick: max(x) + log(sum_i exp(x[i] - max(x))). */
static double gex_model_logsumexp(double *x, int n) {
    int i;
    double max_x = -HUGE_VAL;
    double sum = 0.0;

    /* Find the maximum value in the array */
    for (i = 0; i < n; i++) {
        if (x[i] > max_x)
            max_x = x[i];
    }
    if (!isfinite(max_x))
        return max_x;

    /* Sum the exponentials */
    for (i = 0; i < n; i++)
        sum += exp(x[i] - max_x);

    /* Unlikely numerical stability check */
    if (sum <= 0.0)
        return -HUGE_VAL;

    /* Return the log-sum-exp */
    return max_x + log(sum);
}

/* Compute the mixture-of-Brownian Gaussian prior contribution for the latent
factors Z across all latent dimensions. Returns the contribution to the
objective, adds the prior gradient to Z, and computes gradients with respect
to log(sigma2_latent) for each latent factor. */
static double latent_brownian_prior_term(GexLatentBrownianModel *model,
                                         Matrix **Sigma_invs,
                                         double *logdet_sigmas,
                                         int n_sigmas,
                                         Matrix *grad_Z,
                                         double *grad_log_sigma_latent) {
    int i, d, t;
    int n = model->n_cells;
    int k = model->k;
    double obj = 0.0;
    double *prior_log_terms = NULL;
    double *prior_weights = NULL;
    double *quad_terms = NULL;
    double *z_d = NULL;
    double **sigma_inv_z_cache = NULL;

    prior_log_terms = scalloc(n_sigmas, sizeof(double));
    prior_weights = scalloc(n_sigmas, sizeof(double));
    quad_terms = scalloc(n_sigmas, sizeof(double));
    z_d = scalloc(n, sizeof(double));
    sigma_inv_z_cache = scalloc(n_sigmas, sizeof(double *));
    if (prior_log_terms == NULL || prior_weights == NULL ||
        quad_terms == NULL || z_d == NULL || sigma_inv_z_cache == NULL) {
        free(prior_log_terms);
        free(prior_weights);
        free(quad_terms);
        free(z_d);
        free(sigma_inv_z_cache);
        return HUGE_VAL;
    }

    for (t = 0; t < n_sigmas; t++) {
        sigma_inv_z_cache[t] = scalloc(n, sizeof(double));
        if (sigma_inv_z_cache[t] == NULL) {
            for (i = 0; i < t; i++)
                free(sigma_inv_z_cache[i]);
            free(prior_log_terms);
            free(prior_weights);
            free(quad_terms);
            free(z_d);
            free(sigma_inv_z_cache);
            return HUGE_VAL;
        }
    }

    /* Add the Brownian motion multivariate Gaussian mixture prior on Z for each
    latent dimension z_d, marginalizing over a set of candidate trees. */
    for (d = 0; d < k; d++) {
        double sigma2_d = model->sigma2_latent[d];
        double log_mix;
        double expected_quad_over_sigma2 = 0.0;

        for (i = 0; i < n; i++)
            z_d[i] = mat_get(model->Z, i, d);

        /* Compute full per-tree Brownian Gaussian log densities and cache
           Sigma_t^{-1} z_d and z_d^T Sigma_t^{-1} z_d for reuse. */
        for (t = 0; t < n_sigmas; t++) {
            prior_log_terms[t] = log_mvn_vec(z_d,
                                             Sigma_invs[t],
                                             logdet_sigmas[t],
                                             n,
                                             sigma2_d,
                                             sigma_inv_z_cache[t],
                                             &quad_terms[t]);
        }

        /* Marginalize over tree uncertainty with log-sum-exp. */
        log_mix = gex_model_logsumexp(prior_log_terms, n_sigmas);

        /* Add the full negative log marginal prior for z_d under the mixture:
           -log sum_t N(z_d | 0, sigma2_d * Sigma_t). */
        obj += -log_mix;

        /* Compute responsibility-weighted gradients over trees. */
        for (t = 0; t < n_sigmas; t++) {
            double weight = exp(prior_log_terms[t] - log_mix);
            prior_weights[t] = weight;

            for (i = 0; i < n; i++) {
                double old_grad = mat_get(grad_Z, i, d);
                mat_set(grad_Z, i, d,
                        old_grad + weight * sigma_inv_z_cache[t][i] / sigma2_d);
            }

            expected_quad_over_sigma2 += weight * quad_terms[t] / sigma2_d;
        }

        /* Gradient with respect to log(sigma2_d). */
        grad_log_sigma_latent[d] =
            0.5 * (double)n - 0.5 * expected_quad_over_sigma2;
    }

    for (t = 0; t < n_sigmas; t++)
        free(sigma_inv_z_cache[t]);
    free(sigma_inv_z_cache);
    free(prior_log_terms);
    free(prior_weights);
    free(quad_terms);
    free(z_d);

    return obj;
}

/* Compute the gradient with respect to L under the Gaussian observation model
   and add the L2 regularization penalty on L. Returns the contribution of the
   L2 penalty to the objective and fills grad_L. */
static double l2_regularized_L_term(GexLatentBrownianModel *model,
                                    Matrix *resid,
                                    Matrix *grad_L,
                                    double sigma2_obs,
                                    double lambda_L) {
    int i, j, d;
    int n = model->n_cells;
    int p = model->n_genes;
    int k = model->k;
    double obj = 0.0;

    /* Compute gradient w.r.t. L under Gaussian likelihood with L2 regularization. */
    for (d = 0; d < k; d++) {
        for (j = 0; j < p; j++) {
            double gl = 0.0;

            /* Accumulate gradient from data likelihood:
               -(1/sigma2_obs) * sum_i Z_{i,d} * r_{i,j}
               where r_{i,j} = X_{i,j} - (ZL)_{i,j} */
            for (i = 0; i < n; i++)
                gl += -mat_get(model->Z, i, d) *
                      mat_get(resid, i, j) / sigma2_obs;

            /* Only add the L2 penalty and gradient when regularization is enabled. */
            if (lambda_L > 0.0)
                gl += lambda_L * mat_get(model->L, d, j);

            /* Store gradient for L_{d,j} */
            mat_set(grad_L, d, j, gl);

            if (lambda_L > 0.0) {
                /* Add L2 penalty to objective: (lambda_L / 2) * L_{d,j}^2 */
                obj += 0.5 * lambda_L *
                       mat_get(model->L, d, j) * mat_get(model->L, d, j);
            }
        }
    }

    return obj;
}

/* Compute the negative log-posterior objective and its gradients for the
latent Brownian factor model:
    X ≈ ZL + ε,      ε_ij ~ N(0, sigma2_obs)
    z_d ~ N(0, sigma2_latent[d] * Sigma) independently for each latent factor d

The objective (up to constants) is the sum of three terms for the data likelihood,
the latent factor Brownian prior, and the L2 regularization on loadings L (latent factors x genes):
    Matrix factorization data term (Gaussian likelihood):
    (1 / (2 sigma2_obs)) ||X - ZL||_F^2 + (np/2) log(sigma2_obs)

    Brownian motion prior on latent factors z_d as columns in Z (cells x latent factors),
    marginalized over a set of trees:
    sum_d [ (n/2) log(sigma2_d)
            - log sum_t exp( - (1 / (2 sigma2_d)) z_d^T Sigma_t^{-1} z_d
                             - (1/2) log|Sigma_t| ) ]

    L2 regularization on loadings to encourage distributed latent factors (prevents 
    overfitting to few genes):
    (lambda_L / 2) ||L||_F^2

Returns the total objective value (negative log-posterior) and fills gradients for
Z, L, log(sigma2_latent), and log(sigma2_obs). */
static double gex_model_objective_and_grad(GexLatentBrownianModel *model,
                                           Matrix *Xc,
                                           Matrix **Sigma_invs,
                                           double *logdet_sigmas,
                                           int n_sigmas,
                                           Matrix *grad_Z,
                                           Matrix *grad_L,
                                           double *grad_log_sigma_latent,
                                           double *grad_log_sigma_obs) {
    int d;
    int n = model->n_cells; /* Number of cells */
    int p = model->n_genes; /* Number of genes */
    int k = model->k;   /* Number of latent factors */
    double lambda_L = model->l2_strength;   /* L2 regularization strength for L */
    double sigma2_obs = model->sigma2_obs;  /* Observation noise variance */
    double obj = 0.0;   /* Objective function value */
    Matrix *resid = NULL;   /* Residual matrix */

    /* Initialize the residual matrix as cells x genes */
    resid = mat_new(n, p);
    if (resid == NULL)
        return HUGE_VAL;

    /* Initialize and zero the gradients for Z, L, log(sigma2_latent) for all 
    latent factors, and log(sigma2_obs) */
    mat_zero(grad_Z);
    mat_zero(grad_L);
    for (d = 0; d < k; d++)
        grad_log_sigma_latent[d] = 0.0;
    *grad_log_sigma_obs = 0.0;

    /* Add the likelihood from the gaussian observation model X_ij ~ N((ZL)_ij, sigma2_obs)
    and accumulate the gradients w.r.t. Z and log(sigma2_obs). */
    model->observation_objective =
        gaussian_observation_term(model, Xc, sigma2_obs, resid, grad_Z, grad_log_sigma_obs);
    obj += model->observation_objective;

    /* Add the mixture-of-Brownian prior contribution on Z and accumulate the
    gradients w.r.t. Z and log(sigma2_latent). */
    model->brownian_prior_objective =
        latent_brownian_prior_term(model, Sigma_invs, logdet_sigmas, n_sigmas,
                                    grad_Z, grad_log_sigma_latent);
    obj += model->brownian_prior_objective;
    if (!isfinite(obj)) {
        mat_free(resid);
        return HUGE_VAL;
    }

    /* Add the L2 regularization penalty on L and compute the gradient w.r.t. L. */
    model->l2_objective = l2_regularized_L_term(model, resid, grad_L, sigma2_obs, lambda_L);
    obj += model->l2_objective;

    mat_free(resid);
    return obj;
}

/* Compute the L2 (Euclidean) norm of the gradient ||g||_2 = sqrt( sum_i g_i^2 ),
treating all parameter gradients (Z, L, log(sigma2_latent), log(sigma2_obs))
as a single concatenated vector g. Returns the overall gradient magnitude. */
static double gex_model_grad_norm(Matrix *grad_Z,
                                  Matrix *grad_L,
                                  double *grad_log_sigma_latent,
                                  double grad_log_sigma_obs,
                                  int k) {
    int i, j;
    double ss = 0.0;
    /* Add the squared gradients for Z */
    for (i = 0; i < grad_Z->nrows; i++)
        for (j = 0; j < grad_Z->ncols; j++)
            ss += pow(mat_get(grad_Z, i, j), 2.0);
    /* Add the squared gradients for L */
    for (i = 0; i < grad_L->nrows; i++)
        for (j = 0; j < grad_L->ncols; j++)
            ss += pow(mat_get(grad_L, i, j), 2.0);
    /* Add the squared gradients for log(sigma2_latent) for all latent factors */
    for (i = 0; i < k; i++)
        ss += pow(grad_log_sigma_latent[i], 2.0);
    /* Add the squared gradient for log(sigma2_obs) */
    ss += grad_log_sigma_obs * grad_log_sigma_obs;
    return sqrt(ss);
}

/* Scale all gradients by a constant factor. */
static void gex_model_scale_grads(Matrix *grad_Z,
                                  Matrix *grad_L,
                                  double *grad_log_sigma_latent,
                                  double *grad_log_sigma_obs,
                                  int k,
                                  double scale) {
    int i, j;
    for (i = 0; i < grad_Z->nrows; i++)
        for (j = 0; j < grad_Z->ncols; j++)
            mat_set(grad_Z, i, j, mat_get(grad_Z, i, j) * scale);
    for (i = 0; i < grad_L->nrows; i++)
        for (j = 0; j < grad_L->ncols; j++)
            mat_set(grad_L, i, j, mat_get(grad_L, i, j) * scale);
    for (i = 0; i < k; i++)
        grad_log_sigma_latent[i] *= scale;
    *grad_log_sigma_obs *= scale;
}

/* Perform one Adam optimization update for a matrix parameter.

For each entry (i,j), update the first moment (m) and second moment (v)
estimates using the current gradient, apply bias correction to obtain
mhat and vhat, and then update the parameter using:

    param -= lr * mhat / (sqrt(vhat) + eps)

where m tracks the exponential moving average of gradients,
v tracks the exponential moving average of squared gradients,
and bias correction accounts for initialization at early steps. */
static void gex_model_adam_update_matrix(Matrix *param,
                                         Matrix *grad,
                                         Matrix *m,
                                         Matrix *v,
                                         int step,
                                         double lr) {
    int i, j;
    for (i = 0; i < param->nrows; i++) {
        for (j = 0; j < param->ncols; j++) {
            double g = mat_get(grad, i, j);
            double m_new = ADAM_BETA1 * mat_get(m, i, j) + (1.0 - ADAM_BETA1) * g;
            double v_new = ADAM_BETA2 * mat_get(v, i, j) + (1.0 - ADAM_BETA2) * g * g;
            double mhat = m_new / (1.0 - pow(ADAM_BETA1, step));
            double vhat = v_new / (1.0 - pow(ADAM_BETA2, step));
            mat_set(m, i, j, m_new);
            mat_set(v, i, j, v_new);
            mat_set(param, i, j, mat_get(param, i, j) - lr * mhat / (sqrt(vhat) + ADAM_EPS));
        }
    }
}

/* Perform one Adam optimization update for a vector parameter.

For each element i, update the first moment (m[i]) and second moment (v[i])
estimates using the current gradient, apply bias correction to obtain
mhat and vhat, and update the parameter using:

    param[i] -= lr * mhat / (sqrt(vhat) + eps)

where m stores the exponential moving average of gradients,
v stores the exponential moving average of squared gradients,
and bias correction accounts for initialization at early steps. */
static void gex_model_adam_update_vector(double *param,
                                         double *grad,
                                         double *m,
                                         double *v,
                                         int n,
                                         int step,
                                         double lr) {
    int i;
    for (i = 0; i < n; i++) {
        double m_new = ADAM_BETA1 * m[i] + (1.0 - ADAM_BETA1) * grad[i];
        double v_new = ADAM_BETA2 * v[i] + (1.0 - ADAM_BETA2) * grad[i] * grad[i];
        double mhat = m_new / (1.0 - pow(ADAM_BETA1, step));
        double vhat = v_new / (1.0 - pow(ADAM_BETA2, step));
        m[i] = m_new;
        v[i] = v_new;
        param[i] -= lr * mhat / (sqrt(vhat) + ADAM_EPS);
    }
}

/* Compute the Frobenius norm of a matrix.
This is equivalent to the Euclidean (l2) norm of all entries
treated as a single vector. */
static double frobenius_norm(Matrix *M) {
    int i, j;
    double ss = 0.0;

    for (i = 0; i < M->nrows; i++) {
        for (j = 0; j < M->ncols; j++)
            ss += pow(mat_get(M, i, j), 2.0);
    }

    return sqrt(ss);
}

/* Main model entry point. Fit a low-rank factorization of the centered gene
expression matrix that is regularized by a phylogenetic Brownian-motion prior.
The data are modeled as X ≈ ZL + E, where Z (cells × latent factors) contains
latent factors whose values across cells are constrained to vary smoothly
according to a Brownian-motion Gaussian prior with phylogenetic covariance
matrix Sigma and factor-specific variance parameters, L (latent factors × genes)
is the gene loading matrix that defines the factors, and E is Gaussian observation 
noise capturing variation not explained by the low-rank structure. Returns a 
pointer to the fitted model or NULL on failure. */
GexLatentBrownianModel *gex_fit_latent_brownian_model(GexMatrix *gex,
                                                      Matrix **Sigmas,
                                                      int n_sigmas,
                                                      PCA *pca,
                                                      double l2_strength,
                                                      unsigned int seed,
                                                      const char *outprefix) {
    /* Optimization related */
    int step;   /* Optimization step */
    int max_steps = 100000;   /* Maximum number of optimization steps */
    int min_steps = 500;    /* Minimum number of optimization steps before allowing convergence */
    int stable_steps_needed = 100;   /* Number of consecutive stable steps required for convergence */
    int running_avg_window_long = 500;   /* Number of recent steps used for the long running objective average */
    int running_avg_window_short = 100;   /* Number of recent steps used for the short running objective average */
    int stable_steps = 0;   /* Running count of consecutive near-converged steps */
    int converged = 0;   /* Whether the optimization stopped by satisfying the convergence rule. Assumes 0 (not converged) to start */
    Scheduler *sched = NULL;    /* Scheduler for managing optimization steps */
    SchedState *sched_state = NULL; /* State for the scheduler */
    SchedDirectives directives; /* Directives for each optimization step */
    SchedMetrics metrics;   /* Metrics for each optimization step */
    double running_objective_avg_long = HUGE_VAL; /* Running average over the long objective window */
    double running_objective_avg_short = HUGE_VAL; /* Running average over the short objective window */
    double rel_objective_change = HUGE_VAL; /* Relative difference between the short and long running averages */
    double running_objective_sum_long = 0.0; /* Running sum for the long moving-average window */
    double running_objective_sum_short = 0.0; /* Running sum for the short moving-average window */
    double *objective_hist_long = NULL;   /* Circular buffer for the long objective history */
    double *objective_hist_short = NULL;   /* Circular buffer for the short objective history */
    int objective_hist_size_long = 0;   /* Current number of values stored in the long window */
    int objective_hist_size_short = 0;   /* Current number of values stored in the short window */
    int objective_hist_idx_long = 0;    /* Next insertion position in the long history buffer */
    int objective_hist_idx_short = 0;    /* Next insertion position in the short history buffer */
    const double objective_tol = 1e-3;  /* Relative objective tolerance used for convergence */

    /* Model related */
    int k;  /* Number of latent dimensions */
    int n_cells = gex->X->nrows;
    int n_genes = gex->X->ncols;
    GexLatentBrownianModel *model = NULL;   /* Fitted model */
    Matrix *Xc = NULL;   /* Centered expression matrix */
    Matrix **Sigma_invs = NULL;   /* Inverses of regularized tree covariance matrices */
    double *logdet_sigmas = NULL; /* Log determinants of regularized tree covariances */

    /* Gradients */
    double *grad_log_sigma_latent = NULL;   /* Gradients of log latent noise standard deviations */
    double grad_log_sigma_obs = 0.0;    /* Gradient of log observation noise standard deviation */

    /* Distribution summary stats */
    double *m_log_sigma_latent = NULL;  /* Adam optimizer moment estimates for latent noise */
    double *v_log_sigma_latent = NULL;  /* Adam optimizer variance estimates for latent noise */
    double m_log_sigma_obs = 0.0;   /* Adam optimizer moment estimate for observation noise */
    double v_log_sigma_obs = 0.0;   /* Adam optimizer variance estimate for observation noise */
    double log_sigma_obs;   /* Log of observation noise standard deviation */
    double *log_sigma_latent = NULL;    /* Log of latent noise standard deviations */
    Matrix *grad_Z = NULL, *grad_L = NULL, *mZ = NULL, *vZ = NULL, *mL = NULL, *vL = NULL;  /* Gradients and optimizer states */

    /* Other */
    int i, j, d, n;    /* Loop indices */
    Matrix *L = NULL;   /* Temporary Cholesky factor for covariance calculations */
    FILE *logf = NULL;  /* Optimization log file */
    char log_path[4096]; /* Path to optimization log file */

    /* Input validation */
    if (gex == NULL || gex->X == NULL || Sigmas == NULL || n_sigmas <= 0 || pca == NULL ||
        pca->K <= 0 || outprefix == NULL)
        return NULL;

    /* Open a log file to record the optimization trajectory while fitting
    the latent Brownian model. */
    snprintf(log_path, sizeof(log_path), "%s.model.log", outprefix);
    logf = fopen(log_path, "w");
    fprintf(logf, "step\tobjective\tlong_objective_running_avg\tshort_objective_running_avg\trel_objective_running_avg_change\tstable_steps\tgrad_norm\tobservation_negll\tbrownian_neglprior\tl2_penalty\tsigma_obs");
    for (i = 0; i < pca->K; i++)
        fprintf(logf, "\tsigma_latent_LF%d", i + 1);
    fprintf(logf, "\tZ_norm\tL_norm\n");

    /* Center the expression matrix by subtracting the mean of each gene.
    This ensures the latent factor model is fit to the residual structure
    after removing per-gene offsets. */
    Xc = center_matrix(gex->X);

    /* Build regularized versions of the phylogenetic covariance matrices and
    precompute the inverse and log-determinant for each tree. */
    n = gex->X->nrows;
    Sigma_invs = scalloc(n_sigmas, sizeof(Matrix *));
    logdet_sigmas = scalloc(n_sigmas, sizeof(double));

    for (i = 0; i < n_sigmas; i++) {
        Matrix *Sigma_reg = NULL;
        double max_diag = 0.0;
        double jitter;

        if (Sigmas[i] == NULL || Sigmas[i]->nrows != n || Sigmas[i]->ncols != n)
            return NULL;

        Sigma_invs[i] = mat_new(n, n);
        Sigma_reg = mat_create_copy(Sigmas[i]);
        L = mat_new(n, n);
        if (Sigma_invs[i] == NULL || Sigma_reg == NULL || L == NULL) {
            if (Sigma_reg != NULL) mat_free(Sigma_reg);
            return NULL;
        }

        for (j = 0; j < n; j++) {
            double d = mat_get(Sigmas[i], j, j);
            if (d > max_diag)
                max_diag = d;
        }
        jitter = (max_diag > 0.0 ? 1e-8 * max_diag : 1e-8);

        for (j = 0; j < n; j++)
            mat_set(Sigma_reg, j, j, mat_get(Sigma_reg, j, j) + jitter);

        if (mat_invert(Sigma_invs[i], Sigma_reg) != 0 ||
            mat_cholesky(L, Sigma_reg) != 0) {
            mat_free(Sigma_reg);
            return NULL;
        }

        logdet_sigmas[i] = 0.0;
        for (j = 0; j < n; j++) {
            double diag = mat_get(L, j, j);
            if (diag <= 0.0) {
                mat_free(Sigma_reg);
                return NULL;
            }
            logdet_sigmas[i] += 2.0 * log(diag);
        }

        mat_free(Sigma_reg);
        mat_free(L);
        L = NULL;
    }

    /* Use the number of input PCA components as the number of latent dimensions */
    k = pca->K; 

    /* Allocate the model object and its core parameter matrices. */
    model = scalloc(1, sizeof(GexLatentBrownianModel));
    model->n_cells = n_cells;
    model->n_genes = n_genes;
    model->k = k;
    model->Z = mat_new(n_cells, k); /* Allocate the latent factors matrix: cells × latent factors */
    model->L = mat_new(k, n_genes); /* Allocate the factor loading matrix: latent factors × genes */
    model->l2_strength = l2_strength;

    /* Initialize the latent variance parameters to a desired tip variance on the given tree scale (assuming an ultrametric tree) */
    model->sigma2_latent = scalloc(k, sizeof(double)); /* Allocate latent variance parameters */
    double desired_tip_variance = 1.0;
    for (i = 0; i < k; i++)
        model->sigma2_latent[i] = desired_tip_variance / mat_get(Sigmas[0], 0, 0);

    /* Initialize the factor loading matrix directly from the retained PCA
    components, then initialize latent coordinates as the corresponding PCA
    scores Z = X_centered * L^T. */
    for (d = 0; d < model->k; d++) {
        for (j = 0; j < model->n_genes; j++)
            mat_set(model->L, d, j, mat_get(pca->components, d, j));
    }

    for (i = 0; i < model->n_cells; i++) {
        for (d = 0; d < model->k; d++) {
            double score = 0.0;
            for (j = 0; j < model->n_genes; j++)
                score += mat_get(Xc, i, j) * mat_get(model->L, d, j);
            mat_set(model->Z, i, d, score);
        }
    }

    /* Initialize the observation variance parameter by computing the residual sum of squares.
    This accounts for the difference between the observed and predicted values given
    the retention of only a subset of PCA components. */
    double sse = 0.0;
    for (i = 0; i < model->n_cells; i++) {
        for (j = 0; j < model->n_genes; j++) {
            double pred = 0.0;
            for (d = 0; d < model->k; d++)
                pred += mat_get(model->Z, i, d) * mat_get(model->L, d, j);
            sse += pow(mat_get(Xc, i, j) - pred, 2.0);
        }
    }
    model->sigma2_obs = sse / ((double)model->n_cells * model->n_genes);
    if (model->sigma2_obs < 1e-6)
        model->sigma2_obs = 1e-6;

    /* Allocate gradients, moments, and variances that Adam needs for each parameter.
    Use a log-variance parameterization during optimization for the variance parameters 
    to enforce positivity and for numerical stability. */
    log_sigma_obs = log(model->sigma2_obs); /* Log of observation variance, since we optimize in log-space */
    log_sigma_latent = scalloc(k, sizeof(double)); /* Log of latent variances, since we optimize in log-space */
    grad_log_sigma_latent = scalloc(k, sizeof(double));    /* Gradient of log latent variances */
    m_log_sigma_latent = scalloc(k, sizeof(double));   /* First moment of log latent variances */
    v_log_sigma_latent = scalloc(k, sizeof(double));   /* Second moment of log latent variances */
    grad_Z = mat_new(model->n_cells, k);    /* Gradient of latent coordinates */
    grad_L = mat_new(k, model->n_genes);    /* Gradient of factor loadings */
    mZ = mat_new(model->n_cells, k);    /* First moment of latent coordinates */
    vZ = mat_new(model->n_cells, k);    /* Second moment of latent coordinates */
    mL = mat_new(k, model->n_genes);    /* First moment of factor loadings */
    vL = mat_new(k, model->n_genes);    /* Second moment of factor loadings */

    /* Initialize history arrays for running averages of the objective function */
    objective_hist_long = scalloc(running_avg_window_long, sizeof(double));
    objective_hist_short = scalloc(running_avg_window_short, sizeof(double));

    /* Initialize the log-variance parameters from the initial variance values. */
    for (step = 0; step < k; step++)
        log_sigma_latent[step] = log(model->sigma2_latent[step]);

    /* Zero the gradient matrices. */
    mat_zero(mZ); mat_zero(vZ); mat_zero(mL); mat_zero(vL);
    (void)seed;

    /* Initialize the scheduler that controls learning-rate and clipping
    directives across optimization steps.*/
    sched = sched_new(model->n_genes, model->n_genes, 1000, 0.03, 1, 1, 5); /* Parameter order: n_genes, n_genes, max_steps, lr, clip_norm, decay_rate, momentum */
    sched_state = sched_new_state(sched);
    if (sched == NULL || sched_state == NULL)
        return NULL;
    metrics.grad_norm = 0.0;    /* Initialize the gradient norm */

    /* Run gradient-based optimization of latent coordinates, gene loadings,
    and the variance parameters using Adam updates until the objective and
    gradient norm stabilize, while still enforcing a maximum number of steps. */
    for (step = 1; step <= max_steps; step++) {
        int d;

        /* Update the learning rate and clipping directives for this optimization step. */
        sched_next(sched, sched_state, (step == 1 ? NULL : &metrics), &directives);

        /* Update the variance parameters in the model object from their log-space 
        optimization representations. */
        model->sigma2_obs = exp(log_sigma_obs);
        if (model->sigma2_obs < 1e-8) 
            model->sigma2_obs = 1e-8;
        for (d = 0; d < k; d++) {
            model->sigma2_latent[d] = exp(log_sigma_latent[d]);
            if (model->sigma2_latent[d] < 1e-8) 
                model->sigma2_latent[d] = 1e-8;
        }

        /* Compute the objective function and gradients */
        model->objective = gex_model_objective_and_grad(model, Xc, Sigma_invs,
                                                        logdet_sigmas,
                                                        n_sigmas, grad_Z, grad_L,
                                                        grad_log_sigma_latent,
                                                        &grad_log_sigma_obs);
        
        /* Compute the gradient norm */
        metrics.grad_norm = gex_model_grad_norm(grad_Z, grad_L,
                                                grad_log_sigma_latent,
                                                grad_log_sigma_obs, k);

        /* Compare the short and long running averages of the objective so
        that convergence is judged using denoised trends at two time scales. */
        if (objective_hist_size_long > 0)
            running_objective_avg_long = running_objective_sum_long / (double)objective_hist_size_long;
        else
            running_objective_avg_long = HUGE_VAL;

        if (objective_hist_size_short > 0)
            running_objective_avg_short = running_objective_sum_short / (double)objective_hist_size_short;
        else
            running_objective_avg_short = HUGE_VAL;

        if (objective_hist_size_long > 0 && objective_hist_size_short > 0) {
            rel_objective_change = fabs(running_objective_avg_short - running_objective_avg_long) /
                                   max(1.0, fabs(running_objective_avg_long));
        }
        else {
            rel_objective_change = HUGE_VAL;
        }

        /* Re-scale the gradients if their norm exceeds the clipping threshold. */
        if (directives.clip_norm > 0.0 && metrics.grad_norm > directives.clip_norm) {
            double scale = directives.clip_norm / metrics.grad_norm;
            gex_model_scale_grads(grad_Z, grad_L, grad_log_sigma_latent,
                                  &grad_log_sigma_obs, k, scale);
            metrics.grad_norm = directives.clip_norm;
        }

        /* Update the model parameters using Adam optimization steps. */
        gex_model_adam_update_matrix(model->Z, grad_Z, mZ, vZ, step, directives.lr);
        gex_model_adam_update_matrix(model->L, grad_L, mL, vL, step, directives.lr);
        gex_model_adam_update_vector(log_sigma_latent, grad_log_sigma_latent,
                                     m_log_sigma_latent, v_log_sigma_latent,
                                     k, step, directives.lr);
        
        /* Update the log-space variance parameters. */
        {
            double grad_arr[1], m_arr[1], v_arr[1], param_arr[1];
            grad_arr[0] = grad_log_sigma_obs;
            m_arr[0] = m_log_sigma_obs;
            v_arr[0] = v_log_sigma_obs;
            param_arr[0] = log_sigma_obs;
            gex_model_adam_update_vector(param_arr, grad_arr, m_arr, v_arr, 1, step, directives.lr);
            log_sigma_obs = param_arr[0];
            m_log_sigma_obs = m_arr[0];
            v_log_sigma_obs = v_arr[0];
        }

        /* Track whether the optimizer has entered a stable regime where the
        objective changes by less than the target relative tolerance from one
        step to the next. Once this persists for enough consecutive steps
        beyond the minimum step count, stop the optimization early. */
        if (step >= min_steps && rel_objective_change < objective_tol) {
            stable_steps++;
        }
        else {
            stable_steps = 0;
        }

        /* Log the scalar parameters and compact summaries of Z and L at
        each optimization step without writing the full matrices. */
        fprintf(logf, "%d\t%.17g\t%.17g\t%.17g\t%.17g\t%d\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g",
                step,
                model->objective,
                running_objective_avg_long,
                running_objective_avg_short,
                rel_objective_change,
                stable_steps,
                metrics.grad_norm,
                model->observation_objective,
                model->brownian_prior_objective,
                model->l2_objective,
                model->sigma2_obs);
        for (d = 0; d < k; d++)
            fprintf(logf, "\t%.17g", model->sigma2_latent[d]);
        fprintf(logf, "\t%.17g\t%.17g\n",
                frobenius_norm(model->Z),
                frobenius_norm(model->L));
        fflush(logf);

        /* Update both moving-average histories. */
        if (objective_hist_size_long < running_avg_window_long) {
            objective_hist_long[objective_hist_idx_long] = model->objective;
            running_objective_sum_long += model->objective;
            objective_hist_size_long++;
        }
        else {
            running_objective_sum_long -= objective_hist_long[objective_hist_idx_long];
            objective_hist_long[objective_hist_idx_long] = model->objective;
            running_objective_sum_long += model->objective;
        }
        objective_hist_idx_long = (objective_hist_idx_long + 1) % running_avg_window_long;

        if (objective_hist_size_short < running_avg_window_short) {
            objective_hist_short[objective_hist_idx_short] = model->objective;
            running_objective_sum_short += model->objective;
            objective_hist_size_short++;
        }
        else {
            running_objective_sum_short -= objective_hist_short[objective_hist_idx_short];
            objective_hist_short[objective_hist_idx_short] = model->objective;
            running_objective_sum_short += model->objective;
        }
        objective_hist_idx_short = (objective_hist_idx_short + 1) % running_avg_window_short;

        /* Check the convergence rule and break if satisfied. */
        if (stable_steps >= stable_steps_needed) {
            converged = 1;
            break;
        }

    }
    
    /* Update the model object with the final variance parameter values from their log-space 
    representations. */
    model->sigma2_obs = exp(log_sigma_obs);
    if (model->sigma2_obs < 1e-8) model->sigma2_obs = 1e-8;
    for (step = 0; step < k; step++) {
        model->sigma2_latent[step] = exp(log_sigma_latent[step]);
        if (model->sigma2_latent[step] < 1e-8)
            model->sigma2_latent[step] = 1e-8;
    }

    /* Compute the final state objective and gradients. */
    model->objective = gex_model_objective_and_grad(model, Xc, Sigma_invs,
                                                    logdet_sigmas,
                                                    n_sigmas, grad_Z, grad_L,
                                                    grad_log_sigma_latent,
                                                    &grad_log_sigma_obs);

    /* Write a final footer line describing why optimization terminated. */
    fprintf(logf, "# termination\t%s\n", (converged ? "converged" : "max_steps_reached"));
    fflush(logf);

    /* Free memory */
    if (log_sigma_latent != NULL) free(log_sigma_latent);
    if (grad_log_sigma_latent != NULL) free(grad_log_sigma_latent);
    if (m_log_sigma_latent != NULL) free(m_log_sigma_latent);
    if (v_log_sigma_latent != NULL) free(v_log_sigma_latent);
    if (objective_hist_long != NULL) free(objective_hist_long);
    if (objective_hist_short != NULL) free(objective_hist_short);
    if (grad_Z != NULL) mat_free(grad_Z);
    if (grad_L != NULL) mat_free(grad_L);
    if (mZ != NULL) mat_free(mZ);
    if (vZ != NULL) mat_free(vZ);
    if (mL != NULL) mat_free(mL);
    if (vL != NULL) mat_free(vL);
    if (sched_state != NULL) free(sched_state);
    if (sched != NULL) free(sched);
    if (L != NULL) mat_free(L);
    if (logf != NULL) fclose(logf);
    if (Xc != NULL) mat_free(Xc);
    if (Sigma_invs != NULL) {
        for (i = 0; i < n_sigmas; i++) {
            if (Sigma_invs[i] != NULL)
                mat_free(Sigma_invs[i]);
        }
        free(Sigma_invs);
    }
    if (logdet_sigmas != NULL) free(logdet_sigmas);

    return model;
}

void gex_free_latent_brownian_model(GexLatentBrownianModel *model) {
    if (model == NULL)
        return;
        
    if (model->Z != NULL) 
        mat_free(model->Z);
    if (model->L != NULL) 
        mat_free(model->L);
    if (model->sigma2_latent != NULL) 
        free(model->sigma2_latent);
    if (model->latent_mvn != NULL) 
        mvn_free(model->latent_mvn);

    free(model);
}
