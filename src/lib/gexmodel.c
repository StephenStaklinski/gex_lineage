#include "gexmodel.h"

#include "gexpca.h"
#include "gexmisc.h"

#include <variational.h>

#include <phast/matrix.h>
#include <phast/misc.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>


/* Compute Gaussian observation negative log-likelihood and gradients for
X ~ N(FL, sigma2_obs). Returns the full contribution to the objective,
including the Gaussian normalization constant, fills residual matrix,
and computes gradients w.r.t. F and log(sigma2_obs). */
double gaussian_observation_term(Matrix *FL,
                                    Matrix *F,
                                    Matrix *L,
                                    double log_sigma2_obs,
                                    Matrix *Xc,
                                    Matrix *grad_F,
                                    Matrix *grad_L,
                                    double *grad_log_sigma_obs) {
    int i, j, d;
    int n = Xc->nrows;
    int p = Xc->ncols;
    int k = F->ncols;
    double obj = 0.0;
    double sigma2_obs = exp(log_sigma2_obs);
    Matrix *resid = mat_new(n, p);   /* Residual matrix */

    /* Initialize gradient accumulator for log(sigma2_obs) */
    if (grad_log_sigma_obs != NULL)
        *grad_log_sigma_obs = 0.0;

    /* Gaussian observation model X_ij ~ N((FL)_ij, sigma2_obs).
       Compute residuals r_ij = X_ij - (FL)_ij and accumulate the
       negative log-likelihood and its gradient w.r.t. log(sigma2_obs). */
    for (i = 0; i < n; i++) {
        double *Xc_i = Xc->data[i]; /* Row of centered data for cell i */
        double *resid_i = resid->data[i]; /* Row of residuals for cell i */
        double r;
        for (j = 0; j < p; j++) {
            /* Residual is observed minus predicted */
            r = Xc_i[j] - FL->data[i][j];
            resid_i[j] = r;

            /* Quadratic term of Gaussian negative log-likelihood */
            obj += 0.5 * r * r / sigma2_obs;

            /* Gradient w.r.t. log(sigma2_obs) from quadratic term */
            if (grad_log_sigma_obs != NULL)
                *grad_log_sigma_obs += -0.5 * r * r / sigma2_obs;
        }

        /* Compute gradient w.r.t. F */
        if (grad_F != NULL) {
            double *grad_F_i = grad_F->data[i]; /* Row of gradient for cell i */
            for (d = 0; d < k; d++) {
                double gz = 0.0;

                /* Accumulate gradient contribution across features */
                double *L_d = L->data[d]; /* Row of L for latent factor d */
                for (j = 0; j < p; j++)
                    gz += -resid_i[j] * L_d[j] / sigma2_obs;

                grad_F_i[d] = gz;
            }
        }
    }

    /* Log-determinant term of Gaussian likelihood: (np/2) log(σ²) */
    double n_entries = (double)(n * p);
    obj += 0.5 * n_entries * log(sigma2_obs);

    /* Gaussian normalization constant: (np/2) log(2π). This is not
    dependent on parameters so it does not matter for optimization but is
    included here to obtain full likelihood values. */
    obj += 0.5 * n_entries * log(2.0 * M_PI);

    /* Gradient w.r.t. L */
    if (grad_L != NULL) {
        for (d = 0; d < k; d++) {
            double *grad_L_d = grad_L->data[d]; /* Row of gradient for latent factor d */
            for (j = 0; j < p; j++) {
                double grad = 0.0;
                for (i = 0; i < n; i++)
                    grad += -F->data[i][d] * resid->data[i][j] / sigma2_obs;
                grad_L_d[j] = grad;
            }
        }
    }

    /* Gradient w.r.t. log(sigma2_obs) */
    if (grad_log_sigma_obs != NULL) {
        *grad_log_sigma_obs += 0.5 * n_entries;
        /* Normalize the gradient to be per observation (only for stability with Adam steps) */
        *grad_log_sigma_obs /= n_entries;
    }

    /* Free memory */
    mat_free(resid);

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
    int i, j;
    double quad = 0.0;
    double **S = Sigma_inv->data;

    if (sigma2 <= 0.0)
        return -HUGE_VAL;

    for (i = 0; i < n; i++) {
        double *row = S[i];
        double val = 0.0;

        for (j = 0; j < n; j++)
            val += row[j] * z[j];

        if (sigma_inv_z != NULL)
            sigma_inv_z[i] = val;

        quad += z[i] * val;
    }

    if (quad_out != NULL)
        *quad_out = quad;

    return -0.5 * (double)n * log(2.0 * M_PI)
         - 0.5 * (double)n * log(sigma2)
         - 0.5 * logdet_sigma
         - 0.5 * quad / sigma2;
}

/* Compute the mixture-of-Brownian Gaussian prior contribution for the latent
factors F across all latent dimensions. Returns the contribution to the
objective, adds the prior gradient to F, and computes gradients with respect
to log(sigma2_latent) for each latent factor. */
double latent_brownian_prior_term(Matrix *F,
                                        double *log_sigma2_latent,
                                        Matrix **Sigma_invs,
                                        double *logdet_sigmas,
                                        int n_sigmas,
                                        Matrix *grad_F,
                                        double *grad_log_sigma_latent) {
    int i, d, t;
    int n = F->nrows;   /* Number of cells */
    int k = F->ncols;   /* Number of latent factors */
    double obj = 0.0;
    double *prior_log_terms = NULL;
    double *prior_weights = NULL;
    double *quad_terms = NULL;
    double *f_d = NULL;
    double **sigma_inv_z_cache = NULL;

    prior_log_terms = scalloc(n_sigmas, sizeof(double));
    prior_weights = scalloc(n_sigmas, sizeof(double));
    quad_terms = scalloc(n_sigmas, sizeof(double));
    f_d = scalloc(n, sizeof(double));
    sigma_inv_z_cache = scalloc(n_sigmas, sizeof(double *));
    for (t = 0; t < n_sigmas; t++)
        sigma_inv_z_cache[t] = scalloc(n, sizeof(double));

    /* Add the Brownian motion multivariate Gaussian mixture prior on F for each
    latent dimension f_d, marginalizing over a set of candidate trees. */
    for (d = 0; d < k; d++) {
        double sigma2_d = exp(log_sigma2_latent[d]);
        double log_mix;
        double expected_quad_over_sigma2 = 0.0;

        for (i = 0; i < n; i++)
            f_d[i] = mat_get(F, i, d);

        /* Compute full per-tree Brownian Gaussian log densities */
        for (t = 0; t < n_sigmas; t++) {
            prior_log_terms[t] = log_mvn_vec(f_d,
                                             Sigma_invs[t],
                                             logdet_sigmas[t],
                                             n,
                                             sigma2_d,
                                             sigma_inv_z_cache[t],
                                             &quad_terms[t]);
        }

        /* Marginalize over tree uncertainty with log-sum-exp. */
        log_mix = logsumexp(prior_log_terms, n_sigmas);

        /* Add the full negative log marginal prior for f_d under the mixture:
           -log sum_t N(f_d | 0, sigma2_d * Sigma_t). */
        obj += -log_mix;

        /* Compute responsibility-weighted gradients over trees. */
        for (t = 0; t < n_sigmas; t++) {
            double weight = exp(prior_log_terms[t] - log_mix);
            prior_weights[t] = weight;

            if (grad_F != NULL) {
                for (i = 0; i < n; i++) {
                    double old_grad = mat_get(grad_F, i, d);
                    mat_set(grad_F, i, d,
                            old_grad + weight * sigma_inv_z_cache[t][i] / sigma2_d);
                }
            }

            expected_quad_over_sigma2 += weight * quad_terms[t] / sigma2_d;
        }

        if (grad_log_sigma_latent != NULL) {
            /* Gradient with respect to log(sigma2_d). */
            grad_log_sigma_latent[d] = 0.5 * (double)n - 0.5 * expected_quad_over_sigma2;
        }
    }

    for (t = 0; t < n_sigmas; t++)
        free(sigma_inv_z_cache[t]);
    free(sigma_inv_z_cache);
    free(prior_log_terms);
    free(prior_weights);
    free(quad_terms);
    free(f_d);

    return obj;
}

/* Compute the gradient with respect to L under the Gaussian observation model
   and add the L1 regularization penalty on L. Returns the contribution of the
   L1 penalty to the objective and fills grad_L. */
static double l1_regularized_L_term(Matrix *L,
                                    Matrix *grad_L,
                                    double L_lambda_l1) {
    int j, d;
    int p = L->ncols;
    int k = L->nrows;
    double abs_sum = 0.0;
    double val, grad, subgrad;

    for (d = 0; d < k; d++) {
        for (j = 0; j < p; j++) {

            val = mat_get(L, d, j);

            /* Add |L| to objective */
            abs_sum += fabs(val);

            if (grad_L != NULL) {
                grad = mat_get(grad_L, d, j);

                /* L1 subgradient. The derivative is undefined at zero,
                so we use a subgradient value of 0 there. */
                if (val > 0.0)
                    subgrad = 1.0;
                else if (val < 0.0)
                    subgrad = -1.0;
                else
                    subgrad = 0.0;

                grad += L_lambda_l1 * subgrad;
                mat_set(grad_L, d, j, grad);
            }
        }
    }

    return L_lambda_l1 * abs_sum;
}

/* Compute the negative log-posterior objective and its gradients for the
latent Brownian factor model:
    X ≈ FL + ε,      ε_ij ~ N(0, sigma2_obs)
    f_d ~ N(0, sigma2_latent[d] * Sigma) independently for each latent factor d
The objective (up to constants) is the sum of three terms for the data likelihood,
the latent factor Brownian prior, and the L1 regularization on loadings L (latent factors x genes).
*/
static double gex_model_objective_and_grad(GexLatentBrownianModel *model,
                                           Matrix *Xc,
                                           Matrix **Sigma_invs,
                                           double *logdet_sigmas,
                                           int n_sigmas,
                                           Matrix *grad_F,
                                           Matrix *grad_L,
                                           double *grad_log_sigma_latent,
                                           double *grad_log_sigma_obs) {
    int d;
    int k = model->k;   /* Number of latent factors */
    double L_lambda_l1 = model->l1_strength;   /* L1 regularization strength for L */
    double obj = 0.0;   /* Objective function value */

    /* Zero gradients */
    mat_zero(grad_F);
    mat_zero(grad_L);
    if (grad_log_sigma_latent != NULL) {
        for (d = 0; d < k; d++)
            grad_log_sigma_latent[d] = 0.0;
    }
    *grad_log_sigma_obs = 0.0;

    /* Add the likelihood from the gaussian observation model X_ij ~ N((FL)_ij, sigma2_obs)
    and accumulate the gradients w.r.t. F and log(sigma2_obs). */
    model->observation_objective =
        gaussian_observation_term(model->FL, model->F, model->L, model->log_sigma2_obs, Xc, grad_F, grad_L, grad_log_sigma_obs);
    obj += model->observation_objective;

    /* Add the mixture-of-Brownian prior contribution on F and accumulate the
    gradients w.r.t. F and log(sigma2_latent). */
    model->brownian_prior_objective =
        latent_brownian_prior_term(model->F, model->log_sigma2_latent, Sigma_invs, logdet_sigmas, n_sigmas,
                                    grad_F, grad_log_sigma_latent);
    obj += model->brownian_prior_objective;
    if (!isfinite(obj)) {
        return HUGE_VAL;
    }

    /* Add the L1 regularization penalty on L and compute the gradient w.r.t. L. */
    model->l1_objective = 0.0;
    if (L_lambda_l1 > 0.0) {
        model->l1_objective = l1_regularized_L_term(model->L, grad_L, L_lambda_l1);
        obj += model->l1_objective;
    }

    return obj;
}

/* Cosine decay of learning rate */
static double cosine_lr(double curr_lr, double base_lr, int step, int max_steps) {
    if (step <= 1 || step >= max_steps)
        return curr_lr; /* Stay at the current learning rate after decay ends */

    double progress = (double)step / (double)max_steps;
    double lr = base_lr * 0.5 * (1.0 + cos(M_PI * progress));
    double min_lr = 1e-6;  // floor for learning rate to prevent it from going to zero
    lr = fmax(lr, min_lr);

    return lr;
}

/* Perform one Adam optimization update for a scalar parameter in place.
Updates the first and second moment estimates using the current gradient,
applies bias correction to obtain mhat and vhat, and then updates the
parameter using:

    param -= lr * mhat / (sqrt(vhat) + eps)

where m tracks the exponential moving average of gradients,
v tracks the exponential moving average of squared gradients,
and mhat and vhat are the corresponding bias-corrected estimates used
for the parameter update. Bias correction accounts for initialization
at early steps. */
static void adam_step_scalar(double *param,
                               double grad,
                               double *m,
                               double *v,
                               double pow_beta1,
                               double pow_beta2,
                               double lr) {
    double m_new = ADAM_BETA1 * (*m) + (1.0 - ADAM_BETA1) * grad;
    double v_new = ADAM_BETA2 * (*v) + (1.0 - ADAM_BETA2) * grad * grad;
    double mhat = m_new / (1.0 - pow_beta1);
    double vhat = v_new / (1.0 - pow_beta2);

    *m = m_new;
    *v = v_new;
    *param -= lr * mhat / (sqrt(vhat) + ADAM_EPS);
}

/* Adam update wrapper for a Matrix parameter */
static void adam_step_matrix(Matrix *param,
                                         Matrix *grad,
                                         Matrix *m,
                                         Matrix *v,
                                         double pow_beta1,
                                         double pow_beta2,
                                         double lr) {
    int i, j;
    for (i = 0; i < param->nrows; i++) {
        for (j = 0; j < param->ncols; j++) {
            double p = mat_get(param, i, j);
            double g = mat_get(grad, i, j);
            double m_ij = mat_get(m, i, j);
            double v_ij = mat_get(v, i, j);

            adam_step_scalar(&p, g, &m_ij, &v_ij, pow_beta1, pow_beta2, lr);

            mat_set(param, i, j, p);
            mat_set(m, i, j, m_ij);
            mat_set(v, i, j, v_ij);
        }
    }
}

/* Adam update wrapper for a vector parameter */
static void adam_step_vector(double *param,
                                double *grad,
                                double *m,
                                double *v,
                                int n,
                                double pow_beta1,
                                double pow_beta2,
                                double lr) {
    int i;
    for (i = 0; i < n; i++) {
        adam_step_scalar(&param[i], grad[i], &m[i], &v[i], pow_beta1, pow_beta2, lr);
    }
}

static void normalize_L_rows_and_rescale_F(Matrix *L,
                                           Matrix *F,
                                           Matrix *mL,
                                           Matrix *vL,
                                           Matrix *mF,
                                           Matrix *vF,
                                           double target_row_norm)
{
    int d, i, j;
    int k = L->nrows;
    int p = L->ncols;
    int n = F->nrows;
    const double eps = 1e-12;

    for (d = 0; d < k; d++) {
        double ss = 0.0;
        double norm, scale_L, scale_F;

        /* Compute current L2 norm of row d of L */
        for (j = 0; j < p; j++) {
            double x = mat_get(L, d, j);
            ss += x * x;
        }
        norm = sqrt(ss);

        /* Skip pathological zero rows */
        if (norm < eps)
            continue;

        /* Multiply row d of L by scale_L so that ||L_d|| = target_row_norm */
        scale_L = target_row_norm / norm;

        /* Multiply column d of F by scale_F to preserve FL exactly */
        scale_F = 1.0 / scale_L;

        /* Update row d of L and its Adam states */
        for (j = 0; j < p; j++) {
            mat_set(L, d, j, mat_get(L, d, j) * scale_L);
            if (mL != NULL)
                mat_set(mL, d, j, mat_get(mL, d, j) * scale_L);
            if (vL != NULL)
                mat_set(vL, d, j, mat_get(vL, d, j) * scale_L * scale_L);
        }

        /* Update column d of F and its Adam states */
        for (i = 0; i < n; i++) {
            mat_set(F, i, d, mat_get(F, i, d) * scale_F);
            if (mF != NULL)
                mat_set(mF, i, d, mat_get(mF, i, d) * scale_F);
            if (vF != NULL)
                mat_set(vF, i, d, mat_get(vF, i, d) * scale_F * scale_F);
        }
    }
}

/* Uses an exponential moving average (EMA) to update the clipping threshold 
to be clip_factor * normal gradient levels from the EMA */
static double update_clip_threshold(double grad_norm,
                                    double *ema_grad_norm,
                                    int step,
                                    int clip_warmup,
                                    double clip_beta,
                                    double clip_factor,
                                    double clip_floor) {
    double clip;

    /* Update EMA from current block norm */
    if (grad_norm > 0.0) {
        if (*ema_grad_norm == 0.0)
            *ema_grad_norm = grad_norm;
        else
            *ema_grad_norm = clip_beta * (*ema_grad_norm) +
                             (1.0 - clip_beta) * grad_norm;
    }

    /* Skip clipping during warmup */
    if (step < clip_warmup) {
        return HUGE_VAL;
    }

    /* After warmup, allow adaptive thresholding */
    double adaptive_clip = clip_factor * (*ema_grad_norm);
    clip = fmax(clip_floor, adaptive_clip);

    return clip;
}

static int clip_matrix_by_norm(Matrix *grad, double norm, double clip_norm) {
    if (clip_norm > 0.0 && norm > clip_norm) {
        double scale = clip_norm / norm;
        mat_scale(grad, scale);
        return 1;
    }
    return 0;

}

static int clip_vector_by_norm(double *grad, int n, double norm, double clip_norm) {
    int i;
    if (clip_norm > 0.0 && norm > clip_norm) {
        double scale = clip_norm / norm;
        for (i = 0; i < n; i++)
            grad[i] *= scale;
        return 1;
    }
    return 0;
}

void post_hoc_sign_identifiability(Matrix *L, Matrix *F) {
    int d, j;
    int k = L->nrows;
    int p = L->ncols;

    int use_sum = 1; /* Whether to use the sum of loadings (1) or the max loading (0) for sign identifiability.*/

    for (d = 0; d < k; d++) {
        /* Find the index of the largest loading in absolute value for this factor */
        double sign_val = 0.0;
        for (j = 0; j < p; j++) {
            double val = mat_get(L, d, j);

            if (use_sum) {
                sign_val += val;
            } else {
                if (fabs(val) > fabs(sign_val)) {
                    sign_val = val;
                }
            }
        }

        /* Decide whether or not to flip the sign of the loadings */
        if (sign_val < 0.0) {
            int i;
            for (i = 0; i < F->nrows; i++) {
                mat_set(F, i, d, -mat_get(F, i, d));
            }
            for (j = 0; j < p; j++) {
                mat_set(L, d, j, -mat_get(L, d, j));
            }
        }
    }
}

/* Sorts an array in decreasing order in-place and updates the corresponding indices 
to match the new order */
static void selection_sort_decreasing(double *arr, int *indices, int n) {
    int i, j;

    /* Set the current indices */
    for (i = 0; i < n; i++)
        indices[i] = i;

    for (i = 0; i < n - 1; i++) {
        int best = i;
        for (j = i + 1; j < n; j++) {
            if (arr[j] > arr[best])
                best = j;
        }
        if (best != i) {
            double temp_val = arr[i];
            arr[i] = arr[best];
            arr[best] = temp_val;

            int temp_idx = indices[i];
            indices[i] = indices[best];
            indices[best] = temp_idx;
        }
    }
}

void reorder_factors_by_row_norm(Matrix *L, Matrix *F) {
    int d, j, i;
    int k = L->nrows;
    int p = L->ncols;
    int n = F->nrows;

    /* Compute row squared norms of L */
    double *norms = smalloc(k * sizeof(double));
    int *order = smalloc(k * sizeof(int));

    for (d = 0; d < k; d++) {
        double ss = 0.0;
        for (j = 0; j < p; j++) {
            double val = mat_get(L, d, j);
            ss += val * val;
        }
        norms[d] = ss;   /* squared norm is sufficient for ranking */
    }

    /* Sort indices by decreasing norm */
    selection_sort_decreasing(norms, order, k);

    /* Create reordered copies */
    Matrix *L_new = mat_new(k, p);
    Matrix *F_new = mat_new(n, k);

    for (d = 0; d < k; d++) {
        int old_d = order[d];

        /* Copy row old_d of L into row d */
        for (j = 0; j < p; j++)
            mat_set(L_new, d, j, mat_get(L, old_d, j));

        /* Copy column old_d of F into column d */
        for (i = 0; i < n; i++)
            mat_set(F_new, i, d, mat_get(F, i, old_d));
    }

    /* Overwrite originals */
    mat_copy(L, L_new);
    mat_copy(F, F_new);

    /* Free memory */
    mat_free(L_new);
    mat_free(F_new);
    free(norms);
    free(order);
}

void reorder_factors_by_sigma2_latent(Matrix *L, Matrix *F, double *log_sigma2_latent) {
    int d, j, i;
    int k = L->nrows;
    int p = L->ncols;
    int n = F->nrows;

    /* Sort indices by decreasing latent noise variance */
    int *order = smalloc(k * sizeof(int));
    selection_sort_decreasing(log_sigma2_latent, order, k); /* Sorts log_sigma2_latent in-place */

    /* Create reordered copies */
    Matrix *L_new = mat_new(k, p);
    Matrix *F_new = mat_new(n, k);

    for (d = 0; d < k; d++) {
        int old_d = order[d];

        /* Copy row old_d of L into row d */
        for (j = 0; j < p; j++)
            mat_set(L_new, d, j, mat_get(L, old_d, j));

        /* Copy column old_d of F into column d */
        for (i = 0; i < n; i++)
            mat_set(F_new, i, d, mat_get(F, i, old_d));
    }

    /* Overwrite originals */
    mat_copy(L, L_new);
    mat_copy(F, F_new);

    /* Free memory */
    mat_free(L_new);
    mat_free(F_new);
    free(order);
}

/* Main model entry point. Fit a low-rank factorization of the centered gene
expression matrix that is regularized by a phylogenetic Brownian-motion prior.
The data are modeled as X ≈ FL + E, where F (cells × latent factors) contains
latent factors whose values across cells are constrained to vary smoothly
according to a Brownian-motion Gaussian prior with phylogenetic covariance
matrix Sigma and factor-specific variance parameters, L (latent factors × genes)
is the gene loading matrix that defines the factors, and E is Gaussian observation 
noise capturing variation not explained by the low-rank structure. Returns a 
pointer to the fitted model or NULL on failure. */
GexLatentBrownianModel *gex_fit_latent_brownian_model(GexMatrix *gex,
                                                      Matrix **Sigmas,
                                                      int n_sigmas,
                                                      int k,
                                                      PCA *pca,
                                                      GexScaleInvarConstraint scale_invar_constraint,
                                                      double L_l1_strength,
                                                      const char *outprefix) {
    /* Optimization related */
    int step;   /* Optimization step */
    int max_steps = 100000;   /* Maximum number of optimization steps to prevent infinite run */
    int min_steps = 500;    /* Minimum number of optimization steps before allowing convergence */
    int stable_steps_needed = 100;   /* Number of consecutive stable steps required for convergence */
    int running_avg_window_long = 500;   /* Number of recent steps used for the long running objective average */
    int running_avg_window_short = 100;   /* Number of recent steps used for the short running objective average */
    int stable_steps = 0;   /* Running count of consecutive near-converged steps */
    int converged = 0;   /* Whether the optimization stopped by satisfying the convergence rule. Assumes 0 (not converged) to start */
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
    const double objective_tol = 1e-4;  /* Relative objective tolerance used for convergence */

    /* Model related */
    int n_cells = gex->X->nrows;
    int n_genes = gex->X->ncols;
    GexLatentBrownianModel *model = NULL;   /* Fitted model */
    Matrix **Sigma_invs = NULL;   /* Inverses of regularized tree covariance matrices */
    double *logdet_sigmas = NULL; /* Log determinants of regularized tree covariances */
    const double sigma_jitter = 1e-12; /* Diagonal jitter for singular tree covariance matrices */

    /* Gradients */
    double *grad_log_sigma_latent = NULL;   /* Gradients of log latent noise standard deviations */
    double grad_log_sigma_obs = 0.0;    /* Gradient of log observation noise standard deviation */

    /* Distribution summary stats */
    double *m_log_sigma_latent = NULL;  /* Adam optimizer moment estimates for latent noise */
    double *v_log_sigma_latent = NULL;  /* Adam optimizer variance estimates for latent noise */
    double m_log_sigma_obs = 0.0;   /* Adam optimizer moment estimate for observation noise */
    double v_log_sigma_obs = 0.0;   /* Adam optimizer variance estimate for observation noise */
    Matrix *grad_F = NULL, *grad_L = NULL, *mF = NULL, *vF = NULL, *mL = NULL, *vL = NULL;  /* Gradients and optimizer states */

    /* Other */
    int i, d, n;    /* Loop indices */
    FILE *logf = NULL;  /* Optimization log file */
    char log_path[4096]; /* Path to optimization log file */

    /* Open the log file an write a header */
    snprintf(log_path, sizeof(log_path), "%s.log", outprefix);
    logf = fopen(log_path, "w");
    fprintf(logf, "step\tobjective\tlong_avg\tshort_avg\trel_change\tstable_steps\tclipping_on\tgrad_norm\tF_grad_norm\tL_grad_norm\tlog_sigma2_obs_grad_norm\tlog_sigma2_latent_grad_norm\tobservation_negll\tbrownian_neglprior\tl1_penalty\tsigma2_obs");
    for (i = 0; i < k; i++)
        fprintf(logf, "\tsigma2_latent_LF%d", i + 1);
    fprintf(logf, "\tF_frobenius_norm\tL_frobenius_norm\tFL_frobenius_norm\n");

    /* Pre-compute the inverse and log-determinant for each tree */
    n = gex->X->nrows;
    Sigma_invs = scalloc(n_sigmas, sizeof(Matrix *));
    logdet_sigmas = scalloc(n_sigmas, sizeof(double));
    for (i = 0; i < n_sigmas; i++) {
        Sigma_invs[i] = mat_new(n, n);
        mat_add_diag(Sigmas[i], sigma_jitter);
        mat_invert(Sigma_invs[i], Sigmas[i]);
        logdet_sigmas[i] = mat_logdet(Sigmas[i]);
        mat_add_diag(Sigmas[i], -sigma_jitter);
    }

    /* Allocate the model object and its core parameter matrices. */
    model = scalloc(1, sizeof(GexLatentBrownianModel));
    model->n_cells = n_cells;
    model->n_genes = n_genes;
    model->k = k;
    model->F = mat_new(n_cells, k); /* Allocate the latent factors matrix: cells × latent factors */
    model->L = mat_new(k, n_genes); /* Allocate the factor loading matrix: latent factors × genes */
    model->FL = mat_new(n_cells, n_genes); /* Allocate the product FL for efficient likelihood computation */
    model->l1_strength = L_l1_strength;

    if (pca != NULL) {
        /* L comes from the PCA where the rows of components are the eigenvectors */
        mat_copy(model->L, pca->components);
    } else {
        /* Initialize L with random values */
        mat_set_all(model->L, 0.0);
        mat_add_gaussian_noise(model->L, 1);
    }

    /* Initialize F = X * L^T */
    Matrix *Lt = mat_transpose(model->L);
    mat_mult_lapack(model->F, gex->X, Lt);
    mat_free(Lt);

    /* Initialize sigma2_obs from the residual sum of squares */
    mat_mult_lapack(model->FL, model->F, model->L);
    double sse = mat_sum_squared_entries(model->FL);

    model->log_sigma2_obs = log(sse / ((double)model->n_cells * model->n_genes));
    model->log_sigma2_obs = max(model->log_sigma2_obs, log(1e-6));

    /* Initialize the latent variance parameters to the desired tip variance implied 
    by the PCA initialization of F for the given tree scale (assuming an ultrametric tree) */
    model->log_sigma2_latent = scalloc(k, sizeof(double)); /* Allocate latent variance parameters */
    if (scale_invar_constraint == GEX_SCALE_INVAR_SIGMA2S) {
        /* Fix latent variances to 1.0 */
        for (d = 0; d < k; d++)
            model->log_sigma2_latent[d] = 0.0;
    }
    else {
        /* Initialize latent variances based on the PCA initialization of F */
        double tip_var_scale = mat_get(Sigmas[0], 0, 0);
        double log_sigma2_latent_init;
        for (d = 0; d < k; d++) {
            double mean_z = 0.0;
            double var_z = 0.0;

            for (i = 0; i < n_cells; i++)
                mean_z += mat_get(model->F, i, d);
            mean_z /= (double)n_cells;

            for (i = 0; i < n_cells; i++) {
                double diff = mat_get(model->F, i, d) - mean_z;
                var_z += diff * diff;
            }
            var_z /= (double)n_cells;

            log_sigma2_latent_init = log(var_z / tip_var_scale);
            model->log_sigma2_latent[d] = log_sigma2_latent_init;
            model->log_sigma2_latent[d] = max(model->log_sigma2_latent[d], log(1e-6));
        }
    }

    /* Allocate gradients, moments, and variances for Adam */
    if (scale_invar_constraint != GEX_SCALE_INVAR_SIGMA2S) {
        grad_log_sigma_latent = scalloc(k, sizeof(double));    /* Gradient of log latent variances */
        m_log_sigma_latent = scalloc(k, sizeof(double));   /* First moment of log latent variances */
        v_log_sigma_latent = scalloc(k, sizeof(double));   /* Second moment of log latent variances */
    }
    grad_F = mat_new(model->n_cells, k);    /* Gradient of latent coordinates */
    grad_L = mat_new(k, model->n_genes);    /* Gradient of factor loadings */
    mF = mat_new(model->n_cells, k);    /* First moment of latent coordinates */
    vF = mat_new(model->n_cells, k);    /* Second moment of latent coordinates */
    mL = mat_new(k, model->n_genes);    /* First moment of factor loadings */
    vL = mat_new(k, model->n_genes);    /* Second moment of factor loadings */
    mat_zero(mF); mat_zero(vF); mat_zero(mL); mat_zero(vL); /* Zero the gradient matrices */

    /* Initialize running average histories */
    objective_hist_long = scalloc(running_avg_window_long, sizeof(double));
    objective_hist_short = scalloc(running_avg_window_short, sizeof(double));

    /* Adam hyperparameters */
    double base_lr = 0.1;   /* Base learning rate for Adam */
    double lr = base_lr;
    int lr_decay_max_steps = max_steps / 2; /* Decay lr to try to finish in half the total max time */
    double clip_beta = 0.98;
    double clip_factor = 2.0;
    double clip_floor = 1.0;
    int clip_warmup = 100;
    double clip_F = 0.0;
    double clip_L = 0.0;
    double clip_sigma_obs = 0.0;
    double clip_sigma_latent = 0.0;
    int clipping_on = 0;
    double grad_norm = 0.0;
    double ema_F_norm = 0.0;
    double ema_L_norm = 0.0;
    double ema_log_sigma_obs_norm = 0.0;
    double ema_log_sigma_latent_norm = 0.0;
    double grad_F_norm = 0.0;
    double grad_L_norm = 0.0;
    double grad_log_sigma_obs_norm = 0.0;
    double grad_log_sigma_latent_norm = 0.0;
    double pow_beta1;
    double pow_beta2;
    double rel_L_lr;
    double rel_sigma2_lr;

    /* Run Adam */
    for (step = 1; step <= max_steps; step++) {
        int d;

        /* Reset clipping flag */
        clipping_on = 0;

        /* Compute the objective function and gradients */
        model->objective = gex_model_objective_and_grad(model, gex->X, Sigma_invs,
                                                        logdet_sigmas,
                                                        n_sigmas, grad_F, grad_L,
                                                        grad_log_sigma_latent,
                                                        &grad_log_sigma_obs);
        
        /* Compute the gradient l2 norms */
        grad_F_norm = mat_frobenius_norm(grad_F);
        grad_L_norm = mat_frobenius_norm(grad_L);
        grad_log_sigma_obs_norm = fabs(grad_log_sigma_obs);
        grad_log_sigma_latent_norm = 0.0;
        if (scale_invar_constraint != GEX_SCALE_INVAR_SIGMA2S) {
            for (d = 0; d < k; d++) {
                grad_log_sigma_latent_norm += grad_log_sigma_latent[d] * grad_log_sigma_latent[d];
            }
            grad_log_sigma_latent_norm = sqrt(grad_log_sigma_latent_norm);
        }
        grad_norm = grad_F_norm + grad_L_norm + grad_log_sigma_obs_norm + grad_log_sigma_latent_norm;

        /* Update gradient clipping thresholds */
        clip_F = update_clip_threshold(grad_F_norm, &ema_F_norm, step, clip_warmup,
                                        clip_beta, clip_factor, clip_floor);
        clip_L = update_clip_threshold(grad_L_norm, &ema_L_norm, step, clip_warmup,
                                        clip_beta, clip_factor, clip_floor);
        clip_sigma_obs = update_clip_threshold(grad_log_sigma_obs_norm, &ema_log_sigma_obs_norm, step,
                                                clip_warmup, clip_beta, clip_factor, clip_floor);
        if (scale_invar_constraint != GEX_SCALE_INVAR_SIGMA2S) {
            clip_sigma_latent = update_clip_threshold(grad_log_sigma_latent_norm, &ema_log_sigma_latent_norm,
                                                    step, clip_warmup, clip_beta, clip_factor, clip_floor);
        }

        /* Re-scale the gradients if their norm exceeds the clipping threshold and 
        recompute the norm for those that were rescales */
        if (clip_matrix_by_norm(grad_F, grad_F_norm, clip_F)) {
            grad_F_norm = mat_frobenius_norm(grad_F);
            clipping_on |= 1;
        }
        if (clip_matrix_by_norm(grad_L, grad_L_norm, clip_L)) {
            grad_L_norm = mat_frobenius_norm(grad_L);
            clipping_on |= 1;
        }
        if (clip_sigma_obs > 0.0 && grad_log_sigma_obs_norm > clip_sigma_obs) {
            grad_log_sigma_obs *= clip_sigma_obs / grad_log_sigma_obs_norm;
            grad_log_sigma_obs_norm = fabs(grad_log_sigma_obs);
            clipping_on |= 1;
        }
        if (scale_invar_constraint != GEX_SCALE_INVAR_SIGMA2S) {
            if (clip_vector_by_norm(grad_log_sigma_latent, k,
                                        grad_log_sigma_latent_norm, clip_sigma_latent)) {
                grad_log_sigma_latent_norm = 0.0;
                for (d = 0; d < k; d++) {
                    grad_log_sigma_latent_norm += grad_log_sigma_latent[d] * grad_log_sigma_latent[d];
                }
                grad_log_sigma_latent_norm = sqrt(grad_log_sigma_latent_norm);
                clipping_on |= 1;
            }
        }
        
        grad_norm = grad_F_norm + grad_L_norm + grad_log_sigma_obs_norm + grad_log_sigma_latent_norm;
        

        /* Perturb the model parameters */
        lr = cosine_lr(lr, base_lr, step, lr_decay_max_steps);
        pow_beta1 = pow(ADAM_BETA1, step);
        pow_beta2 = pow(ADAM_BETA2, step);
        adam_step_matrix(model->F, grad_F, mF, vF, pow_beta1, pow_beta2, lr);
        rel_L_lr = lr;
        adam_step_matrix(model->L, grad_L, mL, vL, pow_beta1, pow_beta2, rel_L_lr);
        rel_sigma2_lr = lr * 0.01;
        if (scale_invar_constraint != GEX_SCALE_INVAR_SIGMA2S) {
            /* Step Brownian variance parameters if they are not fixed for scale invariance */
            adam_step_vector(model->log_sigma2_latent, grad_log_sigma_latent,
                                        m_log_sigma_latent, v_log_sigma_latent,
                                        k, pow_beta1, pow_beta2, rel_sigma2_lr);
        }
        adam_step_scalar(&model->log_sigma2_obs, grad_log_sigma_obs,
                           &m_log_sigma_obs, &v_log_sigma_obs,
                           pow_beta1, pow_beta2, rel_sigma2_lr);
        
        if (scale_invar_constraint == GEX_SCALE_INVAR_LROWS) {
            /* Normalize L rows and rescale F to prevent scale invariance */
            normalize_L_rows_and_rescale_F(model->L, model->F,
                               mL, vL, mF, vF, 1.0);
        }
        
        /* Compare the short and long running averages of the objective so
        that convergence is judged using denoised trends at two time scales. */
        if (objective_hist_size_long > 0 && objective_hist_size_short > 0) {
            running_objective_avg_long = running_objective_sum_long / (double)objective_hist_size_long;
            running_objective_avg_short = running_objective_sum_short / (double)objective_hist_size_short;
            rel_objective_change = fabs(running_objective_avg_short - running_objective_avg_long) / fabs(running_objective_avg_long);
        }

        /* Track whether the optimizer has entered a stable regime */
        if (step >= min_steps && rel_objective_change < objective_tol)
            stable_steps++;
        else
            stable_steps = 0;
        
        /* Reconstruct F*L */
        mat_mult_lapack(model->FL, model->F, model->L);   /* Compute predicted values FL */

        /* Log the scalar parameters and compact summaries of F and L at
        each optimization step without writing the full matrices. */
        fprintf(logf, "%d\t%.17g\t%.17g\t%.17g\t%.17g\t%d\t%d\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g",
                step,
                model->objective,
                running_objective_avg_long,
                running_objective_avg_short,
                rel_objective_change,
                stable_steps,
                clipping_on,
                grad_norm,
                grad_F_norm,
                grad_L_norm,
                grad_log_sigma_obs_norm,
                grad_log_sigma_latent_norm,
                model->observation_objective,
                model->brownian_prior_objective,
                model->l1_objective,
                exp(model->log_sigma2_obs));
        for (d = 0; d < k; d++)
            fprintf(logf, "\t%.17g", exp(model->log_sigma2_latent[d]));
        fprintf(logf, "\t%.17g\t%.17g\t%.17g\n", 
                    mat_frobenius_norm(model->F), 
                    mat_frobenius_norm(model->L),
                    mat_frobenius_norm(model->FL));
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

    /* Compute the final state objective and gradients. */
    model->objective = gex_model_objective_and_grad(model, gex->X, Sigma_invs,
                                                    logdet_sigmas,
                                                    n_sigmas, grad_F, grad_L,
                                                    grad_log_sigma_latent,
                                                    &grad_log_sigma_obs);
    
    /* Prevent permutation invariance by reordering factors */
    if (scale_invar_constraint == GEX_SCALE_INVAR_SIGMA2S) {
        reorder_factors_by_row_norm(model->L, model->F);
    }
    else {
        reorder_factors_by_sigma2_latent(model->L, model->F, model->log_sigma2_latent);
    }

    /* Prevent sign invariance by making the largest loading of L positive */
    post_hoc_sign_identifiability(model->L, model->F);

    /* Write a final footer line describing why optimization terminated. */
    fprintf(logf, "# termination\t%s\n", (converged ? "converged" : "max_steps_reached"));
    fflush(logf);

    /* Free memory */
    if (grad_log_sigma_latent != NULL) free(grad_log_sigma_latent);
    if (m_log_sigma_latent != NULL) free(m_log_sigma_latent);
    if (v_log_sigma_latent != NULL) free(v_log_sigma_latent);
    if (objective_hist_long != NULL) free(objective_hist_long);
    if (objective_hist_short != NULL) free(objective_hist_short);
    if (grad_F != NULL) mat_free(grad_F);
    if (grad_L != NULL) mat_free(grad_L);
    if (mF != NULL) mat_free(mF);
    if (vF != NULL) mat_free(vF);
    if (mL != NULL) mat_free(mL);
    if (vL != NULL) mat_free(vL);
    if (logf != NULL) fclose(logf);
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
    if (model->F != NULL) 
        mat_free(model->F);
    if (model->L != NULL) 
        mat_free(model->L);
    if (model->FL != NULL) 
        mat_free(model->FL);
    if (model->log_sigma2_latent != NULL) 
        free(model->log_sigma2_latent);
    if (model->latent_mvn != NULL) 
        mvn_free(model->latent_mvn);
    free(model);
}

/* Select a subset of covariance matrices (trees) by sampling without replacement. */
Matrix **downsample_sigmas(Matrix **Sigmas,
                                int n_sigmas,
                                int n_keep) {
    int i;
    Matrix **selected = NULL;
    int *indices = NULL;

    if (Sigmas == NULL || n_sigmas <= 0)
        return NULL;

    if (n_keep <= 0 || n_keep >= n_sigmas)
        n_keep = n_sigmas;

    selected = scalloc(n_keep, sizeof(Matrix *));

    if (n_keep == n_sigmas) {
        for (i = 0; i < n_sigmas; i++)
            selected[i] = Sigmas[i];
        return selected;
    }

    indices = smalloc(n_sigmas * sizeof(int));

    for (i = 0; i < n_sigmas; i++)
        indices[i] = i;

    for (i = 0; i < n_keep; i++) {
        int j = i + (int)(random() % (n_sigmas - i));
        int tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
        selected[i] = Sigmas[indices[i]];
    }

    free(indices);
    return selected;
}

void write_summary_tsv(const char *path,
                        int n_cells,
                        int n_genes,
                        double brownian_negll,
                        double observation_negll,
                        double sigma2_obs,
                        double *sigma2_latent,
                        double *L_row_norms,
                        int k,
                        char **factor_names) {
    int j;
    FILE *summary_out = NULL;

    summary_out = fopen(path, "w");

    fprintf(summary_out, "parameter\tvalue\n");
    fprintf(summary_out, "n_cells\t%d\n", n_cells);
    fprintf(summary_out, "n_genes\t%d\n", n_genes);
    fprintf(summary_out, "brownian_negll\t%.17g\n", brownian_negll);
    fprintf(summary_out, "observation_negll\t%.17g\n", observation_negll);
    fprintf(summary_out, "k\t%d\n", k);
    fprintf(summary_out, "sigma2_obs\t%.17g\n", sigma2_obs);
    for (j = 0; j < k; j++)
        fprintf(summary_out, "sigma2_latent_LF%d\t%.17g\n", j + 1, sigma2_latent[j]);
    for (j = 0; j < k; j++)
        fprintf(summary_out, "L_LF%d_l2_norm\t%.17g\n", j + 1, L_row_norms[j]);

    fclose(summary_out);
    summary_out = NULL;
}

void write_model(const char *outprefix,
                                GexMatrix *gex,
                                Matrix *L,
                                Matrix *F,
                                char **cell_names,
                                char **gene_names,
                                char **factor_names,
                                int k,
                                double brownian_negll,
                                double observation_negll,
                                double sigma2_obs,
                                double *sigma2_latent) {
    char summary_path[4096];
    char x_path[4096];
    char l_path[4096];
    char f_path[4096];

    snprintf(summary_path, sizeof(summary_path), "%s.summary.tsv", outprefix);
    snprintf(x_path, sizeof(x_path), "%s.X.tsv", outprefix);
    snprintf(l_path, sizeof(l_path), "%s.L.tsv", outprefix);
    snprintf(f_path, sizeof(f_path), "%s.F.tsv", outprefix);

    /* Calculate the row norms of L */
    double *L_row_norms = mat_row_l2_norms(L);

    write_summary_tsv(summary_path, gex->X->nrows, gex->X->ncols, 
                        brownian_negll, observation_negll,
                        sigma2_obs, sigma2_latent, L_row_norms, k, factor_names);
    
    /* Free memory */
    free(L_row_norms);

    /* Write out the simulated matrices */
    write_labeled_matrix_tsv(x_path, gex->X, cell_names, gex->X->nrows,
                                     gene_names, gex->X->ncols, "cell");
    write_labeled_matrix_tsv(l_path, L, factor_names, k,
                                     gene_names, gex->X->ncols, "factor");
    write_labeled_matrix_tsv(f_path, F, cell_names, gex->X->nrows,
                                     factor_names, k, "cell");
}

/* Simulate L and X from the input F and sigma_obs */
void simulate_factorization_and_reconstruction(Matrix *F,
                                     char **cell_names,
                                     int n_cells,
                                     int k,
                                     int n_genes,
                                     double sigma2_obs,
                                     Vector *L_row_norms,
                                     Matrix *L_out,
                                     GexMatrix *gex_out) {
    int i, j, d;    /* Loop indices */
    unsigned int rng_state;

    /* Draw gene loadings L ~ N(0,1) and rescale each row to have norm
    sqrt(n_genes / k), ensuring each latent dimension contributes
    equal expected magnitude to the noiseless gene expression data. */
    double row_ss;
    double target_norm; /* Set target norm for each row for stability */
    for (d = 0; d < k; d++) {
        row_ss = 0.0;    /* Sum of squares for the current row */
        target_norm = vec_get(L_row_norms, d); /* Get the target norm for the current row of L */
        for (j = 0; j < n_genes; j++) {
            rng_state = (unsigned int)random();
            double val = rand_normal(&rng_state);
            mat_set(L_out, d, j, val);
            row_ss += val * val;
        }
        if (row_ss > 0.0) {
            double row_scale = target_norm / sqrt(row_ss);
            for (j = 0; j < n_genes; j++)
                mat_set(L_out, d, j, row_scale * mat_get(L_out, d, j));
        }
    }

    /* Compute the noiseless expression matrix from the 
    simulated F and L matrix factorization. */
    mat_mult_lapack(gex_out->X, F, L_out);

    /* Add noise to the noiseless expression matrix based on 
    the sigma2_obs parameter input as new_val ~N(curr_val, sigma2_obs). */
    for (i = 0; i < n_cells; i++) {
        for (j = 0; j < n_genes; j++) {
            double val = mat_get(gex_out->X, i, j);
            if (sigma2_obs > 0.0) {
                rng_state = (unsigned int)random();
                val += sqrt(sigma2_obs) * rand_normal(&rng_state);
            }
            mat_set(gex_out->X, i, j, val);
        }
    }
}
