#include "gex_model.h"

#include <adam_scheduler.h>
#include <variational.h>

#include <math.h>
#include <phast/misc.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    Matrix *Xc;
    Matrix *Sigma_reg;
    Matrix *Sigma_inv;
    double logdet_sigma;
} GexLatentBrownianWorkspace;

/* Center the columns of a matrix by subtracting the mean of each column.
Returns a newly allocated centered matrix or NULL on failure. */
static Matrix *gex_model_center_matrix(Matrix *X) {
    int i, j;
    int n = X->nrows;
    int p = X->ncols;
    double *means = NULL;
    Matrix *Xc = NULL;

    means = (double *)calloc(p, sizeof(double));
    Xc = mat_new(n, p);
    if (means == NULL || Xc == NULL) {
        free(means);
        if (Xc != NULL)
            mat_free(Xc);
        return NULL;
    }

    for (j = 0; j < p; j++) {
        for (i = 0; i < n; i++)
            means[j] += mat_get(X, i, j);
        means[j] /= (double)n;
    }

    for (i = 0; i < n; i++) {
        for (j = 0; j < p; j++)
            mat_set(Xc, i, j, mat_get(X, i, j) - means[j]);
    }

    free(means);
    return Xc;
}

static double gex_model_objective_and_grad(GexLatentBrownianModel *model,
                                           GexLatentBrownianWorkspace *ws,
                                           Matrix *grad_Z,
                                           Matrix *grad_L,
                                           double *grad_log_sigma_latent,
                                           double *grad_log_sigma_obs) {
    const double lambda_L = 1e-3;
    int i, j, d, t;
    int n = model->n_cells;
    int p = model->n_genes;
    int k = model->k;
    double sigma2_obs = model->sigma2_obs;
    double obj = 0.0;
    Matrix *resid = NULL;

    resid = mat_new(n, p);
    if (resid == NULL)
        return HUGE_VAL;

    mat_zero(grad_Z);
    mat_zero(grad_L);
    for (d = 0; d < k; d++)
        grad_log_sigma_latent[d] = 0.0;
    *grad_log_sigma_obs = 0.0;

    for (i = 0; i < n; i++) {
        for (j = 0; j < p; j++) {
            double pred = 0.0;
            double r;
            for (d = 0; d < k; d++)
                pred += mat_get(model->Z, i, d) * mat_get(model->L, d, j);
            r = mat_get(ws->Xc, i, j) - pred;
            mat_set(resid, i, j, r);
            obj += 0.5 * r * r / sigma2_obs;
            *grad_log_sigma_obs += -0.5 * r * r / sigma2_obs;
        }
    }
    obj += 0.5 * (double)(n * p) * log(sigma2_obs);
    *grad_log_sigma_obs += 0.5 * (double)(n * p);

    for (i = 0; i < n; i++) {
        for (d = 0; d < k; d++) {
            double gz = 0.0;
            for (j = 0; j < p; j++)
                gz += -mat_get(resid, i, j) * mat_get(model->L, d, j) / sigma2_obs;
            mat_set(grad_Z, i, d, gz);
        }
    }

    for (d = 0; d < k; d++) {
        double quad = 0.0;
        double sigma2_d = model->sigma2_latent[d];
        for (i = 0; i < n; i++) {
            double val = 0.0;
            for (t = 0; t < n; t++)
                val += mat_get(ws->Sigma_inv, i, t) * mat_get(model->Z, t, d);
            quad += mat_get(model->Z, i, d) * val;
            mat_set(grad_Z, i, d, mat_get(grad_Z, i, d) + val / sigma2_d);
        }
        obj += 0.5 * quad / sigma2_d;
        obj += 0.5 * (double)n * log(sigma2_d);
        obj += 0.5 * ws->logdet_sigma;
        grad_log_sigma_latent[d] = -0.5 * quad / sigma2_d + 0.5 * (double)n;
    }

    for (d = 0; d < k; d++) {
        for (j = 0; j < p; j++) {
            double gl = 0.0;
            for (i = 0; i < n; i++)
                gl += -mat_get(model->Z, i, d) * mat_get(resid, i, j) / sigma2_obs;
            gl += lambda_L * mat_get(model->L, d, j);
            mat_set(grad_L, d, j, gl);
            obj += 0.5 * lambda_L * mat_get(model->L, d, j) * mat_get(model->L, d, j);
        }
    }

    mat_free(resid);
    return obj;
}

static double gex_model_grad_norm(Matrix *grad_Z,
                                  Matrix *grad_L,
                                  double *grad_log_sigma_latent,
                                  double grad_log_sigma_obs,
                                  int k) {
    int i, j;
    double ss = 0.0;
    for (i = 0; i < grad_Z->nrows; i++)
        for (j = 0; j < grad_Z->ncols; j++)
            ss += pow(mat_get(grad_Z, i, j), 2.0);
    for (i = 0; i < grad_L->nrows; i++)
        for (j = 0; j < grad_L->ncols; j++)
            ss += pow(mat_get(grad_L, i, j), 2.0);
    for (i = 0; i < k; i++)
        ss += pow(grad_log_sigma_latent[i], 2.0);
    ss += grad_log_sigma_obs * grad_log_sigma_obs;
    return sqrt(ss);
}

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
Returns the Euclidean norm across all matrix entries. */
static double gex_model_matrix_norm(Matrix *M) {
    int i, j;
    double ss = 0.0;

    for (i = 0; i < M->nrows; i++) {
        for (j = 0; j < M->ncols; j++)
            ss += pow(mat_get(M, i, j), 2.0);
    }

    return sqrt(ss);
}

/* Main model entry point. Fit a latent Brownian model to the given gene expression data.
Returns a pointer to the fitted model or NULL on failure. */
GexLatentBrownianModel *gex_fit_latent_brownian_model(GexMatrix *gex,
                                                      Matrix *Sigma,
                                                      GexPCA *pca,
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
    int converged = 0;   /* Whether the optimization stopped by satisfying the convergence rule */
    int final_step = 0;   /* Final optimization step reached before termination */
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
    const double objective_tol = 1e-4;  /* Relative objective tolerance used for convergence */

    /* Model related */
    int k;  /* Number of latent dimensions */
    int n_cells = gex->n_cells;
    int n_genes = gex->n_genes;
    GexLatentBrownianWorkspace ws;  /* Workspace for precomputed matrices and intermediate calculations */
    GexLatentBrownianModel *model = NULL;   /* Fitted model */

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
    int success = 0;   /* Whether the model fitting completed successfully */
    Matrix *L = NULL;   /* Temporary Cholesky factor for covariance calculations */
    double max_diag = 0.0;  /* Maximum diagonal element of Sigma */
    double jitter;  /* Diagonal jitter used for numerical stability */
    FILE *logf = NULL;  /* Optimization log file */
    char log_path[4096]; /* Path to optimization log file */

    /* Input validation */
    if (gex == NULL || gex->X == NULL || Sigma == NULL || pca == NULL ||
        pca->K <= 0 || outprefix == NULL)
        return NULL;
    if (Sigma->nrows != Sigma->ncols || Sigma->nrows != gex->n_cells)
        return NULL;

    /* Initialize workspace containers for the centered data matrix and the
    inverse/log-determinant calculations based on the phylogenetic covariance. */
    memset(&ws, 0, sizeof(ws));

    /* Open a log file to record the optimization trajectory while fitting
    the latent Brownian model. */
    snprintf(log_path, sizeof(log_path), "%s.model.log", outprefix);
    logf = fopen(log_path, "w");
    if (logf == NULL)
        goto cleanup_fit_latent_brownian_model;
    fprintf(logf, "step\tobjective\tlong_objective_running_avg\tshort_objective_running_avg\trel_objective_running_avg_change\tstable_steps\tgrad_norm\tsigma_obs");
    for (i = 0; i < pca->K; i++)
        fprintf(logf, "\tsigma_latent_LF%d", i + 1);
    fprintf(logf, "\tZ_norm\tL_norm\tstable_steps\n");

    /* Center the expression matrix by subtracting the mean of each gene.
    This ensures the latent factor model is fit to the residual structure
    after removing per-gene offsets. */
    ws.Xc = gex_model_center_matrix(gex->X);
    if (ws.Xc == NULL)
        goto cleanup_fit_latent_brownian_model;

    /* Build a regularized version of the phylogenetic covariance matrix and
    precompute its inverse and log-determinant for repeated use during fitting. */
    n = Sigma->nrows;
    ws.Sigma_reg = mat_new(n, n);
    ws.Sigma_inv = mat_new(n, n);
    L = mat_new(n, n);
    if (ws.Sigma_reg == NULL || ws.Sigma_inv == NULL || L == NULL)
        goto cleanup_fit_latent_brownian_model;

    /* Find the maximum diagonal element of the covariance matrix */
    for (i = 0; i < n; i++) {
        double d = mat_get(Sigma, i, i);
        if (d > max_diag)
            max_diag = d;
    }
    jitter = (max_diag > 0.0 ? 1e-8 * max_diag : 1e-8);

    /* Regularize the covariance matrix by adding jitter to the diagonal */
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++)
            mat_set(ws.Sigma_reg, i, j, mat_get(Sigma, i, j));
        mat_set(ws.Sigma_reg, i, i, mat_get(ws.Sigma_reg, i, i) + jitter);
    }

    /* Compute the inverse and Cholesky decomposition of the regularized covariance matrix */
    if (mat_invert(ws.Sigma_inv, ws.Sigma_reg) != 0 ||
        mat_cholesky(L, ws.Sigma_reg) != 0)
        goto cleanup_fit_latent_brownian_model;

    /* Compute the log-determinant of the regularized covariance matrix from the Cholesky factor */
    ws.logdet_sigma = 0.0;
    for (i = 0; i < n; i++) {
        double diag = mat_get(L, i, i);
        if (diag <= 0.0)
            goto cleanup_fit_latent_brownian_model;
        ws.logdet_sigma += 2.0 * log(diag);
    }

    /* Free the Cholesky factor since we only needed it for the log-determinant calculation. */
    mat_free(L);
    L = NULL;

    /* Use the number of input PCA components as the number of latent dimensions */
    k = pca->K; 

    /* Allocate the model object and its core parameter matrices. */
    model = (GexLatentBrownianModel *)calloc(1, sizeof(GexLatentBrownianModel));
    if (model == NULL)
        printf("ERROR: failed to allocate memory for the model object.\n");
        goto cleanup_fit_latent_brownian_model;
    model->n_cells = n_cells;
    model->n_genes = n_genes;
    model->k = k;
    model->Z = mat_new(n_cells, k); /* Allocate the latent factors matrix: cells × latent factors */
    model->L = mat_new(k, n_genes); /* Allocate the factor loading matrix: latent factors × genes */
    model->sigma2_latent = (double *)calloc(k, sizeof(double)); /* Allocate latent variance parameters */
    if (model->Z == NULL || model->L == NULL || model->sigma2_latent == NULL)
        goto cleanup_fit_latent_brownian_model;
    for (i = 0; i < k; i++)
        model->sigma2_latent[i] = 1.0;  /* Initialize the latent variance parameters to 1.0 */
    model->sigma2_obs = 1.0;    /* Initialize the observation variance parameter to 1.0 */

    /* Initialize the latent coordinates from the centered data matrix using
    simple standardized gene-derived starting vectors. This provides a stable
    non-degenerate starting point for the optimizer. */
    for (d = 0; d < model->k; d++) {
        int src_gene = d % model->n_genes;  /* Select a gene to initialize the latent factor */
        double mean = 0.0;
        double var = 0.0;
        /* Compute the mean and variance of the selected gene (already centered) across all cells */
        for (i = 0; i < model->n_cells; i++)
            mean += mat_get(ws.Xc, i, src_gene);
        mean /= (double)model->n_cells;
        for (i = 0; i < model->n_cells; i++) {
            double z = mat_get(ws.Xc, i, src_gene) - mean;
            mat_set(model->Z, i, d, z);
            var += z * z;
        }
        var /= (double)model->n_cells;

        /* If the variance of the selected gene is very small, initialize the latent factor 
        to a simple binary vector to avoid numerical issues. Otherwise, standardize 
        the latent factor to have unit variance. */
        if (var < 1e-8) {
            for (i = 0; i < model->n_cells; i++)
                mat_set(model->Z, i, d, (i == d % model->n_cells) ? 1.0 : 0.0);
        }
        else {
            double scale = 1.0 / sqrt(var);
            for (i = 0; i < model->n_cells; i++)
                mat_set(model->Z, i, d, mat_get(model->Z, i, d) * scale);
        }
    }

    /* Initialize the loading matrix by regressing each gene onto each latent
    factor independently. This gives a reasonable first approximation to the
    low-rank reconstruction before joint optimization. */
    for (d = 0; d < model->k; d++) {
        double denom = 1e-8;
        for (i = 0; i < model->n_cells; i++) {
            double zid = mat_get(model->Z, i, d);
            denom += zid * zid;
        }
        for (j = 0; j < model->n_genes; j++) {
            double numer = 0.0;
            for (i = 0; i < model->n_cells; i++)
                numer += mat_get(model->Z, i, d) * mat_get(ws.Xc, i, j);
            mat_set(model->L, d, j, numer / denom);
        }
    }
    {
        double sse = 0.0;
        for (i = 0; i < model->n_cells; i++) {
            for (j = 0; j < model->n_genes; j++) {
                double pred = 0.0;
                for (d = 0; d < model->k; d++)
                    pred += mat_get(model->Z, i, d) * mat_get(model->L, d, j);
                sse += pow(mat_get(ws.Xc, i, j) - pred, 2.0);
            }
        }
        model->sigma2_obs = sse / ((double)model->n_cells * model->n_genes);
        if (model->sigma2_obs < 1e-6)
            model->sigma2_obs = 1e-6;
    }

    /* Allocate gradients, optimizer state, and log-variance parameterization
    used during Adam optimization of the model parameters. */
    log_sigma_obs = log(model->sigma2_obs);
    log_sigma_latent = (double *)calloc(k, sizeof(double));
    grad_log_sigma_latent = (double *)calloc(k, sizeof(double));
    m_log_sigma_latent = (double *)calloc(k, sizeof(double));
    v_log_sigma_latent = (double *)calloc(k, sizeof(double));
    grad_Z = mat_new(model->n_cells, k);
    grad_L = mat_new(k, model->n_genes);
    mZ = mat_new(model->n_cells, k);
    vZ = mat_new(model->n_cells, k);
    mL = mat_new(k, model->n_genes);
    vL = mat_new(k, model->n_genes);
    if (log_sigma_latent == NULL || grad_log_sigma_latent == NULL ||
        m_log_sigma_latent == NULL || v_log_sigma_latent == NULL ||
        grad_Z == NULL || grad_L == NULL || mZ == NULL || vZ == NULL ||
        mL == NULL || vL == NULL)
        goto cleanup_fit_latent_brownian_model;

    objective_hist_long = (double *)calloc(running_avg_window_long, sizeof(double));
    objective_hist_short = (double *)calloc(running_avg_window_short, sizeof(double));
    if (objective_hist_long == NULL || objective_hist_short == NULL)
        goto cleanup_fit_latent_brownian_model;

    for (step = 0; step < k; step++)
        log_sigma_latent[step] = log(model->sigma2_latent[step]);

    mat_zero(mZ); mat_zero(vZ); mat_zero(mL); mat_zero(vL);
    (void)seed;

    /* Initialize the scheduler that controls learning-rate and clipping
    directives across optimization steps. */
    sched = sched_new(model->n_genes, model->n_genes, 1000, 0.03, 1, 1, 5);
    sched_state = sched_new_state(sched);
    if (sched == NULL || sched_state == NULL)
        goto cleanup_fit_latent_brownian_model;
    metrics.grad_norm = 0.0;

    /* Run gradient-based optimization of latent coordinates, gene loadings,
    and the variance parameters using Adam updates until the objective and
    gradient norm stabilize, while still enforcing a maximum number of steps. */
    for (step = 1; step <= max_steps; step++) {
        int d;

        sched_next(sched, sched_state, (step == 1 ? NULL : &metrics), &directives);

        model->sigma2_obs = exp(log_sigma_obs);
        if (model->sigma2_obs < 1e-8) model->sigma2_obs = 1e-8;
        for (d = 0; d < k; d++) {
            model->sigma2_latent[d] = exp(log_sigma_latent[d]);
            if (model->sigma2_latent[d] < 1e-8) model->sigma2_latent[d] = 1e-8;
        }

        model->objective = gex_model_objective_and_grad(model, &ws, grad_Z, grad_L,
                                                        grad_log_sigma_latent,
                                                        &grad_log_sigma_obs);
        metrics.grad_norm = gex_model_grad_norm(grad_Z, grad_L,
                                                grad_log_sigma_latent,
                                                grad_log_sigma_obs, k);

        /* Compare the short and long running averages of the objective so
        that convergence is judged using denoised trends at two time scales
        rather than noisy single-iteration values. */
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
        if (directives.clip_norm > 0.0 && metrics.grad_norm > directives.clip_norm) {
            double scale = directives.clip_norm / metrics.grad_norm;
            gex_model_scale_grads(grad_Z, grad_L, grad_log_sigma_latent,
                                  &grad_log_sigma_obs, k, scale);
            metrics.grad_norm = directives.clip_norm;
        }

        gex_model_adam_update_matrix(model->Z, grad_Z, mZ, vZ, step, directives.lr);
        gex_model_adam_update_matrix(model->L, grad_L, mL, vL, step, directives.lr);
        gex_model_adam_update_vector(log_sigma_latent, grad_log_sigma_latent,
                                     m_log_sigma_latent, v_log_sigma_latent,
                                     k, step, directives.lr);
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

        /* Record the scalar parameters and compact summaries of Z and L at
        each optimization step without writing the full matrices. */
        fprintf(logf, "%d\t%.17g\t%.17g\t%.17g\t%.17g\t%d\t%.17g\t%.17g",
                step,
                model->objective,
                running_objective_avg_long,
                running_objective_avg_short,
                rel_objective_change,
                stable_steps,
                metrics.grad_norm,
                model->sigma2_obs);
        for (d = 0; d < k; d++)
            fprintf(logf, "\t%.17g", model->sigma2_latent[d]);
        fprintf(logf, "\t%.17g\t%.17g\n",
                gex_model_matrix_norm(model->Z),
                gex_model_matrix_norm(model->L));
        fflush(logf);

        /* Update both moving-average histories online in O(1) time so the
        denoised convergence diagnostics do not slow optimization. */
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

        if (stable_steps >= stable_steps_needed) {
            converged = 1;
            final_step = step;
            break;
        }

    }
    if (!converged)
        final_step = max_steps;
    model->sigma2_obs = exp(log_sigma_obs);
    if (model->sigma2_obs < 1e-8) model->sigma2_obs = 1e-8;
    for (step = 0; step < k; step++) {
        model->sigma2_latent[step] = exp(log_sigma_latent[step]);
        if (model->sigma2_latent[step] < 1e-8)
            model->sigma2_latent[step] = 1e-8;
    }
    model->objective = gex_model_objective_and_grad(model, &ws, grad_Z, grad_L,
                                                    grad_log_sigma_latent,
                                                    &grad_log_sigma_obs);

    /* Write a final footer line describing why optimization terminated and
    the convergence settings used for the run. */
    fprintf(logf, "# termination\t%s\tfinal_step\t%d\tstable_steps\t%d\tstable_steps_needed\t%d\tmin_steps\t%d\trel_objective_tol\t%.17g\n",
            (converged ? "converged" : "max_steps_reached"),
            final_step,
            stable_steps,
            stable_steps_needed,
            min_steps,
            objective_tol);
    fflush(logf);

    success = 1;

    cleanup_fit_latent_brownian_model:
    free(log_sigma_latent);
    free(grad_log_sigma_latent);
    free(m_log_sigma_latent);
    free(v_log_sigma_latent);
    free(objective_hist_long);
    free(objective_hist_short);
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
    if (ws.Xc != NULL) mat_free(ws.Xc);
    if (ws.Sigma_reg != NULL) mat_free(ws.Sigma_reg);
    if (ws.Sigma_inv != NULL) mat_free(ws.Sigma_inv);

    if (!success) {
        if (model != NULL)
            gex_free_latent_brownian_model(model);
        return NULL;
    }

    return model;
}

int gex_write_latent_brownian_model(const char *outprefix,
                                    GexLatentBrownianModel *model,
                                    GexMatrix *gex) {
    FILE *summary_out = NULL, *z_out = NULL, *l_out = NULL;
    char summary_path[4096], z_path[4096], l_path[4096];
    int i, j;

    if (outprefix == NULL || model == NULL || gex == NULL)
        return -1;
    if (gex->n_cells != model->n_cells || gex->n_genes != model->n_genes)
        return -1;

    snprintf(summary_path, sizeof(summary_path), "%s.model.summary.tsv", outprefix);
    snprintf(z_path, sizeof(z_path), "%s.model.Z.tsv", outprefix);
    snprintf(l_path, sizeof(l_path), "%s.model.L.tsv", outprefix);

    summary_out = fopen(summary_path, "w");
    z_out = fopen(z_path, "w");
    l_out = fopen(l_path, "w");
    if (summary_out == NULL || z_out == NULL || l_out == NULL) {
        if (summary_out != NULL) fclose(summary_out);
        if (z_out != NULL) fclose(z_out);
        if (l_out != NULL) fclose(l_out);
        return -1;
    }

    fprintf(summary_out, "parameter\tvalue\n");
    fprintf(summary_out, "n_cells\t%d\n", model->n_cells);
    fprintf(summary_out, "n_genes\t%d\n", model->n_genes);
    fprintf(summary_out, "k\t%d\n", model->k);
    fprintf(summary_out, "objective\t%.17g\n", model->objective);
    fprintf(summary_out, "sigma_obs\t%.17g\n", model->sigma2_obs);
    for (j = 0; j < model->k; j++)
        fprintf(summary_out, "sigma_latent_LF%d\t%.17g\n", j + 1, model->sigma2_latent[j]);

    fprintf(z_out, "cell");
    for (j = 0; j < model->k; j++)
        fprintf(z_out, "\tLF%d", j + 1);
    fprintf(z_out, "\n");
    for (i = 0; i < model->n_cells; i++) {
        fprintf(z_out, "%s", gex->cell_names[i]);
        for (j = 0; j < model->k; j++)
            fprintf(z_out, "\t%.17g", mat_get(model->Z, i, j));
        fprintf(z_out, "\n");
    }

    fprintf(l_out, "factor");
    for (j = 0; j < model->n_genes; j++)
        fprintf(l_out, "\t%s", gex->gene_names[j]);
    fprintf(l_out, "\n");
    for (i = 0; i < model->k; i++) {
        fprintf(l_out, "LF%d", i + 1);
        for (j = 0; j < model->n_genes; j++)
            fprintf(l_out, "\t%.17g", mat_get(model->L, i, j));
        fprintf(l_out, "\n");
    }

    fclose(summary_out);
    fclose(z_out);
    fclose(l_out);
    return 0;
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
