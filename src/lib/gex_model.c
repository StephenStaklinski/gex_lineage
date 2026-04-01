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
        if (Xc != NULL) mat_free(Xc);
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

static int gex_model_setup_workspace(Matrix *Sigma, GexLatentBrownianWorkspace *ws) {
    int i, j, n;
    Matrix *L = NULL;
    double max_diag = 0.0;
    double jitter;
    if (Sigma == NULL || Sigma->nrows != Sigma->ncols)
        return -1;

    n = Sigma->nrows;
    ws->Sigma_reg = mat_new(n, n);
    ws->Sigma_inv = mat_new(n, n);
    L = mat_new(n, n);
    if (ws->Sigma_reg == NULL || ws->Sigma_inv == NULL || L == NULL) {
        if (ws->Sigma_reg != NULL) mat_free(ws->Sigma_reg);
        if (ws->Sigma_inv != NULL) mat_free(ws->Sigma_inv);
        if (L != NULL) mat_free(L);
        return -1;
    }

    for (i = 0; i < n; i++) {
        double d = mat_get(Sigma, i, i);
        if (d > max_diag)
            max_diag = d;
    }
    jitter = (max_diag > 0.0 ? 1e-8 * max_diag : 1e-8);

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++)
            mat_set(ws->Sigma_reg, i, j, mat_get(Sigma, i, j));
        mat_set(ws->Sigma_reg, i, i, mat_get(ws->Sigma_reg, i, i) + jitter);
    }

    if (mat_invert(ws->Sigma_inv, ws->Sigma_reg) != 0 ||
        mat_cholesky(L, ws->Sigma_reg) != 0) {
        mat_free(ws->Sigma_reg);
        mat_free(ws->Sigma_inv);
        mat_free(L);
        return -1;
    }

    ws->logdet_sigma = 0.0;
    for (i = 0; i < n; i++)
        ws->logdet_sigma += 2.0 * log(mat_get(L, i, i));

    mat_free(L);
    return 0;
}

static void gex_model_free_workspace(GexLatentBrownianWorkspace *ws) {
    if (ws->Xc != NULL) mat_free(ws->Xc);
    if (ws->Sigma_reg != NULL) mat_free(ws->Sigma_reg);
    if (ws->Sigma_inv != NULL) mat_free(ws->Sigma_inv);
}

static GexLatentBrownianModel *gex_model_new(int n_cells, int n_genes, int k) {
    int i;
    GexLatentBrownianModel *model = NULL;

    model = (GexLatentBrownianModel *)calloc(1, sizeof(GexLatentBrownianModel));
    if (model == NULL)
        return NULL;

    model->n_cells = n_cells;
    model->n_genes = n_genes;
    model->k = k;
    model->Z = mat_new(n_cells, k);
    model->L = mat_new(k, n_genes);
    model->sigma2_latent = (double *)calloc(k, sizeof(double));
    if (model->Z == NULL || model->L == NULL || model->sigma2_latent == NULL) {
        gex_free_latent_brownian_model(model);
        return NULL;
    }
    for (i = 0; i < k; i++)
        model->sigma2_latent[i] = 1.0;
    model->sigma2_obs = 1.0;

    return model;
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

GexLatentBrownianModel *gex_fit_latent_brownian_model(GexMatrix *gex,
                                                      Matrix *Sigma,
                                                      GexPCA *pca,
                                                      unsigned int seed) {
    int i, j, d;
    int step;
    int total_steps = 250;
    int k;
    double log_sigma_obs;
    double *log_sigma_latent = NULL;
    double *grad_log_sigma_latent = NULL;
    double *m_log_sigma_latent = NULL;
    double *v_log_sigma_latent = NULL;
    double grad_log_sigma_obs = 0.0;
    double m_log_sigma_obs = 0.0;
    double v_log_sigma_obs = 0.0;
    GexLatentBrownianWorkspace ws;
    GexLatentBrownianModel *model = NULL;
    Matrix *grad_Z = NULL, *grad_L = NULL, *mZ = NULL, *vZ = NULL, *mL = NULL, *vL = NULL;
    Scheduler *sched = NULL;
    SchedState *sched_state = NULL;
    SchedDirectives directives;
    SchedMetrics metrics;

    if (gex == NULL || Sigma == NULL || pca == NULL || pca->K <= 0)
        return NULL;

    memset(&ws, 0, sizeof(ws));
    ws.Xc = gex_model_center_matrix(gex->X);
    if (ws.Xc == NULL || gex_model_setup_workspace(Sigma, &ws) != 0) {
        gex_model_free_workspace(&ws);
        return NULL;
    }

    k = pca->K;
    model = gex_model_new(gex->n_cells, gex->n_genes, k);
    if (model == NULL) {
        gex_model_free_workspace(&ws);
        return NULL;
    }
    for (d = 0; d < model->k; d++) {
        int src_gene = d % model->n_genes;
        double mean = 0.0;
        double var = 0.0;
        for (i = 0; i < model->n_cells; i++)
            mean += mat_get(ws.Xc, i, src_gene);
        mean /= (double)model->n_cells;
        for (i = 0; i < model->n_cells; i++) {
            double z = mat_get(ws.Xc, i, src_gene) - mean;
            mat_set(model->Z, i, d, z);
            var += z * z;
        }
        var /= (double)model->n_cells;
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
        mL == NULL || vL == NULL) {
        gex_free_latent_brownian_model(model);
        gex_model_free_workspace(&ws);
        free(log_sigma_latent);
        free(grad_log_sigma_latent);
        free(m_log_sigma_latent);
        free(v_log_sigma_latent);
        if (grad_Z != NULL) mat_free(grad_Z);
        if (grad_L != NULL) mat_free(grad_L);
        if (mZ != NULL) mat_free(mZ);
        if (vZ != NULL) mat_free(vZ);
        if (mL != NULL) mat_free(mL);
        if (vL != NULL) mat_free(vL);
        return NULL;
    }

    for (step = 0; step < k; step++)
        log_sigma_latent[step] = log(model->sigma2_latent[step]);

    mat_zero(mZ); mat_zero(vZ); mat_zero(mL); mat_zero(vL);
    (void)seed;

    sched = sched_new(model->n_genes, model->n_genes, 1000, 0.03, 1, 1, 5);
    sched_state = sched_new_state(sched);
    metrics.grad_norm = 0.0;

    for (step = 1; step <= total_steps; step++) {
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

    }
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

    free(log_sigma_latent);
    free(grad_log_sigma_latent);
    free(m_log_sigma_latent);
    free(v_log_sigma_latent);
    mat_free(grad_Z); mat_free(grad_L); mat_free(mZ); mat_free(vZ); mat_free(mL); mat_free(vL);
    free(sched_state);
    free(sched);
    gex_model_free_workspace(&ws);
    return model;
}

int gex_write_latent_brownian_model(const char *filename,
                                    GexLatentBrownianModel *model) {
    FILE *out;
    int i, j;

    if (filename == NULL || model == NULL)
        return -1;

    out = fopen(filename, "w");
    if (out == NULL)
        return -1;

    fprintf(out, "param_type\tindex1\tindex2\tvalue\n");
    fprintf(out, "summary\tn_cells\t.\t%d\n", model->n_cells);
    fprintf(out, "summary\tn_genes\t.\t%d\n", model->n_genes);
    fprintf(out, "summary\tk\t.\t%d\n", model->k);
    fprintf(out, "summary\tobjective\t.\t%.17g\n", model->objective);
    fprintf(out, "sigma_obs\t0\t.\t%.17g\n", model->sigma2_obs);
    for (j = 0; j < model->k; j++)
        fprintf(out, "sigma_latent\t%d\t.\t%.17g\n", j, model->sigma2_latent[j]);
    for (i = 0; i < model->n_cells; i++) {
        for (j = 0; j < model->k; j++)
            fprintf(out, "Z\t%d\t%d\t%.17g\n", i, j, mat_get(model->Z, i, j));
    }
    for (i = 0; i < model->k; i++) {
        for (j = 0; j < model->n_genes; j++)
            fprintf(out, "L\t%d\t%d\t%.17g\n", i, j, mat_get(model->L, i, j));
    }

    fclose(out);
    return 0;
}

void gex_free_latent_brownian_model(GexLatentBrownianModel *model) {
    if (model == NULL)
        return;
    if (model->Z != NULL) mat_free(model->Z);
    if (model->L != NULL) mat_free(model->L);
    if (model->sigma2_latent != NULL) free(model->sigma2_latent);
    if (model->latent_mvn != NULL) mvn_free(model->latent_mvn);
    free(model);
}
