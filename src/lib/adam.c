#include "adam.h"

#include "matrix.h"

#include <phast/matrix.h>

#include <math.h>

/* Cosine decay of learning rate */
double adam_cosine_lr(double curr_lr,
                      double base_lr,
                      double min_lr,
                      int step,
                      int max_steps) {
    if (step <= 1 || step >= max_steps)
        return curr_lr; /* Stay at the current learning rate after decay ends */

    double progress = (double)step / (double)max_steps;
    double lr = base_lr * 0.5 * (1.0 + cos(M_PI * progress));
    lr = fmax(lr, min_lr);

    return lr;
}

/* Perform one Adam optimization update for a scalar parameter in place. */
void adam_step_scalar(double *param,
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
void adam_step_matrix(Matrix *param,
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
void adam_step_vector(double *param,
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

/* Uses an exponential moving average (EMA) to update the clipping threshold
to be clip_factor * normal gradient levels from the EMA */
double adam_update_clip_threshold(double grad_norm,
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

int adam_clip_matrix_by_norm(Matrix *grad, double norm, double clip_norm) {
    if (clip_norm > 0.0 && norm > clip_norm) {
        double scale = clip_norm / norm;
        mat_scale(grad, scale);
        return 1;
    }
    return 0;
}

int adam_clip_vector_by_norm(double *grad, int n, double norm, double clip_norm) {
    int i;
    if (clip_norm > 0.0 && norm > clip_norm) {
        double scale = clip_norm / norm;
        for (i = 0; i < n; i++)
            grad[i] *= scale;
        return 1;
    }
    return 0;
}
