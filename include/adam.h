#ifndef GEX_LINEAGE_ADAM_H
#define GEX_LINEAGE_ADAM_H

#include <phast/matrix.h>

#define ADAM_BETA1 0.9
#define ADAM_BETA2 0.9
#define ADAM_EPS 1e-8

double adam_cosine_lr(double curr_lr,
                      double base_lr,
                      double min_lr,
                      int step,
                      int max_steps);

void adam_step_scalar(double *param,
                      double grad,
                      double *m,
                      double *v,
                      double pow_beta1,
                      double pow_beta2,
                      double lr);

void adam_step_matrix(Matrix *param,
                      Matrix *grad,
                      Matrix *m,
                      Matrix *v,
                      double pow_beta1,
                      double pow_beta2,
                      double lr);

void adam_step_vector(double *param,
                      double *grad,
                      double *m,
                      double *v,
                      int n,
                      double pow_beta1,
                      double pow_beta2,
                      double lr);

double adam_update_clip_threshold(double grad_norm,
                                  double *ema_grad_norm,
                                  int step,
                                  int clip_warmup,
                                  double clip_beta,
                                  double clip_factor,
                                  double clip_floor);

int adam_clip_matrix_by_norm(Matrix *grad, double norm, double clip_norm);

int adam_clip_vector_by_norm(double *grad, int n, double norm, double clip_norm);

#endif
