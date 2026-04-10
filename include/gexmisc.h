#ifndef GECXMISC_H
#define GECXMISC_H

unsigned int rand_u32(unsigned int *state);

double uniform_open(unsigned int *state);

double rand_normal(unsigned int *state);

double logsumexp(double *x, int n);

void calculate_mean_variance(double *y, int n, double *mean_out, double *sigma2_out);

#endif