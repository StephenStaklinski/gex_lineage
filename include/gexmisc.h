#ifndef GECXMISC_H
#define GECXMISC_H

#include <phast/vector.h>


Vector *expand_input_csv(Vector *input, int expected_size);

unsigned int rand_u32(unsigned int *state);

double uniform_open(unsigned int *state);

double rand_normal(unsigned int *state);

double logsumexp(double *x, int n);

void generate_names(char **names, int n_names, char *prefix);

#endif