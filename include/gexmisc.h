#ifndef GECXMISC_H
#define GECXMISC_H

#include <phast/vector.h>

char **copy_string_array(char **input, int n);

char **copy_string_array_inplace(char **output, char **input, int n);

Vector *expand_input_csv(Vector *input, int expected_size);

unsigned int rand_u32(unsigned int *state);

double uniform_open(unsigned int *state);

double rand_normal(unsigned int *state);

double logsumexp(double *x, int n);

void generate_names(char **names, int n_names, char *prefix);

#endif