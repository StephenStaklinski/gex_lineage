#include "gexmisc.h"

#include <phast/vector.h>
#include <phast/misc.h>

#include <math.h>
#include <stdio.h>
#include <string.h>

#include <stdlib.h>


char **copy_string_array(char **input, int n) {
    int i;
    char **copy = scalloc(n, sizeof(char *));
    for (i = 0; i < n; i++) {
        copy[i] = strdup(input[i]);
    }
    return copy;
}

char **copy_string_array_inplace(char **output, char **input, int n) {
    int i;
    for (i = 0; i < n; i++) {
        output[i] = strdup(input[i]);
    }
    return output;
}

/* Expand a vector to a specified size, filling with the first element if necessary. */
Vector *expand_input_csv(Vector *input, int expected_size) {
    int i;
    Vector *out = vec_new(expected_size);

    if (input->size == expected_size) {
        /* Copy as is for all positions */
        vec_copy(out, input);
    } else {

    /* Copy the first entry to all positions */
    for (i = 0; i < expected_size; i++)
        vec_set(out, i, vec_get(input, 0));
    }

    return out;
}


/* Generate next 32-bit unsigned integer from RNG state */
unsigned int rand_u32(unsigned int *state) {
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

/* Generate uniform random number in (0,1), excluding endpoints */
double uniform_open(unsigned int *state) {
    return ((double)rand_u32(state) + 1.0) / 4294967297.0;
}

/* Generate standard normal random variable (mean 0, variance 1) */
double rand_normal(unsigned int *state) {
    double u1 = uniform_open(state);
    double u2 = uniform_open(state);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2); 
}

/* Compute log(sum_i exp(x[i])) in a numerically stable way using the
log-sum-exp trick: max(x) + log(sum_i exp(x[i] - max(x))). */
double logsumexp(double *x, int n) {
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

    /* Return the log-sum-exp */
    return max_x + log(sum);
}

/* Fill a preallocated array of names of specified length
incrementally with a specified prefix. prefix_1, prefix_2, etc. */
void generate_names(char **names, int n_names, char *prefix) {
    int j;
    for (j = 0; j < n_names; j++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "%s_%d", prefix, j + 1);
        names[j] = strdup(buf);
    }
}
