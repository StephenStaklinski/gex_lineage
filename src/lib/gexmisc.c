#include "gexmisc.h"

#include <math.h>

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

/* Get the mean and (population) variance of vector y. */
void calculate_mean_variance(double *y, int n, double *mean_out, double *sigma2_out) {
    int i;
    double mean = 0.0;
    double sse = 0.0;

    /* Compute the mean of the data */
    for (i = 0; i < n; i++)
        mean += y[i];
    mean /= (double)n;

    /* Compute the sum of squared errors around the mean */
    for (i = 0; i < n; i++) {
        double d = y[i] - mean;
        sse += d * d;
    }

    *mean_out = mean;
    *sigma2_out = sse / (double)n;  /* Variance */
    if (*sigma2_out < 1e-12)
        *sigma2_out = 1e-12;
}

/* Fill a preallocated array of names of specified length
incrementally with a specified prefix. prefix_1, prefix_2, etc. */
void generate_names(char **names, int n_names, char *prefix) {
    int i, j;

    for (j = 0; j < n_names; j++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "%s_%04d", prefix, j + 1);
        names[j] = strdup(buf);
    }
}
