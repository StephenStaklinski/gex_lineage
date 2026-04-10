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