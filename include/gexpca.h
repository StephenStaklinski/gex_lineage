#ifndef GEXPCA_H
#define GEXPCA_H

#include <phast/matrix.h>

typedef enum {
    PCA_METHOD_PCA = 0,
    PCA_METHOD_PHYLOPCA = 1,
    PCA_METHOD_NONE = 2,
} PcaMethod;

typedef struct {
    Matrix *components;   // K x n_genes (principal components / loadings)
    double *var_explained; // length K
    int K;
} PCA;

PCA *compute_pca(Matrix *gex, int k, double variance_threshold);

PCA *compute_phylo_pca(Matrix *gex, Matrix *C, int k, double variance_threshold);

void free_pca(PCA *pca);

void print_pca_summary(PCA *pca);

#endif
