#ifndef GEXPCA_H
#define GEXPCA_H

#include "matrix.h"

#include <phast/matrix.h>

typedef enum {
    PCA_METHOD_PCA = 0,
    PCA_METHOD_PHYLOPCA = 1,
    PCA_METHOD_MAX_PHYLOPCA = 2,
    PCA_METHOD_NONE = 3,
} PcaMethod;

typedef struct {
    Matrix *components;   // K x n_genes (principal components / loadings)
    double *eigenvalues;  // length K
    double *var_explained; // length K
    int K;
} PCA;

PCA *compute_pca(Matrix *gex, int k, double variance_threshold);

PCA *compute_phylo_pca(Matrix *gex, Matrix *C, int k, double variance_threshold);

PCA *compute_max_phylo_pca(Matrix *gex, Matrix *C, int k, double variance_threshold);

void free_pca(PCA *pca);

void print_pca_summary(PCA *pca);

void write_pca_tsv(const char *outprefix, PCA *pca, GexMatrix *gex);

#endif
