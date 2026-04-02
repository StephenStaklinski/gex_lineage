#ifndef PCA_H
#define PCA_H

#include "gex.h"

#include <stdio.h>
#include <phast/matrix.h>
#include <phast/trees.h>


typedef struct {
    Matrix *components;   // K x n_genes (principal components / loadings)
    double *var_explained; // length K
    int K;
} GexPCA;

GexPCA *gex_compute_pca(GexMatrix *gex, double variance_threshold);

void gex_free_pca(GexPCA *pca);

void gex_print_pca_summary(GexPCA *pca);

#endif
