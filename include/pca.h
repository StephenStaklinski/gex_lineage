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
} PCA;

PCA *compute_pca(Matrix *gex, double variance_threshold);

void free_pca(PCA *pca);

void print_pca_summary(PCA *pca);

Matrix *center_matrix(Matrix *X);

#endif
