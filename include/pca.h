#ifndef GEXPCA_H
#define GEXPCA_H

#include "gexmatrix.h"

#include <phast/matrix.h>

typedef struct {
    Matrix *components;   // K x n_genes (principal components / loadings)
    double *eigenvalues;  // length K
    double *var_explained; // length K
    int K;
} PCA;

PCA *compute_pca(Matrix *gex, int k);

PCA *compute_max_phylo_pca(Matrix *gex, Matrix *C, int k);

PCA *read_pca_initialization_tsv(const char *filename, GexMatrix *gex, int k);

void free_pca(PCA *pca);

void print_pca_summary(PCA *pca);

void write_pca_tsv(const char *outprefix, PCA *pca, GexMatrix *gex);

#endif
