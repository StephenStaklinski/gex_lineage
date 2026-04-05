#ifndef BROWNIAN_H
#define BROWNIAN_H

#include <phast/matrix.h>
#include <phast/trees.h>

#include "gex.h"

Matrix *covariance_from_tree(TreeNode *tree, char **names, int n);

Matrix *weight_matrix_from_covariance(Matrix *Sigma);

GexMatrix *brownian_simulate_expression(TreeNode *tree,
                                        char **names,
                                        int n,
                                        int n_tree_genes,
                                        int n_null_genes,
                                        unsigned int seed);

int brownian_run_simulation_check(TreeNode *tree,
                                  char **names,
                                  int n,
                                  int n_tree_genes,
                                  int n_null_genes,
                                  GexFilterMode mode,
                                  GexLRTAltMode lrt_alt_mode,
                                  int n_perm,
                                  double max_q,
                                  double min_i,
                                  Matrix **Sigmas,
                                  int n_sigmas,
                                  unsigned int seed);

void print_covariance_summary(Matrix *Sigma, char **names, int n);

void print_weight_matrix_summary(Matrix *W);

#endif
