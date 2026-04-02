#ifndef BROWNIAN_H
#define BROWNIAN_H

#include <phast/matrix.h>
#include <phast/trees.h>

#include "gex.h"

Matrix *covariance_from_tree(TreeNode *tree, char **names, int n);

Matrix *brownian_weight_matrix_from_covariance(Matrix *Sigma);

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
                                  GexLRTNullMode lrt_null_mode,
                                  int n_perm,
                                  double max_q,
                                  double min_i,
                                  Matrix *Sigma,
                                  Matrix *W,
                                  unsigned int seed);

void brownian_print_covariance_summary(Matrix *Sigma, char **names, int n);

void brownian_print_weight_summary(Matrix *W);

#endif
