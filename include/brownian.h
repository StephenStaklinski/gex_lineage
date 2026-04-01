#ifndef BROWNIAN_H
#define BROWNIAN_H

#include <phast/matrix.h>
#include <phast/trees.h>

#include "gex.h"

/* Build the Brownian-motion covariance matrix for tips in the exact
   order given by names[].
   Sigma[i][j] = shared root-to-MRCA branch length for tips i and j.
   Returns NULL on error. */
Matrix *brownian_covariance_from_tree(TreeNode *tree, char **names, int n);
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
                                  int n_perm,
                                  double max_q,
                                  double min_i,
                                  unsigned int seed);

void brownian_print_covariance_summary(Matrix *Sigma, char **names, int n);
void brownian_print_weight_summary(Matrix *W);

#endif
