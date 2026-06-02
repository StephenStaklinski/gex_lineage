#ifndef GEXBROWNIAN_H
#define GEXBROWNIAN_H

#include "matrix.h"
#include "phylofilter.h"

#include <phast/matrix.h>
#include <phast/trees.h>

Matrix *covariance_from_tree(TreeNode *tree, char **names, int n);

Matrix *gex_average_tree_covariance(TreeNode **trees,
                                    int n_trees,
                                    char **names,
                                    int n_names);

Matrix *brownian_simulate(Matrix **Sigmas, int n_sigmas, Vector *mu, int n_cols,
                          Vector *sigma2s);

void print_covariance_summary(Matrix *Sigma, char **names, int n);

#endif
