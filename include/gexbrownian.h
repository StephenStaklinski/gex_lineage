#ifndef GEXBROWNIAN_H
#define GEXBROWNIAN_H

#include "gexmatrix.h"
#include "gexphylofilter.h"

#include <phast/matrix.h>
#include <phast/trees.h>

Matrix *covariance_from_tree(TreeNode *tree, char **names, int n);

Matrix *gex_average_tree_covariance(TreeNode **trees,
                                    int n_trees,
                                    char **names,
                                    int n_names);

Matrix *brownian_simulate(Matrix **Sigmas, int n_sigmas, Vector *mu, int n_cols,
                          double desired_tip_var);

void print_covariance_summary(Matrix *Sigma, char **names, int n);

int gex_simulate_from_latent_factors(Matrix *Z,
                                     char **cell_names,
                                     int n_cells,
                                     int k,
                                     int n_genes,
                                     double sigma2_obs,
                                     unsigned int seed,
                                     Matrix **L_out,
                                     GexMatrix **gex_out);

#endif
