#ifndef BROWNIAN_H
#define BROWNIAN_H

#include <phast/matrix.h>
#include <phast/trees.h>

#include "gex.h"

Matrix *covariance_from_tree(TreeNode *tree, char **names, int n);

Matrix *gex_average_tree_covariance(TreeNode **trees,
                                    int n_trees,
                                    char **names,
                                    int n_names);

Matrix *weight_matrix_from_covariance(Matrix *Sigma);

int add_matrix_in_place(Matrix *dest, Matrix *src);

int scale_matrix_in_place(Matrix *matrix, double factor);

GexMatrix *brownian_simulate_expression_from_covariance(Matrix *Sigma,
                                                        char **names,
                                                        int n,
                                                        int n_genes,
                                                        double *sigma2,
                                                        int n_sigma2,
                                                        unsigned int seed);

GexMatrix *simulate_standard_normal_expression(char **names,
                                             int n,
                                             int n_genes,
                                             unsigned int seed);

GexMatrix *brownian_combine_expression_matrices(GexMatrix *pos_gex,
                                                GexMatrix *neg_gex);

GexMatrix *brownian_simulate_expression_with_nulls(Matrix *Sigma,
                                                  char **names,
                                                  int n,
                                                  int n_tree_genes,
                                                  int n_null_genes,
                                                  unsigned int seed);

int brownian_run_simulation_check(char **names,
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

int generate_gene_names(char **names, int n_genes);

#endif
