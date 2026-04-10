#ifndef GEXPARSER_H
#define GEXPARSER_H

#include "gexmatrix.h"

#include <phast/matrix.h>
#include <phast/trees.h>

TreeNode **gex_read_nexus(const char *filename, int *n_trees);

int gex_check_trees_ultrametric(TreeNode **trees, int n_trees, double tol);

int gex_rescale_trees_total_height(TreeNode **trees, int n_trees, double target_height);

void gex_free_trees(TreeNode **trees, int n_trees);

GexMatrix *read_gex_matrix(const char *filename);

void gex_free_matrix_data(GexMatrix *gex);

void gex_print_io_summary(TreeNode **trees, int n_trees, GexMatrix *gex);

int gex_reconcile_tree_and_expression(TreeNode **trees,
                                      int n_trees,
                                      GexMatrix **gex_ptr);

#endif
