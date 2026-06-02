#ifndef GEXPARSER_H
#define GEXPARSER_H

#include "matrix.h"

#include <phast/matrix.h>
#include <phast/trees.h>
#include <phast/vector.h>

TreeNode **read_nexus(const char *filename, int *n_trees);

int check_trees_ultrametric(TreeNode **trees, int n_trees);

void uniform_rescale_trees(TreeNode **trees, int n_trees, double target_height);

void gex_free_trees(TreeNode **trees, int n_trees);

GexMatrix *read_gex_matrix(const char *filename);

void gex_print_io_summary(TreeNode **trees, int n_trees, GexMatrix *gex);

int gex_reconcile_tree_and_expression(TreeNode **trees,
                                      int n_trees,
                                      GexMatrix **gex_ptr);

Vector *parse_csv_to_vec(const char *csv);

#endif
