#ifndef GEX_LINEAGE_PARSER_H
#define GEX_LINEAGE_PARSER_H

#include "gexmatrix.h"

#include <phast/matrix.h>
#include <phast/trees.h>
#include <phast/vector.h>

TreeNode **read_nexus(const char *filename, int *n_trees, int max_trees);

int check_trees_ultrametric(TreeNode **trees, int n_trees);

void uniform_rescale_trees(TreeNode **trees, int n_trees, double target_height);

void gex_free_trees(TreeNode **trees, int n_trees);

GexMatrix *read_gex_matrix(const char *filename);

int gex_reconcile_tree_and_expression(TreeNode **trees,
                                      int n_trees,
                                      GexMatrix **gex_ptr);

int load_and_reconcile_tree_gex_inputs(const char *trees_file,
                                       const char *expr_file,
                                       int max_trees,
                                       TreeNode ***trees_out,
                                       int *n_trees_out,
                                       GexMatrix **gex_out);

Vector *parse_csv_to_vec(const char *csv);

#endif
