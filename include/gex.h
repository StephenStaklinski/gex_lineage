#ifndef GEX_H
#define GEX_H

#include <stdio.h>
#include <phast/matrix.h>
#include <phast/trees.h>

typedef struct {
    Matrix *X;
    char **cell_names;
    char **gene_names;
    int n_cells;
    int n_genes;
} GexMatrix;

/* Read a NEXUS file containing TREE lines into an array of PHAST trees.
   Returns NULL on error.
   On success, *n_trees is set to the number of trees loaded. */
TreeNode **gex_read_nexus(const char *filename, int *n_trees);
void gex_free_trees(TreeNode **trees, int n_trees);

/* Read a numeric tab-delimited matrix file into a GexMatrix.
   Returns NULL on error. */
GexMatrix *gex_read_labeled_matrix(const char *filename);
void gex_free_matrix_data(GexMatrix *gex);

void gex_print_summary(TreeNode **trees, int n_trees, GexMatrix *gex);

#endif