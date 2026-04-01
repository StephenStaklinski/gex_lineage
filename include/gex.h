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

typedef struct {
    Matrix *corr;
    double *morans_i;
    double *pvals;
    double *qvals;
    int n_genes;
    int n_significant;
} GexMoransResult;

/* Read a NEXUS file containing TREE lines into an array of PHAST trees.
   On success, *n_trees is set to the number of trees loaded. */
TreeNode **gex_read_nexus(const char *filename, int *n_trees);
void gex_free_trees(TreeNode **trees, int n_trees);

/* Read a numeric tab-delimited matrix file into a GexMatrix. */
GexMatrix *gex_read_labeled_matrix(const char *filename);
void gex_free_matrix_data(GexMatrix *gex);

void gex_print_io_summary(TreeNode **trees, int n_trees, GexMatrix *gex);

GexMoransResult *gex_compute_morans_i(GexMatrix *gex,
                                      Matrix *W,
                                      int n_perm,
                                      unsigned int seed);
void gex_print_morans_summary(GexMoransResult *res,
                              GexMatrix *gex,
                              double max_q,
                              double min_i);
void gex_free_morans_result(GexMoransResult *res);

GexMatrix *gex_filter_genes_by_morans_result(GexMatrix *gex,
                                             GexMoransResult *res,
                                             double max_q,
                                             double min_i);

#endif
