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

typedef struct {
    double *ll_null;
    double *ll_alt;
    double *lrt_stat;
    double *pvals;
    double *qvals;
    int n_genes;
    int n_significant;
} GexLRTResult;

typedef enum {
    GEX_FILTER_MORAN = 0,
    GEX_FILTER_LRT = 1,
    GEX_FILTER_BOTH = 2
} GexFilterMode;

typedef enum {
    GEX_LRT_NULL_MONTECARLO = 0,
    GEX_LRT_NULL_CHI2 = 1
} GexLRTNullMode;

TreeNode **gex_read_nexus(const char *filename, int *n_trees);

int gex_check_trees_ultrametric(TreeNode **trees, int n_trees, double tol);

int gex_rescale_trees_total_height(TreeNode **trees, int n_trees, double target_height);

void gex_free_trees(TreeNode **trees, int n_trees);

GexMatrix *gex_read_labeled_matrix(const char *filename);

void gex_free_matrix_data(GexMatrix *gex);

void gex_print_io_summary(TreeNode **trees, int n_trees, GexMatrix *gex);

int gex_reconcile_tree_and_expression(TreeNode **trees,
                                      int n_trees,
                                      GexMatrix **gex_ptr);

GexMoransResult *gex_compute_morans_i(GexMatrix *gex,
                                      Matrix *W,
                                      int n_perm,
                                      unsigned int seed);

void gex_print_morans_summary(GexMoransResult *res,
                              GexMatrix *gex,
                              double max_q,
                              double min_i);

int gex_write_morans_tsv(const char *filename,
                         GexMoransResult *res,
                         GexMatrix *gex,
                         double max_q,
                         double min_i);

void gex_free_morans_result(GexMoransResult *res);

GexLRTResult *gex_compute_brownian_lrt(GexMatrix *gex,
                                       Matrix *Sigma,
                                       int n_mc,
                                       unsigned int seed);

void gex_print_lrt_summary(GexLRTResult *res,
                           GexMatrix *gex,
                           double max_q);

int gex_write_lrt_tsv(const char *filename,
                      GexLRTResult *res,
                      GexMatrix *gex,
                      double max_q);
                      
void gex_free_lrt_result(GexLRTResult *res);

GexMatrix *gex_filter_genes_by_results(GexMatrix *gex,
                                       GexMoransResult *morans,
                                       GexLRTResult *lrt,
                                       GexFilterMode mode,
                                       double max_q,
                                       double min_i);

#endif
