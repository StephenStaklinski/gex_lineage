#ifndef GEX_H
#define GEX_H

#include <stdio.h>
#include <phast/matrix.h>
#include <phast/trees.h>

typedef struct {
    Matrix *X;  /* cells (nrows) x genes (ncols) */
    char **cell_names;
    char **gene_names;
} GexMatrix;

typedef enum {
    GEX_LRT_NULL_MONTECARLO = 0,
    GEX_LRT_NULL_CHI2 = 1
} GexLRTNullMode;

typedef enum {
    GEX_LRT_ALT_FULL = 0,
    GEX_LRT_ALT_LAMBDA = 1
} GexLRTAltMode;

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
    double *lambda_hat;
    double *lrt_stat;
    double *pvals;
    double *qvals;
    GexLRTAltMode alt_mode;
    int n_genes;
    int n_significant;
} GexLRTResult;

typedef enum {
    GEX_FILTER_MORAN = 0,
    GEX_FILTER_LRT = 1,
    GEX_FILTER_BOTH = 2
} GexFilterMode;

TreeNode **gex_read_nexus(const char *filename, int *n_trees);

int gex_check_trees_ultrametric(TreeNode **trees, int n_trees, double tol);

int gex_rescale_trees_total_height(TreeNode **trees, int n_trees, double target_height);

void gex_free_trees(TreeNode **trees, int n_trees);

GexMatrix *read_gex_matrix(const char *filename);

void gex_free_matrix_data(GexMatrix *gex);

int normalize_by_row_sums(Matrix *X);

int apply_scaling_factor_elementwise(Matrix *X, double scaling_factor);

int log1p_transform(Matrix *X);

int center_matrix_inplace(Matrix *X);

void gex_print_io_summary(TreeNode **trees, int n_trees, GexMatrix *gex);

int gex_reconcile_tree_and_expression(TreeNode **trees,
                                      int n_trees,
                                      GexMatrix **gex_ptr);

GexMoransResult *gex_compute_morans_i(GexMatrix *gex,
                                      Matrix **Sigmas,
                                      int n_sigmas,
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
                                       Matrix **Sigmas,
                                       int n_sigmas,
                                       int n_mc,
                                       unsigned int seed,
                                       GexLRTAltMode alt_mode);

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

int gex_write_labeled_matrix_tsv(const char *filename,
                                 Matrix *X,
                                 char **row_names,
                                 int n_rows,
                                 char **col_names,
                                 int n_cols,
                                 const char *corner_label);

int gex_write_model(const char *outprefix,
                               GexMatrix *gex,
                               Matrix *L,
                               Matrix *Z,
                               char **cell_names,
                               char **gene_names,
                               int k,
                               double sigma2_obs,
                               double *sigma2_latent);

int gex_simulate_from_latent_factors(Matrix *Z,
                                     char **cell_names,
                                     int n_cells,
                                     int k,
                                     int n_genes,
                                     double sigma2_obs,
                                     unsigned int seed,
                                     Matrix **L_out,
                                     GexMatrix **gex_out);

/* Helpers */
void free_string_array(char **arr, int n);

unsigned int rand_u32(unsigned int *state);

double uniform_open(unsigned int *state);

double rand_normal(unsigned int *state);

#endif
