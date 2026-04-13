#ifndef GEXPHYLOFILTER_H
#define GEXPHYLOFILTER_H

#include "gexmatrix.h"

#include <phast/matrix.h>


typedef enum {
    GEX_LRT_NULL_MONTECARLO = 0,
    GEX_LRT_NULL_CHI2 = 1
} GexLRTNullMode;

typedef enum {
    GEX_LRT_ALT_FULL = 0,
    GEX_LRT_ALT_LAMBDA = 1
} GexLRTAltMode;

typedef struct {
    double *morans_i;
    double *zscores;
    double *pvals;
    double *qvals;
    int n_genes;
    int n_significant;
} MoranResult;

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

Matrix *weight_matrix_from_covariance(Matrix *Sigma);

void print_weight_matrix_summary(Matrix *W);

MoranResult *gex_compute_morans_i(Matrix *X,
                                      Matrix **Sigmas,
                                      int n_sigmas);

void write_moran_tsv(const char *filename,
                         MoranResult *res,
                         GexMatrix *gex,
                         double max_q,
                         double min_i);

void free_moran_result(MoranResult *res);

GexLRTResult *gex_compute_brownian_lrt(GexMatrix *gex,
                                       Matrix **Sigmas,
                                       int n_sigmas,
                                       int n_mc,
                                       unsigned int seed,
                                       GexLRTAltMode alt_mode);

void write_lrt_tsv(const char *filename,
                      GexLRTResult *res,
                      GexMatrix *gex,
                      double max_q);
                      
void gex_free_lrt_result(GexLRTResult *res);

GexMatrix *gex_filter_genes_by_results(GexMatrix *gex,
                                       MoranResult *morans,
                                       GexLRTResult *lrt,
                                       GexFilterMode mode,
                                       double max_q,
                                       double min_i);

#endif