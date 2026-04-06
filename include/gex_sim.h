#ifndef GEX_SIM_H
#define GEX_SIM_H

#include "gex.h"

typedef struct {
    int n_cells;
    int n_genes;
    int k;
    Matrix *Z;
    Matrix *L;
    double *sigma2_latent;
    double sigma2_obs;
} GexSimulationTruth;

int gex_get_shared_leaf_names(TreeNode **trees,
                              int n_trees,
                              char ***names_out,
                              int *n_names_out);

GexSimulationTruth *gex_simulate_latent_brownian_expression(Matrix *Sigma,
                                                            char **cell_names,
                                                            int n_cells,
                                                            int k,
                                                            int n_genes,
                                                            const double *sigma2_latent,
                                                            double sigma2_obs,
                                                            unsigned int seed,
                                                            GexMatrix **gex_out);

int gex_write_labeled_matrix_tsv(const char *filename,
                                 Matrix *X,
                                 char **row_names,
                                 int n_rows,
                                 char **col_names,
                                 int n_cols,
                                 const char *corner_label);

int gex_write_simulation_truth(const char *outprefix,
                               GexSimulationTruth *truth,
                               char **cell_names,
                               char **gene_names);

void gex_free_simulation_truth(GexSimulationTruth *truth);

#endif
