#ifndef GEX_SIM_H
#define GEX_SIM_H

#include "gex.h"


int gex_get_shared_leaf_names(TreeNode **trees,
                              int n_trees,
                              char ***names_out,
                              int *n_names_out);

int gex_simulate_from_latent_factors(GexMatrix *Z,
                                     char **cell_names,
                                     int n_cells,
                                     int k,
                                     int n_genes,
                                     double sigma2_obs,
                                     unsigned int seed,
                                     GexMatrix **L_out,
                                     GexMatrix **gex_out);

int gex_write_labeled_matrix_tsv(const char *filename,
                                 Matrix *X,
                                 char **row_names,
                                 int n_rows,
                                 char **col_names,
                                 int n_cols,
                                 const char *corner_label);

int gex_write_simulation_truth(const char *outprefix,
                               GexMatrix *gex,
                               GexMatrix *L,
                               GexMatrix *Z,
                               char **cell_names,
                               char **gene_names,
                               int k,
                               double sigma2_obs,
                               double *sigma2_latent);

#endif
