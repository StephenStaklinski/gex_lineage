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

#endif
