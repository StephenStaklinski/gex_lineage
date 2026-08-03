#ifndef GEXLATENTFLOW_H
#define GEXLATENTFLOW_H

#include "gexmatrix.h"
#include "model.h"

#include <phast/trees.h>

int gex_write_latent_flow_from_factors(const char *outprefix,
                                       TreeNode *tree,
                                       Matrix *F,
                                       double *log_sigma2_latent,
                                       char **cell_names);

int gex_write_latent_flow_outputs(const char *outprefix,
                                  TreeNode **trees,
                                  int n_trees,
                                  GexMatrix *gex,
                                  GexLatentBrownianModel *model);

#endif
