#ifndef GEXLATENTFLOW_H
#define GEXLATENTFLOW_H

#include "gexmatrix.h"
#include "gexmodel.h"

#include <phast/trees.h>

int gex_write_latent_flow_outputs(const char *outprefix,
                                  TreeNode **trees,
                                  int n_trees,
                                  GexMatrix *gex,
                                  GexLatentBrownianModel *model);

#endif
