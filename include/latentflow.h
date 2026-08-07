#ifndef GEXLATENTFLOW_H
#define GEXLATENTFLOW_H

#include "gexmatrix.h"
#include "model.h"

#include <phast/trees.h>

int gex_write_latent_flow_outputs(const char *outprefix,
                                  TreeNode *tree,
                                  GexMatrix *gex,
                                  GexLatentBrownianModel *model);

#endif
