#ifndef GEX_LINEAGE_LATENTFLOW_H
#define GEX_LINEAGE_LATENTFLOW_H

#include "gexmatrix.h"
#include "model.h"

#include <phast/trees.h>

int gex_write_latent_flow_outputs(const char *outprefix,
                                  TreeNode *tree,
                                  GexMatrix *gex,
                                  GexLatentBrownianModel *model);

#endif
