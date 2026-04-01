#ifndef GEX_MODEL_H
#define GEX_MODEL_H

#include "gex.h"
#include "pca.h"

#include <mvn.h>

typedef struct {
    int n_cells;
    int n_genes;
    int k;
    Matrix *Z;              /* n_cells x k latent cell coordinates */
    Matrix *L;              /* k x n_genes loadings */
    double *sigma2_latent;  /* length k */
    double sigma2_obs;
    double objective;
    MVN *latent_mvn;
} GexLatentBrownianModel;

GexLatentBrownianModel *gex_fit_latent_brownian_model(GexMatrix *gex,
                                                      Matrix *Sigma,
                                                      GexPCA *pca,
                                                      unsigned int seed);
int gex_write_latent_brownian_model(const char *filename,
                                    GexLatentBrownianModel *model);
void gex_free_latent_brownian_model(GexLatentBrownianModel *model);

#endif
