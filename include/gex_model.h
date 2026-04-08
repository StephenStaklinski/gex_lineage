#ifndef GEX_MODEL_H
#define GEX_MODEL_H

#include "gex.h"
#include "pca.h"

#include <mvn.h>

typedef struct {
    Matrix *Sigma_inv;
    double logdet_sigma;
} GexLatentBrownianTreePrior;

typedef struct {
    Matrix *Xc;
    GexLatentBrownianTreePrior *tree_priors;
    int n_tree_priors;
    double *prior_log_terms;
    double *prior_weights;
} GexLatentBrownianWorkspace;

typedef struct {
    int n_cells;
    int n_genes;
    int k;
    Matrix *Z;              /* n_cells x k latent factors */
    Matrix *L;              /* k latent factors x n_genes */
    double *sigma2_latent;  /* length k */
    double sigma2_obs;
    double l2_strength;
    double objective;
    double observation_objective;
    double brownian_prior_objective;
    double l2_objective;
    MVN *latent_mvn;
} GexLatentBrownianModel;

GexLatentBrownianModel *gex_fit_latent_brownian_model(GexMatrix *gex,
                                                      Matrix **Sigmas,
                                                      int n_sigmas,
                                                      GexPCA *pca,
                                                      double l2_strength,
                                                      unsigned int seed,
                                                      const char *outprefix);

void gex_free_latent_brownian_model(GexLatentBrownianModel *model);

#endif
