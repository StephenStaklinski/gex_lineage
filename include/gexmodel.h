#ifndef GEXMODEL_H
#define GEXMODEL_H

#include "gexmatrix.h"
#include "gexpca.h"
#include "gexmodel.h"

#include "mvn.h"

#include <phast/matrix.h>

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
                                                      PCA *pca,
                                                      double l2_strength,
                                                      unsigned int seed,
                                                      const char *outprefix);

void gex_free_latent_brownian_model(GexLatentBrownianModel *model);

int gex_write_labeled_matrix_tsv(const char *filename,
                                 Matrix *X,
                                 char **row_names,
                                 int n_rows,
                                 char **col_names,
                                 int n_cols,
                                 const char *corner_label);

int gex_write_model(const char *outprefix,
                               GexMatrix *gex,
                               Matrix *L,
                               Matrix *Z,
                               char **cell_names,
                               char **gene_names,
                               int k,
                               double sigma2_obs,
                               double *sigma2_latent);

#endif
