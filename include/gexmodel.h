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
    double *log_sigma2_latent;  /* length k */
    double log_sigma2_obs;
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
                                                      const char *outprefix);

void gex_free_latent_brownian_model(GexLatentBrownianModel *model);

double gaussian_observation_term(Matrix *Z,
                                        Matrix *L,
                                        double log_sigma2_obs,
                                        Matrix *Xc,
                                        Matrix *grad_Z,
                                        Matrix *grad_L,
                                        double *grad_log_sigma_obs);

double latent_brownian_prior_term(Matrix *Z,
                                        double *log_sigma2_latent,
                                        Matrix **Sigma_invs,
                                        double *logdet_sigmas,
                                        int n_sigmas,
                                        Matrix *grad_Z,
                                        double *grad_log_sigma_latent);

Matrix **downsample_sigmas(Matrix **Sigmas,
                                    int n_sigmas,
                                    int n_keep);

void write_summary_tsv(const char *path,
                        int n_cells,
                        int n_genes,
                        double brownian_negll,
                        double observation_negll,
                        double sigma2_obs,
                        double *sigma2_latent,
                        int k,
                        char **factor_names);

void write_model(const char *outprefix,
                    GexMatrix *gex,
                    Matrix *L,
                    Matrix *Z,
                    char **cell_names,
                    char **gene_names,
                    char ** factor_names,
                    int k,
                    double brownian_negll,
                    double observation_negll,
                    double sigma2_obs,
                    double *sigma2_latent);

void simulate_factorization_and_reconstruction(Matrix *Z,
                                                char **cell_names,
                                                int n_cells,
                                                int k,
                                                int n_genes,
                                                double sigma2_obs,
                                                Matrix *L_out,
                                                GexMatrix *gex_out);

#endif
