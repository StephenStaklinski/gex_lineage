#ifndef GEXMODEL_H
#define GEXMODEL_H

#include "gexmatrix.h"
#include "gexpca.h"
#include "gexmodel.h"

#include "mvn.h"

#include <phast/matrix.h>

typedef enum {
    GEX_SCALE_INVAR_SIGMA2S = 0,
    GEX_SCALE_INVAR_LROWS = 1
} GexScaleInvarConstraint;

typedef struct {
    int n_cells;
    int n_genes;
    int k;
    Matrix *F;              /* n_cells x k latent factors */
    Matrix *L;              /* k latent factors x n_genes */
    double *log_sigma2_latent;  /* length k */
    double log_sigma2_obs;
    double l1_strength;
    double objective;
    double observation_objective;
    double brownian_prior_objective;
    double l1_objective;
    MVN *latent_mvn;
} GexLatentBrownianModel;

GexLatentBrownianModel *gex_fit_latent_brownian_model(GexMatrix *gex,
                                                        Matrix **Sigmas,
                                                        int n_sigmas,
                                                        int k,
                                                        PCA *pca,
                                                        GexScaleInvarConstraint scale_invar_constraint,
                                                        double L_l1_strength,
                                                        const char *outprefix);

void gex_free_latent_brownian_model(GexLatentBrownianModel *model);

void post_hoc_sign_identifiability(Matrix *L, Matrix *F);

void reorder_factors_by_row_norm(Matrix *L, Matrix *F);

void reorder_factors_by_sigma2_latent(Matrix *L, Matrix *F, double *log_sigma2_latent);

double gaussian_observation_term(Matrix *F,
                                        Matrix *L,
                                        double log_sigma2_obs,
                                        Matrix *Xc,
                                        Matrix *grad_F,
                                        Matrix *grad_L,
                                        double *grad_log_sigma_obs);

double latent_brownian_prior_term(Matrix *F,
                                        double *log_sigma2_latent,
                                        Matrix **Sigma_invs,
                                        double *logdet_sigmas,
                                        int n_sigmas,
                                        Matrix *grad_F,
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
                        double *L_row_norms,
                        int k,
                        char **factor_names);

void write_model(const char *outprefix,
                    GexMatrix *gex,
                    Matrix *L,
                    Matrix *F,
                    char **cell_names,
                    char **gene_names,
                    char ** factor_names,
                    int k,
                    double brownian_negll,
                    double observation_negll,
                    double sigma2_obs,
                    double *sigma2_latent);

void simulate_factorization_and_reconstruction(Matrix *F,
                                                char **cell_names,
                                                int n_cells,
                                                int k,
                                                int n_genes,
                                                double sigma2_obs,
                                                Vector *L_row_norms,
                                                Matrix *L_out,
                                                GexMatrix *gex_out);

#endif
