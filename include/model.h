#ifndef GEXMODEL_H
#define GEXMODEL_H

#include "gexmatrix.h"
#include "pca.h"
#include "model.h"

#include "mvn.h"

#include <phast/matrix.h>
#include <phast/trees.h>

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
    Matrix *FL;             /* n_cells x n_genes reconstructed expression (F*L) */
    double *log_sigma2_latent;  /* length k */
    double log_sigma2_obs;
    double l1_strength;
    int final_absorbing_factor;
    double absorbing_l2_strength;
    double F_orthogonality_strength;
    double F_correlation_strength;
    double objective;
    double observation_objective;
    double brownian_prior_objective;
    double l1_objective;
    double l2_objective;
    double F_orthogonality_objective;
    double F_correlation_objective;
    double FL_frobenius_norm;
    MVN *latent_mvn;
} GexLatentBrownianModel;

GexLatentBrownianModel *gex_fit_latent_brownian_model(GexMatrix *gex,
                                                        TreeNode **trees,
                                                        int n_trees,
                                                        int k,
                                                        PCA *pca,
                                                        GexScaleInvarConstraint scale_invar_constraint,
                                                        double L_l1_strength,
                                                        int final_absorbing_factor,
                                                        double L_absorbing_l2_strength,
                                                        double F_orthogonality_strength,
                                                        double F_correlation_strength,
                                                        const char *outprefix,
                                                        int max_iter,
                                                        int verbose_log);

void gex_free_latent_brownian_model(GexLatentBrownianModel *model);

void post_hoc_sign_identifiability(Matrix *L, Matrix *F);

void reorder_factors_by_row_norm(Matrix *L, Matrix *F);

void reorder_factors_by_row_norm_prefix(Matrix *L, Matrix *F, int n_reorder);

void reorder_factors_by_sigma2_latent(Matrix *L, Matrix *F, double *log_sigma2_latent);

void reorder_factors_by_sigma2_latent_prefix(Matrix *L, Matrix *F, double *log_sigma2_latent, int n_reorder);

void varimax_rotate_model_factors(Matrix *L, Matrix *F, const char *outprefix, int max_iter, double tol);

void varimax_rotate_model_factors_prefix(Matrix *L, Matrix *F, int n_rotate, const char *outprefix, int max_iter, double tol);

void write_varimax_summary_tsv(const char *outprefix,
                                Matrix *L_before,
                                Matrix *F_before,
                                Matrix *FL_before,
                                Matrix *L_after,
                                Matrix *F_after,
                                Matrix *FL_after,
                                int n_iters);

double gaussian_observation_term(Matrix *FL,
                                        Matrix *F,
                                        Matrix *L,
                                        double log_sigma2_obs,
                                        Matrix *Xc,
                                        Matrix *grad_F,
                                        Matrix *grad_L,
                                        double *grad_log_sigma_obs,
                                        double *FL_frobenius_norm);

double gex_brownian_prior_from_trees(Matrix *F,
                                        double *log_sigma2_latent,
                                        TreeNode **trees,
                                        int n_trees,
                                        char **cell_names,
                                        Matrix *grad_F,
                                        double *grad_log_sigma_latent);

Matrix *gex_reconstruct_latent_tree_states(TreeNode *tree,
                                            Matrix *F,
                                            double *log_sigma2_latent,
                                            char **cell_names);

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
