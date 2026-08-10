#ifndef GEX_LINEAGE_MODEL_H
#define GEX_LINEAGE_MODEL_H

#include "gexmatrix.h"
#include "pca.h"

#include <phast/matrix.h>
#include <phast/trees.h>

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
    double L_loading_overlap_strength;
    double objective;
    double observation_objective;
    double brownian_prior_objective;
    double l1_objective;
    double L_loading_overlap_objective;
    double FL_frobenius_norm;
} GexLatentBrownianModel;

GexLatentBrownianModel *gex_fit_latent_brownian_model(GexMatrix *gex,
                                                        TreeNode **trees,
                                                        int n_trees,
                                                        int k,
                                                        PCA *pca,
                                                        int constrain_L_scale,
                                                        double L_l1_strength,
                                                        double L_loading_overlap_strength,
                                                        int apply_post_hoc_identifiability,
                                                        const char *outprefix);

void gex_free_latent_brownian_model(GexLatentBrownianModel *model);

void post_hoc_sign_identifiability(Matrix *L, Matrix *F);

void reorder_factors_by_row_norm(Matrix *L, Matrix *F);

void reorder_factors_by_sigma2_latent(Matrix *L, Matrix *F, double *log_sigma2_latent);

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
                                                int n_cells,
                                                int k,
                                                int n_genes,
                                                double sigma2_obs,
                                                Vector *L_row_norms,
                                                Matrix *L_out,
                                                GexMatrix *gex_out);

#endif
