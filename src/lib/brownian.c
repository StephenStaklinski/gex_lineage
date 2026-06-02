#include "brownian.h"

#include "matrix.h"
#include "misc.h"

#include "mvn.h"

#include <phast/trees.h>
#include <phast/matrix.h>
#include <phast/misc.h>
#include <phast/vector.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>




/* Find the most recent common ancestor (MRCA) of two nodes in a tree.
Returns a pointer to the MRCA node or NULL if no common ancestor is found. */
static TreeNode *find_mrca(TreeNode *a, TreeNode *b) {
    TreeNode *anc;
    TreeNode *cur;

    if (a == NULL || b == NULL)
        return NULL;

    for (anc = a; anc != NULL; anc = anc->parent) {
        for (cur = b; cur != NULL; cur = cur->parent) {
            if (anc == cur)
                return anc;
        }
    }

    return NULL;
}

/* Fill the depth array with the depth from the origin to each node in the tree. 
Sets the depth for each node in the array. Returns 0 on success, -1 on failure. */
static int fill_node_depths(TreeNode *node, double *depth_by_id, int nnodes, double depth) {
    if (node == NULL)
        return 0;

    if (node->id < 0 || node->id >= nnodes)
        return -1;

    depth_by_id[node->id] = depth;

    if (fill_node_depths(node->lchild,
                                  depth_by_id,
                                  nnodes,
                                  depth + (node->lchild ? node->lchild->dparent : 0.0)) != 0)
        return -1;
    if (fill_node_depths(node->rchild,
                                  depth_by_id,
                                  nnodes,
                                  depth + (node->rchild ? node->rchild->dparent : 0.0)) != 0)
        return -1;

    return 0;
}

static void fill_tip_map(TreeNode *node,
                            char **names,
                            int n,
                            TreeNode **tips) {
    int i;

    if (node == NULL)
        return;

    /* Check if the node is a leaf */
    if (node->lchild == NULL && node->rchild == NULL) {
        for (i = 0; i < n; i++) {
            if (tips[i] == NULL && node->name != NULL &&
                strcmp(node->name, names[i]) == 0) {
                tips[i] = node;
                break;
            }
        }
        return;
    }

    fill_tip_map(node->lchild, names, n, tips);
    fill_tip_map(node->rchild, names, n, tips);
}


/* Calculate the phylogenetic covariance matrix for an input tree.
Covariance is the distance from root to MRCA for each pair of tips. 
Tips are matched to the order of the input names.
Returns a pointer to the allocated covariance matrix or NULL on failure. */
Matrix *covariance_from_tree(TreeNode *tree, char **names, int n) {
    int i, j;
    Matrix *Sigma = NULL;   /* Phylogenetic covariance matrix */
    TreeNode **tips = NULL;
    double *depth_by_id = NULL;

    /* Check that the leading origin to root node branch exists */
    if (tree->dparent < 0.0) {
        fprintf(stderr, "ERROR: origin to root has invalid or no branch length.\n");
        return NULL;
    }

    /* Fill the tip mapping from the input names to tips in the tree */
    tips = scalloc(n, sizeof(TreeNode *));
    fill_tip_map(tree, names, n, tips);
    for (i = 0; i < n; i++) {
        if (tips[i] == NULL) {
            fprintf(stderr,
                    "ERROR: could not find tip '%s' in tree.\n",
                    names[i]);
            return NULL;
        }
    }

    /* Fill the node depth array with the depth from the origin to each node in the tree. 
    This allows for fast lookup of MRCA depths when building the covariance matrix. */
    tr_set_nnodes(tree);
    depth_by_id = smalloc(tree->nnodes * sizeof(double));
    if (fill_node_depths(tree, depth_by_id, tree->nnodes, tree->dparent) != 0) {
        fprintf(stderr, "ERROR: failed to compute node depths.\n");
        return NULL;
    }

    /* Fill the covariance matrix based on the depth to MRCA for each pair of tips.
    The covariance between two tips is the depth from the origin to their MRCA. */
    Sigma = mat_new(n, n);
    mat_zero(Sigma);
    double depth;
    for (i = 0; i < n; i++) {
        mat_set(Sigma, i, i, depth_by_id[tips[i]->id]); /* Diagonal compares with self */
        for (j = i + 1; j < n; j++) {
            TreeNode *mrca = find_mrca(tips[i], tips[j]);

            if (mrca == NULL || mrca->id < 0 || mrca->id >= tree->nnodes) {
                fprintf(stderr, "ERROR: failed MRCA for '%s' and '%s'\n",
                        tips[i]->name, tips[j]->name);
                return NULL;
            }

            depth = depth_by_id[mrca->id];
            mat_set(Sigma, i, j, depth);
            mat_set(Sigma, j, i, depth); /* Make the covariance matrix symmetric */
        }
    }

    free(depth_by_id);
    free(tips);
    return Sigma;
}

Matrix *gex_average_tree_covariance(TreeNode **trees,
                                    int n_trees,
                                    char **names,
                                    int n_names) {
    int i;
    Matrix *avg = NULL;

    if (trees == NULL || n_trees <= 0 || names == NULL || n_names <= 0)
        return NULL;

    avg = mat_new(n_names, n_names);
    if (avg == NULL)
        return NULL;
    mat_zero(avg);

    for (i = 0; i < n_trees; i++) {
        int r, c;
        Matrix *Sigma = NULL;

        if (trees[i] == NULL) {
            mat_free(avg);
            return NULL;
        }

        Sigma = covariance_from_tree(trees[i], names, n_names);
        if (Sigma == NULL) {
            mat_free(avg);
            return NULL;
        }

        for (r = 0; r < n_names; r++) {
            for (c = 0; c < n_names; c++)
                mat_set(avg, r, c, mat_get(avg, r, c) + mat_get(Sigma, r, c));
        }
        mat_free(Sigma);
    }

    mat_scale(avg, 1.0 / (double)n_trees);
    return avg;
}

/* Print a summary of the covariance matrix. */
void print_covariance_summary(Matrix *Sigma, char **names, int n) {
    int i, j;

    if (Sigma == NULL || names == NULL || n <= 0) {
        fprintf(stderr, "ERROR: cannot summarize NULL covariance matrix\n");
        return;
    }
    if (Sigma->nrows != n || Sigma->ncols != n) {
        fprintf(stderr, "ERROR: covariance summary got mismatched dimensions\n");
        return;
    }

    printf("\n");
    printf("First few entries of covariance matrix:\n");
    for (i = 0; i < n && i < 10; i++) {
        printf("%s", names[i]);
        for (j = 0; j < n && j < 10; j++)
            printf("\t%g", mat_get(Sigma, i, j));
        printf("\n");
    }
    printf("\n");
}

Matrix *brownian_simulate(Matrix **Sigmas, int n_sigmas, Vector *mu, int n_cols,
                          Vector *sigma2s) {

    if (Sigmas == NULL || n_sigmas <= 0 || n_cols <= 0 || sigma2s == NULL || 
        sigma2s->size != n_cols || mu == NULL || mu->size != Sigmas[0]->nrows)
        return NULL;

    int i, j;
    int tol = 1e-12;  /* Tolerance for checking if sigma2s are the same across cols for MVN reuse */
    int n_rows = Sigmas[0]->nrows; /* Assume all Sigmas have the same number of rows */
    Matrix *res = mat_new(n_rows, n_cols);
    mat_zero(res);
    Matrix *cur_sim = mat_new(n_rows, n_cols);
    Vector *sim_vec = vec_new(n_rows);

    for (i = 0; i < n_sigmas; i++) {

        /* Draw the simulation results per desired col (gene or latent factor) from the MVN */
        mat_zero(cur_sim);
        int reuse_mvn = 0;  /* If sigma2 is the same across a col, then reuse the MVN object */
        MVN *mvn_obj = NULL;
        for (j = 0; j < n_cols; j++) {

            if (!reuse_mvn) {
                /* Scale the input Sigma by the Brownian variance parameter sigma2 to get the covariance for this simulation */
                Matrix *scaled_Sigma = mat_create_copy(Sigmas[i]);  /* Copy to scale and since it will be freed automatically by mvn_free */
                mat_scale(scaled_Sigma, vec_get(sigma2s, j));

                /* Create MVN object from the scaled covariance matrix */
                Vector *mu_copy = vec_create_copy(mu);  /* Copy since mvn_free will free the supplied mu directly */
                mvn_obj = mvn_new(n_rows, mu_copy, scaled_Sigma);
                mvn_preprocess(mvn_obj, FALSE);
            }

            mvn_sample(mvn_obj, sim_vec);
            mat_set_col(cur_sim, j, sim_vec);

            if ((j != (n_cols - 1) && (fabs(vec_get(sigma2s, j) - vec_get(sigma2s, j + 1)) <= tol))) {
                reuse_mvn = 1;
            } else {
                reuse_mvn = 0;
            }

            if (!reuse_mvn) {
                /* Free memory */
                mvn_free(mvn_obj);
            }
        }

        /* Accumulate results over each Sigma */
        mat_add_mat(res, cur_sim);
    }

    /* Finish the expectation over n_sigmas */
    double scale_factor = 1.0 / (double)n_sigmas;
    mat_scale(res, scale_factor);

    /* Free memory */
    vec_free(sim_vec);
    mat_free(cur_sim);

    return res;
}

