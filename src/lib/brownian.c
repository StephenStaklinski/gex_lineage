#include "brownian.h"

#include "gexmatrix.h"
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

typedef struct {
    const char *name;
    int index;
} TipNameIndex;

typedef struct {
    int *indices;
    int size;
} TipIndexSet;

static int compare_tip_names(const void *a, const void *b) {
    const TipNameIndex *x = a;
    const TipNameIndex *y = b;
    return strcmp(x->name, y->name);
}

static int lookup_tip_index(TipNameIndex *name_index, int n, const char *name) {
    int lo = 0;
    int hi = n - 1;

    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        int cmp = strcmp(name, name_index[mid].name);
        if (cmp == 0)
            return name_index[mid].index;
        if (cmp < 0)
            hi = mid - 1;
        else
            lo = mid + 1;
    }
    return -1;
}

/* A pair of tips first occurs in opposite child sets at its MRCA. Filling
   those cross-products during one postorder traversal avoids a separate
   ancestor search for every covariance entry. */
static void fill_covariance_postorder(TreeNode *node,
                                      double depth,
                                      TipNameIndex *name_index,
                                      int n,
                                      int *matched,
                                      Matrix *Sigma,
                                      TipIndexSet *out) {
    TipIndexSet left = {NULL, 0};
    TipIndexSet right = {NULL, 0};
    int i, j;

    out->indices = NULL;
    out->size = 0;
    if (node == NULL)
        return;

    if (node->lchild == NULL && node->rchild == NULL) {
        int index = lookup_tip_index(name_index, n, node->name);
        if (index >= 0) {
            if (matched[index])
                die("ERROR: duplicate tree tip '%s'.\n", node->name);
            matched[index] = 1;
            mat_set(Sigma, index, index, depth);
            out->indices = smalloc(sizeof(int));
            out->indices[0] = index;
            out->size = 1;
        }
        return;
    }

    fill_covariance_postorder(node->lchild,
                              depth + (node->lchild ?
                                  max(node->lchild->dparent, 1e-8) : 0.0),
                              name_index, n, matched, Sigma, &left);
    fill_covariance_postorder(node->rchild,
                              depth + (node->rchild ?
                                  max(node->rchild->dparent, 1e-8) : 0.0),
                              name_index, n, matched, Sigma, &right);

    for (i = 0; i < left.size; i++) {
        for (j = 0; j < right.size; j++) {
            mat_set(Sigma, left.indices[i], right.indices[j], depth);
            mat_set(Sigma, right.indices[j], left.indices[i], depth);
        }
    }

    out->size = left.size + right.size;
    if (out->size > 0) {
        out->indices = smalloc((size_t)out->size * sizeof(int));
        if (left.size > 0)
            memcpy(out->indices, left.indices, (size_t)left.size * sizeof(int));
        if (right.size > 0)
            memcpy(out->indices + left.size, right.indices,
                   (size_t)right.size * sizeof(int));
    }
    free(left.indices);
    free(right.indices);
}


/* Calculate the phylogenetic covariance matrix for an input tree.
Covariance is the distance from root to MRCA for each pair of tips. Non-root
branches are floored at 1e-8, matching the Brownian pruning calculations.
Tips are matched to the order of the input names.
Returns a pointer to the allocated covariance matrix or NULL on failure. */
Matrix *covariance_from_tree(TreeNode *tree, char **names, int n) {
    int i;
    Matrix *Sigma = NULL;   /* Phylogenetic covariance matrix */
    TipNameIndex *name_index = NULL;
    TipIndexSet root_tips = {NULL, 0};
    int *matched = NULL;

    if (tree == NULL || names == NULL || n <= 0)
        return NULL;

    /* Check that the leading origin to root node branch exists */
    if (tree->dparent < 0.0) {
        fprintf(stderr, "ERROR: origin to root has invalid or no branch length.\n");
        return NULL;
    }

    name_index = smalloc((size_t)n * sizeof(TipNameIndex));
    matched = scalloc((size_t)n, sizeof(int));
    for (i = 0; i < n; i++) {
        name_index[i].name = names[i];
        name_index[i].index = i;
    }
    qsort(name_index, (size_t)n, sizeof(TipNameIndex), compare_tip_names);
    for (i = 1; i < n; i++) {
        if (strcmp(name_index[i - 1].name, name_index[i].name) == 0)
            die("ERROR: duplicate requested tip name '%s'.\n", name_index[i].name);
    }

    Sigma = mat_new(n, n);
    mat_zero(Sigma);
    fill_covariance_postorder(tree, tree->dparent, name_index, n, matched,
                              Sigma, &root_tips);

    for (i = 0; i < n; i++) {
        if (!matched[i])
            die("ERROR: could not find tip '%s' in tree.\n", names[i]);
    }

    free(root_tips.indices);
    free(matched);
    free(name_index);
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

Matrix *brownian_simulate(Matrix **Sigmas, int n_sigmas, Vector *mu, int n_cols,
                          Vector *sigma2s) {

    if (Sigmas == NULL || n_sigmas <= 0 || n_cols <= 0 || sigma2s == NULL || 
        sigma2s->size != n_cols || mu == NULL || mu->size != Sigmas[0]->nrows)
        return NULL;

    int i, j;
    double tol = 1e-12;  /* Tolerance for checking if sigma2s are the same across cols for MVN reuse */
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
