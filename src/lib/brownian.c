#include "brownian.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* -------------------- helpers -------------------- */

static int brownian_is_leaf(TreeNode *node) {
    if (node == NULL) return 0;
    return (node->lchild == NULL && node->rchild == NULL);
}

static TreeNode *brownian_find_tip_by_name(TreeNode *node, const char *name) {
    TreeNode *left_hit;
    TreeNode *right_hit;

    if (node == NULL || name == NULL)
        return NULL;
    if (brownian_is_leaf(node) && node->name != NULL &&
        strcmp(node->name, name) == 0)
        return node;

    left_hit = brownian_find_tip_by_name(node->lchild, name);
    if (left_hit != NULL)
        return left_hit;

    right_hit = brownian_find_tip_by_name(node->rchild, name);
    if (right_hit != NULL)
        return right_hit;

    return NULL;
}

static double brownian_depth_to_root(TreeNode *node) {
    double depth = 0.0;

    while (node != NULL && node->parent != NULL) {
        depth += node->dparent;
        node = node->parent;
    }

    return depth;
}

static TreeNode *brownian_mrca(TreeNode *a, TreeNode *b) {
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

static char *brownian_strdup(const char *s) {
    size_t n;
    char *out;

    if (s == NULL)
        return NULL;
    n = strlen(s);
    out = (char *)malloc(n + 1);
    if (out == NULL)
        return NULL;
    memcpy(out, s, n + 1);
    return out;
}

static unsigned int brownian_rand_u32(unsigned int *state) {
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static double brownian_uniform_open(unsigned int *state) {
    return ((double)brownian_rand_u32(state) + 1.0) / 4294967297.0;
}

static double brownian_rand_normal(unsigned int *state) {
    double u1 = brownian_uniform_open(state);
    double u2 = brownian_uniform_open(state);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

static void brownian_fill_tip_map(TreeNode *node,
                                  char **names,
                                  int n,
                                  TreeNode **tips) {
    int i;

    if (node == NULL)
        return;
    if (brownian_is_leaf(node)) {
        for (i = 0; i < n; i++) {
            if (tips[i] == NULL && node->name != NULL &&
                strcmp(node->name, names[i]) == 0) {
                tips[i] = node;
                break;
            }
        }
        return;
    }

    brownian_fill_tip_map(node->lchild, names, n, tips);
    brownian_fill_tip_map(node->rchild, names, n, tips);
}

static double brownian_max_root_to_tip_height(TreeNode *node) {
    double left_h, right_h;
    double here = 0.0;

    if (node == NULL)
        return 0.0;

    if (node->parent != NULL) {
        here = node->dparent;
        if (here < 0.0)
            here = 0.0;
    }

    if (brownian_is_leaf(node))
        return here;

    left_h = brownian_max_root_to_tip_height(node->lchild);
    right_h = brownian_max_root_to_tip_height(node->rchild);
    return here + (left_h > right_h ? left_h : right_h);
}

static void brownian_simulate_gene_recursive(TreeNode *node,
                                             double curval,
                                             double sigma2,
                                             TreeNode **tips,
                                             int n,
                                             double *out,
                                             unsigned int *state) {
    int i;

    if (node == NULL)
        return;

    if (node->parent != NULL) {
        double bl = node->dparent;
        if (bl < 0.0)
            bl = 0.0;
        curval += sqrt(sigma2 * bl) * brownian_rand_normal(state);
    }

    if (brownian_is_leaf(node)) {
        for (i = 0; i < n; i++) {
            if (tips[i] == node) {
                out[i] = curval;
                return;
            }
        }
        return;
    }

    brownian_simulate_gene_recursive(node->lchild, curval, sigma2, tips, n, out, state);
    brownian_simulate_gene_recursive(node->rchild, curval, sigma2, tips, n, out, state);
}

/* Calculate the phylogenetic covariance matrix for an input tree.
Returns a pointer to the allocated covariance matrix or NULL on failure. */
Matrix *covariance_from_tree(TreeNode *tree, char **names, int n) {
    int i, j;
    Matrix *Sigma = NULL;
    TreeNode **tips = NULL;

    if (tree == NULL || names == NULL || n <= 0) {
        fprintf(stderr, "ERROR: covariance_from_tree got invalid input\n");
        return NULL;
    }

    Sigma = mat_new(n, n);
    if (Sigma == NULL) {
        fprintf(stderr, "ERROR: failed to allocate Brownian covariance matrix\n");
        return NULL;
    }

    tips = (TreeNode **)malloc(n * sizeof(TreeNode *));
    if (tips == NULL) {
        fprintf(stderr, "ERROR: out of memory allocating Brownian tip map\n");
        mat_free(Sigma);
        return NULL;
    }

    for (i = 0; i < n; i++) {
        tips[i] = brownian_find_tip_by_name(tree, names[i]);
        if (tips[i] == NULL) {
            fprintf(stderr,
                    "ERROR: could not find tip '%s' in tree while building Brownian covariance\n",
                    names[i]);
            free(tips);
            mat_free(Sigma);
            return NULL;
        }
    }

    /* Fill covariance matrix */
    for (i = 0; i < n; i++) {
        for (j = i; j < n; j++) {
            TreeNode *mrca = brownian_mrca(tips[i], tips[j]);
            double cov;

            if (mrca == NULL) {
                fprintf(stderr,
                        "ERROR: failed to compute MRCA for '%s' and '%s'\n",
                        tips[i]->name, tips[j]->name);
                free(tips);
                mat_free(Sigma);
                return NULL;
            }

            cov = brownian_depth_to_root(mrca);
            mat_set(Sigma, i, j, cov);
            mat_set(Sigma, j, i, cov);
        }
    }

    free(tips);
    return Sigma;
}

Matrix *brownian_weight_matrix_from_covariance(Matrix *Sigma) {
    int i, j;
    int n;
    double max_dist = 0.0;
    double eps;
    double total = 0.0;
    Matrix *W = NULL;

    if (Sigma == NULL || Sigma->nrows != Sigma->ncols || Sigma->nrows <= 0) {
        fprintf(stderr, "ERROR: brownian_weight_matrix_from_covariance got invalid input\n");
        return NULL;
    }

    n = Sigma->nrows;
    W = mat_new(n, n);
    if (W == NULL) {
        fprintf(stderr, "ERROR: failed to allocate Brownian weight matrix\n");
        return NULL;
    }

    for (i = 0; i < n; i++) {
        for (j = i + 1; j < n; j++) {
            double dij = mat_get(Sigma, i, i) + mat_get(Sigma, j, j) -
                         2.0 * mat_get(Sigma, i, j);
            if (dij < 0.0 && fabs(dij) < 1e-12)
                dij = 0.0;
            if (dij > max_dist)
                max_dist = dij;
        }
    }

    eps = (max_dist > 0.0 ? 1e-8 * max_dist : 1e-8);

    for (i = 0; i < n; i++) {
        mat_set(W, i, i, 0.0);
        for (j = i + 1; j < n; j++) {
            double dij = mat_get(Sigma, i, i) + mat_get(Sigma, j, j) -
                         2.0 * mat_get(Sigma, i, j);
            double wij;

            if (dij < 0.0 && fabs(dij) < 1e-12)
                dij = 0.0;
            if (dij < 0.0) {
                fprintf(stderr, "ERROR: Brownian covariance implied negative distance\n");
                mat_free(W);
                return NULL;
            }

            wij = 1.0 / (dij + eps);
            mat_set(W, i, j, wij);
            mat_set(W, j, i, wij);
            total += 2.0 * wij;
        }
    }

    if (total <= 0.0) {
        fprintf(stderr, "ERROR: Brownian weight matrix normalization failed\n");
        mat_free(W);
        return NULL;
    }

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++)
            mat_set(W, i, j, mat_get(W, i, j) / total);
    }

    return W;
}

GexMatrix *brownian_simulate_expression(TreeNode *tree,
                                        char **names,
                                        int n,
                                        int n_tree_genes,
                                        int n_null_genes,
                                        unsigned int seed) {
    int i, j;
    int ngenes;
    double tree_height;
    double sigma2;
    unsigned int rng_state;
    TreeNode **tips = NULL;
    GexMatrix *gex = NULL;

    if (tree == NULL || names == NULL || n <= 0 ||
        n_tree_genes < 0 || n_null_genes < 0) {
        fprintf(stderr, "ERROR: brownian_simulate_expression got invalid input\n");
        return NULL;
    }

    ngenes = n_tree_genes + n_null_genes;
    if (ngenes <= 0) {
        fprintf(stderr, "ERROR: brownian_simulate_expression needs at least one gene\n");
        return NULL;
    }

    /* Set sigma2 based on input tree height, as 1/tree_height, to get reasonable Moran's I values */
    tree_height = brownian_max_root_to_tip_height(tree);
    sigma2 = (tree_height > 0.0 ? 1.0 / tree_height : 1.0);

    tips = (TreeNode **)calloc(n, sizeof(TreeNode *));
    gex = (GexMatrix *)calloc(1, sizeof(GexMatrix));
    if (tips == NULL || gex == NULL) {
        free(tips);
        free(gex);
        return NULL;
    }

    brownian_fill_tip_map(tree, names, n, tips);
    for (i = 0; i < n; i++) {
        if (tips[i] == NULL) {
            fprintf(stderr, "ERROR: could not match simulated tip '%s'\n", names[i]);
            free(tips);
            gex_free_matrix_data(gex);
            return NULL;
        }
    }

    gex->n_cells = n;
    gex->n_genes = ngenes;
    gex->X = mat_new(n, ngenes);
    gex->cell_names = (char **)calloc(n, sizeof(char *));
    gex->gene_names = (char **)calloc(ngenes, sizeof(char *));
    if (gex->X == NULL || gex->cell_names == NULL || gex->gene_names == NULL) {
        free(tips);
        gex_free_matrix_data(gex);
        return NULL;
    }

    for (i = 0; i < n; i++) {
        gex->cell_names[i] = brownian_strdup(names[i]);
        if (gex->cell_names[i] == NULL) {
            free(tips);
            gex_free_matrix_data(gex);
            return NULL;
        }
    }

    for (j = 0; j < n_tree_genes; j++) {
        char gene_name[64];
        snprintf(gene_name, sizeof(gene_name), "sim_pos_%02d", j + 1);
        gex->gene_names[j] = brownian_strdup(gene_name);
        if (gex->gene_names[j] == NULL) {
            free(tips);
            gex_free_matrix_data(gex);
            return NULL;
        }
    }
    for (j = 0; j < n_null_genes; j++) {
        char gene_name[64];
        snprintf(gene_name, sizeof(gene_name), "sim_neg_%02d", j + 1);
        gex->gene_names[n_tree_genes + j] = brownian_strdup(gene_name);
        if (gex->gene_names[n_tree_genes + j] == NULL) {
            free(tips);
            gex_free_matrix_data(gex);
            return NULL;
        }
    }

    rng_state = (seed == 0u ? 1u : seed);
    for (j = 0; j < n_tree_genes; j++) {
        double root = brownian_rand_normal(&rng_state);
        double *vals = (double *)calloc(n, sizeof(double));
        if (vals == NULL) {
            free(tips);
            gex_free_matrix_data(gex);
            return NULL;
        }

        brownian_simulate_gene_recursive(tree, root, sigma2, tips, n, vals, &rng_state);
        for (i = 0; i < n; i++)
            mat_set(gex->X, i, j, vals[i]);
        free(vals);
    }

    for (j = 0; j < n_null_genes; j++) {
        int col = n_tree_genes + j;
        for (i = 0; i < n; i++)
            mat_set(gex->X, i, col, brownian_rand_normal(&rng_state));
    }

    free(tips);
    return gex;
}

int brownian_run_simulation_check(TreeNode *tree,
                                  char **names,
                                  int n,
                                  int n_tree_genes,
                                  int n_null_genes,
                                  GexFilterMode mode,
                                  GexLRTNullMode lrt_null_mode,
                                  int n_perm,
                                  double max_q,
                                  double min_i,
                                  Matrix *Sigma,
                                  Matrix *W,
                                  unsigned int seed) {
    int j;
    int tp = 0, fn = 0, fp = 0, tn = 0;
    GexMatrix *sim = NULL;
    GexMoransResult *morans = NULL;
    GexLRTResult *lrt = NULL;

    sim = brownian_simulate_expression(tree, names, n,
                                       n_tree_genes, n_null_genes, seed);
    if (sim == NULL)
        return 0;

    if (mode == GEX_FILTER_MORAN || mode == GEX_FILTER_BOTH)
        morans = (W == NULL ? NULL : gex_compute_morans_i(sim, W, n_perm, seed + 17u));
    if (mode == GEX_FILTER_LRT || mode == GEX_FILTER_BOTH)
        lrt = (Sigma == NULL ? NULL : gex_compute_brownian_lrt(sim, Sigma,
                                                               lrt_null_mode,
                                                               n_perm,
                                                               seed + 31u));
    if (Sigma == NULL || W == NULL ||
        ((mode == GEX_FILTER_MORAN || mode == GEX_FILTER_BOTH) && morans == NULL) ||
        ((mode == GEX_FILTER_LRT || mode == GEX_FILTER_BOTH) && lrt == NULL)) {
        gex_free_matrix_data(sim);
        if (morans != NULL) gex_free_morans_result(morans);
        if (lrt != NULL) gex_free_lrt_result(lrt);
        return 0;
    }

    for (j = 0; j < n_tree_genes; j++) {
        int keep = 0;
        if (mode == GEX_FILTER_MORAN || mode == GEX_FILTER_BOTH)
            keep = keep || (morans->qvals[j] <= max_q && morans->morans_i[j] > min_i);
        if (mode == GEX_FILTER_LRT || mode == GEX_FILTER_BOTH)
            keep = keep || (lrt->qvals[j] <= max_q && lrt->lrt_stat[j] > 0.0);
        if (keep) tp++;
        else fn++;
    }
    for (j = 0; j < n_null_genes; j++) {
        int idx = n_tree_genes + j;
        int keep = 0;
        if (mode == GEX_FILTER_MORAN || mode == GEX_FILTER_BOTH)
            keep = keep || (morans->qvals[idx] <= max_q && morans->morans_i[idx] > min_i);
        if (mode == GEX_FILTER_LRT || mode == GEX_FILTER_BOTH)
            keep = keep || (lrt->qvals[idx] <= max_q && lrt->lrt_stat[idx] > 0.0);
        if (keep) fp++;
        else tn++;
    }

    printf("Brownian simulation check:\n");
    printf("  positives simulated: %d, detected: %d, missed: %d\n",
           n_tree_genes, tp, fn);
    printf("  negatives simulated: %d, rejected: %d, false positives: %d\n",
           n_null_genes, tn, fp);
    printf("\n");

    gex_free_matrix_data(sim);
    gex_free_morans_result(morans);
    gex_free_lrt_result(lrt);

    return (fn == 0 && fp == 0);
}

void brownian_print_covariance_summary(Matrix *Sigma, char **names, int n) {
    int i, j;

    if (Sigma == NULL || names == NULL || n <= 0) {
        fprintf(stderr, "ERROR: cannot summarize NULL Brownian covariance matrix\n");
        return;
    }
    if (Sigma->nrows != n || Sigma->ncols != n) {
        fprintf(stderr, "ERROR: Brownian covariance summary got mismatched dimensions\n");
        return;
    }

    printf("First few entries of Brownian covariance matrix:\n");
    for (i = 0; i < n && i < 10; i++) {
        printf("%s", names[i]);
        for (j = 0; j < n && j < 10; j++)
            printf("\t%g", mat_get(Sigma, i, j));
        printf("\n");
    }
    printf("\n");
}

void brownian_print_weight_summary(Matrix *W) {
    int i, j;

    if (W == NULL) {
        fprintf(stderr, "ERROR: cannot summarize NULL Brownian weight matrix\n");
        return;
    }

    printf("First few entries of Brownian covariance-based weight matrix:\n");
    for (i = 0; i < W->nrows && i < 10; i++) {
        for (j = 0; j < W->ncols && j < 10; j++)
            printf(" %g", mat_get(W, i, j));
        printf("\n");
    }
    printf("\n");
}
