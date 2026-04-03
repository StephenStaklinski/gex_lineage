#include "brownian.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


static int is_leaf(TreeNode *node) {
    if (node == NULL) return 0;
    return (node->lchild == NULL && node->rchild == NULL);
}

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

static void fill_tip_map(TreeNode *node,
                                  char **names,
                                  int n,
                                  TreeNode **tips) {
    int i;

    if (node == NULL)
        return;
    if (is_leaf(node)) {
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

/* Compute the maximum height from the origin to any tip in the tree. 
Does a depth-first search from the origin to find the maximum distance. */
static double max_origin_to_tip_height(TreeNode *node) {
    double left_h, right_h;
    double here = 0.0;

    if (node == NULL)
        return 0.0;

    /* Use all branch lengths including the branch from the origin to the root where the root node parent would be NULL */
    here = node->dparent;
    if (here < 0.0)
        here = 0.0;

    if (is_leaf(node))
        return here;

    left_h = max_origin_to_tip_height(node->lchild);
    right_h = max_origin_to_tip_height(node->rchild);
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

    if (is_leaf(node)) {
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
Covariance is the distance from root to MRCA for each pair of tips. 
Tips are matched to the order of the input names.
Returns a pointer to the allocated covariance matrix or NULL on failure. */
Matrix *covariance_from_tree(TreeNode *tree, char **names, int n) {
    int i, j;
    Matrix *Sigma = NULL;   /* Phylogenetic covariance matrix */
    TreeNode **tips = NULL;
    double *depth_by_id = NULL;

    if (tree == NULL || names == NULL || n <= 0) {
        fprintf(stderr, "ERROR: covariance_from_tree got invalid input\n");
        return NULL;
    }

    Sigma = mat_new(n, n);  /* Allocate the covariance matrix based on the input n tips */
    if (Sigma == NULL) {
        fprintf(stderr, "ERROR: failed to allocate Brownian covariance matrix\n");
        return NULL;
    }
    mat_zero(Sigma);

    /* Allocate an array to hold pointers to the tree tips in the order of the input names */
    tips = (TreeNode **)calloc(n, sizeof(TreeNode *));
    if (tips == NULL) {
        fprintf(stderr, "ERROR: out of memory allocating Brownian tip map\n");
        mat_free(Sigma);
        return NULL;
    }

    /* Fill the tip mapping from the input names to tips in the tree */
    fill_tip_map(tree, names, n, tips);
    for (i = 0; i < n; i++) {
        if (tips[i] == NULL) {
            fprintf(stderr,
                    "ERROR: could not find tip '%s' in tree while building phylogenetic covariance\n",
                    names[i]);
            free(tips);
            mat_free(Sigma);
            return NULL;
        }
    }

    /* Check that the leading origin to root node branch exists */
    if (tree->dparent < 0.0) {
        fprintf(stderr, "ERROR: origin node has invalid branch length or does not exist, so depths are incorrect\n");
        free(tips);
        mat_free(Sigma);
        return NULL;
    }

    /* Set the number of nodes in the tree */
    tr_set_nnodes(tree);
    if (tree->nnodes <= 0) {
        fprintf(stderr, "ERROR: tree has invalid node count\n");
        free(tips);
        mat_free(Sigma);
        return NULL;
    }

    /* Allocate an array to hold the depths from the origin to each node in the tree */
    depth_by_id = (double *)malloc(tree->nnodes * sizeof(double));
    if (depth_by_id == NULL) {
        fprintf(stderr, "ERROR: out of memory allocating Brownian node depths\n");
        free(tips);
        mat_free(Sigma);
        return NULL;
    }

    /* Fill the node depth array with the depth from the origin to each node in the tree. 
    This allows for fast lookup of MRCA depths when building the covariance matrix. */
    if (fill_node_depths(tree, depth_by_id, tree->nnodes, tree->dparent) != 0) {
        fprintf(stderr, "ERROR: failed to compute node depths for phylogenetic covariance\n");
        free(depth_by_id);
        free(tips);
        mat_free(Sigma);
        return NULL;
    }

    /* Fill the covariance matrix based on the depth to MRCA for each pair of tips.
    The covariance between two tips is the depth from the origin to their MRCA. */
    for (i = 0; i < n; i++) {
        for (j = i + 1; j < n; j++) {
            TreeNode *mrca = find_mrca(tips[i], tips[j]);

            if (mrca == NULL || mrca->id < 0 || mrca->id >= tree->nnodes) {
                fprintf(stderr,
                        "ERROR: failed to compute MRCA for '%s' and '%s'\n",
                        tips[i]->name, tips[j]->name);
                free(depth_by_id);
                free(tips);
                mat_free(Sigma);
                return NULL;
            }

            mat_set(Sigma, i, j, depth_by_id[mrca->id]);
        }
    }

    /* Make the covariance matrix symmetric */
    for (i = 0; i < n; i++) {
        for (j = i + 1; j < n; j++) {
            mat_set(Sigma, j, i, mat_get(Sigma, i, j));
        }
    }

    /* Fill the diagonal elements for each tip paired with itself */
    for (i = 0; i < n; i++) {
        mat_set(Sigma, i, i, depth_by_id[tips[i]->id]);
    }

    free(depth_by_id);
    free(tips);
    return Sigma;
}

/* Calculate the weight matrix from a phylogenetic covariance matrix.
This weight matrix approach is based on the PATH method by Schiffman et al. 2024 
Nature Genetics (PMID: 39317739) and is calculated as the element-wise inverse pairwise distance matrix.
The weight W_ij = 1/(d_ij + eps) where d_ij is the pairwise distance between tips i and j which can be
calculated from the covariance matrix as d_ij = Sigma_ii + Sigma_jj - 2*Sigma_ij and eps is a small 
constant to avoid division by zero. The weight matrix is then normalized to sum to 1.
Returns a pointer to the allocated weight matrix or NULL on failure. */
Matrix *weight_matrix_from_covariance(Matrix *Sigma) {
    int i, j;
    int n;
    double max_dist = 0.0;
    double eps;
    double total = 0.0;
    Matrix *W = NULL;

    if (Sigma == NULL || Sigma->nrows != Sigma->ncols || Sigma->nrows <= 0) {
        fprintf(stderr, "ERROR: weight_matrix_from_covariance got invalid input\n");
        return NULL;
    }

    /* Allocate the weight matrix with the same dimensions as the covariance matrix*/
    n = Sigma->nrows;  
    W = mat_new(n, n);
    if (W == NULL) {
        fprintf(stderr, "ERROR: failed to allocate weight matrix\n");
        return NULL;
    }

    /* Calculate the maximum pairwise distance from the covariance matrix to use for setting eps
    and simultaneously fill the weight matrix with initial pairwise distance values */
    for (i = 0; i < n; i++) {
        mat_set(W, i, i, 0.0);  /* Set diagonal elements (comparing each tip to itself) to zero pairwise distance */
        for (j = i + 1; j < n; j++) {
            double dij = mat_get(Sigma, i, i) + mat_get(Sigma, j, j) -
                         (2.0 * mat_get(Sigma, i, j));

            if (dij < 0.0) {
                fprintf(stderr, "ERROR: covariance implied negative distance\n");
                mat_free(W);
                return NULL;
            }

            /* Handle numerical precision issues */
            if (dij < 0.0 && fabs(dij) < 1e-12)
                dij = 0.0;
            
            /* Update the maximum distance for setting relative eps */
            if (dij > max_dist)
                max_dist = dij;

            /* Store the pairwise distance in the weight matrix temporarily for now, will convert to weights after setting eps */
            mat_set(W, i, j, dij);
            mat_set(W, j, i, dij);
        }
    }

    /* Set the epsilon value as a relative tolerance based on the maximum distance */
    eps = (max_dist > 0.0 ? 1e-8 * max_dist : 1e-8);

    /* Fill the weight matrix */
    for (i = 0; i < n; i++) {
        mat_set(W, i, i, 0.0);
        for (j = i + 1; j < n; j++) {
            double dij = mat_get(W, i, j);
            double wij = 1.0 / (dij + eps);
            mat_set(W, i, j, wij);
            mat_set(W, j, i, wij);
            total += 2.0 * wij;
        }
    }

    if (total <= 0.0) {
        fprintf(stderr, "ERROR: weight matrix normalization failed\n");
        mat_free(W);
        return NULL;
    }

    /* Normalize the weight matrix to sum to 1 */
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
    tree_height = max_origin_to_tip_height(tree);
    sigma2 = (tree_height > 0.0 ? 1.0 / tree_height : 1.0);

    tips = (TreeNode **)calloc(n, sizeof(TreeNode *));
    gex = (GexMatrix *)calloc(1, sizeof(GexMatrix));
    if (tips == NULL || gex == NULL) {
        free(tips);
        free(gex);
        return NULL;
    }

    fill_tip_map(tree, names, n, tips);
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

    printf("First few entries of covariance matrix:\n");
    for (i = 0; i < n && i < 10; i++) {
        printf("%s", names[i]);
        for (j = 0; j < n && j < 10; j++)
            printf("\t%g", mat_get(Sigma, i, j));
        printf("\n");
    }
    printf("\n");
}

/* Print a summary of the covariance-based weight matrix. */
void print_weight_matrix_summary(Matrix *W) {
    int i, j;

    if (W == NULL) {
        fprintf(stderr, "ERROR: cannot summarize NULL weight matrix\n");
        return;
    }

    printf("First few entries of covariance-based weight matrix:\n");
    for (i = 0; i < W->nrows && i < 10; i++) {
        for (j = 0; j < W->ncols && j < 10; j++)
            printf(" %g", mat_get(W, i, j));
        printf("\n");
    }
    printf("\n");
}
