#include "brownian.h"

#include "gex.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <phast/misc.h>

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

/* Copy cell names to gex matrix. */
void copy_cell_names(char **src, char **dst, int n) {
    int i;
    for (i = 0; i < n; i++) {
        dst[i] = strdup(src[i]);
    }
}

/* Fill a preallocated array of gene names of length n_genes. 
Returns 0 on success, -1 on failure. */
int generate_gene_names(char **names, int n_genes, char *gene_name_prefix) {
    int j;

    if (names == NULL || n_genes <= 0)
        return -1;

    for (j = 0; j < n_genes; j++) {
        char buf[64];
        if (gene_name_prefix == NULL)
            gene_name_prefix = "gene";
        snprintf(buf, sizeof(buf), "%s_%04d", gene_name_prefix, j + 1);
        names[j] = strdup(buf);
        if (names[j] == NULL) {
            free_string_array(names, j);
            return -1;
        }
    }

    return 0;
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
    mat_zero(Sigma);

    /* Allocate an array to hold pointers to the tree tips in the order of the input names */
    tips = scalloc(n, sizeof(TreeNode *));

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
    depth_by_id = smalloc(tree->nnodes * sizeof(double));

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

/* Simulate Brownian motion expression draws directly from a covariance matrix. */
GexMatrix *brownian_simulate_expression_from_covariance(Matrix *Sigma,
                                                        char **names,
                                                        int n,
                                                        int n_genes,
                                                        double *sigma2,
                                                        int n_sigma2,
                                                        unsigned int seed) {
    int i, j;
    int ngenes;
    GexMatrix *gex = NULL;
    Matrix *Sigma_reg = NULL;
    Matrix *chol = NULL;
    double *std_normals = NULL;
    unsigned int rng_state = (seed == 0u ? 1u : seed);
    double max_diag = 0.0;
    double jitter;

    if (Sigma == NULL || names == NULL || n <= 0 ||
        Sigma->nrows != n || Sigma->ncols != n ||
        n_genes < 0 || sigma2 == NULL) {
        fprintf(stderr, "ERROR: brownian_simulate_expression_from_covariance got invalid input\n");
        return NULL;
    }

    ngenes = n_genes;
    if (ngenes <= 0) {
        fprintf(stderr, "ERROR: brownian_simulate_expression_from_covariance needs at least one gene\n");
        return NULL;
    }

    /* Check that either 1 or all sigma2 values are provided */
    if (n_sigma2 != 1 && n_sigma2 != n_genes) {
        fprintf(stderr, "ERROR: brownian_simulate_expression_from_covariance got invalid number of sigma2 values\n");
        return NULL;
    }

    /* Check the input sigma2 values are valid */
    for (j = 0; j < n_sigma2; j++) {
        if (sigma2[j] <= 0.0) {
            fprintf(stderr, "ERROR: brownian_simulate_expression_from_covariance got invalid sigma2\n");
            return NULL;
        }
    }

    gex = scalloc(1, sizeof(GexMatrix));
    Sigma_reg = mat_create_copy(Sigma);
    chol = mat_new(n, n);

    for (i = 0; i < n; i++) {
        double diag = mat_get(Sigma, i, i);
        if (diag > max_diag)
            max_diag = diag;
    }
    jitter = (max_diag > 0.0 ? 1e-8 * max_diag : 1e-8);
    for (i = 0; i < n; i++)
        mat_set(Sigma_reg, i, i, mat_get(Sigma_reg, i, i) + jitter);
    if (mat_cholesky(chol, Sigma_reg) != 0) {
        fprintf(stderr, "ERROR: covariance-based simulation failed because the covariance was not positive definite enough\n");
        gex_free_matrix_data(gex);
        mat_free(Sigma_reg);
        mat_free(chol);
        return NULL;
    }

    gex->X = mat_new(n, ngenes);
    gex->cell_names = scalloc(n, sizeof(char *));
    copy_cell_names(names, gex->cell_names, n);
    gex->gene_names = scalloc(ngenes, sizeof(char *));
    generate_gene_names(gex->gene_names, ngenes, NULL);

    std_normals = scalloc(n, sizeof(double));

    for (j = 0; j < ngenes; j++) {
        double sigma2_j = (n_sigma2 == 1 ? sigma2[0] : sigma2[j]);
        double sigma_scale = sqrt(sigma2_j);
        for (i = 0; i < n; i++)
            std_normals[i] = rand_normal(&rng_state);
        for (i = 0; i < n; i++) {
            double sum = 0.0;
            int m;
            for (m = 0; m <= i; m++)
                sum += mat_get(chol, i, m) * std_normals[m];
            mat_set(gex->X, i, j, sigma_scale * sum);
        }
    }

    free(std_normals);
    mat_free(Sigma_reg);
    mat_free(chol);
    return gex;
}

GexMatrix *simulate_standard_normal_expression(char **names,
                                             int n,
                                             int n_genes,
                                             unsigned int seed) {
    int i, j;
    GexMatrix *gex = NULL;
    unsigned int rng_state = (seed == 0u ? 1u : seed);

    if (names == NULL || n <= 0 || n_genes <= 0) {
        fprintf(stderr, "ERROR: simulate_standard_normal_expression got invalid input\n");
        return NULL;
    }

    gex = scalloc(1, sizeof(GexMatrix));
    gex->X = mat_new(n, n_genes);
    gex->cell_names= scalloc(n, sizeof(char *));
    copy_cell_names(names, gex->cell_names, n);
    gex->gene_names = scalloc(n_genes, sizeof(char *));
    generate_gene_names(gex->gene_names, n_genes, "neg");

    /* Draw expression values from standard normal distribution */
    for (j = 0; j < n_genes; j++) {
        for (i = 0; i < n; i++)
            mat_set(gex->X, i, j, rand_normal(&rng_state));
    }

    return gex;
}

/* Append columns from two expression matrices */
GexMatrix *combine_expression_matrices(GexMatrix *pos_gex,
                                                GexMatrix *neg_gex) {
    int i, j;
    GexMatrix *combined = NULL;

    if (pos_gex == NULL || neg_gex == NULL ||
        pos_gex->X == NULL || neg_gex->X == NULL ||
        pos_gex->X->nrows != neg_gex->X->nrows ||
        pos_gex->X->ncols <= 0 || neg_gex->X->ncols <= 0)
        return NULL;

    combined = scalloc(1, sizeof(GexMatrix));
    combined->X = mat_new(pos_gex->X->nrows, pos_gex->X->ncols + neg_gex->X->ncols);
    combined->cell_names = scalloc(pos_gex->X->nrows, sizeof(char *));
    copy_cell_names(pos_gex->cell_names, combined->cell_names, pos_gex->X->nrows);
    combined->gene_names = scalloc(combined->X->ncols, sizeof(char *));

    for (j = 0; j < pos_gex->X->ncols; j++) {
        combined->gene_names[j] = strdup(pos_gex->gene_names[j]);
        if (combined->gene_names[j] == NULL) {
            gex_free_matrix_data(combined);
            return NULL;
        }
    }
    for (j = 0; j < neg_gex->X->ncols; j++) {
        int col = pos_gex->X->ncols + j;
        combined->gene_names[col] = strdup(neg_gex->gene_names[j]);
        if (combined->gene_names[col] == NULL) {
            gex_free_matrix_data(combined);
            return NULL;
        }
    }

    for (i = 0; i < combined->X->nrows; i++) {
        for (j = 0; j < pos_gex->X->ncols; j++)
            mat_set(combined->X, i, j, mat_get(pos_gex->X, i, j));
        for (j = 0; j < neg_gex->X->ncols; j++)
            mat_set(combined->X, i, pos_gex->X->ncols + j, mat_get(neg_gex->X, i, j));
    }

    return combined;
}

GexMatrix *brownian_simulate_expression_with_nulls(Matrix *Sigma,
                                                   char **names,
                                                   int n,
                                                   int n_tree_genes,
                                                   int n_null_genes,
                                                   unsigned int seed) {
    GexMatrix *pos_gex = NULL;
    GexMatrix *neg_gex = NULL;
    GexMatrix *combined = NULL;

    if (Sigma == NULL || names == NULL || n <= 0 ||
        n_tree_genes <= 0 || n_null_genes <= 0)
        return NULL;

    /* Set sigma2 value based on tree height */
    int n_sigma2 = 1;
    double sigma2[n_sigma2];
    double desired_tip_variance = 5.0;
    sigma2[0] = desired_tip_variance / mat_get(Sigma, 0, 0);  /* Set sigma2 relative to an assumed ultrametric tree height T to achieve a desired tip variance */

    pos_gex = brownian_simulate_expression_from_covariance(Sigma, names, n,
                                                           n_tree_genes, sigma2, 
                                                           n_sigma2, seed);

    neg_gex = simulate_standard_normal_expression(names, n, n_null_genes,
                                                   seed + 7919u);

    combined = combine_expression_matrices(pos_gex, neg_gex);

    /* Free memory */
    gex_free_matrix_data(pos_gex);
    gex_free_matrix_data(neg_gex);

    return combined;
}

/* Add one matrix to another in place element-wise. */
int add_matrix_in_place(Matrix *dest, Matrix *src) {
    int i, j;

    if (dest == NULL || src == NULL ||
        dest->nrows != src->nrows || dest->ncols != src->ncols)
        return -1;

    for (i = 0; i < dest->nrows; i++) {
        for (j = 0; j < dest->ncols; j++) {
            mat_set(dest, i, j, mat_get(dest, i, j) + mat_get(src, i, j));
        }
    }

    return 0;
}

/* Scale a matrix in place element-wise. */
int scale_matrix_in_place(Matrix *matrix, double factor) {
    int i, j;

    if (matrix == NULL)
        return -1;

    for (i = 0; i < matrix->nrows; i++) {
        for (j = 0; j < matrix->ncols; j++) {
            mat_set(matrix, i, j, factor * mat_get(matrix, i, j));
        }
    }

    return 0;
}

/* Run a simulation check to evaluate the performance of the phylogenetic signal filter(s). 
Sets up a simulation with the specified number of tree and null genes, runs the specified 
filter(s), and evaluates how many tree genes are correctly identified as true positives and how 
many null genes are incorrectly identified as false positives. Prints a summary of the results.
Returns 1 if successful, 0 if failed. */
int brownian_run_simulation_check(char **names,
                                  int n,
                                  int n_tree_genes,
                                  int n_null_genes,
                                  GexFilterMode mode,
                                  GexLRTAltMode lrt_alt_mode,
                                  int n_perm,
                                  double max_q,
                                  double min_i,
                                  Matrix **Sigmas,
                                  int n_sigmas,
                                  char *filter_sims_path,
                                  unsigned int seed) {
    int i, j;
    int tp = 0, fn = 0, fp = 0, tn = 0;
    GexMatrix *sim = NULL;  /* Simulated gene expression matrix */
    GexMatrix *tree_sim = NULL; /* Per-covariance simulated matrix */
    GexMoransResult *morans = NULL; /* Results from Moran's I calculation on simulated data */
    GexLRTResult *lrt = NULL;   /* Results from Brownian LRT calculation on simulated data */

    if (Sigmas == NULL || n_sigmas <= 0) {
        fprintf(stderr, "ERROR: brownian_run_simulation_check got invalid covariance set\n");
        return 0;
    }

    for (i = 0; i < n_sigmas; i++) {
        if (Sigmas[i] == NULL) {
            fprintf(stderr, "ERROR: brownian_run_simulation_check got NULL covariance at index %d\n", i);
            return 1;
        }

        tree_sim = brownian_simulate_expression_with_nulls(Sigmas[i], names, n,
                                                           n_tree_genes, n_null_genes,
                                                           seed + (unsigned int)(104729u * i));
        if (tree_sim == NULL)
            return 1;

        if (sim == NULL) {
            sim = tree_sim;
            tree_sim = NULL;
        } else {
            if (add_matrix_in_place(sim->X, tree_sim->X) != 0) {
                fprintf(stderr, "ERROR: failed to accumulate simulated expression matrices\n");
                return 1;
            }
            gex_free_matrix_data(tree_sim);
            tree_sim = NULL;
        }
    }

    if (sim == NULL || scale_matrix_in_place(sim->X, 1.0 / (double)n_sigmas) != 0) {
        fprintf(stderr, "ERROR: failed to build expected simulated expression matrix\n");
        return 1;
    }

    /* Optionally write the simulation expr matrix to a file. */
    if (filter_sims_path != NULL) {
        if (gex_write_labeled_matrix_tsv(filter_sims_path, sim->X,sim->cell_names, sim->X->nrows, sim->gene_names, sim->X->ncols, "cell") != 0) {
            fprintf(stderr, "ERROR: failed to write filter simulation results to %s\n", filter_sims_path);
            return 1;
        }
    }

    /* Run the specified filter(s) on the simulated data. */
    if (mode == GEX_FILTER_MORAN || mode == GEX_FILTER_BOTH)
        morans = (Sigmas == NULL ? NULL : gex_compute_morans_i(sim, Sigmas, n_sigmas, n_perm, seed + 17u));
    
    if (mode == GEX_FILTER_LRT || mode == GEX_FILTER_BOTH)
        lrt = (Sigmas == NULL ? NULL : gex_compute_brownian_lrt(sim, Sigmas,
                                                               n_sigmas,
                                                               n_perm,
                                                               seed + 31u,
                                                               lrt_alt_mode));

    if (Sigmas == NULL || n_sigmas <= 0 ||
        ((mode == GEX_FILTER_MORAN || mode == GEX_FILTER_BOTH) && morans == NULL) ||
        ((mode == GEX_FILTER_LRT || mode == GEX_FILTER_BOTH) && lrt == NULL)) {
        return 1;
    }

    /* Evaluate the performance of the filter(s) */
    for (j = 0; j < n_tree_genes; j++) {
        int keep = 0;
        if (mode == GEX_FILTER_MORAN) {
            keep = (morans->qvals[j] <= max_q && morans->morans_i[j] > min_i);
        } else if (mode == GEX_FILTER_LRT) {
            keep = (lrt->qvals[j] <= max_q && lrt->lrt_stat[j] > 0.0);
        } else if (mode == GEX_FILTER_BOTH) {
            keep = (morans->qvals[j] <= max_q && morans->morans_i[j] > min_i) &&
                (lrt->qvals[j] <= max_q && lrt->lrt_stat[j] > 0.0);
        }
        if (keep) tp++;
        else fn++;
    }

    for (j = 0; j < n_null_genes; j++) {
        int idx = n_tree_genes + j;
        int keep = 0;
        if (mode == GEX_FILTER_MORAN) {
            keep = (morans->qvals[idx] <= max_q && morans->morans_i[idx] > min_i);
        } else if (mode == GEX_FILTER_LRT) {
            keep = (lrt->qvals[idx] <= max_q && lrt->lrt_stat[idx] > 0.0);
        } else if (mode == GEX_FILTER_BOTH) {
            keep = (morans->qvals[idx] <= max_q && morans->morans_i[idx] > min_i) &&
                (lrt->qvals[idx] <= max_q && lrt->lrt_stat[idx] > 0.0);
        }
        if (keep) fp++;
        else tn++;
    }

    /* Print a summary of the simulation check results. */
    printf("Phylogenetic signal filter(s) simulation check:\n");
    printf("  positives simulated: %d, detected: %d, missed: %d\n",
           n_tree_genes, tp, fn);
    printf("  negatives simulated: %d, rejected: %d, false positives: %d\n",
           n_null_genes, tn, fp);

    /* Free memory */
    gex_free_matrix_data(tree_sim);
    gex_free_matrix_data(sim);
    gex_free_morans_result(morans);
    gex_free_lrt_result(lrt);

    return (fn == 0 && fp == 0); /* Return 1 if performance is perfect (no false negatives or false positives), 0 otherwise */
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

/* Print a summary of the covariance-based weight matrix. */
void print_weight_matrix_summary(Matrix *W) {
    int i, j;

    if (W == NULL) {
        fprintf(stderr, "ERROR: cannot summarize NULL weight matrix\n");
        return;
    }

    printf("\n");
    printf("First few entries of covariance-based weight matrix:\n");
    for (i = 0; i < W->nrows && i < 10; i++) {
        for (j = 0; j < W->ncols && j < 10; j++)
            printf(" %g", mat_get(W, i, j));
        printf("\n");
    }
    printf("\n");
}
