#include "gex_sim.h"

#include "brownian.h"
#include "gex.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <phast/eigen.h>
#include <phast/lists.h>
#include <phast/stringsplus.h>

static unsigned int gexsim_rand_u32(unsigned int *state) {
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static double gexsim_uniform_open(unsigned int *state) {
    return ((double)gexsim_rand_u32(state) + 1.0) / 4294967297.0;
}

static double gexsim_rand_normal(unsigned int *state) {
    double u1 = gexsim_uniform_open(state);
    double u2 = gexsim_uniform_open(state);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

static int gexsim_name_in_list(const char *name, List *names) {
    int i;

    if (name == NULL || names == NULL)
        return 0;
    for (i = 0; i < lst_size(names); i++) {
        String *s = lst_get_ptr(names, i);
        if (s != NULL && strcmp(name, s->chars) == 0)
            return 1;
    }
    return 0;
}

static char **gexsim_make_gene_names(int n_genes) {
    int j;
    char **names = NULL;

    if (n_genes <= 0)
        return NULL;
    names = (char **)calloc(n_genes, sizeof(char *));
    if (names == NULL)
        return NULL;

    for (j = 0; j < n_genes; j++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "sim_gene_%04d", j + 1);
        names[j] = strdup(buf);
        if (names[j] == NULL) {
            free_string_array(names, j);
            return NULL;
        }
    }

    return names;
}

int gex_get_shared_leaf_names(TreeNode **trees,
                              int n_trees,
                              char ***names_out,
                              int *n_names_out) {
    int i, j;
    List *first = NULL;
    List **name_lists = NULL;
    char **names = NULL;
    int names_capacity = 0;
    int n_keep = 0;
    int status = -1;

    if (names_out == NULL || n_names_out == NULL || trees == NULL || n_trees <= 0)
        return -1;

    *names_out = NULL;
    *n_names_out = 0;

    name_lists = (List **)calloc(n_trees, sizeof(List *));
    if (name_lists == NULL)
        return -1;

    for (i = 0; i < n_trees; i++) {
        if (trees[i] == NULL)
            goto cleanup;
        name_lists[i] = tr_leaf_names(trees[i]);
        if (name_lists[i] == NULL)
            goto cleanup;
    }

    first = name_lists[0];
    if (first == NULL || lst_size(first) <= 0)
        goto cleanup;

    names_capacity = lst_size(first);
    names = (char **)calloc(names_capacity, sizeof(char *));
    if (names == NULL)
        goto cleanup;

    for (i = 0; i < lst_size(first); i++) {
        String *s = lst_get_ptr(first, i);
        int present_in_all = 1;
        if (s == NULL)
            continue;
        for (j = 1; j < n_trees; j++) {
            if (!gexsim_name_in_list(s->chars, name_lists[j])) {
                present_in_all = 0;
                break;
            }
        }
        if (present_in_all) {
            names[n_keep] = strdup(s->chars);
            if (names[n_keep] == NULL)
                goto cleanup;
            n_keep++;
        }
    }

    if (n_keep <= 0)
        goto cleanup;

    *names_out = names;
    *n_names_out = n_keep;
    names = NULL;
    status = 0;

cleanup:
    if (name_lists != NULL) {
        for (i = 0; i < n_trees; i++) {
            if (name_lists[i] != NULL) {
                lst_free_strings(name_lists[i]);
                lst_free(name_lists[i]);
            }
        }
        free(name_lists);
    }
    if (names != NULL)
        free_string_array(names, names_capacity);
    return status;
}

/* Use the provides latent factors . */
int gex_simulate_from_latent_factors(GexMatrix *Z,
                                     char **cell_names,
                                     int n_cells,
                                     int k,
                                     int n_genes,
                                     double sigma2_obs,
                                     unsigned int seed,
                                     GexMatrix **L_out,
                                     GexMatrix **gex_out) {
    int i, j, d;    /* Loop indices */
    int success = 1; /* Whether the simulation succeeded; Failure by default */
    GexMatrix *gex = NULL;  /* Simulation output gene expression object */
    GexMatrix *L = NULL;    /* Simulation output gene loadings object */
    char **gene_names = NULL;   /* Gene names */
    unsigned int rng_state = (seed == 0u ? 1u : seed);  /* Random number generator state */

    if (Z == NULL || Z->X == NULL || L_out == NULL || gex_out == NULL || cell_names == NULL || n_cells <= 0 ||
        k <= 0 || n_genes <= 0 || sigma2_obs < 0.0)
        return 1;
    if (Z->n_cells != n_cells || Z->n_genes != k)
        return 1;

    /* Make sure the simulation output is initialized as empty */
    *gex_out = NULL;
    *L_out = NULL;

    /* Allocate objects in memory */
    gex = (GexMatrix *)calloc(1, sizeof(GexMatrix));
    L = (GexMatrix *)calloc(1, sizeof(GexMatrix));
    gene_names = gexsim_make_gene_names(n_genes);
    if (L == NULL || gex == NULL || gene_names == NULL)
        goto cleanup;

    L->n_cells = k;
    L->n_genes = n_genes;
    L->X = mat_new(k, n_genes);
    if (L->X == NULL)
        goto cleanup;

    /* Draw gene loadings L ~ N(0,1) and rescale each row to have norm
    sqrt(n_genes / k), ensuring each latent dimension contributes
    equal expected magnitude to the noiseless gene expression data. */
    for (d = 0; d < k; d++) {
        double row_ss = 0.0;
        double target_norm = sqrt((double)n_genes / (double)k);
        for (j = 0; j < n_genes; j++) {
            double val = gexsim_rand_normal(&rng_state);
            mat_set(L->X, d, j, val);
            row_ss += val * val;
        }
        if (row_ss > 0.0) {
            double row_scale = target_norm / sqrt(row_ss);
            for (j = 0; j < n_genes; j++)
                mat_set(L->X, d, j, row_scale * mat_get(L->X, d, j));
        }
    }

    /* Initialize the gene expression matrix */
    gex->n_cells = n_cells;
    gex->n_genes = n_genes;
    gex->X = mat_new(n_cells, n_genes);
    if (gex->X == NULL)
        goto cleanup;

    /* Compute the noiseless expression matrix from the 
    simulated Z and L matrix factorization. */
    mat_mult(gex->X, Z->X, L->X);

    /* Initialize the cell and gene names */
    gex->cell_names = (char **)calloc(n_cells, sizeof(char *));
    gex->gene_names = (char **)calloc(n_genes, sizeof(char *));
    if (gex->X == NULL || gex->cell_names == NULL || gex->gene_names == NULL)
        goto cleanup;

    for (i = 0; i < n_cells; i++) {
        gex->cell_names[i] = strdup(cell_names[i]);
        if (gex->cell_names[i] == NULL)
            goto cleanup;
    }
    for (j = 0; j < n_genes; j++) {
        gex->gene_names[j] = gene_names[j];
        gene_names[j] = NULL;
    }

    /* Add noise to the noiseless expression matrix based on 
    the sigma2_obs parameter input. */
    for (i = 0; i < n_cells; i++) {
        for (j = 0; j < n_genes; j++) {
            double val = mat_get(gex->X, i, j);
            if (sigma2_obs > 0.0)
                val += sqrt(sigma2_obs) * gexsim_rand_normal(&rng_state);
            mat_set(gex->X, i, j, val);
        }
    }

    success = 0; /* Simulation succeeded */

    *L_out = L;
    L = NULL;
    *gex_out = gex;
    gex = NULL;
    free_string_array(gene_names, n_genes);
    gene_names = NULL;

    cleanup:
    if (gene_names != NULL)
        free_string_array(gene_names, n_genes);
    if (L != NULL)
        gex_free_matrix_data(L);
    if (gex != NULL)
        gex_free_matrix_data(gex);
    return success;
}
