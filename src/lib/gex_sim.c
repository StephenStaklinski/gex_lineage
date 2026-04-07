#include "gex_sim.h"

#include "brownian.h"

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

static void gexsim_free_string_array(char **names, int n) {
    int i;

    if (names == NULL)
        return;
    for (i = 0; i < n; i++)
        free(names[i]);
    free(names);
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

static char **gexsim_make_factor_names(int k) {
    int i;
    char **names = NULL;

    if (k <= 0)
        return NULL;
    names = (char **)calloc(k, sizeof(char *));
    if (names == NULL)
        return NULL;

    for (i = 0; i < k; i++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "LF%d", i + 1);
        names[i] = strdup(buf);
        if (names[i] == NULL) {
            gexsim_free_string_array(names, i);
            return NULL;
        }
    }

    return names;
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
            gexsim_free_string_array(names, j);
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
        gexsim_free_string_array(names, names_capacity);
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
    gexsim_free_string_array(gene_names, n_genes);
    gene_names = NULL;

    cleanup:
    if (gene_names != NULL)
        gexsim_free_string_array(gene_names, n_genes);
    if (L != NULL)
        gex_free_matrix_data(L);
    if (gex != NULL)
        gex_free_matrix_data(gex);
    return success;
}

int gex_write_labeled_matrix_tsv(const char *filename,
                                 Matrix *X,
                                 char **row_names,
                                 int n_rows,
                                 char **col_names,
                                 int n_cols,
                                 const char *corner_label) {
    int i, j;
    FILE *out = NULL;

    if (filename == NULL || X == NULL || row_names == NULL || col_names == NULL ||
        n_rows <= 0 || n_cols <= 0 || X->nrows != n_rows || X->ncols != n_cols)
        return -1;

    out = fopen(filename, "w");
    if (out == NULL)
        return -1;

    fprintf(out, "%s", corner_label == NULL ? "id" : corner_label);
    for (j = 0; j < n_cols; j++)
        fprintf(out, "\t%s", col_names[j]);
    fprintf(out, "\n");

    for (i = 0; i < n_rows; i++) {
        fprintf(out, "%s", row_names[i]);
        for (j = 0; j < n_cols; j++)
            fprintf(out, "\t%.17g", mat_get(X, i, j));
        fprintf(out, "\n");
    }

    fclose(out);
    return 0;
}

int gex_write_simulation_truth(const char *outprefix,
                                GexMatrix *gex,
                                GexMatrix *L,
                                GexMatrix *Z,
                                char **cell_names,
                                char **gene_names,
                                int k,
                                double sigma2_obs,
                                double *sigma2_latent) {
    char summary_path[4096];
    char z_path[4096];
    char l_path[4096];
    char expr_path[4096];
    char **factor_names = NULL;
    FILE *summary_out = NULL;
    int j;
    int status = 1;

    if (outprefix == NULL || gex == NULL || L == NULL || Z == NULL ||
        L->X == NULL || Z->X == NULL || cell_names == NULL || gene_names == NULL ||
        k <= 0 || sigma2_latent == NULL)
        goto cleanup;

    factor_names = gexsim_make_factor_names(k);
    if (factor_names == NULL)
        goto cleanup;

    snprintf(summary_path, sizeof(summary_path), "%s.summary.tsv", outprefix);
    snprintf(z_path, sizeof(z_path), "%s.Z.tsv", outprefix);
    snprintf(l_path, sizeof(l_path), "%s.L.tsv", outprefix);
    snprintf(expr_path, sizeof(expr_path), "%s.expr.tsv", outprefix);

    /* Write out the summary parameters file to match the format used
    by model fitting output */
    summary_out = fopen(summary_path, "w");
    if (summary_out == NULL)
        goto cleanup;
    fprintf(summary_out, "parameter\tvalue\n");
    fprintf(summary_out, "n_cells\t%d\n", gex->n_cells);
    fprintf(summary_out, "n_genes\t%d\n", gex->n_genes);
    fprintf(summary_out, "k\t%d\n", k);
    fprintf(summary_out, "sigma_obs\t%.17g\n", sigma2_obs);
    for (j = 0; j < k; j++)
        fprintf(summary_out, "sigma_latent_LF%d\t%.17g\n", j + 1, sigma2_latent[j]);
    fclose(summary_out);
    summary_out = NULL;

    /* Write out the simulated matrices */
    if (gex_write_labeled_matrix_tsv(expr_path, gex->X, cell_names, gex->n_cells,
                                     gene_names, gex->n_genes, "cell") != 0)
        goto cleanup;
    if (gex_write_labeled_matrix_tsv(z_path, Z->X, cell_names, gex->n_cells,
                                     factor_names, k, "cell") != 0)
        goto cleanup;
    if (gex_write_labeled_matrix_tsv(l_path, L->X, factor_names, k,
                                     gene_names, gex->n_genes, "factor") != 0)
        goto cleanup;

    status = 0;

cleanup:
    if (summary_out != NULL)
        fclose(summary_out);
    if (factor_names != NULL)
        gexsim_free_string_array(factor_names, k);
    return status;
}
