#include "gex.h"

#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <phast/eigen.h>

/* -------------------- helpers -------------------- */

static char *gex_strdup(const char *s) {
    size_t n;
    char *out;

    if (s == NULL) return NULL;
    n = strlen(s);

    out = (char *)malloc(n + 1);
    if (out == NULL) return NULL;

    memcpy(out, s, n + 1);
    return out;
}

static char *gex_lstrip(char *s) {
    while (*s != '\0' && isspace((unsigned char)*s))
        s++;
    return s;
}

static void gex_rstrip_inplace(char *s) {
    size_t n;

    if (s == NULL) return;
    n = strlen(s);

    while (n > 0 && isspace((unsigned char)s[n - 1])) {
        s[n - 1] = '\0';
        n--;
    }
}

static int gex_starts_with_tree_keyword(const char *s) {
    if (s == NULL) return 0;
    if (strlen(s) < 4) return 0;

    return (tolower((unsigned char)s[0]) == 't' &&
            tolower((unsigned char)s[1]) == 'r' &&
            tolower((unsigned char)s[2]) == 'e' &&
            tolower((unsigned char)s[3]) == 'e');
}

static char *gex_extract_newick_from_tree_line(const char *line) {
    const char *eq;
    char *tmp;
    char *s;
    char *out;

    if (line == NULL) return NULL;

    eq = strchr(line, '=');
    if (eq == NULL) return NULL;

    tmp = gex_strdup(eq + 1);
    if (tmp == NULL) return NULL;

    s = gex_lstrip(tmp);
    gex_rstrip_inplace(s);

    if (strncmp(s, "[&R]", 4) == 0 || strncmp(s, "[&U]", 4) == 0) {
        s += 4;
        s = gex_lstrip(s);
    }

    out = gex_strdup(s);
    free(tmp);
    return out;
}

/* Split a line on tabs/newlines only. Returns number of fields. */
static int gex_split_tab_fields(char *line, char ***fields_out) {
    int count = 0;
    int capacity = 8;
    char **fields = NULL;
    char *token = NULL;

    fields = (char **)malloc(capacity * sizeof(char *));
    if (fields == NULL) return -1;

    token = strtok(line, "\t\r\n");
    while (token != NULL) {
        if (count == capacity) {
            char **tmp;
            capacity *= 2;
            tmp = (char **)realloc(fields, capacity * sizeof(char *));
            if (tmp == NULL) {
                free(fields);
                return -1;
            }
            fields = tmp;
        }

        fields[count++] = token;
        token = strtok(NULL, "\t\r\n");
    }

    *fields_out = fields;
    return count;
}

typedef struct {
    double pval;
    int idx;
} GexPvalPair;

static int gex_cmp_pval_asc(const void *a, const void *b) {
    const GexPvalPair *pa = (const GexPvalPair *)a;
    const GexPvalPair *pb = (const GexPvalPair *)b;

    if (pa->pval < pb->pval) return -1;
    if (pa->pval > pb->pval) return 1;
    return 0;
}

static unsigned int gex_rand_u32(unsigned int *state) {
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static void gex_shuffle_double(double *x, int n, unsigned int *state) {
    int i;

    for (i = n - 1; i > 0; i--) {
        int j = (int)(gex_rand_u32(state) % (unsigned int)(i + 1));
        double tmp = x[i];
        x[i] = x[j];
        x[j] = tmp;
    }
}

static double gex_weighted_quadratic(Matrix *W, double *x, int n) {
    int i, j;
    double out = 0.0;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++)
            out += x[i] * mat_get(W, i, j) * x[j];
    }

    return out;
}

static Matrix *gex_standardize_columns(GexMatrix *gex) {
    int i, j;
    Matrix *Z;

    if (gex == NULL || gex->X == NULL)
        return NULL;

    Z = mat_new(gex->n_cells, gex->n_genes);
    if (Z == NULL)
        return NULL;

    for (j = 0; j < gex->n_genes; j++) {
        double mean = 0.0;
        double var = 0.0;
        double sd;

        for (i = 0; i < gex->n_cells; i++)
            mean += mat_get(gex->X, i, j);
        mean /= (double)gex->n_cells;

        for (i = 0; i < gex->n_cells; i++) {
            double d = mat_get(gex->X, i, j) - mean;
            var += d * d;
        }
        var /= (double)gex->n_cells;
        sd = sqrt(var);

        if (sd < 1e-12) {
            for (i = 0; i < gex->n_cells; i++)
                mat_set(Z, i, j, 0.0);
        }
        else {
            for (i = 0; i < gex->n_cells; i++) {
                double z = (mat_get(gex->X, i, j) - mean) / sd;
                mat_set(Z, i, j, z);
            }
        }
    }

    return Z;
}

static void gex_bh_adjust(double *pvals, double *qvals, int n) {
    GexPvalPair *pairs;
    int i;
    double running;

    pairs = (GexPvalPair *)malloc(n * sizeof(GexPvalPair));
    if (pairs == NULL) {
        for (i = 0; i < n; i++)
            qvals[i] = 1.0;
        return;
    }

    for (i = 0; i < n; i++) {
        pairs[i].pval = pvals[i];
        pairs[i].idx = i;
    }

    qsort(pairs, n, sizeof(GexPvalPair), gex_cmp_pval_asc);

    running = 1.0;
    for (i = n - 1; i >= 0; i--) {
        double rank = (double)(i + 1);
        double val = pairs[i].pval * (double)n / rank;
        if (val > 1.0) val = 1.0;
        if (val < running) running = val;
        qvals[pairs[i].idx] = running;
    }

    free(pairs);
}

static int gex_count_kept_genes(GexMoransResult *res, double max_q, double min_i) {
    int j;
    int nkeep = 0;

    for (j = 0; j < res->n_genes; j++) {
        if (res->qvals[j] <= max_q && res->morans_i[j] > min_i)
            nkeep++;
    }

    return nkeep;
}

/* -------------------- tree reading -------------------- */

TreeNode **gex_read_nexus(const char *filename, int *n_trees) {
    FILE *f;
    char line[100000];
    TreeNode **trees = NULL;
    int capacity = 0;
    int count = 0;

    if (n_trees == NULL || filename == NULL)
        return NULL;

    *n_trees = 0;

    f = fopen(filename, "r");
    if (f == NULL) {
        fprintf(stderr, "ERROR: could not open NEXUS file: %s\n", filename);
        return NULL;
    }

    while (fgets(line, sizeof(line), f) != NULL) {
        char *trimmed;
        char *newick;
        TreeNode *tree;

        trimmed = gex_lstrip(line);

        if (!gex_starts_with_tree_keyword(trimmed))
            continue;

        newick = gex_extract_newick_from_tree_line(trimmed);
        if (newick == NULL)
            continue;

        tree = tr_new_from_string(newick);
        free(newick);

        if (tree == NULL) {
            fprintf(stderr, "ERROR: failed to parse tree from file: %s\n", filename);
            gex_free_trees(trees, count);
            fclose(f);
            return NULL;
        }

        if (count == capacity) {
            int new_capacity = (capacity == 0 ? 8 : 2 * capacity);
            TreeNode **tmp = (TreeNode **)realloc(trees, new_capacity * sizeof(TreeNode *));
            if (tmp == NULL) {
                fprintf(stderr, "ERROR: out of memory while storing trees\n");
                tr_free(tree);
                gex_free_trees(trees, count);
                fclose(f);
                return NULL;
            }
            trees = tmp;
            capacity = new_capacity;
        }

        trees[count++] = tree;
    }

    fclose(f);

    if (count == 0) {
        fprintf(stderr, "ERROR: no TREE lines found in NEXUS file: %s\n", filename);
        free(trees);
        return NULL;
    }

    *n_trees = count;
    return trees;
}

void gex_free_trees(TreeNode **trees, int n_trees) {
    int i;

    if (trees == NULL) return;

    for (i = 0; i < n_trees; i++) {
        if (trees[i] != NULL)
            tr_free(trees[i]);
    }

    free(trees);
}

/* -------------------- labeled matrix reading -------------------- */

GexMatrix *gex_read_labeled_matrix(const char *filename) {
    FILE *f;
    char line[100000];
    GexMatrix *gex = NULL;
    int n_cells = 0;
    int n_genes = 0;

    if (filename == NULL) return NULL;

    f = fopen(filename, "r");
    if (f == NULL) {
        fprintf(stderr, "ERROR: could not open matrix file: %s\n", filename);
        return NULL;
    }

    /* ---------- first pass: read header ---------- */
    if (fgets(line, sizeof(line), f) == NULL) {
        fprintf(stderr, "ERROR: matrix file is empty: %s\n", filename);
        fclose(f);
        return NULL;
    }

    {
        char *line_copy = gex_strdup(line);
        char **fields = NULL;
        int nfields, i;

        if (line_copy == NULL) {
            fclose(f);
            return NULL;
        }

        nfields = gex_split_tab_fields(line_copy, &fields);
        if (nfields < 2) {
            fprintf(stderr, "ERROR: header must contain row label column plus at least one gene\n");
            free(line_copy);
            fclose(f);
            return NULL;
        }

        n_genes = nfields - 1;

        gex = (GexMatrix *)calloc(1, sizeof(GexMatrix));
        if (gex == NULL) {
            free(fields);
            free(line_copy);
            fclose(f);
            return NULL;
        }

        gex->gene_names = (char **)malloc(n_genes * sizeof(char *));
        if (gex->gene_names == NULL) {
            free(fields);
            free(line_copy);
            free(gex);
            fclose(f);
            return NULL;
        }

        for (i = 0; i < n_genes; i++) {
            gex->gene_names[i] = gex_strdup(fields[i + 1]);
            if (gex->gene_names[i] == NULL) {
                free(fields);
                free(line_copy);
                fclose(f);
                return NULL;
            }
        }

        free(fields);
        free(line_copy);
    }

    /* ---------- count number of cell rows ---------- */
    while (fgets(line, sizeof(line), f) != NULL) {
        char *trimmed = gex_lstrip(line);
        if (*trimmed == '\0' || *trimmed == '\n')
            continue;
        n_cells++;
    }

    if (n_cells == 0) {
        fprintf(stderr, "ERROR: no data rows found in matrix file: %s\n", filename);
        fclose(f);
        return NULL;
    }

    gex->n_cells = n_cells;
    gex->n_genes = n_genes;

    gex->cell_names = (char **)malloc(n_cells * sizeof(char *));
    if (gex->cell_names == NULL) {
        fclose(f);
        return NULL;
    }

    gex->X = mat_new(n_cells, n_genes);
    if (gex->X == NULL) {
        fclose(f);
        return NULL;
    }

    /* ---------- second pass: fill names and matrix ---------- */
    rewind(f);

    /* skip header */
    if (fgets(line, sizeof(line), f) == NULL) {
        fclose(f);
        return NULL;
    }

    {
        int row = 0;

        while (fgets(line, sizeof(line), f) != NULL) {
            char *line_copy;
            char **fields = NULL;
            int nfields, j;

            char *trimmed = gex_lstrip(line);
            if (*trimmed == '\0' || *trimmed == '\n')
                continue;

            line_copy = gex_strdup(line);
            if (line_copy == NULL) {
                fclose(f);
                return NULL;
            }

            nfields = gex_split_tab_fields(line_copy, &fields);
            if (nfields != n_genes + 1) {
                fprintf(stderr, "ERROR: row %d has wrong number of columns in %s\n", row + 1, filename);
                free(line_copy);
                fclose(f);
                return NULL;
            }

            gex->cell_names[row] = gex_strdup(fields[0]);
            if (gex->cell_names[row] == NULL) {
                free(fields);
                free(line_copy);
                fclose(f);
                return NULL;
            }

            for (j = 0; j < n_genes; j++) {
                mat_set(gex->X, row, j, atof(fields[j + 1]));
            }

            free(fields);
            free(line_copy);
            row++;
        }
    }

    fclose(f);
    return gex;
}

void gex_free_matrix_data(GexMatrix *gex) {
    int i;

    if (gex == NULL) return;

    if (gex->cell_names != NULL) {
        for (i = 0; i < gex->n_cells; i++)
            free(gex->cell_names[i]);
        free(gex->cell_names);
    }

    if (gex->gene_names != NULL) {
        for (i = 0; i < gex->n_genes; i++)
            free(gex->gene_names[i]);
        free(gex->gene_names);
    }

    if (gex->X != NULL)
        mat_free(gex->X);

    free(gex);
}

/* Summary of tree set and expr matrix i/o */
void gex_print_io_summary(TreeNode **trees, int n_trees, GexMatrix *gex) {
    int i;

    printf("Loaded %d tree(s)\n", n_trees);

    if (gex != NULL && gex->X != NULL) {
        printf("Loaded matrix with %d cell(s) and %d gene(s)\n",
               gex->n_cells, gex->n_genes);

        printf("First few cell names:\n");
        for (i = 0; i < gex->n_cells && i < 10; i++)
            printf("  %s\n", gex->cell_names[i]);

        printf("First few gene names:\n");
        for (i = 0; i < gex->n_genes && i < 10; i++)
            printf("  %s\n", gex->gene_names[i]);

        printf("First few entries of matrix:\n");
        for (i = 0; i < gex->n_cells && i < 10; i++) {
            int j;
            printf("  %s:", gex->cell_names[i]);
            for (j = 0; j < gex->n_genes && j < 10; j++)
                printf(" %g", mat_get(gex->X, i, j));
            printf("\n");
        }
    }

    if (n_trees > 0 && trees != NULL && trees[0] != NULL) {
        printf("First tree: ");
        tr_print(stdout, trees[0], 1);
        printf("\n");
    }
}

GexMoransResult *gex_compute_morans_i(GexMatrix *gex,
                                      Matrix *W,
                                      int n_perm,
                                      unsigned int seed) {
    int i, j, k;
    int n_cells;
    int n_genes;
    unsigned int rng_state;
    Matrix *Z = NULL;
    Matrix *B = NULL;
    GexMoransResult *res = NULL;
    double *zcol = NULL;
    double *perm = NULL;

    if (gex == NULL || gex->X == NULL || W == NULL || n_perm <= 0) {
        fprintf(stderr, "ERROR: gex_compute_morans_i got invalid input\n");
        return NULL;
    }

    n_cells = gex->n_cells;
    n_genes = gex->n_genes;

    if (W->nrows != n_cells || W->ncols != n_cells) {
        fprintf(stderr, "ERROR: weight matrix dimensions do not match number of cells\n");
        return NULL;
    }

    Z = gex_standardize_columns(gex);
    if (Z == NULL) {
        fprintf(stderr, "ERROR: failed to standardize gene expression matrix\n");
        return NULL;
    }

    B = mat_new(n_cells, n_genes);
    if (B == NULL) {
        mat_free(Z);
        return NULL;
    }

    for (i = 0; i < n_cells; i++) {
        for (j = 0; j < n_genes; j++) {
            double sum = 0.0;
            for (k = 0; k < n_cells; k++)
                sum += mat_get(W, i, k) * mat_get(Z, k, j);
            mat_set(B, i, j, sum);
        }
    }

    res = (GexMoransResult *)calloc(1, sizeof(GexMoransResult));
    if (res == NULL) {
        mat_free(Z);
        mat_free(B);
        return NULL;
    }

    res->corr = mat_new(n_genes, n_genes);
    res->morans_i = (double *)calloc(n_genes, sizeof(double));
    res->pvals = (double *)calloc(n_genes, sizeof(double));
    res->qvals = (double *)calloc(n_genes, sizeof(double));
    res->n_genes = n_genes;
    if (res->corr == NULL || res->morans_i == NULL ||
        res->pvals == NULL || res->qvals == NULL) {
        gex_free_morans_result(res);
        mat_free(Z);
        mat_free(B);
        return NULL;
    }

    for (j = 0; j < n_genes; j++) {
        for (k = j; k < n_genes; k++) {
            double sum = 0.0;
            for (i = 0; i < n_cells; i++)
                sum += mat_get(Z, i, j) * mat_get(B, i, k);
            mat_set(res->corr, j, k, sum);
            mat_set(res->corr, k, j, sum);
        }
        res->morans_i[j] = mat_get(res->corr, j, j);
    }

    zcol = (double *)malloc(n_cells * sizeof(double));
    perm = (double *)malloc(n_cells * sizeof(double));
    if (zcol == NULL || perm == NULL) {
        free(zcol);
        free(perm);
        gex_free_morans_result(res);
        mat_free(Z);
        mat_free(B);
        return NULL;
    }

    rng_state = (seed == 0u ? 1u : seed);
    for (j = 0; j < n_genes; j++) {
        int ge_count = 0;

        for (i = 0; i < n_cells; i++) {
            zcol[i] = mat_get(Z, i, j);
            perm[i] = zcol[i];
        }

        for (k = 0; k < n_perm; k++) {
            double perm_i;
            memcpy(perm, zcol, n_cells * sizeof(double));
            gex_shuffle_double(perm, n_cells, &rng_state);
            perm_i = gex_weighted_quadratic(W, perm, n_cells);
            if (perm_i >= res->morans_i[j])
                ge_count++;
        }

        res->pvals[j] = ((double)ge_count + 1.0) / ((double)n_perm + 1.0);
    }

    gex_bh_adjust(res->pvals, res->qvals, n_genes);
    res->n_significant = gex_count_kept_genes(res, 0.05, 0.0);

    free(zcol);
    free(perm);
    mat_free(Z);
    mat_free(B);
    return res;
}

void gex_print_morans_summary(GexMoransResult *res,
                              GexMatrix *gex,
                              double max_q,
                              double min_i) {
    int i, j;

    if (res == NULL || gex == NULL) {
        fprintf(stderr, "ERROR: cannot summarize NULL Moran's I result\n");
        return;
    }

    printf("Computed Moran's I correlation matrix for %d gene(s)\n", res->n_genes);
    printf("Genes passing filter (q <= %.4f and I > %.4f): %d\n",
           max_q, min_i, gex_count_kept_genes(res, max_q, min_i));

    printf("First few entries of Moran's I gene-gene correlation matrix:\n");
    for (i = 0; i < res->n_genes && i < 10; i++) {
        printf("%s", gex->gene_names[i]);
        for (j = 0; j < res->n_genes && j < 10; j++)
            printf("\t%g", mat_get(res->corr, i, j));
        printf("\n");
    }

    printf("\nFirst few gene-wise Moran's I statistics:\n");
    printf("gene\tI\tp\tq\tkeep\n");
    for (j = 0; j < 10; j++) {
        int keep = (res->qvals[j] <= max_q && res->morans_i[j] > min_i);
        printf("%s\t%g\t%g\t%g\t%s\n",
               gex->gene_names[j],
               res->morans_i[j],
               res->pvals[j],
               res->qvals[j],
               (keep ? "yes" : "no"));
    }
    printf("\n");
}

int gex_write_morans_tsv(const char *filename,
                         GexMoransResult *res,
                         GexMatrix *gex,
                         double max_q,
                         double min_i) {
    FILE *out;
    int i;

    if (filename == NULL || res == NULL || gex == NULL) {
        fprintf(stderr, "ERROR: gex_write_morans_tsv got invalid input\n");
        return -1;
    }

    out = fopen(filename, "w");
    if (out == NULL) {
        fprintf(stderr, "ERROR: could not open Moran output file: %s\n", filename);
        return -1;
    }

    fprintf(out, "gene\tmorans_I\tp_value\tq_value\tkeep\n");
    for (i = 0; i < res->n_genes; i++) {
        int keep = (res->qvals[i] <= max_q && res->morans_i[i] > min_i);
        fprintf(out, "%s\t%.17g\t%.17g\t%.17g\t%s\n",
                gex->gene_names[i],
                res->morans_i[i],
                res->pvals[i],
                res->qvals[i],
                (keep ? "yes" : "no"));
    }

    fclose(out);
    return 0;
}

void gex_free_morans_result(GexMoransResult *res) {
    if (res == NULL)
        return;

    if (res->corr != NULL)
        mat_free(res->corr);
    if (res->morans_i != NULL)
        free(res->morans_i);
    if (res->pvals != NULL)
        free(res->pvals);
    if (res->qvals != NULL)
        free(res->qvals);

    free(res);
}

GexMatrix *gex_filter_genes_by_morans_result(GexMatrix *gex,
                                             GexMoransResult *res,
                                             double max_q,
                                             double min_i) {
    int i, j;
    int out_j = 0;
    int nkeep;
    GexMatrix *out = NULL;

    if (gex == NULL || res == NULL || gex->n_genes != res->n_genes)
        return NULL;

    nkeep = gex_count_kept_genes(res, max_q, min_i);
    if (nkeep <= 0) {
        fprintf(stderr, "ERROR: Moran's I filtering removed all genes\n");
        return NULL;
    }

    out = (GexMatrix *)calloc(1, sizeof(GexMatrix));
    if (out == NULL)
        return NULL;

    out->n_cells = gex->n_cells;
    out->n_genes = nkeep;
    out->X = mat_new(out->n_cells, out->n_genes);
    out->cell_names = (char **)malloc(out->n_cells * sizeof(char *));
    out->gene_names = (char **)malloc(out->n_genes * sizeof(char *));
    if (out->X == NULL || out->cell_names == NULL || out->gene_names == NULL) {
        gex_free_matrix_data(out);
        return NULL;
    }

    for (i = 0; i < out->n_cells; i++) {
        out->cell_names[i] = gex_strdup(gex->cell_names[i]);
        if (out->cell_names[i] == NULL) {
            gex_free_matrix_data(out);
            return NULL;
        }
    }

    for (j = 0; j < gex->n_genes; j++) {
        if (res->qvals[j] <= max_q && res->morans_i[j] > min_i) {
            out->gene_names[out_j] = gex_strdup(gex->gene_names[j]);
            if (out->gene_names[out_j] == NULL) {
                gex_free_matrix_data(out);
                return NULL;
            }
            for (i = 0; i < gex->n_cells; i++)
                mat_set(out->X, i, out_j, mat_get(gex->X, i, j));
            out_j++;
        }
    }

    return out;
}
