#include "gex.h"

#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <phast/eigen.h>
#include <phast/lists.h>
#include <phast/stringsplus.h>

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

/* Strip leading whitespace from a string. 
Returns pointer to first non-whitespace character. */
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

/* Extract a Newick string from a NEXUS tree line.
Returns a pointer to the extracted string or NULL on failure. */
static char *gex_extract_newick_from_tree_line(const char *line) {
    const char *eq;
    char *tmp;
    char *s;
    char *out;

    if (line == NULL) return NULL;

    eq = strchr(line, '='); /* Find the '=' character that separates the tree name from the Newick string */
    if (eq == NULL) return NULL;

    tmp = gex_strdup(eq + 1);   /* Duplicate the substring after '=' for manipulation */
    if (tmp == NULL) return NULL;

    s = gex_lstrip(tmp);    /* Strip leading whitespace */
    gex_rstrip_inplace(s);  /* Strip trailing whitespace */

    /* Strip any leading '[&' and trailing ']' annotation characters */
    if (strncmp(s, "[&R]", 4) == 0 || strncmp(s, "[&U]", 4) == 0) {
        s += 4;
        s = gex_lstrip(s);
    }

    out = gex_strdup(s);
    free(tmp);
    return out;
}

/* Split a line on tabs/newlines only. Sets the fields_out pointer to an array of field strings.
Returns number of fields if successful, -1 on failure. */
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

static double gex_uniform_open(unsigned int *state) {
    return ((double)gex_rand_u32(state) + 1.0) / 4294967297.0;
}

static double gex_rand_normal(unsigned int *state) {
    double u1 = gex_uniform_open(state);
    double u2 = gex_uniform_open(state);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
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

static double gex_chisq1_sf(double x) {
    if (x <= 0.0)
        return 1.0;
    return erfc(sqrt(0.5 * x));
}

static double gex_loglik_centered_gaussian_identity(double *y, int n) {
    int i;
    double mean = 0.0;
    double sse = 0.0;
    double sigma2;

    for (i = 0; i < n; i++)
        mean += y[i];
    mean /= (double)n;

    for (i = 0; i < n; i++) {
        double d = y[i] - mean;
        sse += d * d;
    }

    sigma2 = sse / (double)n;
    if (sigma2 < 1e-12)
        sigma2 = 1e-12;

    return -0.5 * ((double)n * (log(2.0 * M_PI * sigma2) + 1.0));
}

static void gex_fit_gaussian_identity(double *y, int n, double *mean_out, double *sigma2_out) {
    int i;
    double mean = 0.0;
    double sse = 0.0;

    for (i = 0; i < n; i++)
        mean += y[i];
    mean /= (double)n;

    for (i = 0; i < n; i++) {
        double d = y[i] - mean;
        sse += d * d;
    }

    *mean_out = mean;
    *sigma2_out = sse / (double)n;
    if (*sigma2_out < 1e-12)
        *sigma2_out = 1e-12;
}

static double gex_loglik_centered_gaussian_cov(double *y,
                                               Matrix *Sigma,
                                               Matrix *Sigma_inv,
                                               double logdet_sigma) {
    int i, j, n;
    double *Sinv1 = NULL;
    double *Sinvy = NULL;
    double quad = 0.0;
    double ones_Sinv_ones = 0.0;
    double ones_Sinv_y = 0.0;
    double muhat;
    double sigma2;
    double ll;

    n = Sigma->nrows;
    Sinv1 = (double *)calloc(n, sizeof(double));
    Sinvy = (double *)calloc(n, sizeof(double));
    if (Sinv1 == NULL || Sinvy == NULL) {
        free(Sinv1);
        free(Sinvy);
        return -HUGE_VAL;
    }

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            Sinv1[i] += mat_get(Sigma_inv, i, j);
            Sinvy[i] += mat_get(Sigma_inv, i, j) * y[j];
        }
        ones_Sinv_ones += Sinv1[i];
        ones_Sinv_y += Sinvy[i];
    }

    if (ones_Sinv_ones <= 0.0) {
        free(Sinv1);
        free(Sinvy);
        return -HUGE_VAL;
    }

    muhat = ones_Sinv_y / ones_Sinv_ones;
    for (i = 0; i < n; i++) {
        double yi = y[i] - muhat;
        for (j = 0; j < n; j++)
            quad += yi * mat_get(Sigma_inv, i, j) * (y[j] - muhat);
    }

    sigma2 = quad / (double)n;
    if (sigma2 < 1e-12)
        sigma2 = 1e-12;

    ll = -0.5 * ((double)n * log(2.0 * M_PI * sigma2) +
                 logdet_sigma +
                 (double)n);

    free(Sinv1);
    free(Sinvy);
    return ll;
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

static int gex_count_kept_lrt_genes(GexLRTResult *res, double max_q) {
    int j;
    int nkeep = 0;

    for (j = 0; j < res->n_genes; j++) {
        if (res->qvals[j] <= max_q && res->lrt_stat[j] > 0.0)
            nkeep++;
    }

    return nkeep;
}

static int gex_name_in_char_array(const char *name, char **names, int n) {
    int i;
    for (i = 0; i < n; i++) {
        if (strcmp(name, names[i]) == 0)
            return 1;
    }
    return 0;
}

static int gex_name_in_string_list(const char *name, List *names) {
    int i;
    for (i = 0; i < lst_size(names); i++) {
        String *s = lst_get_ptr(names, i);
        if (strcmp(name, s->chars) == 0)
            return 1;
    }
    return 0;
}

static void gex_free_string_ptr_list(List *l) {
    int i;
    if (l == NULL)
        return;
    for (i = 0; i < lst_size(l); i++) {
        String *s = lst_get_ptr(l, i);
        if (s != NULL)
            str_free(s);
    }
    lst_free(l);
}

static int gex_is_leaf(TreeNode *node) {
    return (node != NULL && node->lchild == NULL && node->rchild == NULL);
}

/* Collect the depth range of all tips in a tree.
Updates the min_depth, max_depth, and n_tips pointers. Returns nothing. */
static void gex_collect_tip_depth_range(TreeNode *node,
                                        double depth,
                                        double *min_depth,
                                        double *max_depth,
                                        int *n_tips) {
    double next_depth = depth;

    if (node == NULL)
        return;

    if (node->parent != NULL)
        next_depth += node->dparent;    /* Add the parent distance to the depth */

    /* For leaf, increment n_tips and optionally update depth ranges */
    if (gex_is_leaf(node)) {
        if (*n_tips == 0 || next_depth < *min_depth)
            *min_depth = next_depth;
        if (*n_tips == 0 || next_depth > *max_depth)
            *max_depth = next_depth;
        (*n_tips)++;
        return;
    }

    /* Recursive depth-first traversal to collect depth ranges for left and right children */
    gex_collect_tip_depth_range(node->lchild, next_depth, min_depth, max_depth, n_tips);
    gex_collect_tip_depth_range(node->rchild, next_depth, min_depth, max_depth, n_tips);
}

/* Scale the distances in a tree uniformly by a given factor. */
static void gex_scale_tree_recursive(TreeNode *node, double scale) {
    if (node == NULL)
        return;

    if (node->parent != NULL)
        node->dparent *= scale;

    gex_scale_tree_recursive(node->lchild, scale);
    gex_scale_tree_recursive(node->rchild, scale);
}

static int gex_keep_gene(GexMoransResult *morans,
                         GexLRTResult *lrt,
                         int gene_idx,
                         GexFilterMode mode,
                         double max_q,
                         double min_i) {
    int keep_moran = 0;
    int keep_lrt = 0;

    if (morans != NULL)
        keep_moran = (morans->qvals[gene_idx] <= max_q &&
                      morans->morans_i[gene_idx] > min_i);
    if (lrt != NULL)
        keep_lrt = (lrt->qvals[gene_idx] <= max_q &&
                    lrt->lrt_stat[gene_idx] > 0.0);

    if (mode == GEX_FILTER_MORAN)
        return keep_moran;
    if (mode == GEX_FILTER_LRT)
        return keep_lrt;
    /* Return the intersection if both tests are run */
    return (keep_moran && keep_lrt);
}

/* Read in NEXUS tree file and parse trees.
Updates the n_trees pointer. Returns a pointer to the 
array of tree pointers or NULL on failure. 

TODO: Make the function map names from the nexus file header
the trees block since many NEXUS files have renamed taxa. */
TreeNode **gex_read_nexus(const char *filename, int *n_trees) {
    FILE *f;
    char line[100000];
    TreeNode **trees = NULL;    /* Array of tree pointers to fill */
    int capacity = 0;
    int count = 0;

    if (n_trees == NULL || filename == NULL)
        return NULL;

    *n_trees = 0;   /* Reset the number of trees to 0 */

    f = fopen(filename, "r");
    if (f == NULL) {
        fprintf(stderr, "ERROR: could not open NEXUS file: %s\n", filename);
        return NULL;
    }

    while (fgets(line, sizeof(line), f) != NULL) {
        char *trimmed;
        char *newick;
        TreeNode *tree;

        trimmed = gex_lstrip(line); /* Strip leading whitespace */

        if (!gex_starts_with_tree_keyword(trimmed))
            continue;   /* Skip lines that don't start with "TREE" */

        newick = gex_extract_newick_from_tree_line(trimmed);    /* Get Newick string */
        if (newick == NULL)
            continue;

        tree = tr_new_from_string(newick);  /* Parse the Newick string into a tree structure */
        free(newick);

        /* Check if tree parsing was successful */
        if (tree == NULL) {
            fprintf(stderr, "ERROR: failed to parse tree from file: %s\n", filename);
            gex_free_trees(trees, count);
            fclose(f);
            return NULL;
        }

        /* Ensure capacity in the trees array */
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

/* Check if all trees in an array are ultrametric.
Returns 0 if all trees are ultrametric, -1 otherwise. */
int gex_check_trees_ultrametric(TreeNode **trees, int n_trees, double tol) {
    int i;

    if (trees == NULL || n_trees < 0 || tol < 0.0) {
        fprintf(stderr, "ERROR: gex_check_trees_ultrametric received invalid input\n");
        return -1;
    }

    for (i = 0; i < n_trees; i++) {
        double min_depth = 0.0;
        double max_depth = 0.0;
        int n_tips = 0;

        if (trees[i] == NULL)
            continue;

        gex_collect_tip_depth_range(trees[i], 0.0, &min_depth, &max_depth, &n_tips);
        if (n_tips == 0) {
            fprintf(stderr, "ERROR: tree %d has no tips\n", i + 1);
            return -1;
        }

        if (fabs(max_depth - min_depth) > tol) {
            fprintf(stderr,
                    "ERROR: tree %d is not ultrametric; min root-to-tip depth=%.12g max root-to-tip depth=%.12g diff=%.12g exceeds tolerance %.12g\n",
                    i + 1, min_depth, max_depth, fabs(max_depth - min_depth), tol);
            return -1;
        }
    }

    return 0;
}

int gex_rescale_trees_total_height(TreeNode **trees, int n_trees, double target_height) {
    int i;

    if (trees == NULL || n_trees < 0 || target_height <= 0.0) {
        fprintf(stderr, "ERROR: gex_rescale_trees_total_height received invalid input\n");
        return -1;
    }

    for (i = 0; i < n_trees; i++) {
        double min_depth = 0.0;
        double max_depth = 0.0;
        int n_tips = 0;
        double scale;

        if (trees[i] == NULL)
            continue;

        gex_collect_tip_depth_range(trees[i], 0.0, &min_depth, &max_depth, &n_tips);
        if (n_tips == 0) {
            fprintf(stderr, "ERROR: tree %d has no tips\n", i + 1);
            return -1;
        }
        if (max_depth <= 0.0) {
            fprintf(stderr, "ERROR: tree %d has non-positive total height and cannot be rescaled\n",
                    i + 1);
            return -1;
        }

        scale = target_height / max_depth;
        gex_scale_tree_recursive(trees[i], scale);
    }

    return 0;
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

/* Read in a labeled expression matrix from a tab-delimited file. 
Returns a pointer to the allocated matrix or NULL on failure. */
GexMatrix *gex_read_labeled_matrix(const char *filename) {
    FILE *f;
    char line[100000];
    GexMatrix *gex = NULL;  /* Pointer to the allocated gex matrix struct */
    int n_cells = 0;
    int n_genes = 0;

    if (filename == NULL) return NULL;

    f = fopen(filename, "r");
    if (f == NULL) {
        fprintf(stderr, "ERROR: could not open matrix file: %s\n", filename);
        return NULL;
    }

    /* First pass: Read header and allocate data structures based on input data dimensions */
    if (fgets(line, sizeof(line), f) == NULL) {
        fprintf(stderr, "ERROR: matrix file is empty: %s\n", filename);
        fclose(f);
        return NULL;
    }

    char *line_copy = gex_strdup(line);
    char **fields = NULL;
    int nfields, i;

    if (line_copy == NULL) {
        fclose(f);
        return NULL;
    }

    nfields = gex_split_tab_fields(line_copy, &fields); /* Split the header line into fields */
    if (nfields < 2) {
        fprintf(stderr, "ERROR: header must contain row label column plus at least one gene\n");
        free(line_copy);
        fclose(f);
        return NULL;
    }

    n_genes = nfields - 1;  /* Number of genes is the number of fields minus the row label first column */

    gex = (GexMatrix *)calloc(1, sizeof(GexMatrix));    /* Allocate matrix structure */
    if (gex == NULL) {
        free(fields);
        free(line_copy);
        fclose(f);
        return NULL;
    }

    gex->gene_names = (char **)malloc(n_genes * sizeof(char *));    /* Allocate array for gene names */
    if (gex->gene_names == NULL) {
        free(fields);
        free(line_copy);
        free(gex);
        fclose(f);
        return NULL;
    }

    for (i = 0; i < n_genes; i++) {
        gex->gene_names[i] = gex_strdup(fields[i + 1]); /* Duplicate gene name strings from header fields */
        if (gex->gene_names[i] == NULL) {
            free(fields);
            free(line_copy);
            fclose(f);
            return NULL;
        }
    }

    free(fields);
    free(line_copy);

    /* Count the number of cell rows */
    while (fgets(line, sizeof(line), f) != NULL) {
        char *trimmed = gex_lstrip(line); /* Strip leading whitespace from the line */
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

    gex->cell_names = (char **)malloc(n_cells * sizeof(char *));    /* Allocate array for cell names */
    if (gex->cell_names == NULL) {
        fclose(f);
        return NULL;
    }

    gex->X = mat_new(n_cells, n_genes);    /* Allocate expression matrix */
    if (gex->X == NULL) {
        fclose(f);
        return NULL;
    }

    /* Second pass: Fill data structures */
    rewind(f);

    /* Skip header */
    if (fgets(line, sizeof(line), f) == NULL) {
        fclose(f);
        return NULL;
    }

    int row = 0;

    while (fgets(line, sizeof(line), f) != NULL) {
        char *line_copy;
        char **fields = NULL;
        int nfields, j;

        char *trimmed = gex_lstrip(line); /* Strip leading whitespace from the line */
        if (*trimmed == '\0' || *trimmed == '\n')
            continue;

        line_copy = gex_strdup(line);   /* Duplicate the line for tokenization since strtok modifies the string */
        if (line_copy == NULL) {
            fclose(f);
            return NULL;
        }

        nfields = gex_split_tab_fields(line_copy, &fields); /* Split the line into fields */
        if (nfields != n_genes + 1) {
            fprintf(stderr, "ERROR: row %d has wrong number of columns in %s\n", row + 1, filename);
            free(line_copy);
            fclose(f);
            return NULL;
        }

        gex->cell_names[row] = gex_strdup(fields[0]);   /* Duplicate the cell name from the first field of the line */
        if (gex->cell_names[row] == NULL) {
            free(fields);
            free(line_copy);
            fclose(f);
            return NULL;
        }

        for (j = 0; j < n_genes; j++) {
            mat_set(gex->X, row, j, atof(fields[j + 1]));   /* Convert the expression value from string to double and store in the matrix */
        }

        free(fields);
        free(line_copy);
        row++;
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

/* Print a summary of the tree set and expr matrix i/o results */
void gex_print_io_summary(TreeNode **trees, int n_trees, GexMatrix *gex) {
    int i;
    int n_print = 10; /* Number of variable entries to print in each summary */

    printf("Loaded %d tree(s)\n", n_trees);

    if (gex != NULL && gex->X != NULL) {
        printf("Loaded matrix with %d cell(s) and %d gene(s)\n",
               gex->n_cells, gex->n_genes);

        printf("First few cell names:\n");
        for (i = 0; i < gex->n_cells && i < n_print; i++)
            printf("  %s\n", gex->cell_names[i]);

        printf("First few gene names:\n");
        for (i = 0; i < gex->n_genes && i < n_print; i++)
            printf("  %s\n", gex->gene_names[i]);

        printf("First few entries of matrix:\n");
        for (i = 0; i < gex->n_cells && i < n_print; i++) {
            int j;
            printf("  %s:", gex->cell_names[i]);
            for (j = 0; j < gex->n_genes && j < n_print; j++)
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

/* Reconcile the tree tip names with the expression matrix cell names.
Prune trees and subset matrix to the shared names. Updates the gex_ptr 
to point to the new subsetted matrix. Returns 0 on success, -1 on failure. */
int gex_reconcile_tree_and_expression(TreeNode **trees,
                                      int n_trees,
                                      GexMatrix **gex_ptr) {
    int i, j;
    int tree_missing_from_expr = 0; /* Number of tree tips missing from expression matrix */
    int expr_missing_from_tree = 0; /* Number of expression matrix cells missing from tree tips */
    int n_keep = 0; /* Number of shared names between tree tips and expression matrix cells */
    List *tree_names = NULL;
    List *keep_names = NULL;
    GexMatrix *gex;
    GexMatrix *subset = NULL;

    /* Check inputs are valid */
    if (trees == NULL || n_trees <= 0 || trees[0] == NULL ||
        gex_ptr == NULL || *gex_ptr == NULL) {
        fprintf(stderr, "ERROR: gex_reconcile_tree_and_expression got invalid input\n");
        return -1;
    }

    gex = *gex_ptr; /* Get the expression matrix pointer */
    tree_names = tr_leaf_names(trees[0]);   /* Collect the names of all tree tips */
    if (tree_names == NULL) {
        fprintf(stderr, "ERROR: failed to collect tree tip names\n");
        return -1;
    }

    keep_names = lst_new_ptr(gex->n_cells > 0 ? gex->n_cells : 1);  /* List to hold the names of the shared tree tips and expression matrix cells */
    if (keep_names == NULL) {
        gex_free_string_ptr_list(tree_names);
        return -1;
    }

    /* Check which tree tips are missing from the expression matrix */
    for (i = 0; i < lst_size(tree_names); i++) {
        String *s = lst_get_ptr(tree_names, i);
        if (!gex_name_in_char_array(s->chars, gex->cell_names, gex->n_cells))
            tree_missing_from_expr++;
    }

    /* Check which expression matrix cells are missing from the tree tips */
    for (i = 0; i < gex->n_cells; i++) {
        if (!gex_name_in_string_list(gex->cell_names[i], tree_names))
            expr_missing_from_tree++;
        else {
            String *s = str_new_charstr(gex->cell_names[i]);
            if (s == NULL) {
                gex_free_string_ptr_list(tree_names);
                gex_free_string_ptr_list(keep_names);
                return -1;
            }
            lst_push_ptr(keep_names, s);
            n_keep++;
        }
    }

    if (tree_missing_from_expr > 0 || expr_missing_from_tree > 0) {
        fprintf(stderr,
                "WARNING: tree/expression names do not match perfectly; %d tree tip(s) missing from expression matrix and %d expression cell(s) missing from tree. Using the %d shared name(s).\n",
                tree_missing_from_expr, expr_missing_from_tree, n_keep);
    }

    if (n_keep <= 0) {
        fprintf(stderr, "ERROR: no shared names between first tree tips and expression matrix\n");
        gex_free_string_ptr_list(tree_names);
        gex_free_string_ptr_list(keep_names);
        return -1;
    }

    subset = (GexMatrix *)calloc(1, sizeof(GexMatrix)); /* Allocate memory for the subsetted matrix */
    if (subset == NULL) {
        gex_free_string_ptr_list(tree_names);
        gex_free_string_ptr_list(keep_names);
        return -1;
    }

    /* Initialize the subsetted matrix */
    subset->n_cells = n_keep;
    subset->n_genes = gex->n_genes;
    subset->X = mat_new(n_keep, gex->n_genes);
    subset->cell_names = (char **)calloc(n_keep, sizeof(char *));
    subset->gene_names = (char **)calloc(gex->n_genes, sizeof(char *));
    if (subset->X == NULL || subset->cell_names == NULL || subset->gene_names == NULL) {
        gex_free_matrix_data(subset);
        gex_free_string_ptr_list(tree_names);
        gex_free_string_ptr_list(keep_names);
        return -1;
    }

    /* Fill the gene names in the subsetted matrix (same as original since we keep all genes) */
    for (j = 0; j < gex->n_genes; j++) {
        subset->gene_names[j] = gex_strdup(gex->gene_names[j]);
        if (subset->gene_names[j] == NULL) {
            gex_free_matrix_data(subset);
            gex_free_string_ptr_list(tree_names);
            gex_free_string_ptr_list(keep_names);
            return -1;
        }
    }

    /* Fill the cell names and expression values in the subsetted matrix for the shared names */
    for (i = 0, j = 0; i < gex->n_cells; i++) {
        if (gex_name_in_string_list(gex->cell_names[i], tree_names)) {
            int g;
            subset->cell_names[j] = gex_strdup(gex->cell_names[i]);
            if (subset->cell_names[j] == NULL) {
                gex_free_matrix_data(subset);
                gex_free_string_ptr_list(tree_names);
                gex_free_string_ptr_list(keep_names);
                return -1;
            }
            for (g = 0; g < gex->n_genes; g++)
                mat_set(subset->X, j, g, mat_get(gex->X, i, g));
            j++;
        }
    }

    /* Prune the trees to keep only the shared names */
    for (i = 0; i < n_trees; i++) {
        if (trees[i] != NULL)
            tr_prune(&trees[i], keep_names, 1, NULL);
    }

    gex_free_matrix_data(gex);
    *gex_ptr = subset;

    gex_free_string_ptr_list(tree_names);
    gex_free_string_ptr_list(keep_names);
    return 0;
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
    for (j = 0; j < res->n_genes && j < 10; j++) {
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

GexLRTResult *gex_compute_brownian_lrt(GexMatrix *gex,
                                       Matrix *Sigma,
                                       GexLRTNullMode null_mode,
                                       int n_mc,
                                       unsigned int seed) {
    int i, j;
    int n;
    Matrix *Sigma_reg = NULL;
    Matrix *Sigma_inv = NULL;
    Matrix *L = NULL;
    GexLRTResult *res = NULL;
    double logdet_sigma = 0.0;
    double max_diag = 0.0;
    double jitter;
    double *y = NULL;
    double *y_sim = NULL;
    unsigned int rng_state;

    if (gex == NULL || gex->X == NULL || Sigma == NULL ||
        Sigma->nrows != Sigma->ncols || Sigma->nrows != gex->n_cells) {
        fprintf(stderr, "ERROR: gex_compute_brownian_lrt got invalid input\n");
        return NULL;
    }
    if (null_mode == GEX_LRT_NULL_MONTECARLO && n_mc <= 0) {
        fprintf(stderr, "ERROR: Monte Carlo LRT requires positive n_mc\n");
        return NULL;
    }

    n = gex->n_cells;
    Sigma_reg = mat_new(n, n);
    Sigma_inv = mat_new(n, n);
    L = mat_new(n, n);
    if (Sigma_reg == NULL || Sigma_inv == NULL || L == NULL) {
        if (Sigma_reg != NULL) mat_free(Sigma_reg);
        if (Sigma_inv != NULL) mat_free(Sigma_inv);
        if (L != NULL) mat_free(L);
        return NULL;
    }

    for (i = 0; i < n; i++) {
        double d = mat_get(Sigma, i, i);
        if (d > max_diag)
            max_diag = d;
    }
    jitter = (max_diag > 0.0 ? 1e-8 * max_diag : 1e-8);

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++)
            mat_set(Sigma_reg, i, j, mat_get(Sigma, i, j));
        mat_set(Sigma_reg, i, i, mat_get(Sigma_reg, i, i) + jitter);
    }

    if (mat_invert(Sigma_inv, Sigma_reg) != 0) {
        fprintf(stderr, "ERROR: failed to invert Brownian covariance matrix for LRT\n");
        mat_free(Sigma_reg);
        mat_free(Sigma_inv);
        mat_free(L);
        return NULL;
    }
    if (mat_cholesky(L, Sigma_reg) != 0) {
        fprintf(stderr, "ERROR: failed to Cholesky Brownian covariance matrix for LRT\n");
        mat_free(Sigma_reg);
        mat_free(Sigma_inv);
        mat_free(L);
        return NULL;
    }

    for (i = 0; i < n; i++) {
        double diag = mat_get(L, i, i);
        if (diag <= 0.0) {
            mat_free(Sigma_reg);
            mat_free(Sigma_inv);
            mat_free(L);
            return NULL;
        }
        logdet_sigma += 2.0 * log(diag);
    }

    res = (GexLRTResult *)calloc(1, sizeof(GexLRTResult));
    y = (double *)malloc(n * sizeof(double));
    y_sim = (double *)malloc(n * sizeof(double));
    if (res == NULL || y == NULL || y_sim == NULL) {
        free(y);
        free(y_sim);
        free(res);
        mat_free(Sigma_reg);
        mat_free(Sigma_inv);
        mat_free(L);
        return NULL;
    }

    res->lrt_stat = (double *)calloc(gex->n_genes, sizeof(double));
    res->pvals = (double *)calloc(gex->n_genes, sizeof(double));
    res->qvals = (double *)calloc(gex->n_genes, sizeof(double));
    res->n_genes = gex->n_genes;
    if (res->lrt_stat == NULL || res->pvals == NULL || res->qvals == NULL) {
        gex_free_lrt_result(res);
        free(y);
        free(y_sim);
        mat_free(Sigma_reg);
        mat_free(Sigma_inv);
        mat_free(L);
        return NULL;
    }

    rng_state = (seed == 0u ? 1u : seed);
    for (j = 0; j < gex->n_genes; j++) {
        double ll_null;
        double ll_alt;
        double mu0;
        double sigma20;
        for (i = 0; i < n; i++)
            y[i] = mat_get(gex->X, i, j);

        gex_fit_gaussian_identity(y, n, &mu0, &sigma20);
        ll_null = gex_loglik_centered_gaussian_identity(y, n);
        ll_alt = gex_loglik_centered_gaussian_cov(y, Sigma_reg, Sigma_inv, logdet_sigma);
        res->lrt_stat[j] = 2.0 * (ll_alt - ll_null);
        if (res->lrt_stat[j] < 0.0 && fabs(res->lrt_stat[j]) < 1e-10)
            res->lrt_stat[j] = 0.0;
        if (res->lrt_stat[j] < 0.0)
            res->lrt_stat[j] = 0.0;

        if (null_mode == GEX_LRT_NULL_CHI2) {
            res->pvals[j] = gex_chisq1_sf(res->lrt_stat[j]);
        }
        else {
            int ge_count = 0;
            int rep;
            for (rep = 0; rep < n_mc; rep++) {
                double ll0_sim;
                double ll1_sim;
                double stat_sim;
                for (i = 0; i < n; i++)
                    y_sim[i] = mu0 + sqrt(sigma20) * gex_rand_normal(&rng_state);
                ll0_sim = gex_loglik_centered_gaussian_identity(y_sim, n);
                ll1_sim = gex_loglik_centered_gaussian_cov(y_sim, Sigma_reg, Sigma_inv, logdet_sigma);
                stat_sim = 2.0 * (ll1_sim - ll0_sim);
                if (stat_sim < 0.0 && fabs(stat_sim) < 1e-10)
                    stat_sim = 0.0;
                if (stat_sim < 0.0)
                    stat_sim = 0.0;
                if (stat_sim >= res->lrt_stat[j])
                    ge_count++;
            }
            res->pvals[j] = ((double)ge_count + 1.0) / ((double)n_mc + 1.0);
        }
    }

    gex_bh_adjust(res->pvals, res->qvals, res->n_genes);
    res->n_significant = gex_count_kept_lrt_genes(res, 0.05);

    free(y);
    free(y_sim);
    mat_free(Sigma_reg);
    mat_free(Sigma_inv);
    mat_free(L);
    return res;
}

void gex_print_lrt_summary(GexLRTResult *res,
                           GexMatrix *gex,
                           double max_q) {
    int j;

    if (res == NULL || gex == NULL) {
        fprintf(stderr, "ERROR: cannot summarize NULL LRT result\n");
        return;
    }

    printf("Computed Brownian LRT for %d gene(s)\n", res->n_genes);
    printf("Genes passing LRT filter (q <= %.4f): %d\n",
           max_q, gex_count_kept_lrt_genes(res, max_q));
    printf("\nFirst few gene-wise Brownian LRT statistics:\n");
    printf("gene\tlrt\tp\tq\tkeep\n");
    for (j = 0; j < res->n_genes && j < 10; j++) {
        int keep = (res->qvals[j] <= max_q && res->lrt_stat[j] > 0.0);
        printf("%s\t%g\t%g\t%g\t%s\n",
               gex->gene_names[j],
               res->lrt_stat[j],
               res->pvals[j],
               res->qvals[j],
               (keep ? "yes" : "no"));
    }
    printf("\n");
}

int gex_write_lrt_tsv(const char *filename,
                      GexLRTResult *res,
                      GexMatrix *gex,
                      double max_q) {
    FILE *out;
    int i;

    if (filename == NULL || res == NULL || gex == NULL) {
        fprintf(stderr, "ERROR: gex_write_lrt_tsv got invalid input\n");
        return -1;
    }

    out = fopen(filename, "w");
    if (out == NULL) {
        fprintf(stderr, "ERROR: could not open LRT output file: %s\n", filename);
        return -1;
    }

    fprintf(out, "gene\tlrt_stat\tp_value\tq_value\tkeep\n");
    for (i = 0; i < res->n_genes; i++) {
        int keep = (res->qvals[i] <= max_q && res->lrt_stat[i] > 0.0);
        fprintf(out, "%s\t%.17g\t%.17g\t%.17g\t%s\n",
                gex->gene_names[i],
                res->lrt_stat[i],
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

void gex_free_lrt_result(GexLRTResult *res) {
    if (res == NULL)
        return;

    if (res->lrt_stat != NULL)
        free(res->lrt_stat);
    if (res->pvals != NULL)
        free(res->pvals);
    if (res->qvals != NULL)
        free(res->qvals);

    free(res);
}

GexMatrix *gex_filter_genes_by_results(GexMatrix *gex,
                                       GexMoransResult *morans,
                                       GexLRTResult *lrt,
                                       GexFilterMode mode,
                                       double max_q,
                                       double min_i) {
    int i, j;
    int out_j = 0;
    int nkeep = 0;
    GexMatrix *out = NULL;

    if (gex == NULL)
        return NULL;
    if ((mode == GEX_FILTER_MORAN || mode == GEX_FILTER_BOTH) &&
        (morans == NULL || gex->n_genes != morans->n_genes))
        return NULL;
    if ((mode == GEX_FILTER_LRT || mode == GEX_FILTER_BOTH) &&
        (lrt == NULL || gex->n_genes != lrt->n_genes))
        return NULL;

    for (j = 0; j < gex->n_genes; j++) {
        if (gex_keep_gene(morans, lrt, j, mode, max_q, min_i))
            nkeep++;
    }
    if (nkeep <= 0) {
        fprintf(stderr, "ERROR: selected gene filter removed all genes\n");
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
        if (gex_keep_gene(morans, lrt, j, mode, max_q, min_i)) {
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
