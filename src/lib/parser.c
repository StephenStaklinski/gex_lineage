#include "parser.h"

#include "gexmatrix.h"

#include <phast/trees.h>
#include <phast/matrix.h>
#include <phast/lists.h>
#include <phast/stringsplus.h>
#include <phast/misc.h>
#include <phast/vector.h>

#include <sys/types.h>
#include <string.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


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

static int gex_append_text(char **buf, size_t *len, size_t *capacity, const char *text) {
    size_t text_len;
    size_t needed;

    if (buf == NULL || len == NULL || capacity == NULL || text == NULL)
        return -1;

    text_len = strlen(text);
    needed = *len + text_len + 1;

    if (needed > *capacity) {
        size_t new_capacity = (*capacity == 0 ? 256 : *capacity);

        while (needed > new_capacity)
            new_capacity *= 2;

        *buf = srealloc(*buf, new_capacity * sizeof(char));
        *capacity = new_capacity;
    }

    memcpy(*buf + *len, text, text_len + 1);
    *len += text_len;
    return 0;
}

/* Extract a Newick string from a NEXUS tree line.
Returns a pointer to the extracted string or NULL on failure. */
static char *gex_extract_newick_from_tree_line(const char *line) {
    const char *eq;
    const char *end;
    char *tmp;
    char *s;
    char *out;
    size_t out_len;

    if (line == NULL) return NULL;

    eq = strchr(line, '='); /* Find the '=' character that separates the tree name from the Newick string */
    if (eq == NULL) return NULL;

    tmp = strdup(eq + 1);   /* Duplicate the substring after '=' for manipulation */
    if (tmp == NULL) return NULL;

    s = gex_lstrip(tmp);    /* Strip leading whitespace */
    gex_rstrip_inplace(s);  /* Strip trailing whitespace */

    /* Strip any leading '[&' and trailing ']' annotation characters */
    if (strncmp(s, "[&R]", 4) == 0 || strncmp(s, "[&U]", 4) == 0) {
        s += 4;
        s = gex_lstrip(s);
    }

    end = strchr(s, ';');
    out_len = (end == NULL ? strlen(s) : (size_t)(end - s + 1));
    out = smalloc((out_len + 1) * sizeof(char));
    memcpy(out, s, out_len);
    out[out_len] = '\0';
    free(tmp);
    return out;
}

/* Split a TSV line while preserving empty fields.
Modifies the input string in place by replacing delimiters with '\0'.
Returns number of fields on success, -1 on failure. */
static int gex_split_tab_fields(char *line, char ***fields_out) {
    int count = 0;
    int capacity = 8;
    char **fields = NULL;
    char *p, *field_start;

    if (line == NULL || fields_out == NULL)
        return -1;

    fields = smalloc(capacity * sizeof(char *));
    if (fields == NULL)
        return -1;

    /* Trim trailing newline / carriage return characters only */
    gex_rstrip_inplace(line);

    p = line;
    field_start = p;

    while (1) {
        if (*p == '\t' || *p == '\0') {
            if (count == capacity) {
                capacity *= 2;
                fields = srealloc(fields, capacity * sizeof(char *));
                if (fields == NULL)
                    return -1;
            }

            fields[count++] = field_start;

            if (*p == '\0')
                break;

            *p = '\0';
            p++;
            field_start = p;
        }
        else {
            p++;
        }
    }

    *fields_out = fields;
    return count;
}

/* Read up to max_trees trees from a NEXUS file. A max_trees value of -1
means that every tree should be read.

TODO: Make the function map names from the nexus file header
the trees block since many NEXUS files have renamed taxa. */
TreeNode **read_nexus(const char *filename, int *n_trees, int max_trees) {
    FILE *f;
    char line[4096];
    char *tree_record = NULL;
    TreeNode **trees = NULL;    /* Array of tree pointers to fill */
    size_t tree_record_len = 0;
    size_t tree_record_capacity = 0;
    int capacity = 0;
    int count = 0;
    int collecting_tree = 0;

    if (n_trees == NULL || filename == NULL ||
        (max_trees != -1 && max_trees <= 0))
        return NULL;

    *n_trees = 0;   /* Reset the number of trees to 0 */

    f = fopen(filename, "r");
    if (f == NULL) {
        fprintf(stderr, "ERROR: could not open NEXUS file: %s\n", filename);
        return NULL;
    }

    while (fgets(line, sizeof(line), f) != NULL) {
        char *trimmed;

        trimmed = gex_lstrip(line); /* Strip leading whitespace */
        gex_rstrip_inplace(trimmed); /* Remove line-ending whitespace before buffering wrapped trees */

        if (!collecting_tree) {
            if (!gex_starts_with_tree_keyword(trimmed))
                continue;   /* Skip lines that don't start with "TREE" */

            tree_record_len = 0;
            if (gex_append_text(&tree_record, &tree_record_len, &tree_record_capacity, trimmed) != 0) {
                fprintf(stderr, "ERROR: failed to buffer TREE line from file: %s\n", filename);
                gex_free_trees(trees, count);
                fclose(f);
                return NULL;
            }
            collecting_tree = 1;
        }
        else {
            if (gex_append_text(&tree_record, &tree_record_len, &tree_record_capacity, trimmed) != 0) {
                fprintf(stderr, "ERROR: failed to buffer wrapped TREE line from file: %s\n", filename);
                free(tree_record);
                gex_free_trees(trees, count);
                fclose(f);
                return NULL;
            }
        }

        if (strchr(tree_record, ';') == NULL)
            continue;

        {
            char *newick = gex_extract_newick_from_tree_line(tree_record);    /* Get Newick string */
            TreeNode *tree;

            collecting_tree = 0;

            if (newick == NULL)
                continue;

            tree = tr_new_from_string(newick);  /* Parse the Newick string into a tree structure */
            free(newick);

            /* Check if tree parsing was successful */
            if (tree == NULL) {
                fprintf(stderr, "ERROR: failed to parse tree from file: %s\n", filename);
                free(tree_record);
                gex_free_trees(trees, count);
                fclose(f);
                return NULL;
            }

            /* Ensure capacity in the trees array */
            if (count == capacity) {
                int new_capacity = (capacity == 0 ? 8 : 2 * capacity);
                TreeNode **tmp = srealloc(trees, new_capacity * sizeof(TreeNode *));
                trees = tmp;
                capacity = new_capacity;
            }

            trees[count++] = tree;
            if (max_trees != -1 && count >= max_trees)
                break;
        }
    }

    fclose(f);

    if (collecting_tree) {
        fprintf(stderr, "ERROR: unterminated TREE entry in NEXUS file: %s\n", filename);
        free(tree_record);
        gex_free_trees(trees, count);
        return NULL;
    }

    free(tree_record);

    if (count == 0) {
        fprintf(stderr, "ERROR: no TREE lines found in NEXUS file: %s\n", filename);
        free(trees);
        return NULL;
    }

    *n_trees = count;

    return trees;
}

static int is_leaf(TreeNode *node) {
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
    if (is_leaf(node)) {
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

/* Check if all trees in an array are ultrametric.
Returns 0 if all trees are ultrametric, -1 otherwise. */
int check_trees_ultrametric(TreeNode **trees, int n_trees) {
    int i;
    double tol = 1e-3;   /* Tolerance for total height comparisons */

    if (trees == NULL || n_trees < 0 || tol < 0.0) {
        fprintf(stderr, "ERROR: check_trees_ultrametric received invalid input\n");
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

/* Scale the distances in a tree uniformly by a given factor. */
static void gex_scale_tree_recursive(TreeNode *node, double scale) {
    if (node == NULL)
        return;

    if (node->parent != NULL)
        node->dparent *= scale;

    gex_scale_tree_recursive(node->lchild, scale);
    gex_scale_tree_recursive(node->rchild, scale);
}

void uniform_rescale_trees(TreeNode **trees, int n_trees, double target_height) {
    int i;

    if (trees == NULL || n_trees < 0 || target_height <= 0.0) {
        fprintf(stderr, "ERROR: uniform_rescale_trees received invalid input\n");
        return;
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
            return;
        }
        if (max_depth <= 0.0) {
            fprintf(stderr, "ERROR: tree %d has non-positive total height and cannot be rescaled\n",
                    i + 1);
            return;
        }

        scale = target_height / max_depth;
        gex_scale_tree_recursive(trees[i], scale);
    }
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
GexMatrix *read_gex_matrix(const char *filename) {
    FILE *f;
    char *line = NULL;
    size_t linecap = 0;
    ssize_t linelen;
    GexMatrix *gex = NULL;
    int n_cells = 0;
    int n_genes = 0;
    int i;

    if (filename == NULL)
        return NULL;

    f = fopen(filename, "r");
    if (f == NULL) {
        fprintf(stderr, "ERROR: could not open matrix file: %s\n", filename);
        return NULL;
    }

    /* First pass: read header */
    linelen = getline(&line, &linecap, f);
    if (linelen < 0) {
        fprintf(stderr, "ERROR: matrix file is empty: %s\n", filename);
        fclose(f);
        free(line);
        return NULL;
    }

    {
        char *line_copy = strdup(line);
        char **fields = NULL;
        int nfields;

        if (line_copy == NULL) {
            fclose(f);
            free(line);
            return NULL;
        }

        nfields = gex_split_tab_fields(line_copy, &fields);
        if (nfields < 2) {
            fprintf(stderr, "ERROR: header must contain row label column plus at least one gene\n");
            free(fields);
            free(line_copy);
            fclose(f);
            free(line);
            return NULL;
        }

        n_genes = nfields - 1;

        gex = scalloc(1, sizeof(GexMatrix));
        if (gex == NULL) {
            free(fields);
            free(line_copy);
            fclose(f);
            free(line);
            return NULL;
        }

        gex->gene_names = scalloc(n_genes, sizeof(char *));
        if (gex->gene_names == NULL) {
            free(fields);
            free(line_copy);
            gex_free_matrix_data(gex);
            fclose(f);
            free(line);
            return NULL;
        }

        for (i = 0; i < n_genes; i++) {
            gex->gene_names[i] = strdup(fields[i + 1]);
            if (gex->gene_names[i] == NULL) {
                free(fields);
                free(line_copy);
                gex_free_matrix_data(gex);
                fclose(f);
                free(line);
                return NULL;
            }
        }

        free(fields);
        free(line_copy);
    }

    /* Count the number of non-empty data rows */
    while ((linelen = getline(&line, &linecap, f)) >= 0) {
        char *trimmed = gex_lstrip(line);
        if (*trimmed == '\0' || *trimmed == '\n' || *trimmed == '\r')
            continue;
        n_cells++;
    }

    if (n_cells == 0) {
        fprintf(stderr, "ERROR: no data rows found in matrix file: %s\n", filename);
        gex_free_matrix_data(gex);
        fclose(f);
        free(line);
        return NULL;
    }

    gex->cell_names = scalloc(n_cells, sizeof(char *));
    gex->X = mat_new(n_cells, n_genes);

    if (gex->cell_names == NULL || gex->X == NULL) {
        gex_free_matrix_data(gex);
        fclose(f);
        free(line);
        return NULL;
    }

    /* Second pass: fill data */
    rewind(f);

    linelen = getline(&line, &linecap, f);  /* skip header */
    if (linelen < 0) {
        gex_free_matrix_data(gex);
        fclose(f);
        free(line);
        return NULL;
    }

    {
        int row = 0;

        while ((linelen = getline(&line, &linecap, f)) >= 0) {
            char *trimmed = gex_lstrip(line);
            char *line_copy;
            char **fields = NULL;
            int nfields, j;

            if (*trimmed == '\0' || *trimmed == '\n' || *trimmed == '\r')
                continue;

            line_copy = strdup(line);
            if (line_copy == NULL) {
                gex_free_matrix_data(gex);
                fclose(f);
                free(line);
                return NULL;
            }

            nfields = gex_split_tab_fields(line_copy, &fields);
            if (nfields != n_genes + 1) {
                fprintf(stderr,
                        "ERROR: row %d has wrong number of columns in %s. Expected %d, found %d\n",
                        row + 1, filename, n_genes + 1, nfields);
                free(fields);
                free(line_copy);
                gex_free_matrix_data(gex);
                fclose(f);
                free(line);
                return NULL;
            }

            gex->cell_names[row] = strdup(fields[0]);
            if (gex->cell_names[row] == NULL) {
                free(fields);
                free(line_copy);
                gex_free_matrix_data(gex);
                fclose(f);
                free(line);
                return NULL;
            }

            for (j = 0; j < n_genes; j++) {
                char *endptr;
                double val;

                endptr = NULL;
                val = strtod(fields[j + 1], &endptr);

                if (endptr == fields[j + 1] || *endptr != '\0') {
                    fprintf(stderr,
                            "ERROR: row %d, gene column %d contains a non-numeric value: '%s'\n",
                            row + 1, j + 2, fields[j + 1]);
                    free(fields);
                    free(line_copy);
                    gex_free_matrix_data(gex);
                    fclose(f);
                    free(line);
                    return NULL;
                }

                mat_set(gex->X, row, j, val);
            }

            free(fields);
            free(line_copy);
            row++;
        }
    }

    fclose(f);
    free(line);

    return gex;
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

static List *gex_copy_string_ptr_list(List *l) {
    int i;
    List *copy;

    if (l == NULL)
        return NULL;

    copy = lst_new_ptr(lst_size(l) > 0 ? lst_size(l) : 1);
    for (i = 0; i < lst_size(l); i++) {
        String *s = lst_get_ptr(l, i);
        String *s_copy = str_new_charstr(s->chars);
        if (s_copy == NULL) {
            gex_free_string_ptr_list(copy);
            return NULL;
        }
        lst_push_ptr(copy, s_copy);
    }

    return copy;
}

/* Reconcile the tree tip names with the expression matrix cell names.
Prune trees and subset matrix to the shared names. Updates the gex_ptr 
to point to the new subsetted matrix. Returns 0 on success, -1 on failure. */
int gex_reconcile_tree_and_expression(TreeNode **trees,
                                      int n_trees,
                                      GexMatrix **gex_ptr) {
    int i, j;
    int tree_missing_from_expr = 0; /* Total number of tree tips, across all trees, missing from the expression matrix */
    int expr_missing_from_tree = 0; /* Number of expression matrix cells missing from at least one tree */
    int n_keep = 0; /* Number of names shared between the expression matrix and all trees */
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

    /* Get the leaf names from the first tree, assuming all trees have the same tips */
    tree_names = tr_leaf_names(trees[0]);

    /* List to hold the names of the shared tree tips and expression matrix cells */
    keep_names = lst_new_ptr(gex->X->nrows > 0 ? gex->X->nrows : 1);

    /* Check which tree tips are missing from the expression matrix */
    for (j = 0; j < lst_size(tree_names); j++) {
        String *s = lst_get_ptr(tree_names, j);
        if (!gex_name_in_char_array(s->chars, gex->cell_names, gex->X->nrows))
            tree_missing_from_expr++;
    }

    /* Keep only expression cells that are present in the tree */
    for (i = 0; i < gex->X->nrows; i++) {

        if (!gex_name_in_string_list(gex->cell_names[i], tree_names)) {
            expr_missing_from_tree++;
            continue;
        }
        
        String *s = str_new_charstr(gex->cell_names[i]);
        if (s == NULL) {
            return 1;
        }
        lst_push_ptr(keep_names, s);
        n_keep++;
    }

    if (tree_missing_from_expr > 0 || expr_missing_from_tree > 0) {
        printf("WARNING: tree/expression names do not match perfectly; %d tree tip occurrence(s) are missing from the expression matrix and %d expression cell(s) are missing from the first tree. Using the %d shared names.\n",
                tree_missing_from_expr, expr_missing_from_tree, n_keep);
    }

    if (n_keep <= 0) {
        fprintf(stderr, "ERROR: no shared names between the expression matrix and all trees\n");
        return 1;
    }

    /* Initialize the subsetted matrix */
    subset = scalloc(1, sizeof(GexMatrix)); /* Allocate memory for the subsetted matrix */
    subset->X = mat_new(n_keep, gex->X->ncols);
    subset->cell_names = scalloc(n_keep, sizeof(char *));
    subset->gene_names = scalloc(gex->X->ncols, sizeof(char *));

    /* Fill the gene names in the subsetted matrix (same as original since we keep all genes) */
    for (j = 0; j < gex->X->ncols; j++) {
        subset->gene_names[j] = strdup(gex->gene_names[j]);
        if (subset->gene_names[j] == NULL) {
            gex_free_matrix_data(subset);
            subset = NULL;
            return 1;
        }
    }

    /* Fill the cell names and expression values in the subsetted matrix for the shared names */
    for (i = 0, j = 0; i < gex->X->nrows; i++) {
        if (gex_name_in_string_list(gex->cell_names[i], keep_names)) {
            int g;
            subset->cell_names[j] = strdup(gex->cell_names[i]);
            for (g = 0; g < gex->X->ncols; g++)
                mat_set(subset->X, j, g, mat_get(gex->X, i, g));
            j++;
        }
    }

    /* Prune the trees */
    if (tree_missing_from_expr > 0) {
        for (i = 0; i < n_trees; i++) {
            if (trees[i] != NULL) {
                List *tree_keep_names = gex_copy_string_ptr_list(keep_names);
                tr_prune(&trees[i], tree_keep_names, 1, NULL);
                gex_free_string_ptr_list(tree_keep_names);
            }
        }
    }

    *gex_ptr = subset;

    /* Free memory */
    gex_free_matrix_data(gex);
    gex_free_string_ptr_list(tree_names);
    gex_free_string_ptr_list(keep_names);

    return 0;
}

int load_and_reconcile_tree_gex_inputs(const char *trees_file,
                                       const char *expr_file,
                                       int max_trees,
                                       TreeNode ***trees_out,
                                       int *n_trees_out,
                                       GexMatrix **gex_out) {
    TreeNode **trees = NULL;
    GexMatrix *gex = NULL;
    int n_trees = 0;

    if (trees_file == NULL || expr_file == NULL || trees_out == NULL ||
        n_trees_out == NULL || gex_out == NULL)
        return -1;

    *trees_out = NULL;
    *n_trees_out = 0;
    *gex_out = NULL;

    trees = read_nexus(trees_file, &n_trees, max_trees);
    if (trees == NULL || n_trees <= 0 ||
        check_trees_ultrametric(trees, n_trees) != 0) {
        gex_free_trees(trees, n_trees);
        return -1;
    }
    printf("Loaded %d tree(s).\n", n_trees);

    gex = read_gex_matrix(expr_file);
    if (gex == NULL) {
        gex_free_trees(trees, n_trees);
        return -1;
    }
    printf("Loaded matrix with %d cell(s) and %d gene(s).\n",
           gex->X->nrows, gex->X->ncols);

    if (gex_reconcile_tree_and_expression(trees, n_trees, &gex) != 0) {
        fprintf(stderr,
                "ERROR: failed to reconcile tree tips and expression cell names.\n");
        gex_free_matrix_data(gex);
        gex_free_trees(trees, n_trees);
        return -1;
    }

    uniform_rescale_trees(trees, n_trees, 1.0);

    *trees_out = trees;
    *n_trees_out = n_trees;
    *gex_out = gex;
    return 0;
}

Vector *parse_csv_to_vec(const char *csv_string) {
    Vector *v = NULL;
    int count = 0;
    int idx = 0;
    char *copy = NULL;
    char *token = NULL;

    if (!csv_string || *csv_string == '\0') {
        v = vec_new(1);
        return v;
    }

    /* First pass to count components */
    copy = strdup(csv_string);
    if (!copy)
        return NULL;
    
    token = strtok(copy, ",");
    while (token) {
        count++;
        token = strtok(NULL, ",");
    }
    free(copy);

    /* Initialize vector with exact capacity */
    v = vec_new(count);

    /* Second pass to fill vector */
    copy = strdup(csv_string);
    if (!copy) {
        vec_free(v);
        return NULL;
    }

    token = strtok(copy, ",");
    while (token) {
        vec_set(v, idx++, strtod(token, NULL));
        token = strtok(NULL, ",");
    }
    free(copy);

    return v;
}
