#include "gexparser.h"

#include "gexmatrix.h"

#include <phast/trees.h>
#include <phast/matrix.h>
#include <phast/lists.h>
#include <phast/stringsplus.h>
#include <phast/misc.h>

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

/* Split a line on tabs/newlines only. Sets the fields_out pointer to an array of field strings.
Returns number of fields if successful, -1 on failure. */
static int gex_split_tab_fields(char *line, char ***fields_out) {
    int count = 0;
    int capacity = 8;
    char **fields = NULL;
    char *token = NULL;

    fields = smalloc(capacity * sizeof(char *));

    token = strtok(line, "\t\r\n");
    while (token != NULL) {
        if (count == capacity) {
            char **tmp;
            capacity *= 2;
            tmp = srealloc(fields, capacity * sizeof(char *));
            fields = tmp;
        }

        fields[count++] = token;
        token = strtok(NULL, "\t\r\n");
    }

    *fields_out = fields;
    return count;
}

/* Read in NEXUS tree file and parse trees.
Updates the n_trees pointer. Returns a pointer to the 
array of tree pointers or NULL on failure. 

TODO: Make the function map names from the nexus file header
the trees block since many NEXUS files have renamed taxa. */
TreeNode **gex_read_nexus(const char *filename, int *n_trees) {
    FILE *f;
    char line[4096];
    char *tree_record = NULL;
    TreeNode **trees = NULL;    /* Array of tree pointers to fill */
    size_t tree_record_len = 0;
    size_t tree_record_capacity = 0;
    int capacity = 0;
    int count = 0;
    int collecting_tree = 0;

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

    char *line_copy = strdup(line);
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

    gex = scalloc(1, sizeof(GexMatrix));    /* Allocate matrix structure */

    gex->gene_names = scalloc(n_genes, sizeof(char *));    /* Allocate array for gene names */
    for (i = 0; i < n_genes; i++) {
        gex->gene_names[i] = strdup(fields[i + 1]); /* Duplicate gene name strings from header fields */
        if (gex->gene_names[i] == NULL) {
            free(fields);
            free(line_copy);
            gex_free_matrix_data(gex);
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

    gex->cell_names = scalloc(n_cells, sizeof(char *));    /* Allocate array for cell names */
    gex->X = mat_new(n_cells, n_genes);    /* Allocate expression matrix */

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

        line_copy = strdup(line);   /* Duplicate the line for tokenization since strtok modifies the string */
        if (line_copy == NULL) {
            gex_free_matrix_data(gex);
            fclose(f);
            return NULL;
        }

        nfields = gex_split_tab_fields(line_copy, &fields); /* Split the line into fields */
        if (nfields != n_genes + 1) {
            fprintf(stderr, "ERROR: row %d has wrong number of columns in %s\n", row + 1, filename);
            free(line_copy);
            gex_free_matrix_data(gex);
            fclose(f);
            return NULL;
        }

        gex->cell_names[row] = strdup(fields[0]);   /* Duplicate the cell name from the first field of the line */
        if (gex->cell_names[row] == NULL) {
            free(fields);
            free(line_copy);
            gex_free_matrix_data(gex);
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

/* Print a summary of the tree set and expr matrix i/o results */
void gex_print_io_summary(TreeNode **trees, int n_trees, GexMatrix *gex) {
    int i;
    int n_print = 10; /* Number of variable entries to print in each summary */

    if (gex != NULL && gex->X != NULL) {
        printf("\n");
        printf("First few cell names:\n");
        for (i = 0; i < gex->X->nrows && i < n_print; i++)
            printf("  %s\n", gex->cell_names[i]);

        printf("First few gene names:\n");
        for (i = 0; i < gex->X->ncols && i < n_print; i++)
            printf("  %s\n", gex->gene_names[i]);

        printf("First few entries of matrix:\n");
        for (i = 0; i < gex->X->nrows && i < n_print; i++) {
            int j;
            printf("  %s:", gex->cell_names[i]);
            for (j = 0; j < gex->X->ncols && j < n_print; j++)
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

/* Reconcile the tree tip names with the expression matrix cell names.
Prune trees and subset matrix to the shared names. Updates the gex_ptr 
to point to the new subsetted matrix. Returns 0 on success, -1 on failure. */
int gex_reconcile_tree_and_expression(TreeNode **trees,
                                      int n_trees,
                                      GexMatrix **gex_ptr) {
    int i, j;
    int prune_needed = 0; /* Whether any tree or expression names require pruning/subsetting */
    int tree_missing_from_expr = 0; /* Total number of tree tips, across all trees, missing from the expression matrix */
    int expr_missing_from_tree = 0; /* Number of expression matrix cells missing from at least one tree */
    int n_keep = 0; /* Number of names shared between the expression matrix and all trees */
    List **tree_name_lists = NULL;
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
    tree_name_lists = scalloc(n_trees, sizeof(List *));

    /* Get the leaf names from each tree */
    for (i = 0; i < n_trees; i++) {
        if (trees[i] == NULL) {
            fprintf(stderr, "ERROR: tree %d is NULL during reconciliation\n", i + 1);
            return 1;
        }
        tree_name_lists[i] = tr_leaf_names(trees[i]);   /* Collect the leaf names from the tree into a list of strings */
        if (tree_name_lists[i] == NULL) {
            fprintf(stderr, "ERROR: failed to collect tree tip names for tree %d\n", i + 1);
            return 1;
        }
    }

    keep_names = lst_new_ptr(gex->X->nrows > 0 ? gex->X->nrows : 1);  /* List to hold the names of the shared tree tips and expression matrix cells */
    if (keep_names == NULL) {
        return 1;
    }

    /* Check which tree tips are missing from the expression matrix across all trees. */
    for (i = 0; i < n_trees; i++) {
        for (j = 0; j < lst_size(tree_name_lists[i]); j++) {
            String *s = lst_get_ptr(tree_name_lists[i], j);
            if (!gex_name_in_char_array(s->chars, gex->cell_names, gex->X->nrows))
                tree_missing_from_expr++;
        }
    }

    /* Keep only expression cells that are present in every tree. */
    for (i = 0; i < gex->X->nrows; i++) {
        int present_in_all_trees = 1;
        for (j = 0; j < n_trees; j++) {
            if (!gex_name_in_string_list(gex->cell_names[i], tree_name_lists[j])) {
                present_in_all_trees = 0;
                break;
            }
        }
        if (!present_in_all_trees) {
            expr_missing_from_tree++;
        } else {
            String *s = str_new_charstr(gex->cell_names[i]);
            if (s == NULL) {
                return 1;
            }
            lst_push_ptr(keep_names, s);
            n_keep++;
        }
    }

    if (tree_missing_from_expr > 0 || expr_missing_from_tree > 0) {
        prune_needed = 1;
        fprintf(stderr,
                "WARNING: tree/expression names do not match perfectly across the full tree set; %d tree tip occurrence(s) are missing from the expression matrix and %d expression cell(s) are missing from at least one tree. Using the %d shared name(s) present in every tree.\n",
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
            if (subset->cell_names[j] == NULL) {
                gex_free_matrix_data(subset);
                subset = NULL;
                return 1;
            }
            for (g = 0; g < gex->X->ncols; g++)
                mat_set(subset->X, j, g, mat_get(gex->X, i, g));
            j++;
        }
    }

    /* Prune the trees only when the tree and expression names do not already match. */
    if (prune_needed) {
        for (i = 0; i < n_trees; i++) {
            if (trees[i] != NULL)
                tr_prune(&trees[i], keep_names, 1, NULL);
            if (trees[i] == NULL) {
                fprintf(stderr, "ERROR: tree %d became empty after reconciliation across all trees\n", i + 1);
                gex_free_matrix_data(subset);
                subset = NULL;
                return 1;
            }
        }
    }
    *gex_ptr = subset;

    /* Print result summary */
    if (n_keep < gex->X->nrows) {
        printf("After reconciling mismatched tree and expression matrix names, %d tree tip(s) and %d expression cell(s) are shared and kept for downstream analysis.\n\n",
            subset->X->nrows, subset->X->nrows);
    } else {
        printf("All tree tip names and expression cell names match in the input data.\n");
    }

    /* Free memory */
    gex_free_matrix_data(gex);
    for (i = 0; i < n_trees; i++) {
        gex_free_string_ptr_list(tree_name_lists[i]);
    }
    if (tree_name_lists != NULL)
        free(tree_name_lists);
    gex_free_string_ptr_list(keep_names);

    return 0;
}
