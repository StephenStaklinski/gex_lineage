#include "gex.h"

#include <stdlib.h>
#include <string.h>
#include <ctype.h>

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

/* -------------------- summary -------------------- */

void gex_print_summary(TreeNode **trees, int n_trees, GexMatrix *gex) {
    int i;

    printf("Loaded %d tree(s)\n", n_trees);

    if (gex != NULL && gex->X != NULL) {
        printf("Loaded matrix with %d cell(s) and %d gene(s)\n",
               gex->n_cells, gex->n_genes);

        printf("First few cell names:\n");
        for (i = 0; i < gex->n_cells && i < 5; i++)
            printf("  %s\n", gex->cell_names[i]);

        printf("First few gene names:\n");
        for (i = 0; i < gex->n_genes && i < 5; i++)
            printf("  %s\n", gex->gene_names[i]);

        printf("First few entries of matrix:\n");
        for (i = 0; i < gex->n_cells && i < 5; i++) {
            int j;
            printf("  %s:", gex->cell_names[i]);
            for (j = 0; j < gex->n_genes && j < 5; j++)
                printf(" %g", mat_get(gex->X, i, j));
            printf("\n");
        }
    }

    if (n_trees > 0 && trees != NULL && trees[0] != NULL) {
        printf("First tree (Newick): ");
        tr_print(stdout, trees[0], 1);
        printf("\n");
    }
}