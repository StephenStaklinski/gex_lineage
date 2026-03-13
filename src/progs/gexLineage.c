#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gex.h"

static void usage(const char *progname) {
    fprintf(stderr,
        "Usage: %s --trees <trees.nex> --expr <matrix.tsv>\n",
        progname);
}

int main(int argc, char *argv[]) {
    const char *trees_file = NULL;
    const char *expr_file = NULL;
    TreeNode **trees = NULL;
    GexMatrix *gex = NULL;
    int n_trees = 0;
    int i;

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--trees") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            trees_file = argv[++i];
        }
        else if (strcmp(argv[i], "--expr") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            expr_file = argv[++i];
        }
        else if (strcmp(argv[i], "--help") == 0 ||
                 strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        }
        else {
            fprintf(stderr, "ERROR: unknown argument: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (trees_file == NULL || expr_file == NULL) {
        usage(argv[0]);
        return 1;
    }

    trees = gex_read_nexus(trees_file, &n_trees);
    if (trees == NULL) {
        fprintf(stderr, "ERROR: failed to load trees\n");
        return 1;
    }

    gex = gex_read_labeled_matrix(expr_file);
    if (gex == NULL) {
        fprintf(stderr, "ERROR: failed to load expression matrix\n");
        gex_free_trees(trees, n_trees);
        return 1;
    }

    gex_print_summary(trees, n_trees, gex);
    printf("done\n");

    gex_free_trees(trees, n_trees);
    gex_free_matrix_data(gex);

    return 0;
}