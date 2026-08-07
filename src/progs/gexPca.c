#include "brownian.h"
#include "external_libs.h"
#include "gexmatrix.h"
#include "parser.h"
#include "pca.h"

#include <phast/matrix.h>
#include <phast/misc.h>
#include <phast/trees.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *progname) {
    fprintf(stderr,
            "Usage: %s --tree <tree.nex> --expr <filtered.tsv> "
            "--outprefix <prefix> --dim K\n",
            progname);
}

int main(int argc, char *argv[]) {
    const char *tree_file = NULL;
    const char *expr_file = NULL;
    const char *outprefix = NULL;
    TreeNode **trees = NULL;
    GexMatrix *gex = NULL;
    Matrix *tree_covariance = NULL;
    PCA *pca = NULL;
    int k = 0;
    int n_trees = 0;
    int i;

    set_num_threads(1);

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--tree") == 0) {
            if (++i >= argc) {
                usage(argv[0]);
                return 1;
            }
            tree_file = argv[i];
        } else if (strcmp(argv[i], "--expr") == 0) {
            if (++i >= argc) {
                usage(argv[0]);
                return 1;
            }
            expr_file = argv[i];
        } else if (strcmp(argv[i], "--outprefix") == 0) {
            if (++i >= argc) {
                usage(argv[0]);
                return 1;
            }
            outprefix = argv[i];
        } else if (strcmp(argv[i], "--dim") == 0) {
            if (++i >= argc) {
                usage(argv[0]);
                return 1;
            }
            k = atoi(argv[i]);
        } else if (strcmp(argv[i], "--help") == 0 ||
                   strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "ERROR: unknown argument: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (tree_file == NULL || expr_file == NULL || outprefix == NULL || k <= 0) {
        fprintf(stderr,
                "ERROR: --tree, --expr, --outprefix, and a positive --dim are required.\n");
        usage(argv[0]);
        return 1;
    }

    if (load_and_reconcile_tree_gex_inputs(tree_file, expr_file, 1,
                                           &trees, &n_trees, &gex) != 0)
        return 1;

    tree_covariance =
        covariance_from_tree(trees[0], gex->cell_names, gex->X->nrows);
    if (tree_covariance == NULL) {
        gex_free_matrix_data(gex);
        gex_free_trees(trees, n_trees);
        return 1;
    }

    printf("Running maxPhyloPCA with k=%d...\n", k);
    pca = compute_max_phylo_pca(gex->X, tree_covariance, k);
    if (pca == NULL) {
        fprintf(stderr, "ERROR: maxPhyloPCA failed.\n");
        mat_free(tree_covariance);
        gex_free_matrix_data(gex);
        gex_free_trees(trees, n_trees);
        return 1;
    }

    print_pca_summary(pca);
    write_pca_tsv(outprefix, pca, gex);
    write_pca_gram(outprefix, pca);
    printf("Wrote PCA results to outprefix %s\n", outprefix);

    free_pca(pca);
    mat_free(tree_covariance);
    gex_free_matrix_data(gex);
    gex_free_trees(trees, n_trees);
    return 0;
}
