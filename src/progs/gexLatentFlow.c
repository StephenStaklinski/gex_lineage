#include "external_libs.h"
#include "gexmatrix.h"
#include "latentflow.h"
#include "parser.h"

#include <phast/misc.h>
#include <phast/trees.h>
#include <phast/vector.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *progname) {
    fprintf(stderr,
        "Usage:\n"
        "  %s --trees <trees.nex> --fit-prefix <prefix> "
        "[--outprefix <prefix>]\n"
        "  %s --trees <trees.nex> --factors <F.tsv> "
        "(--summary <summary.tsv> | --sigma2-latent <csv>) "
        "--outprefix <prefix>\n",
        progname, progname);
}

static int read_sigma2_latent_summary(const char *path,
                                      double *log_sigma2,
                                      int k) {
    FILE *fh = NULL;
    char *line = NULL;
    size_t capacity = 0;
    int *seen = NULL;
    int n_seen = 0;

    fh = fopen(path, "r");
    if (fh == NULL) {
        fprintf(stderr, "ERROR: could not open summary file: %s\n", path);
        return -1;
    }

    seen = scalloc(k, sizeof(int));
    while (getline(&line, &capacity, fh) >= 0) {
        int factor_index;
        double sigma2;

        if (sscanf(line, "sigma2_latent_LF%d\t%lf", &factor_index, &sigma2) != 2)
            continue;
        factor_index--;
        if (factor_index < 0 || factor_index >= k || seen[factor_index] ||
            !isfinite(sigma2) || sigma2 <= 0.0) {
            fprintf(stderr, "ERROR: invalid latent variance entry in %s: %s",
                    path, line);
            return -1;
        }
        log_sigma2[factor_index] = log(sigma2);
        seen[factor_index] = 1;
        n_seen++;
    }

    if (n_seen != k) {
        fprintf(stderr,
                "ERROR: expected %d sigma2_latent_LF entries in %s, found %d.\n",
                k, path, n_seen);
        return -1;
    }

    free(line);
    free(seen);
    fclose(fh);
    return 0;
}

static int copy_sigma2_latent_csv(const char *csv,
                                  double *log_sigma2,
                                  int k) {
    Vector *values = parse_csv_to_vec(csv);
    int d;

    if (values == NULL || values->size != k) {
        fprintf(stderr,
                "ERROR: --sigma2-latent must contain exactly %d values.\n", k);
        if (values != NULL)
            vec_free(values);
        return -1;
    }

    for (d = 0; d < k; d++) {
        double sigma2 = vec_get(values, d);
        if (!isfinite(sigma2) || sigma2 <= 0.0) {
            fprintf(stderr,
                    "ERROR: latent Brownian variances must be finite and positive.\n");
            vec_free(values);
            return -1;
        }
        log_sigma2[d] = log(sigma2);
    }
    vec_free(values);
    return 0;
}

static int count_tree_tips(TreeNode *tree) {
    List *nodes = tr_preorder(tree);
    int i;
    int n_tips = 0;

    if (nodes == NULL)
        return -1;
    for (i = 0; i < lst_size(nodes); i++) {
        TreeNode *node = lst_get_ptr(nodes, i);
        if (node != NULL && node->lchild == NULL && node->rchild == NULL)
            n_tips++;
    }
    return n_tips;
}

int main(int argc, char *argv[]) {
    const char *trees_file = NULL;
    const char *fit_prefix = NULL;
    const char *factors_file = NULL;
    const char *summary_file = NULL;
    const char *sigma2_csv = NULL;
    const char *outprefix = NULL;
    char factors_path[4096];
    char summary_path[4096];
    TreeNode **trees = NULL;
    GexMatrix *factors = NULL;
    double *log_sigma2 = NULL;
    int n_trees = 0;
    int i;

    set_num_threads(1);

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--trees") == 0 && i + 1 < argc)
            trees_file = argv[++i];
        else if (strcmp(argv[i], "--fit-prefix") == 0 && i + 1 < argc)
            fit_prefix = argv[++i];
        else if (strcmp(argv[i], "--factors") == 0 && i + 1 < argc)
            factors_file = argv[++i];
        else if (strcmp(argv[i], "--summary") == 0 && i + 1 < argc)
            summary_file = argv[++i];
        else if (strcmp(argv[i], "--sigma2-latent") == 0 && i + 1 < argc)
            sigma2_csv = argv[++i];
        else if (strcmp(argv[i], "--outprefix") == 0 && i + 1 < argc)
            outprefix = argv[++i];
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        }
        else {
            fprintf(stderr, "ERROR: unknown or incomplete argument: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (trees_file == NULL) {
        usage(argv[0]);
        return 1;
    }
    if (fit_prefix != NULL) {
        if (factors_file != NULL || summary_file != NULL || sigma2_csv != NULL) {
            fprintf(stderr,
                    "ERROR: --fit-prefix cannot be combined with explicit factor or variance inputs.\n");
            return 1;
        }
        snprintf(factors_path, sizeof(factors_path), "%s.F.tsv", fit_prefix);
        snprintf(summary_path, sizeof(summary_path), "%s.summary.tsv", fit_prefix);
        factors_file = factors_path;
        summary_file = summary_path;
        if (outprefix == NULL)
            outprefix = fit_prefix;
    }
    else if (factors_file == NULL || outprefix == NULL ||
             ((summary_file == NULL) == (sigma2_csv == NULL))) {
        usage(argv[0]);
        return 1;
    }

    trees = read_nexus(trees_file, &n_trees, -1);
    if (trees == NULL || n_trees <= 0)
        return 1;
    if (check_trees_ultrametric(trees, n_trees) != 0)
        return 1;
    if (n_trees > 1)
        fprintf(stderr,
                "WARNING: input contains %d trees; latent flow uses the first tree, matching gexFactor.\n",
                n_trees);
    uniform_rescale_trees(trees, n_trees, 1.0);

    factors = read_gex_matrix(factors_file);
    if (factors == NULL || factors->X == NULL)
        return 1;
    if (count_tree_tips(trees[0]) != factors->X->nrows) {
        fprintf(stderr,
                "ERROR: the first tree has a different number of tips than %s has rows.\n",
                factors_file);
        return 1;
    }

    log_sigma2 = scalloc(factors->X->ncols, sizeof(double));
    if (summary_file != NULL) {
        if (read_sigma2_latent_summary(summary_file, log_sigma2,
                                       factors->X->ncols) != 0)
            return 1;
    }
    else if (copy_sigma2_latent_csv(sigma2_csv, log_sigma2,
                                    factors->X->ncols) != 0) {
        return 1;
    }

    if (gex_write_latent_flow_from_factors(outprefix, trees[0], factors->X,
                                            log_sigma2,
                                            factors->cell_names) != 0)
        return 1;

    printf("Wrote full-tree Brownian latent reconstruction to %s.latent_flow.tsv\n",
           outprefix);

    free(log_sigma2);
    gex_free_matrix_data(factors);
    gex_free_trees(trees, n_trees);
    return 0;
}
