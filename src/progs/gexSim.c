
#include "gexbrownian.h"
#include "gexmatrix.h"
#include "gexmisc.h"
#include "gexmodel.h"
#include "gexparser.h"
#include "gexpca.h"
#include "gexphylofilter.h"

#include <phast/trees.h>
#include <phast/matrix.h>
#include <phast/misc.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


/* Print command line usage information to stderr. */
static void usage(const char *progname) {
    fprintf(stderr,
        "Usage: %s "
        "--trees <trees.nex> "
        "--outprefix <prefix> "
        "[--tree-total-time T] "
        "[--use-n-trees N] "
        "[--identity-cov] "
        "[--verbose] "
        "[--seed S]\n",
        progname);
}

/* Main program entry point for gexLineage. */
int main(int argc, char *argv[]) {
    /* Data structures to store user inputs and other default parameters */
    const char *trees_file = NULL;  /* Path to input NEXUS file containing trees */
    const char *outprefix = NULL;   /* Prefix for all output files */
    double tree_total_time = -1.0;  /* If positive, rescale all trees uniformly to have this total height. */
    int use_n_trees = -1;  /* -1: average covariance, 0: all trees, >0: first N trees */
    int identity_cov = 0; /* If nonzero, use identity covariance (null model) instead of tree-based covariance for simulations. */
    int verbose = 0;    /* If nonzero, print additional progress messages during the run. */
    unsigned int seed = 1u;   /* Random seed (positive) for all stochastic calculations */

    /* Data structures for calculations later */
    TreeNode **trees = NULL;    /* Array of tree pointers */
    Matrix **Sigmas = NULL; /* Phylogenetic covariance matrices, one per tree */
    Matrix *avg_Sigma = NULL; /* Average covariance used when --n-filter-trees=-1 */
    Matrix **use_Sigmas = NULL; /* Covariance matrices used for filtering */
    int n_trees = 0;    /* Number of input trees */
    GexMatrix *gex = NULL;  /* Simulated expression matrix */
    int i;
    

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--trees") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            trees_file = argv[++i];
        }
        else if (strcmp(argv[i], "--outprefix") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            outprefix = argv[++i];
        }
        else if (strcmp(argv[i], "--tree-total-time") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            tree_total_time = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--use-n-trees") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            use_n_trees = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--seed") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return 1;
            }
            seed = (unsigned int)strtoul(argv[++i], NULL, 10);
        }
        else if (strcmp(argv[i], "--verbose") == 0) {
            verbose = 1;
        }
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;   /* Success since user just wants help */
        }
        else {
            fprintf(stderr, "ERROR: unknown argument: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    /* Check that all required inputs are specified */
    if (trees_file == NULL || outprefix == NULL) {
        usage(argv[0]);
        return 1;
    }

    /* Load the input trees */
    trees = gex_read_nexus(trees_file, &n_trees);
    if (trees == NULL || n_trees < 1 || trees[0] == NULL) {
        fprintf(stderr, "ERROR: failed to load tree(s).\n");
        return 1;
    }
    printf("Loaded %d tree(s).\n", n_trees);

    if (use_n_trees < -1) {
        fprintf(stderr, "ERROR: --use-n-trees must be -1, 0, or a positive integer\n");
        return 1;
    }
    if (use_n_trees > n_trees) {
        fprintf(stderr, "ERROR: --use-n-trees (%d) cannot exceed the number of loaded trees (%d)\n",
                use_n_trees, n_trees);
        return 1;
    }

    /* Check that the input trees are ultrametric (required for cell lineage) */
    if (check_trees_ultrametric(trees, n_trees) != 0) {
        return 1;
    }

    if (verbose) {
        /* Print input/output summary for user verification */
        gex_print_io_summary(trees, n_trees, NULL);
    }

    /* Rescale the trees to a specified total height if requested */
    if (tree_total_time > 0.0) {
        uniform_rescale_trees(trees, n_trees, tree_total_time);
        printf("Rescaled tree(s) to total height %.6f.\n", tree_total_time);
    }

    if (!identity_cov) {
        /* Calculate the phylogenetic covariance matrix for each input tree. */
        Sigmas = scalloc(n_trees, sizeof(Matrix *));
        for (i = 0; i < n_trees; i++) {
            Sigmas[i] = covariance_from_tree(trees[i], gex->cell_names, gex->X->nrows);
            if (Sigmas[i] == NULL) {
                fprintf(stderr, "ERROR: failed to compute Brownian covariance matrix for tree %d.\n", i + 1);
                return 1;
            }
        }
        if (verbose) {
            printf("Computed phylogenetic covariance matrix for the first tree:\n");
            print_covariance_summary(Sigmas[0], gex->cell_names, gex->X->nrows);
        }

        /* Compute average covariance matrix over input trees if needed */
        if (use_n_trees == -1) {
            avg_Sigma = gex_average_tree_covariance(trees, n_trees,
                                                        gex->cell_names, gex->X->nrows);
            if (avg_Sigma == NULL) {
                fprintf(stderr, "ERROR: failed to compute average covariance for filtering.\n");
                return 1;
            }
            use_Sigmas = scalloc(1, sizeof(Matrix *));
            use_Sigmas[0] = avg_Sigma;
        }
        else {
            use_Sigmas = Sigmas;
            if (use_n_trees == 0)
                use_n_trees = n_trees;
        }

        if (use_n_trees == -1) {
            printf("Running a simulation check of the phylogenetic signal gene filter(s) using the average covariance across all %d tree(s)...\n",
                    n_trees);
        } else if (use_n_trees == n_trees) {
            printf("Running a simulation check of the phylogenetic signal gene filter(s) using all %d tree(s)...\n",
                    use_n_trees);
        } else {
            printf("Running a simulation check of the phylogenetic signal gene filter(s) using the first %d tree(s)...\n",
                    use_n_trees);
        }

    } else {
        /* Use identity covariance for simulations instead of tree-based covariance if requested */
        use_Sigmas = scalloc(1, sizeof(Matrix *));
        use_Sigmas[0] = mat_identity(gex->X->nrows);
        if (use_Sigmas[0] == NULL) {
            fprintf(stderr, "ERROR: failed to create identity covariance matrix for simulations.\n");
            return 1;
        }
    }

    char filter_sims_buf[4096];
    snprintf(filter_sims_buf, sizeof(filter_sims_buf), "%s.phylo_filter_sims.expr.tsv", outprefix);
    char *filter_sims_path = filter_sims_buf;

    brownian_run_simulation_check(gex->cell_names,
                                        gex->X->nrows,
                                        n_sims,
                                        n_sims,
                                        filter_mode,
                                        lrt_alt_mode,
                                        n_perms,
                                        max_q,
                                        moran_min_i,
                                        filter_Sigmas,
                                        (use_n_trees == -1 ? 1 : use_n_trees),
                                        filter_sims_path,
                                        seed))

    /* Free memory */
    gex_free_trees(trees, n_trees);
    gex_free_matrix_data(gex);
    if (Sigmas != NULL) {
        for (i = 0; i < n_trees; i++) {
            if (Sigmas[i] != NULL)
                mat_free(Sigmas[i]);
        }
        free(Sigmas);
    }
    if (use_Sigmas != NULL && use_Sigmas != Sigmas) {
        free(use_Sigmas);
    }
    if (avg_Sigma != NULL) {
        mat_free(avg_Sigma);
    }

    return 0; /* Success */
}
