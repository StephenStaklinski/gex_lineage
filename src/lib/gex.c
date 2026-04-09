#include "gex.h"
#include "brownian.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <phast/eigen.h>
#include <phast/lists.h>
#include <phast/misc.h>
#include <phast/numerical_opt.h>
#include <phast/stringsplus.h>

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

/* Generate next 32-bit unsigned integer from RNG state */
unsigned int rand_u32(unsigned int *state) {
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

/* Generate uniform random number in (0,1), excluding endpoints */
double uniform_open(unsigned int *state) {
    return ((double)rand_u32(state) + 1.0) / 4294967297.0;
}

/* Generate standard normal random variable (mean 0, variance 1) */
double rand_normal(unsigned int *state) {
    double u1 = uniform_open(state);
    double u2 = uniform_open(state);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2); 
}

void free_string_array(char **names, int n) {
    int i;

    if (names == NULL)
        return;
    for (i = 0; i < n; i++)
        free(names[i]);
    free(names);
}

/* Helper to generate latent factor names incrementally */
static char **generate_factor_names(int k) {
    int i;
    char **names = NULL;

    if (k <= 0)
        return NULL;
    names = scalloc(k, sizeof(char *));

    for (i = 0; i < k; i++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "LF%d", i + 1);
        names[i] = strdup(buf);
        if (names[i] == NULL) {
            free_string_array(names, i);
            return NULL;
        }
    }

    return names;
}

/* Shuffle an array of doubles in place using the Fisher-Yates algorithm. */
static void gex_shuffle_double(double *x, int n, unsigned int *state) {
    int i;

    /* Loop backwards from the last element to the second */
    for (i = n - 1; i > 0; i--) {
        int j = (int)(rand_u32(state) % (unsigned int)(i + 1)); /* Pick a random index */
        /* Swap elements at indices i and j */
        double tmp = x[i];
        x[i] = x[j];
        x[j] = tmp;
    }
}

/* Compute the weighted quadratic form (x^T * W * x) for a symmetric matrix W 
and vector x.*/
static double gex_weighted_quadratic(Matrix *W, double *x, int n) {
    int i, j;
    double out = 0.0;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++)
            out += x[i] * mat_get(W, i, j) * x[j];
    }

    return out;
}

static double gex_loglik_centered_gaussian_identity(int n, double sigma2) {
    return -0.5 * ((double)n * (log(2.0 * M_PI * sigma2) + 1.0));
}

/* Get the mean and (population) variance of vector y. */
static void calculate_mean_variance(double *y, int n, double *mean_out, double *sigma2_out) {
    int i;
    double mean = 0.0;
    double sse = 0.0;

    /* Compute the mean of the data */
    for (i = 0; i < n; i++)
        mean += y[i];
    mean /= (double)n;

    /* Compute the sum of squared errors around the mean */
    for (i = 0; i < n; i++) {
        double d = y[i] - mean;
        sse += d * d;
    }

    *mean_out = mean;
    *sigma2_out = sse / (double)n;  /* Variance */
    if (*sigma2_out < 1e-12)
        *sigma2_out = 1e-12;
}

/* Compute the log-likelihood of y under N(mu 1, sigma2 * Sigma),
   profiling out mu and sigma2, using the Cholesky factor L of Sigma. */
static double gex_loglik_centered_gaussian_chol(double *y,
                                                Matrix *L,
                                                double logdet_sigma) {
    int i, n;
    Vector *ones = NULL;
    Vector *yvec = NULL;
    Vector *tmp = NULL;
    Vector *Sinv1 = NULL;
    Vector *Sinvy = NULL;
    Vector *resid = NULL;
    Vector *Sinv_resid = NULL;
    double ones_Sinv_ones = 0.0;
    double ones_Sinv_y = 0.0;
    double quad = 0.0;
    double muhat;
    double sigma2;
    double ll;

    n = L->nrows;

    if (n != L->ncols)
        return NAN;

    ones = vec_new(n);
    yvec = vec_new(n);
    tmp = vec_new(n);
    Sinv1 = vec_new(n);
    Sinvy = vec_new(n);
    resid = vec_new(n);
    Sinv_resid = vec_new(n);

    for (i = 0; i < n; i++) {
        vec_set(ones, i, 1.0);
        vec_set(yvec, i, y[i]);
    }

    /* Solve Sigma * Sinv1 = 1 via L tmp = 1, then L^T Sinv1 = tmp */
    mat_forward_subst(L, ones, tmp);
    mat_backward_subst(L, tmp, Sinv1);

    /* Solve Sigma * Sinvy = y via L tmp = y, then L^T Sinvy = tmp */
    mat_forward_subst(L, yvec, tmp);
    mat_backward_subst(L, tmp, Sinvy);

    for (i = 0; i < n; i++) {
        ones_Sinv_ones += vec_get(Sinv1, i);
        ones_Sinv_y += vec_get(Sinvy, i);
    }

    if (ones_Sinv_ones <= 0.0)
        return NAN;

    /* GLS estimate of the mean */
    muhat = ones_Sinv_y / ones_Sinv_ones;

    for (i = 0; i < n; i++)
        vec_set(resid, i, y[i] - muhat);

    /* Solve Sigma * Sinv_resid = resid via L tmp = resid, then L^T Sinv_resid = tmp */
    mat_forward_subst(L, resid, tmp);
    mat_backward_subst(L, tmp, Sinv_resid);

    /* quad = resid^T Sigma^{-1} resid */
    for (i = 0; i < n; i++)
        quad += vec_get(resid, i) * vec_get(Sinv_resid, i);

    sigma2 = quad / (double)n;
    if (sigma2 < 1e-12)
        sigma2 = 1e-12;

    ll = -0.5 * ((double)n * log(2.0 * M_PI * sigma2) +
                 logdet_sigma +
                 (double)n);

    vec_free(ones);
    vec_free(yvec);
    vec_free(tmp);
    vec_free(Sinv1);
    vec_free(Sinvy);
    vec_free(resid);
    vec_free(Sinv_resid);
    return ll;
}

typedef struct {
    double *y;
    Matrix *Sigma;
    Matrix *Sigma_lambda;
    Matrix *L;
    double diag_mean;
    double jitter;
} GexPagelsLambdaOptData;

/* Compute the log-likelihood under a Pagel's lambda style covariance model.
The diagonal entries are held fixed at the common tip variance and the
off-diagonal entries are scaled by lambda. */
static double gex_loglik_pagels_lambda_cov(double *y,
                                           Matrix *Sigma,
                                           double diag_mean,
                                           double jitter,
                                           double lambda,
                                           Matrix *Sigma_lambda,
                                           Matrix *L) {
    int i, j, n;    /* Loop indices */
    double logdet_sigma = 0.0;  /* Log determinant of the lambda-transformed covariance matrix */

    /* Get the number of rows in the covariance matrix */
    n = Sigma->nrows;
    if (lambda < 0.0 || lambda > 1.0)
        return -HUGE_VAL;

    /* Build the lambda-transformed covariance matrix.
    For ultrametric trees, all diagonals should match, so keeping a common
    diagonal value makes lambda = 0 correspond to the null model up to scale. */
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i == j)
                mat_set(Sigma_lambda, i, j, diag_mean + jitter);
            else
                mat_set(Sigma_lambda, i, j, lambda * mat_get(Sigma, i, j));
        }
    }

    /* Compute the Cholesky decomposition of the lambda-transformed covariance matrix */
    if (mat_cholesky(L, Sigma_lambda) != 0)
        return -HUGE_VAL;

    /* Compute the log determinant of the lambda-transformed covariance matrix */
    for (i = 0; i < n; i++) {
        double diag = mat_get(L, i, i);
        if (diag <= 0.0)
            return -HUGE_VAL;
        logdet_sigma += 2.0 * log(diag);
    }

    return gex_loglik_centered_gaussian_chol(y, L, logdet_sigma);
}

/* Objective function for fitting Pagel's lambda by one-dimensional numerical optimization. */
static double gex_pagels_lambda_negloglik(double lambda, void *data) {
    GexPagelsLambdaOptData *d = (GexPagelsLambdaOptData *)data;
    double ll = gex_loglik_pagels_lambda_cov(d->y,
                                             d->Sigma,
                                             d->diag_mean,
                                             d->jitter,
                                             lambda,
                                             d->Sigma_lambda,
                                             d->L);

    if (!isfinite(ll))
        return HUGE_VAL;
    return -ll;
}

/* Fit Pagel's lambda using PHAST's bounded one-dimensional optimizer.
Returns the optimized log-likelihood and sets lambda_hat to the MLE. */
static double gex_fit_pagels_lambda_loglik(double *y,
                                           Matrix *Sigma,
                                           double jitter,
                                           Matrix *Sigma_lambda,
                                           Matrix *L,
                                           double *lambda_hat) {
    int i, status;  /* Loop index and optimization status */
    double diag_mean = 0.0; /* Mean of the diagonal elements of the covariance matrix */
    double lambda;  /* Parameter for Pagel's lambda */
    double fx;  /* Objective function value */
    double fx0; /* Objective function value at lambda = 0 (null model) */
    double fx1; /* Objective function value at lambda = 1 (full model) */
    double best_lambda; /* Best lambda value found among boundary and interior evaluations */
    double best_fx; /* Best objective function value found among boundary and interior evaluations */
    GexPagelsLambdaOptData data;    /* Data structure to pass to the objective function for optimization */

    /* Compute the mean of the diagonal elements of the covariance matrix */
    for (i = 0; i < Sigma->nrows; i++)
        diag_mean += mat_get(Sigma, i, i);
    diag_mean /= (double)Sigma->nrows;

    /* Set up the data structure for optimization */
    data.y = y;
    data.Sigma = Sigma;
    data.Sigma_lambda = Sigma_lambda;
    data.L = L;
    data.diag_mean = diag_mean;
    data.jitter = jitter;

    /* Explicitly evaluate both boundaries because lambda is allowed to sit on
    the edge of the parameter space and the mixture null depends on that case. */
    fx0 = gex_pagels_lambda_negloglik(0.0, &data);
    fx1 = gex_pagels_lambda_negloglik(1.0, &data);
    if (fx0 <= fx1) {
        best_fx = fx0;
        best_lambda = 0.0;
    }
    else {
        best_fx = fx1;
        best_lambda = 1.0;
    }

    /* Evaluate the objective function starting at an interior point */
    lambda = 0.5;
    fx = gex_pagels_lambda_negloglik(lambda, &data);

    /* Optimize the objective function using Newton's method */
    status = opt_newton_1d(gex_pagels_lambda_negloglik,
                           &lambda,
                           &data,
                           &fx,
                           6,
                           0.0,
                           1.0,
                           NULL,
                           NULL,
                           NULL);

    /* If the optimizer converges to a finite value that is better than the boundary values, update the best solution */
    if (isfinite(fx) && fx < best_fx) {
        best_fx = fx;
        best_lambda = lambda;
    }

    /* If the optimizer does not fully converge, still retain the best valid
    value found among the boundary and interior evaluations. */
    if (status != 0 && !isfinite(best_fx))
        return -HUGE_VAL;

    *lambda_hat = best_lambda;
    return -best_fx;
}

/* Compute log(sum_i exp(x[i])) in a numerically stable way using the
log-sum-exp trick: max(x) + log(sum_i exp(x[i] - max(x))). */
double logsumexp(double *x, int n) {
    int i;
    double max_x = -HUGE_VAL;
    double sum = 0.0;

    /* Find the maximum value in the array */
    for (i = 0; i < n; i++) {
        if (x[i] > max_x)
            max_x = x[i];
    }
    if (!isfinite(max_x))
        return max_x;

    /* Sum the exponentials */
    for (i = 0; i < n; i++)
        sum += exp(x[i] - max_x);

    /* Return the log-sum-exp */
    return max_x + log(sum);
}

/* Standardize the columns of a gene expression matrix by
subtracting the mean and dividing by the standard deviation */
static Matrix *gex_standardize_columns(GexMatrix *gex) {
    int i, j;
    Matrix *Z;
    double *col = NULL; /* Array to store column values from the expression matrix to use the helper function for mean and variance calculation */

    if (gex == NULL || gex->X == NULL)
        return NULL;

    Z = mat_new(gex->X->nrows, gex->X->ncols);    /* Initialize the standardized matrix */
    if (Z == NULL)
        return NULL;
    col = smalloc(gex->X->nrows * sizeof(double));

    /* Standardize each column (gene) of the expression matrix */
    for (j = 0; j < gex->X->ncols; j++) {
        double mean = 0.0;
        double var = 0.0;
        double sd;

        for (i = 0; i < gex->X->nrows; i++)
            col[i] = mat_get(gex->X, i, j);

        calculate_mean_variance(col, gex->X->nrows, &mean, &var);
        sd = sqrt(var);

        /* If the standard deviation is very small, set all values to 0. Otherwise, standardize the values 
        by subtracting the mean and dividing by the standard deviation */
        if (sd < 1e-12) {
            for (i = 0; i < gex->X->nrows; i++)
                mat_set(Z, i, j, 0.0);
        }
        else {
            for (i = 0; i < gex->X->nrows; i++) {
                double z = (col[i] - mean) / sd;
                mat_set(Z, i, j, z);
            }
        }
    }

    free(col);
    return Z;
}

/* Adjust p-values for multiple testing using the Benjamini-Hochberg procedure.
Fills the qvals array with the adjusted p-values. */
static void gex_bh_adjust(double *pvals, double *qvals, int n) {
    GexPvalPair *pairs;
    int i;
    double running;

    /* Initialize the pairs array */
    pairs = smalloc(n * sizeof(GexPvalPair));

    /* Fill the pairs array with p-values and their original indices */
    for (i = 0; i < n; i++) {
        pairs[i].pval = pvals[i];
        pairs[i].idx = i;
    }

    /* Sort the pairs array in ascending order of p-values */
    qsort(pairs, n, sizeof(GexPvalPair), gex_cmp_pval_asc);

    /* Adjust p-values using the Benjamini-Hochberg procedure.
    For sorted p-values p_(i), compute (n / i) * p_(i),
   then apply a reverse cumulative minimum to ensure
   the adjusted p-values are non-decreasing. */
    running = 1.0;
    for (i = n - 1; i >= 0; i--) {
        double rank = (double)(i + 1);
        double val = pairs[i].pval * (double)n / rank;
        if (val > 1.0) 
            val = 1.0;
        if (val < running) 
            running = val;
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

/* Determine if a gene should be kept based on the specified 
filter mode and thresholds. */
static int gex_keep_gene(GexMoransResult *morans,
                         GexLRTResult *lrt,
                         int gene_idx,
                         GexFilterMode mode,
                         double max_q,
                         double min_i) {
    int keep_moran = 0; /* Flag indicating if the gene passes the Moran's I filter */
    int keep_lrt = 0;   /* Flag indicating if the gene passes the LRT filter */

    /* Apply the filters based on if the provided objects are not NULL */
    if (morans != NULL)
        keep_moran = (morans->qvals[gene_idx] <= max_q &&
                      morans->morans_i[gene_idx] > min_i);
    if (lrt != NULL)
        keep_lrt = (lrt->qvals[gene_idx] <= max_q &&
                    lrt->lrt_stat[gene_idx] > 0.0);

    /* Return the appropriate filter result based on the filter mode */
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

void gex_free_matrix_data(GexMatrix *gex) {
    int i;

    if (gex == NULL) return;

    if (gex->cell_names != NULL) {
        for (i = 0; i < gex->X->nrows; i++)
            free(gex->cell_names[i]);
        free(gex->cell_names);
    }

    if (gex->gene_names != NULL) {
        for (i = 0; i < gex->X->ncols; i++)
            free(gex->gene_names[i]);
        free(gex->gene_names);
    }

    if (gex->X != NULL)
        mat_free(gex->X);

    free(gex);
}

/* Normalize the entries in a row by the row sum in-place.
Returns 0 on success, -1 on failure. */
void normalize_by_row_sums(Matrix *X) {
    int i, j;
    
    for (i = 0; i < X->nrows; i++) {
        double row_sum = 0.0;
        /* Accumulate row sum */
        for (j = 0; j < X->ncols; j++) {
            row_sum += mat_get(X, i, j);
        } 
        if (row_sum < 1e-12)
            continue;   /* Skip normalization if the row sum is negligibly small */
        /* Apply the normalization to elements of the row */
        for (j = 0; j < X->ncols; j++) {
            double val = mat_get(X, i, j);
            val /= row_sum;
            mat_set(X, i, j, val);
        }
    }
}

/* Transform a matrix using the log1p function (log(1+x))) element-wise
in-place. */
void log1p_transform(Matrix *X) {
    int i, j;

    for (i = 0; i < X->nrows; i++) {
        for (j = 0; j < X->ncols; j++) {
            double val = mat_get(X, i, j);
            val = log1p(val);
            mat_set(X, i, j, val);
        }
    }
}

/* Center the columns of a matrix by subtracting the mean of each column
to get the residuals in-place. */
void center_matrix_inplace(Matrix *X) {
    int i, j;

    for (j = 0; j < X->ncols; j++) {

        /* Get the mean of the column */
        double mean = 0.0;
        for (i = 0; i < X->nrows; i++)
            mean += mat_get(X, i, j);
        mean /= X->nrows;

        /* Center the column */
        for (i = 0; i < X->nrows; i++) {
            double val = mat_get(X, i, j);
            val -= mean;
            mat_set(X, i, j, val);
        }
    }
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

/* Compute Moran's I for each gene in the expression matrix.
Moran's I is a measure of spatial autocorrelation, which is
calculated here as the expectation over trees of Z^T * W_t * Z,
where z is the column-wise standardized gene expression matrix and
W_t is derived internally from the Brownian covariance matrix Sigma_t.
Returns a pointer to the result structure. */
GexMoransResult *gex_compute_morans_i(GexMatrix *gex,
                                      Matrix **Sigmas,
                                      int n_sigmas,
                                      int n_perm,
                                      unsigned int seed) {
    int i, j, k, t;    /* Loop indices */
    int n_cells;    /* Number of cells (rows in the expression matrix) */
    int n_genes;    /* Number of genes (columns in the expression matrix) */
    unsigned int rng_state; /* State for the random number generator used in permutation testing */
    Matrix *Z = NULL;   /* Standardized gene expression matrix (n_cells x n_genes) */
    Matrix **Ws = NULL; /* Weight matrices derived from the Brownian covariance matrices */
    Matrix *B = NULL;   /* Intermediate matrix for E[W] * Z (n_cells x n_genes) */
    GexMoransResult *res = NULL;    /* Result structure for Moran's I computation */
    double *zcol = NULL;    /* Temporary array to hold a single column of the standardized matrix for permutation testing */
    double *perm = NULL;    /* Temporary array to hold the permuted version of the column for permutation testing */

    /* Validate input parameters */
    if (gex == NULL || gex->X == NULL || Sigmas == NULL || n_sigmas <= 0 || n_perm <= 0) {
        fprintf(stderr, "ERROR: gex_compute_morans_i got invalid input\n");
        return NULL;
    }
    n_cells = gex->X->nrows;
    n_genes = gex->X->ncols;
    Ws = scalloc(n_sigmas, sizeof(Matrix *));
    for (t = 0; t < n_sigmas; t++) {
        if (Sigmas[t] == NULL ||
            Sigmas[t]->nrows != n_cells ||
            Sigmas[t]->ncols != n_cells) {
            fprintf(stderr, "ERROR: covariance matrix dimensions do not match number of cells\n");
            return NULL;
        }
        Ws[t] = weight_matrix_from_covariance(Sigmas[t]);
        if (Ws[t] == NULL) {
            fprintf(stderr, "ERROR: failed to derive Moran weight matrix from covariance\n");
            return NULL;
        }
    }

    /* Normalize the gene expression matrix */
    Z = gex_standardize_columns(gex);
    if (Z == NULL) {
        fprintf(stderr, "ERROR: failed to standardize gene expression matrix\n");
        return NULL;
    }

    /* Compute E[W * Z] across the tree set. */
    B = mat_new(n_cells, n_genes);
    if (B == NULL) {
        return NULL;
    }
    mat_zero(B);
    for (i = 0; i < n_cells; i++) {
        for (j = 0; j < n_genes; j++) {
            double sum = 0.0;
            for (t = 0; t < n_sigmas; t++) {
                for (k = 0; k < n_cells; k++)
                    sum += mat_get(Ws[t], i, k) * mat_get(Z, k, j);
            }
            mat_set(B, i, j, sum / (double)n_sigmas);
        }
    }

    /* Initialize the result structure */
    res = scalloc(1, sizeof(GexMoransResult));
    res->corr = mat_new(n_genes, n_genes);
    res->morans_i = scalloc(n_genes, sizeof(double));
    res->pvals = scalloc(n_genes, sizeof(double));
    res->qvals = scalloc(n_genes, sizeof(double));
    res->n_genes = n_genes;

    /* Compute the Moran's I correlation matrix from Z^T x B */
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

    /* Initialize temporary arrays for permutation testing */
    zcol = smalloc(n_cells * sizeof(double));
    perm = smalloc(n_cells * sizeof(double));

    /* Run permutation tests per gene */
    rng_state = (seed == 0u ? 1u : seed);
    for (j = 0; j < n_genes; j++) {
        int ge_count = 0;

        for (i = 0; i < n_cells; i++) {
            zcol[i] = mat_get(Z, i, j);
            perm[i] = zcol[i];
        }

        for (k = 0; k < n_perm; k++) {
            double perm_i;
            memcpy(perm, zcol, n_cells * sizeof(double));   /* Copy the column */
            gex_shuffle_double(perm, n_cells, &rng_state);  /* Shuffle the column */
            perm_i = 0.0;
            for (t = 0; t < n_sigmas; t++)
                perm_i += gex_weighted_quadratic(Ws[t], perm, n_cells);
            perm_i /= (double)n_sigmas;  /* Compute the expected weighted quadratic form */
            if (perm_i >= res->morans_i[j]) /* Track permutations with higher or equal Moran's I */
                ge_count++;
        }

        res->pvals[j] = ((double)ge_count + 1.0) / ((double)n_perm + 1.0);  /* Compute the p-value */
    }

    /* Adjust p-values for multiple testing and count significant genes */
    gex_bh_adjust(res->pvals, res->qvals, n_genes);
    res->n_significant = gex_count_kept_genes(res, 0.05, 0.0);

    /* Free memory */
    if (zcol != NULL)
        free(zcol);
    if (perm != NULL)
        free(perm);
    if (Z != NULL)
        mat_free(Z);
    if (Ws != NULL) {
        for (t = 0; t < n_sigmas; t++) {
            if (Ws[t] != NULL)
                mat_free(Ws[t]);
        }
        free(Ws);
    }
    if (B != NULL)
        mat_free(B);
    
    return res;
}

/* Print a summary of the Moran's I results */
void gex_print_morans_summary(GexMoransResult *res,
                              GexMatrix *gex,
                              double max_q,
                              double min_i) {
    int i, j;
    int n_print = 10; /* Number of variable entries to print in each summary */

    if (res == NULL || gex == NULL) {
        fprintf(stderr, "ERROR: cannot summarize NULL Moran's I result\n");
        return;
    }

    printf("\n");
    printf("Computed Moran's I correlation matrix for %d gene(s)\n", res->n_genes);
    printf("Genes passing filter (q <= %.4f and I > %.4f): %d\n",
           max_q, min_i, gex_count_kept_genes(res, max_q, min_i));

    printf("First few entries of Moran's I gene-gene correlation matrix:\n");
    for (i = 0; i < res->n_genes && i < n_print; i++) {
        printf("%s", gex->gene_names[i]);
        for (j = 0; j < res->n_genes && j < n_print; j++)
            printf("\t%g", mat_get(res->corr, i, j));
        printf("\n");
    }

    printf("\nFirst few gene-wise Moran's I statistics:\n");
    printf("gene\tI\tp\tq\tkeep\n");
    for (j = 0; j < res->n_genes && j < n_print; j++) {
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

/* Write Moran's I results to a TSV file */
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

/* Compute the Brownian LRT for a given expression matrix and tree-set of phylogenetic covariance matrices.
   The LRT compares:

     Null model:   y ~ N(mu, sigma^2 * I)
                   (no phylogenetic structure; identity covariance used)

     Alternative:  y ~ N(mu, sigma^2 * Sigma_t)
                   (Brownian motion on each tree t; the alternative log-likelihood
                   is averaged across the supplied tree set)

   For each gene, the function computes the log-likelihood under both models
   and forms the likelihood ratio statistic LRT = 2 * (logLik_alt - logLik_null).
   P-values are obtained either from Monte Carlo simulation under the null
   for the full Brownian alternative or from the standard 50:50 mixture of
   chi-square with 1 dof and a point mass at zero for the Pagel's lambda
   alternative. Returns a pointer to the result structure or NULL on failure.
*/
GexLRTResult *gex_compute_brownian_lrt(GexMatrix *gex,
                                       Matrix **Sigmas,
                                       int n_sigmas,
                                       int n_mc,
                                       unsigned int seed,
                                       GexLRTAltMode alt_mode) {
    int i, j, t;   /* Loop indices */
    int n;  /* Number of cells */
    Matrix **Sigma_regs = NULL;   /* Regularized covariance matrices */
    Matrix **Ls = NULL;   /* Cholesky factors of regularized covariance matrices */
    Matrix **Sigma_lambdas = NULL;   /* Lambda-transformed covariance matrices */
    GexLRTResult *res = NULL;   /* Result structure for the LRT computation */
    double *logdet_sigmas = NULL;   /* Log determinants of the regularized covariance matrices */
    double *jitters = NULL;   /* Per-tree diagonal jitters for numerical stability */
    double *y = NULL;   /* Vector for storing the expression values */
    double *y_sim = NULL;   /* Vector for storing simulated expression values */
    unsigned int rng_state; /* Random number generator state */

    if (gex == NULL || gex->X == NULL || Sigmas == NULL || n_sigmas <= 0 ||
        alt_mode < GEX_LRT_ALT_FULL || alt_mode > GEX_LRT_ALT_LAMBDA ||
        (alt_mode == GEX_LRT_ALT_FULL && n_mc <= 0)) {
        fprintf(stderr, "ERROR: gex_compute_brownian_lrt got invalid input\n");
        return NULL;
    }

    n = gex->X->nrows;
    Sigma_regs = scalloc(n_sigmas, sizeof(Matrix *));
    Ls = scalloc(n_sigmas, sizeof(Matrix *));
    logdet_sigmas = scalloc(n_sigmas, sizeof(double));
    jitters = scalloc(n_sigmas, sizeof(double));
    if (alt_mode == GEX_LRT_ALT_LAMBDA)
        Sigma_lambdas = scalloc(n_sigmas, sizeof(Matrix *));

    for (t = 0; t < n_sigmas; t++) {
        double max_diag = 0.0;

        if (Sigmas[t] == NULL || Sigmas[t]->nrows != n || Sigmas[t]->ncols != n) {
            fprintf(stderr, "ERROR: covariance matrix dimensions do not match number of cells for tree %d\n",
                    t + 1);
            return NULL;
        }

        Sigma_regs[t] = mat_new(n, n);
        Ls[t] = mat_new(n, n);
        if (alt_mode == GEX_LRT_ALT_LAMBDA)
            Sigma_lambdas[t] = mat_new(n, n);

        for (i = 0; i < n; i++) {
            double d = mat_get(Sigmas[t], i, i);
            if (d > max_diag)
                max_diag = d;
        }
        jitters[t] = (max_diag > 0.0 ? 1e-8 * max_diag : 1e-8);

        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++)
                mat_set(Sigma_regs[t], i, j, mat_get(Sigmas[t], i, j));
            mat_set(Sigma_regs[t], i, i, mat_get(Sigma_regs[t], i, i) + jitters[t]);
        }

        /* Compute the Chloesky factor and log determinant of the covariance matrix once
        and store the results */
        if (mat_cholesky(Ls[t], Sigma_regs[t]) != 0) {
            fprintf(stderr, "ERROR: failed to initialize Brownian covariance matrices for LRT tree %d\n",
                    t + 1);
            return NULL;
        }
        for (i = 0; i < n; i++) {
            double diag = mat_get(Ls[t], i, i);
            if (diag <= 0.0) {
                fprintf(stderr, "ERROR: invalid Cholesky factor for LRT tree %d\n", t + 1);
                return NULL;
            }
            logdet_sigmas[t] += 2.0 * log(diag);
        }
    }

    /* Initialize result structure and temporary vectors for the LRT computation */
    res = scalloc(1, sizeof(GexLRTResult));
    y = smalloc(n * sizeof(double));
    if (alt_mode == GEX_LRT_ALT_FULL)
        y_sim = smalloc(n * sizeof(double));
    res->ll_null = scalloc(gex->X->ncols, sizeof(double));
    res->ll_alt = scalloc(gex->X->ncols, sizeof(double));
    if (alt_mode == GEX_LRT_ALT_LAMBDA)
        res->lambda_hat = scalloc(gex->X->ncols, sizeof(double));
    res->lrt_stat = scalloc(gex->X->ncols, sizeof(double));
    res->pvals = scalloc(gex->X->ncols, sizeof(double));
    res->qvals = scalloc(gex->X->ncols, sizeof(double));
    res->alt_mode = alt_mode;
    res->n_genes = gex->X->ncols;

    /* Run the LRT on each gene */
    rng_state = (seed == 0u ? 1u : seed);
    for (j = 0; j < gex->X->ncols; j++) {
        double ll_null; /* Log-likelihood under the null model */
        double ll_alt;  /* Log-likelihood under the alternative model */
        double *ll_alts; /* Log-likelihoods under the alternative model for each tree */
        double mu0; /* Mean under the null model (estimated from the data) */
        double sigma20; /* Variance under the null model (estimated from the data) */

        /* Extract gene expression data for the current gene across all cells */
        for (i = 0; i < n; i++)
            y[i] = mat_get(gex->X, i, j);

        /* Calculate the mean and variance of the gene expression data */
        calculate_mean_variance(y, n, &mu0, &sigma20);

        /* Compute the log-likelihood under the null model, which is the same
        independent of the alternative model */
        ll_null = gex_loglik_centered_gaussian_identity(n, sigma20);
        res->ll_null[j] = ll_null;

        /* Compute the log-likelihood under the alternative model */
        ll_alts = smalloc(n_sigmas * sizeof(double));
        ll_alt = 0.0;
        if (alt_mode == GEX_LRT_ALT_FULL) {
            for (t = 0; t < n_sigmas; t++) {
                ll_alts[t] = gex_loglik_centered_gaussian_chol(y,
                                                              Ls[t],
                                                              logdet_sigmas[t]);
            }
            ll_alt = logsumexp(ll_alts, n_sigmas) - log((double)n_sigmas);
        }
        else {
            double lambda_hat_sum = 0.0;
            for (t = 0; t < n_sigmas; t++) {
                double lambda_hat = 0.0;
                ll_alts[t] = gex_fit_pagels_lambda_loglik(y,
                                                       Sigmas[t],
                                                       jitters[t],
                                                       Sigma_lambdas[t],
                                                       Ls[t],
                                                       &lambda_hat);
                lambda_hat_sum += lambda_hat;
            }
            ll_alt = logsumexp(ll_alts, n_sigmas) - log((double)n_sigmas);
            res->lambda_hat[j] = lambda_hat_sum / (double)n_sigmas;
        }
        res->ll_alt[j] = ll_alt;

        /* Compute the LRT statistic */
        res->lrt_stat[j] = 2.0 * (ll_alt - ll_null);

        if (alt_mode == GEX_LRT_ALT_FULL) {
            /* Estimate p-values by simulating data under the null model to
            use a monte carlo estimate. */
            int ge_count = 0;
            int rep;
            for (rep = 0; rep < n_mc; rep++) {
                double ll0_sim;
                double ll1_sim;
                double stat_sim;
                for (i = 0; i < n; i++)
                    /* Draw simulated data independently from the null model N(μ0, σ20) */
                    y_sim[i] = mu0 + sqrt(sigma20) * rand_normal(&rng_state);

                /* Re-calculate the mean and variance of the simulated gene expression data.
                This is necessary because we only simulate finite samples, so we are not guaranteed
                to have the generating distribution mean and variance parameters exactly. */
                calculate_mean_variance(y_sim, n, &mu0, &sigma20);

                /* Compute the log-likelihood under the null model */
                ll0_sim = gex_loglik_centered_gaussian_identity(n, sigma20);

                /* Compute the expected log-likelihood under the alternative model */
                ll1_sim = 0.0;
                for (t = 0; t < n_sigmas; t++) {
                    ll1_sim += gex_loglik_centered_gaussian_chol(y_sim,
                                                                Ls[t],
                                                                logdet_sigmas[t]);
                }
                ll1_sim /= (double)n_sigmas;

                /* Compute the LRT statistic for the simulated data */
                stat_sim = 2.0 * (ll1_sim - ll0_sim);
                if (stat_sim >= res->lrt_stat[j])
                    ge_count++;
            }
            res->pvals[j] = ((double)ge_count + 1.0) / ((double)n_mc + 1.0);
        }
        else {
            /* Under the boundary null lambda = 0, use the standard 50:50
            mixture of a point mass at zero and chi-square with 1 dof. */
            res->pvals[j] = half_chisq_cdf(res->lrt_stat[j], 1.0, FALSE);
        }
    }

    gex_bh_adjust(res->pvals, res->qvals, res->n_genes);    /* Adjust p-values for multiple testing */
    res->n_significant = gex_count_kept_lrt_genes(res, 0.05);   /* Number of significant genes to keep*/

    /* Free memory */
    free(y);
    free(y_sim);
    if (Sigma_regs != NULL) {
        for (t = 0; t < n_sigmas; t++) {
            if (Sigma_regs[t] != NULL)
                mat_free(Sigma_regs[t]);
        }
        free(Sigma_regs);
    }
    if (Ls != NULL) {
        for (t = 0; t < n_sigmas; t++) {
            if (Ls[t] != NULL)
                mat_free(Ls[t]);
        }
        free(Ls);
    }
    if (Sigma_lambdas != NULL) {
        for (t = 0; t < n_sigmas; t++) {
            if (Sigma_lambdas[t] != NULL)
                mat_free(Sigma_lambdas[t]);
        }
        free(Sigma_lambdas);
    }
    free(logdet_sigmas);
    free(jitters);

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

    printf("\n");
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

    if (res->alt_mode == GEX_LRT_ALT_LAMBDA && res->lambda_hat != NULL)
        fprintf(out, "gene\tll_null\tll_alt\tlambda_hat\tlrt_stat\tp_value\tq_value\tkeep\n");
    else
        fprintf(out, "gene\tll_null\tll_alt\tlrt_stat\tp_value\tq_value\tkeep\n");

    for (i = 0; i < res->n_genes; i++) {
        int keep = (res->qvals[i] <= max_q && res->lrt_stat[i] > 0.0);
        if (res->alt_mode == GEX_LRT_ALT_LAMBDA && res->lambda_hat != NULL) {
            fprintf(out, "%s\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%s\n",
                    gex->gene_names[i],
                    res->ll_null[i],
                    res->ll_alt[i],
                    res->lambda_hat[i],
                    res->lrt_stat[i],
                    res->pvals[i],
                    res->qvals[i],
                    (keep ? "yes" : "no"));
        }
        else {
            fprintf(out, "%s\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%s\n",
                    gex->gene_names[i],
                    res->ll_null[i],
                    res->ll_alt[i],
                    res->lrt_stat[i],
                    res->pvals[i],
                    res->qvals[i],
                    (keep ? "yes" : "no"));
        }
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
    if (res->ll_null != NULL)
        free(res->ll_null);
    if (res->ll_alt != NULL)
        free(res->ll_alt);
    if (res->lambda_hat != NULL)
        free(res->lambda_hat);
    free(res);
}

/* Filter genes based on LRT and Moran's I results 
to keep only those passing the filter(s) with the 
given significance and signal strength thresholds.
Returns a pointer to a new GexMatrix containing only 
the filtered genes. */
GexMatrix *gex_filter_genes_by_results(GexMatrix *gex,
                                       GexMoransResult *morans,
                                       GexLRTResult *lrt,
                                       GexFilterMode mode,
                                       double max_q,
                                       double min_i) {
    int i, j;   /* Loop indices */
    int out_j = 0;  /* Index for the output matrix */
    int nkeep = 0;  /* Number of genes retained */
    GexMatrix *out = NULL;  /* Output matrix */

    if (gex == NULL)
        return NULL;
    if ((mode == GEX_FILTER_MORAN || mode == GEX_FILTER_BOTH) &&
        (morans == NULL || gex->X->ncols != morans->n_genes))
        return NULL;
    if ((mode == GEX_FILTER_LRT || mode == GEX_FILTER_BOTH) &&
        (lrt == NULL || gex->X->ncols != lrt->n_genes))
        return NULL;

    /* Count how many genes pass the filter(s) to determine the size of the output matrix */
    for (j = 0; j < gex->X->ncols; j++) {
        if (gex_keep_gene(morans, lrt, j, mode, max_q, min_i))
            nkeep++;
    }
    if (nkeep <= 0) {
        fprintf(stderr, "ERROR: no genes passed the filter(s)\n");
        return NULL;
    }

    /* Initialize the output matrix */
    out = scalloc(1, sizeof(GexMatrix));
    out->X = mat_new(gex->X->nrows, nkeep);
    out->cell_names = smalloc(gex->X->nrows * sizeof(char *));
    out->gene_names = smalloc(nkeep * sizeof(char *));

    /* Copy cell names */
    for (i = 0; i < out->X->nrows; i++) {
        out->cell_names[i] = strdup(gex->cell_names[i]);
        if (out->cell_names[i] == NULL) {
            gex_free_matrix_data(out);
            return NULL;
        }
    }

    /* Fill the output matrix with passing genes  */
    for (j = 0; j < gex->X->ncols; j++) {
        if (gex_keep_gene(morans, lrt, j, mode, max_q, min_i)) {
            /* Copy gene name that passed the filter(s) */
            out->gene_names[out_j] = strdup(gex->gene_names[j]);
            if (out->gene_names[out_j] == NULL) {
                gex_free_matrix_data(out);
                return NULL;
            }
            /* Copy expression values for the passing gene */
            for (i = 0; i < gex->X->nrows; i++)
                mat_set(out->X, i, out_j, mat_get(gex->X, i, j));
            out_j++;
        }
    }

    return out;
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
    if (out == NULL) {
        fprintf(stderr, "ERROR: failed to open %s for writing: %s\n",
                filename, strerror(errno));
        return -1;
    }

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

int gex_write_model(const char *outprefix,
                                GexMatrix *gex,
                                Matrix *L,
                                Matrix *Z,
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

    if (outprefix == NULL || gex == NULL || L == NULL || Z == NULL ||
        L == NULL || Z == NULL || cell_names == NULL || gene_names == NULL ||
        k <= 0 || sigma2_latent == NULL)
        return 1;

    /* Draw latent factor names incrementally */
    factor_names = generate_factor_names(k);
    if (factor_names == NULL)
        return 1;

    snprintf(summary_path, sizeof(summary_path), "%s.summary.tsv", outprefix);
    snprintf(z_path, sizeof(z_path), "%s.Z.tsv", outprefix);
    snprintf(l_path, sizeof(l_path), "%s.L.tsv", outprefix);
    snprintf(expr_path, sizeof(expr_path), "%s.expr.tsv", outprefix);

    /* Write out the summary parameters file to match the format used
    by model fitting output */
    summary_out = fopen(summary_path, "w");
    if (summary_out == NULL)
        return 1;
    fprintf(summary_out, "parameter\tvalue\n");
    fprintf(summary_out, "n_cells\t%d\n", gex->X->nrows);
    fprintf(summary_out, "n_genes\t%d\n", gex->X->ncols);
    fprintf(summary_out, "k\t%d\n", k);
    fprintf(summary_out, "sigma2_obs\t%.17g\n", sigma2_obs);
    for (j = 0; j < k; j++)
        fprintf(summary_out, "sigma2_latent_LF%d\t%.17g\n", j + 1, sigma2_latent[j]);
    fclose(summary_out);
    summary_out = NULL;

    /* Write out the simulated matrices */
    if (gex_write_labeled_matrix_tsv(expr_path, gex->X, cell_names, gex->X->nrows,
                                     gene_names, gex->X->ncols, "cell") != 0)
        return 1;
    if (gex_write_labeled_matrix_tsv(z_path, Z, cell_names, gex->X->nrows,
                                     factor_names, k, "cell") != 0)
        return 1;
    if (gex_write_labeled_matrix_tsv(l_path, L, factor_names, k,
                                     gene_names, gex->X->ncols, "factor") != 0)
        return 1;

    /* Free memory */
    if (summary_out != NULL)
        fclose(summary_out);
    if (factor_names != NULL)
        free_string_array(factor_names, k);

    return 0;
}

/* Use the provides latent factors . */
int gex_simulate_from_latent_factors(Matrix *Z,
                                     char **cell_names,
                                     int n_cells,
                                     int k,
                                     int n_genes,
                                     double sigma2_obs,
                                     unsigned int seed,
                                     Matrix **L_out,
                                     GexMatrix **gex_out) {
    int i, j, d;    /* Loop indices */
    GexMatrix *gex = NULL;  /* Simulation output gene expression object */
    Matrix *L = NULL;    /* Simulation output gene loadings object */
    char **gene_names = NULL;   /* Gene names */
    unsigned int rng_state = (seed == 0u ? 1u : seed);  /* Random number generator state */

    if (Z == NULL || L_out == NULL || gex_out == NULL || cell_names == NULL || n_cells <= 0 ||
        k <= 0 || n_genes <= 0 || sigma2_obs < 0.0)
        return 1;
    if (Z->nrows != n_cells || Z->ncols != k)
        return 1;

    /* Make sure the simulation output is initialized as empty */
    *gex_out = NULL;
    *L_out = NULL;

    /* Allocate objects in memory */
    gex = scalloc(1, sizeof(GexMatrix));
    L = mat_new(k, n_genes);

    /* Get simulated gene names */
    gene_names = scalloc(n_genes, sizeof(char *));
    generate_gene_names(gene_names, n_genes, NULL);

    if (L == NULL || gex == NULL || gene_names == NULL)
        return 1;

    /* Setup dimensions of L */
    L->nrows = k;
    L->ncols = n_genes;

    /* Draw gene loadings L ~ N(0,1) and rescale each row to have norm
    sqrt(n_genes / k), ensuring each latent dimension contributes
    equal expected magnitude to the noiseless gene expression data. */
    for (d = 0; d < k; d++) {
        double row_ss = 0.0;
        double target_norm = sqrt((double)n_genes / (double)k);
        for (j = 0; j < n_genes; j++) {
            double val = rand_normal(&rng_state);
            mat_set(L, d, j, val);
            row_ss += val * val;
        }
        if (row_ss > 0.0) {
            double row_scale = target_norm / sqrt(row_ss);
            for (j = 0; j < n_genes; j++)
                mat_set(L, d, j, row_scale * mat_get(L, d, j));
        }
    }

    /* Initialize the gene expression matrix */
    gex->X = mat_new(n_cells, n_genes);
    if (gex->X == NULL)
        return 1;

    /* Compute the noiseless expression matrix from the 
    simulated Z and L matrix factorization. */
    mat_mult(gex->X, Z, L);

    /* Initialize the cell and gene names */
    gex->cell_names = scalloc(n_cells, sizeof(char *));
    gex->gene_names = scalloc(n_genes, sizeof(char *));
    for (i = 0; i < n_cells; i++) {
        gex->cell_names[i] = strdup(cell_names[i]);
        if (gex->cell_names[i] == NULL)
            return 1;
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
                val += sqrt(sigma2_obs) * rand_normal(&rng_state);
            mat_set(gex->X, i, j, val);
        }
    }

    *L_out = L;
    L = NULL;
    *gex_out = gex;
    gex = NULL;

    /* Free memory */
    if (gene_names != NULL)
        free_string_array(gene_names, n_genes);
    if (L != NULL)
        mat_free(L);
    if (gex != NULL)
        gex_free_matrix_data(gex);

    return 0;
}
