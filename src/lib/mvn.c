#include "mvn.h"

#include <phast/eigen.h>
#include <phast/misc.h>

#include <assert.h>
#include <math.h>
#include <stdlib.h>

static void mvn_update_type(MVN *mvn) {
    int mean_zero = TRUE;
    int diag_covar = TRUE;
    int ident_covar = TRUE;
    int i, j;

    assert(mvn->sigma != NULL);

    if (mvn->mu == NULL)
        mean_zero = FALSE;
    else {
        for (i = 0; mean_zero == TRUE && i < mvn->dim; i++) {
            if (vec_get(mvn->mu, i) != 0)
                mean_zero = FALSE;
        }
    }

    for (i = 0; diag_covar == TRUE && i < mvn->dim; i++) {
        for (j = 0; diag_covar == TRUE && j < i; j++) {
            if (mat_get(mvn->sigma, i, j) != 0 ||
                mat_get(mvn->sigma, j, i) != 0)
                diag_covar = FALSE;
        }
    }

    if (diag_covar == FALSE)
        ident_covar = FALSE;
    else {
        for (i = 0; ident_covar == TRUE && i < mvn->dim; i++) {
            if (mat_get(mvn->sigma, i, i) != 1)
                ident_covar = FALSE;
        }
    }

    if (mean_zero && ident_covar)
        mvn->type = MVN_STD;
    else if (ident_covar)
        mvn->type = MVN_IDENTITY;
    else if (diag_covar)
        mvn->type = MVN_DIAG;
    else
        mvn->type = MVN_GEN;
}

static void mvn_sample_std(Vector *retval) {
    int i;

    for (i = 0; i < retval->size; i += 2) {
        double u1 = unif_rand();
        double u2 = unif_rand();
        double radius = sqrt(-2.0 * log(u1));
        double angle = 2.0 * M_PI * u2;

        vec_set(retval, i, radius * cos(angle));
        if (i + 1 < retval->size)
            vec_set(retval, i + 1, radius * sin(angle));
    }
}

static void mvn_map_std(MVN *mvn, Vector *rv) {
    int i, j;

    if (mvn->type == MVN_IDENTITY) {
        vec_plus_eq(rv, mvn->mu);
    }
    else if (mvn->type == MVN_DIAG) {
        for (i = 0; i < mvn->dim; i++) {
            vec_set(rv, i,
                    vec_get(mvn->mu, i) +
                    sqrt(mat_get(mvn->sigma, i, i)) * vec_get(rv, i));
        }
    }
    else if (mvn->type == MVN_GEN) {
        Vector *tmp = vec_create_copy(rv);

        if (mvn->cholL != NULL) {
            for (i = 0; i < mvn->dim; i++) {
                double covarsum = 0.0;
                for (j = 0; j <= i; j++)
                    covarsum += mat_get(mvn->cholL, i, j) * vec_get(tmp, j);
                vec_set(rv, i, vec_get(mvn->mu, i) + covarsum);
            }
        }
        else if (mvn->evals != NULL) {
            for (i = 0; i < mvn->dim; i++) {
                double covarsum = 0.0;
                for (j = 0; j < mvn->dim; j++) {
                    covarsum += mat_get(mvn->evecs, i, j) *
                        sqrt(vec_get(mvn->evals, j)) * vec_get(tmp, j);
                }
                vec_set(rv, i, vec_get(mvn->mu, i) + covarsum);
            }
        }
        else {
            die("ERROR in mvn_map_std: call mvn_preprocess before sampling a general MVN.\n");
        }

        vec_free(tmp);
    }
}

MVN *mvn_new(int dim, Vector *mu, Matrix *sigma) {
    MVN *mvn = smalloc(sizeof(MVN));

    assert(dim > 0);
    mvn->dim = dim;

    if (mu == NULL) {
        mvn->mu = vec_new(dim);
        vec_zero(mvn->mu);
    }
    else {
        if (mu->size != dim)
            die("ERROR in mvn_new: bad dimension in mean vector.\n");
        mvn->mu = mu;
    }

    if (sigma == NULL) {
        mvn->sigma = mat_new(dim, dim);
        mat_set_identity(mvn->sigma);
    }
    else {
        if (sigma->nrows != dim || sigma->ncols != dim)
            die("ERROR in mvn_new: bad dimension in covariance matrix.\n");
        mvn->sigma = sigma;
    }

    mvn->cholL = NULL;
    mvn->evals = NULL;
    mvn->evecs = NULL;
    mvn_update_type(mvn);
    return mvn;
}

void mvn_free(MVN *mvn) {
    if (mvn == NULL)
        return;
    if (mvn->mu != NULL)
        vec_free(mvn->mu);
    if (mvn->sigma != NULL)
        mat_free(mvn->sigma);
    if (mvn->cholL != NULL)
        mat_free(mvn->cholL);
    if (mvn->evals != NULL)
        vec_free(mvn->evals);
    if (mvn->evecs != NULL)
        mat_free(mvn->evecs);
    free(mvn);
}

void mvn_preprocess(MVN *mvn, unsigned int force_eigen) {
    int retval = 1;

    if (mvn->type != MVN_GEN)
        return;

    if (force_eigen == FALSE) {
        if (mvn->cholL == NULL)
            mvn->cholL = mat_new(mvn->dim, mvn->dim);

        retval = mat_cholesky(mvn->cholL, mvn->sigma);
        if (retval == 0) {
            if (mvn->evals != NULL) {
                vec_free(mvn->evals);
                mvn->evals = NULL;
            }
            if (mvn->evecs != NULL) {
                mat_free(mvn->evecs);
                mvn->evecs = NULL;
            }
        }
    }

    if (force_eigen == TRUE || retval != 0) {
        int i;

        if (mvn->cholL != NULL) {
            mat_free(mvn->cholL);
            mvn->cholL = NULL;
        }
        if (mvn->evals == NULL)
            mvn->evals = vec_new(mvn->dim);
        if (mvn->evecs == NULL)
            mvn->evecs = mat_new(mvn->dim, mvn->dim);

        if (mat_diagonalize_sym(mvn->sigma, mvn->evals, mvn->evecs) != 0)
            die("ERROR in mvn_preprocess: matrix diagonalization failed.\n");

        for (i = 0; i < mvn->dim; i++) {
            if (fabs(vec_get(mvn->evals, i)) < 1e-6)
                vec_set(mvn->evals, i, 1e-6);
            if (vec_get(mvn->evals, i) < 0)
                die("ERROR in mvn_preprocess: covariance matrix is not positive definite.\n");
        }
    }
}

void mvn_sample(MVN *mvn, Vector *retval) {
    if (mvn->dim != retval->size)
        die("ERROR in mvn_sample: bad dimensions.\n");

    mvn_sample_std(retval);
    if (mvn->type != MVN_STD)
        mvn_map_std(mvn, retval);
}
