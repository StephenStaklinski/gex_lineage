/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025-2026, Cold Spring Harbor Laboratory
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */

/* ELBO estimation based on Taylor approximation to reduce number of
   NJ calls */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <assert.h>
#include <float.h>
#include <taylor.h>
#include <upgma.h>
#include <gradients.h>
#include <likelihoods.h>
#include <variational.h>
#include <hutchinson.h>
#include <nuisance.h>
#include <backprop.h>
#include <geometry.h>

TaylorData *tay_new(CovarData *data) {
  TaylorData *td = smalloc(sizeof(TaylorData));
  td->covar_data = data;

  /* set dimensionalities */
  td->nseqs = data->nseqs;
  td->nbranches = data->nseqs * 2 - 2; 
  td->dim = data->dim;
  td->fulld = data->nseqs * data->dim;
  td->ndist = data->nseqs * (data->nseqs - 1) / 2;
  
  /* allocate workspace memory */
  td->base_grad = vec_new(td->nbranches);
  vec_zero(td->base_grad);
  td->Jbx = mat_new(td->nbranches, td->fulld);
  td->JbxT = mat_new(td->fulld, td->nbranches);
  td->tmp_x1 = vec_new(td->fulld);
  td->tmp_x2 = vec_new(td->fulld);
  td->tmp_dD = vec_new(td->ndist);
  td->tmp_dy = vec_new(td->fulld);
  td->tmp_extra = vec_new(td->fulld); /* for flows */

  /* auxiliary data stored during gradient computation */
  td->y = vec_new(td->fulld);
  td->nb = nj_new_neighbors(td->nseqs);

  /* these will be set later */
  td->mmvn = NULL;
  td->mod = NULL;

  /* scheduling directives */
  td->iter = 0;
  td->T_cache = 0.0;
  td->elbo_bias = 0.0;
  td->siggrad_cache = NULL; /* will be allocated later */
  td->warmup = 50; /* number of iterations before updates begin */
  td->period = 30; /* update period */
  td->beta = 0.3;  
  
  return td;
}

void tay_free(TaylorData *td) {
  vec_free(td->base_grad);
  mat_free(td->Jbx);
  mat_free(td->JbxT);
  vec_free(td->tmp_x1);
  vec_free(td->tmp_x2);
  vec_free(td->tmp_dD);
  vec_free(td->tmp_dy);
  vec_free(td->tmp_extra);
  if (td->siggrad_cache != NULL)
    vec_free(td->siggrad_cache);
  vec_free(td->y);
  sfree(td);
}

/* estimate key components of the ELBO by a Taylor approximation
   around the mean.  Returns the expected log likelihood.  If
   non-NULL, &half_tr will be populated with 1/2 tr(H S), which is the
   second-order term in the Taylor expansion. */
double nj_elbo_taylor(TreeModel *mod, multi_MVN *mmvn, CovarData *data,
                      Vector *grad, Vector *nuis_grad, double *lprior,
                      double *migll, double *ll_at_mean) {
  double ll;

  /* make sure mmvn and mod are accessible from TaylorData */
  TaylorData *td = data->taylor;
  td->mmvn = mmvn;
  td->mod = mod;

  /* first calculate log likelihood at the mean */
  vec_zero(grad);
  Vector *mu = vec_new(mmvn->n * mmvn->d);
  mmvn_save_mu(mmvn, mu); /* express mean as a single vector */

  ll = nj_compute_model_grad(mod, mmvn, mu, NULL,
                             grad, data, NULL, migll);

  /* do after calling nj_compute_model_grad so tree is defined */
  assert(mod->tree->nnodes - 1 == td->nbranches);  

  if (!isfinite(ll)) {
    vec_free(mu);
    return ll;  /* let caller handle via zero_likl recovery */
  }

  /* also handle log prior and nuisance gradient if needed */
  if (data->treeprior != NULL) {
    Vector *prior_grad = vec_new(grad->size);
    *lprior = tp_compute_log_prior(mod, data, prior_grad);
    vec_plus_eq(grad, prior_grad);
    vec_free(prior_grad);
  }
  else 
    *lprior = 0.0;

  if (nuis_grad != NULL) {
    vec_zero(nuis_grad);
    nj_update_nuis_grad(mod, data, nuis_grad);
  }
  
  /* note that there is no first-order term in the Taylor approximation
     because we are expanding around the mean */
  
  /* now add the second-order terms for the Taylor expansion.  These
     terms are equal to 1/2 tr(H Sigma), where H is the Hessian of the
     ELBO.  But we can simplify this expression by considering the
     chain of transformations from the standard normal to the
     phylogeny and likelihood.  The NJ transformation is linear up to
     a choice of neighbors, the tranformation from z to x is linear.
     If we also assume the distances are locally linear, then all
     curvature comes from the phylogenetic likelihood function, and we
     can approximate tr(H Sigma) by tr(H S), where S is a square
     matrix of dimension nbranches x nbranches representing a product
     of Sigma and the relevant Jacobian matrices [see
     manuscript for detailed derivation] */
 
  /* we will approximate tr(H S) using a Hutchinson trace estimator */
  
  /* CHECK: need to consider curvature of the flows? */
  /* CHECK: do we need to propagate gradients wrt to the variance
     terms through to the migll?  */
  /* CHECK: are there also second order terms to consider for the log prior? */

  int sigdim = grad->size - data->taylor->fulld; /* number of covariance parameters */
  if (td->siggrad_cache == NULL)   /* set up first time */
    td->siggrad_cache = vec_new(sigdim);

  if (td->iter == 0) {
    vec_zero(td->siggrad_cache);
    td->T_cache = 0.0;
  }
  
  /* decide whether and how to update trace and derivatives.  This is expensive
     so we only do it intermittently */
  int do_refresh = (td->iter >= td->warmup) && ((td->iter - td->warmup) % td->period == 0);

  /* if variance has hit its floor, there's no sense in updating the trace */
  if (do_refresh && nj_var_at_floor(mmvn, data))
    do_refresh = FALSE;

  if (do_refresh) {
    Vector *grad_sigma = vec_new(sigdim);
    vec_zero(grad_sigma);

    /* build Jacobians Jbx and JbxT once at the mean point */
    tay_prep_jacobians(data->taylor, mod, mu);
  
    /* Compute scalar T and its covariance gradient */
    double T = hutch_tr_plus_grad(tay_HVP, tay_SVP, tay_JTfun, tay_Sigmafun,
                                  tay_SigmaGradfun, data->taylor,
                                  data->taylor->nbranches, data->taylor->fulld,
                                  NHUTCH_SAMPLES, grad_sigma);

    if (isfinite(T)) { /* guard against runaway values */
      if (td->iter == td->warmup) {
        td->T_cache = T; /* initialize on first update */
        vec_copy(td->siggrad_cache, grad_sigma);
      }
      else {
        td->T_cache = (1.0 - td->beta) * td->T_cache + td->beta * T;
        for (int j = 0; j < sigdim; j++) {
          double old = vec_get(td->siggrad_cache, j);
          double nw  = vec_get(grad_sigma, j);
          vec_set(td->siggrad_cache, j, (1.0 - td->beta)*old + td->beta*nw);
        }
      }
    }

    vec_free(grad_sigma);
  }
  td->iter++;
  
  /* capture likelihood at mean before adding trace correction */
  if (ll_at_mean != NULL) *ll_at_mean = ll;

  /* add 1/2 T to log likelihood and scale gradient by 1/2; always used cached versions */
  ll += 0.5 * td->T_cache;

  /* add covariance part of gradient into grad */
  int offset = data->taylor->fulld; 
  for (int j = 0; j < sigdim; j++)
    vec_set(grad, offset + j, vec_get(grad, offset + j)
                              + 0.5 * data->lambda * vec_get(td->siggrad_cache, j));
  
  /* free everything and return */
  vec_free(mu);

  /* we also have to free the last tree in the tree model to avoid a
     memory leak */
  tr_free(mod->tree);
  mod->tree = NULL;
  
  return ll; 
}

/* helper functions for Hessian-vector product computation */

/* save current branch lengths from tree into vector bl */
static inline
void tr_save_branch_lengths(TreeNode *root, Vector *bl) {
  assert(root->nnodes - 1 == bl->size);
  for (int i = 0; i < bl->size; i++) {
    TreeNode *n = lst_get_ptr(root->nodes, i);
    assert(n->parent != NULL); /* root must be node with id bl->size */
    vec_set(bl, i, n->dparent);
  }
}

/* smooth floor function to avoid discontinuous derivatives */
static inline double smooth_floor(double x, double xmin) {
  /* softness parameter: larger = sharper floor */
  const double alpha = 50.0;   /* safe default */

  double z = alpha * (x - xmin);

  /* numerically stable softplus */
  if (z > 30.0)
    return x;                  /* already well above floor */
  else if (z < -30.0)
    return xmin;               /* well below floor */
  else
    return xmin + log1p(exp(z)) / alpha;
}

/* adjust branch lengths by incrementing scaled values in vector bl;
   excludes root */
static inline
void tr_incr_branch_lengths(TreeNode *root, Vector *bl, double scale) {
  assert(root->nnodes - 1 == bl->size);
  for (int i = 0; i < bl->size; i++) {
    TreeNode *n = lst_get_ptr(root->nodes, i);
    double raw = n->dparent + scale * vec_get(bl, i);
    /* smooth floor instead of hard clamp */
    n->dparent = smooth_floor(raw, 0.0);
  }
}

/* restore branch lengths from vector bl */
static inline
void tr_restore_branch_lengths(TreeNode *root, Vector *bl) {
  assert(root->nnodes - 1 == bl->size);
  for (int i = 0; i < bl->size; i++) {
    TreeNode *n = lst_get_ptr(root->nodes, i);
    n->dparent = vec_get(bl, i);
  }
}

/* Finite-difference version of Hessian-vector product based on
   directional gradient.  Computes H v ≈ (grad(theta + eps*v) -
   grad(theta)) / eps */
void tay_HVP(Vector *out, Vector *v, void *dat)
{
  TaylorData *tay_data = (TaylorData *)dat;
  CovarData  *data     = tay_data->covar_data;
  TreeModel  *mod      = tay_data->mod;

  assert(v->size   == tay_data->nbranches);
  assert(out->size == tay_data->nbranches);

  /* Save original non-root branch lengths */
  Vector *origbl = vec_new(tay_data->nbranches);
  tr_save_branch_lengths(mod->tree, origbl);

  /* retrieve baseline gradient g0 at current b */
  Vector *g0 = tay_data->base_grad;

  /* Pick an eps that is local relative to v */
  double maxabs = 0.0;
  for (int i = 0; i < v->size; i++) {
    double a = fabs(vec_get(v, i));
    if (a > maxabs) maxabs = a;
  }
  double eps_eff = DERIV_EPS;
  if (maxabs > 1.0) eps_eff /= maxabs;     /* keep max perturbation ~ DERIV_EPS */

  /* avoid hitting the branch-length floor clamp.
     If any component would push b_i below the floor, shrink eps. */
  const double safety   = 0.5;   /* stay away from the kink */

  /* Backtrack eps to avoid clamp-triggering perturbations */
  for (int attempt = 0; attempt < 8; attempt++) {
    int would_clamp = 0;

    for (int i = 0; i < v->size; i++) {
      double vi = vec_get(v, i);
      if (vi < 0.0) {
        double bi = vec_get(origbl, i);
        double newb = bi + eps_eff * vi;
        if (newb <= 0.0) {
          would_clamp = 1;
          break;
        }
      }
    }

    if (!would_clamp)
      break;                /* eps_eff is safe */

    eps_eff *= safety;      /* shrink and retry */
  }

  /* If eps_eff gets absurdly small, bail out with zero HVP (better than spikes) */
  if (eps_eff < 1e-16) {
    vec_zero(out);
    tr_restore_branch_lengths(mod->tree, origbl);
    vec_free(origbl);
    return;
  }

  /* ---------- apply perturbation and compute gradient ---------- */

  /* alternate between forward or backward steps so unbiased in
     expectation */
  static int sign = 1;
  sign = -sign;
  double signed_eps = sign * eps_eff;

  tr_incr_branch_lengths(mod->tree, v, signed_eps);

  /* Compute perturbed gradient g1 into out */
  if (data->crispr_mod != NULL)
    cpr_compute_log_likelihood(data->crispr_mod, out);
  else
    nj_compute_log_likelihood(mod, data, out);

  /* out = (g1 - g0)/eps_eff */
  vec_minus_eq(out, g0);
  vec_scale(out, 1.0 / signed_eps);

  /* cap HVP norm to prevent rare FD blow-ups */
  double nrm = vec_norm(out);
  if (isfinite(nrm) && nrm > TAYLOR_HVP_NORM_CAP) 
    vec_scale(out, TAYLOR_HVP_NORM_CAP / nrm);
  
  /* Restore original branch lengths */
  tr_restore_branch_lengths(mod->tree, origbl);

  vec_free(origbl);
}


/* Compute S_b * v, where S_b = Jbx * Sigma_x * Jbx^T.
   v is a vector in branch-length space (dimension = nbranches).
   out is also in branch-length space.
   data_vd is CovarData*, which must contain Jbx, Sigma, and workspace vectors. */
void tay_SVP(Vector *out, Vector *v, void *dat) {
  TaylorData *tay_data = (TaylorData *)dat;
    
  assert(v->size == tay_data->nbranches);
  assert(out->size == tay_data->nbranches);

  /* tmp_x1 = JbxT * v */
  mat_vec_mult(tay_data->tmp_x1, tay_data->JbxT, v);

  /* tmp_x2 = Sigma * tmp_x_1 */
  tay_sigma_vec_mult(tay_data->tmp_x2, tay_data->mmvn, tay_data->tmp_x1, tay_data->covar_data);

  /* out = Jbx * tmp_x2 */
  mat_vec_mult(out, tay_data->Jbx, tay_data->tmp_x2);
}

void tay_prep_jacobians(TaylorData *tay_data, TreeModel *mod, Vector *x_mean) {
  int nb = tay_data->nbranches;
  int fulld = tay_data->fulld;

  if (tay_data->Jbx == NULL)
    tay_data->Jbx  = mat_new(nb, fulld);

  if (tay_data->JbxT == NULL)
    tay_data->JbxT = mat_new(fulld, nb);

  /* Workspace for reverse-mode J^T e_j */
  Vector *dL_dt = vec_new(nb);
  Vector *dL_dx = vec_new(fulld);
  /* CHECK: use CovarData workspaces instead? */

  /* Loop over each branch length index */
  for (int j = 0; j < nb; j++) {

    /* dL_dt = e_j */
    vec_zero(dL_dt);
    vec_set(dL_dt, j, 1.0);

    /* Use existing reverse-mode path: 
       dL_dt → dL_dD → dL_dy → dL_dx 
       evaluated at the mean point.
    */
    tay_dx_from_dt(dL_dt, dL_dx, mod, tay_data);

    /* Fill column j of Jbx^T */
    for (int k = 0; k < fulld; k++)
      mat_set(tay_data->JbxT, k, j, vec_get(dL_dx, k));
  }

  /* Jbx = transpose(JbxT) */
  mat_trans(tay_data->Jbx, tay_data->JbxT);

  vec_free(dL_dt);
  vec_free(dL_dx);
}

/* Compute J_{bx}^T * dL_dt at the MEAN point.
   Inputs:
   dL_dt : (nbranches)    vector in branch-length space (seed direction)
   Output:
   dL_dx : (n*d)           vector in latent coordinate space
   Uses:
   data->nb      neighbor tape (from mean tree)
   data->dist    distances at mean
   data->y       embedding at mean
   data->rf, pf  flows
   Everything else must be precomputed.

   This is JUST the reverse-mode Jacobian chain for the mean point.
*/
void tay_dx_from_dt(Vector *dL_dt, Vector *dL_dx, TreeModel *mod,
                    TaylorData *tay_data) {
  CovarData *data = tay_data->covar_data;
  int n     = tay_data->nseqs;
  int dim   = tay_data->dim;

  /* Workspace from CovarData */
  Vector *dL_dD = tay_data->tmp_dD;   /* size ndist */
  Vector *dL_dy = tay_data->tmp_dy;   /* size fulld    */

  vec_zero(dL_dD);
  vec_zero(dL_dy);
  vec_zero(dL_dx);

  /* Branch lengths → distances (reverse) */

  if (data->ultrametric)
    upgma_dL_dD_from_tree(mod->tree, dL_dt, dL_dD);
  else
    nj_dL_dD_from_neighbors(tay_data->nb, dL_dt, dL_dD);

  /* Distances → embedding y  (reverse) */

  if (data->hyperbolic) {
    /* hyperbolic reverse-mode */
    int i, j, d;

    /* Precompute needed geometric constants */
    double *x0 = (double *)smalloc(n * sizeof(double));
    Vector *y  = tay_data->y;         /* size fulld */
    Matrix *dist = data->dist;    /* n x n  */

    for (i = 0; i < n; i++) {
      double ss = 1.0;
      int base = i * dim;
      for (d = 0; d < dim; d++) {
        double yi = vec_get(y, base + d);
        ss += yi * yi;
      }
      x0[i] = sqrt(ss);
    }

    for (i = 0; i < n; i++) {
      int base_i = i * dim;

      for (j = i+1; j < n; j++) {
        int base_j = j * dim;

        double weight = vec_get(dL_dD, nj_i_j_to_dist(i, j, n));
        if (weight == 0.0) continue;

        double Dij = mat_get(dist, i, j);
        if (Dij > 10) weight *= (10 / Dij);

        /* Compute u and derivatives as in smarter() */
        double dot_spatial = 0.0;
        for (d = 0; d < dim; d++)
          dot_spatial += vec_get(y, base_i + d) *
            vec_get(y, base_j + d);

        double u = x0[i] * x0[j] - dot_spatial;
        double denom_inv = d_acosh_du_stable(u);
        double pref = (1.0 / sqrt(data->negcurvature))
          * (denom_inv / data->pointscale);

        for (d = 0; d < dim; d++) {
          int idx_i = base_i + d;
          int idx_j = base_j + d;

          double yi = vec_get(y, idx_i);
          double yj = vec_get(y, idx_j);

          double gi = pref * (-yj + (x0[j]/x0[i])*yi);
          double gj = pref * (-yi + (x0[i]/x0[j])*yj);

          vec_set(dL_dy, idx_i,
                  vec_get(dL_dy, idx_i) + weight * gi);
          vec_set(dL_dy, idx_j,
                  vec_get(dL_dy, idx_j) + weight * gj);
        }
      }
    }
    sfree(x0);

    /* no flows in hyperbolic case */
    vec_copy(dL_dx, dL_dy);
  }

  else {  /* Euclidean reverse-mode */

    int i, j, d;
    Vector *y = tay_data->y;
    Matrix *dist = data->dist;

    for (i = 0; i < n; i++) {
      int base_i = i * dim;

      for (j = i+1; j < n; j++) {
        int base_j = j * dim;

        double weight = vec_get(dL_dD, nj_i_j_to_dist(i, j, n));
        if (weight == 0.0) continue;

        double Dij = mat_get(dist, i, j);
        if (Dij < 1e-15) Dij = 1e-15;

        for (d = 0; d < dim; d++) {

          int idx_i = base_i + d;
          int idx_j = base_j + d;

          double diff = vec_get(y, idx_i) - vec_get(y, idx_j);
          double grad_contrib =
            weight * diff /
            (Dij * data->pointscale * data->pointscale);

          vec_set(dL_dy, idx_i,
                  vec_get(dL_dy, idx_i) + grad_contrib);
          vec_set(dL_dy, idx_j,
                  vec_get(dL_dy, idx_j) - grad_contrib);
        }
      }
    }

    /* Backprop through flows: y → x */
    if (data->rf != NULL && data->pf != NULL) {
      Vector *tmp = tay_data->tmp_extra;
      rf_backprop(data->rf, y, tmp, dL_dy);
      pf_backprop(data->pf, y, dL_dx, tmp);
    }
    else if (data->rf != NULL) {
      rf_backprop(data->rf, y, dL_dx, dL_dy);
    }
    else if (data->pf != NULL) {
      pf_backprop(data->pf, y, dL_dx, dL_dy);
    }
    else {
      vec_copy(dL_dx, dL_dy);
    }
  }
}

/* Sigma * v depending on parameterization */
void tay_sigma_vec_mult(Vector *out, multi_MVN *mmvn, Vector *v, CovarData *data) {
  int n = mmvn->n;
  int d = mmvn->d;     /* embedding dimension */
  int nx = n * d;

  assert(out->size == nx && v->size == nx);

  if (data->type == CONST || data->type == DIST) {
    for (int i = 0; i < nx; i++)
      vec_set(out, i, data->lambda * vec_get(v, i)); 
    return;
  }

  else if (data->type == DIAG) {
    for (int i = 0; i < nx; i++) {
      double s = mat_get(mmvn->mvn->sigma, i, i);  
      vec_set(out, i, s * vec_get(v, i));
    }
    return;
  }

 
  else if (data->type == LOWR) {

    /* In the LOWR case, a vector v (nx = n*d) is reshaped into d blocks
       v = [v^(1); v^(2); ...; v^(d)] where each v^(k) ∈ R^n Σv = [R(Rᵀ
       v^(1)); ...; R(Rᵀ v^(d))] */
    
    Matrix *R = mmvn->mvn->lowR;   /* n x r */
    int r = R->ncols;
    assert(R->nrows == n);

    /* temporary vector of length r */
    Vector *tmp_r = vec_new(r);

    /* operate dim-by-dim */
    for (int k = 0; k < d; k++) {

      /* pointers to the k-th block (size n) */
      int offset = k * n;

      /* Compute tmp_r = Rᵀ * v_block */
      vec_zero(tmp_r);
      for (int i = 0; i < n; i++) {
        double vi = vec_get(v, offset + i);
        for (int j = 0; j < r; j++) {
          double rij = mat_get(R, i, j);
          vec_set(tmp_r, j, vec_get(tmp_r, j) + rij * vi);
        }
      }

      /* Compute out_block = R * tmp_r */
      for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < r; j++)
          sum += mat_get(R, i, j) * vec_get(tmp_r, j);

        vec_set(out, offset + i, sum);
      }
    }

    vec_free(tmp_r);
    return;
  }

  else 
    die("ERROR in tay_sigma_vec_mult: unknown covariance type.\n");
}

/* Compute gradient of uᵀ Σ u wrt the covariance parameters.  Adds
   results into out (size = data->params->size).  u is the
   latent-space vector Jᵀ z. */
void tay_sigma_grad_mult(Vector *out, Vector *p, Vector *q, multi_MVN *mmvn,
                         CovarData *data) {

  int n = mmvn->n;
  int d = mmvn->d;
  int nx = n * d;

  assert(p->size == nx);
  assert(q->size == nx);
  
  /* CONST and DIST share a scalar λ */
  if (data->type == CONST || data->type == DIST) {
    double dotpq = vec_inner_prod(p, q);
    vec_set(out, 0, vec_get(out, 0) + dotpq);
    return;
  }

  /* DIAG case: Σ = diag(exp(sigma_params)) */
  else if (data->type == DIAG) {
    for (int i = 0; i < nx; i++) {
      double lam = mat_get(mmvn->mvn->sigma, i, i);
      double gi  = lam * vec_get(p, i) * vec_get(q, i);
      vec_set(out, i, vec_get(out, i) + gi);
    }
    return;
  }

  /* LOWR case: Σ = R Rᵀ, params = R[i,j] */
  else if (data->type == LOWR) {
    Matrix *R = mmvn->mvn->lowR; /* n x r */
    int r = R->ncols;

    Vector *Rt_p = vec_new(r);
    Vector *Rt_q = vec_new(r);

    for (int k = 0; k < d; k++) {
      int off = k * n;

      vec_zero(Rt_p);
      vec_zero(Rt_q);

      /* Rt_p = Rᵀ p_block ; Rt_q = Rᵀ q_block */
      for (int i = 0; i < n; i++) {
        double pi = vec_get(p, off + i);
        double qi = vec_get(q, off + i);
        for (int j = 0; j < r; j++) {
          double Rij = mat_get(R, i, j);
          vec_set(Rt_p, j, vec_get(Rt_p, j) + Rij * pi);
          vec_set(Rt_q, j, vec_get(Rt_q, j) + Rij * qi);
        }
      }

      /* d/dR[i,j] = p_i * (Rᵀ q)_j + q_i * (Rᵀ p)_j */
      for (int i = 0; i < n; i++) {
        double pi = vec_get(p, off + i);
        double qi = vec_get(q, off + i);
        for (int j = 0; j < r; j++) {
          int idx = i * r + j;
          double g = pi * vec_get(Rt_q, j) + qi * vec_get(Rt_p, j);
          vec_set(out, idx, vec_get(out, idx) + g);
        }
      }
    }

    vec_free(Rt_p);
    vec_free(Rt_q); 
    return;
  }

  else {
    die("ERROR in tay_sigma_grad_mult: unknown covariance type\n");
  }
}

/* out = Jbx^T * v_branch */
void tay_JTfun(Vector *out, Vector *v, void *userdata)
{
    TaylorData *td = (TaylorData *)userdata;

    /* out ∈ R^{fulld}, v ∈ R^{nbranches} */
    assert(out->size == td->fulld);
    assert(v->size   == td->nbranches);

    mat_vec_mult(out, td->JbxT, v);   /* uses your existing mat_vec_mult */
}

/* out = Sigma * v_latent */
void tay_Sigmafun(Vector *out, Vector *v, void *userdata)
{
    TaylorData *td = (TaylorData *)userdata;
    tay_sigma_vec_mult(out, td->mmvn, v, td->covar_data);
}

/* grad_sigma += ∂/∂σ ( v_lat^T Σ v_lat )
   NOTE: no factor 1/2 and no 1/nprobe scaling here — the Hutchinson driver handles it.
*/
void tay_SigmaGradfun(Vector *grad_sigma, Vector *p_lat, Vector *q_lat, void *userdata)
{
    TaylorData *td = (TaylorData *)userdata;

    /* add contribution of this probe */
    tay_sigma_grad_mult(grad_sigma, p_lat, q_lat, td->mmvn, td->covar_data);
}


/* below is a hybrid of the full Monte Carlo and full Taylor methods
 * that uses Monte Carlo only for variance parameters.  It uses the
 * caching and EMA strategy of the Taylor method to avoid recomputing
 * the full set of gradients at every step. */

/* Blend only the covariance / variance block of gradients.
   Assumes:
     grad_full size = fulld + sigdim
     siggrad_cache size = sigdim
*/
static inline void blend_variance_block(Vector *siggrad_cache,
                                        Vector *mc_grad,
                                        int fulld,
                                        double beta) {
  int sigdim = siggrad_cache->size;

  for (int j = 0; j < sigdim; j++) {
    double old = vec_get(siggrad_cache, j);
    double nw  = vec_get(mc_grad, fulld + j);
    vec_set(siggrad_cache, j, (1.0 - beta) * old + beta * nw);
  }
}

static inline void add_cached_variance_grad(Vector *grad,
                                            Vector *siggrad_cache,
                                            int fulld) {
  int sigdim = siggrad_cache->size;

  for (int j = 0; j < sigdim; j++) {
    vec_set(grad, fulld + j,
            vec_get(grad, fulld + j) + vec_get(siggrad_cache, j));
  }
}

/* Hybrid ELBO:
   - Mean gradient from Taylor (log likelihood at mu)
   - Variance gradient + curvature correction from MC (cached + smoothed)
*/
double nj_elbo_hybrid(TreeModel *mod, multi_MVN *mmvn, CovarData *data,
                      int nminibatch, Vector *grad, Vector *nuis_grad,
                      double *lprior, double *migll, double *ll_at_mean) {
  TaylorData *td = data->taylor;
  double ll_mu;

  int fulld  = td->fulld;
  int sigdim = grad->size - fulld;

  /* ---------------------------------------
   * 1. Log likelihood and gradient at mean
   * --------------------------------------- */

  vec_zero(grad);

  Vector *mu = vec_new(fulld);
  mmvn_save_mu(mmvn, mu);

  ll_mu = nj_compute_model_grad(mod, mmvn,
                                mu,
                                NULL,        /* points_std == NULL → no variance grads */
                                grad,
                                data,
                                NULL,
                                migll);

  if (!isfinite(ll_mu)) {
    vec_free(mu);
    return ll_mu;  /* let caller handle via zero_likl recovery */
  }

  /* Prior + nuisance gradient.  Only add the mean block (first fulld
     elements) here; the sigma block of the prior gradient will come
     from the MC cache to avoid double-counting. */
  if (data->treeprior != NULL) {
    Vector *prior_grad = vec_new(grad->size);
    *lprior = tp_compute_log_prior(mod, data, prior_grad);
    for (int j = 0; j < fulld; j++)
      vec_set(grad, j, vec_get(grad, j) + vec_get(prior_grad, j));
    vec_free(prior_grad);
  }
  else {
    *lprior = 0.0;
  }

  if (nuis_grad != NULL) {
    vec_zero(nuis_grad);
    nj_update_nuis_grad(mod, data, nuis_grad);
  }

  /* ---------------------------------------
   * 2. Decide whether to refresh MC cache
   * --------------------------------------- */

  int do_refresh =
      (td->iter >= td->warmup) &&
      ((td->iter - td->warmup) % td->period == 0);

  if (do_refresh && nj_var_at_floor(mmvn, data))
    do_refresh = FALSE;

  if (do_refresh) {

    Vector *mc_grad = vec_new(grad->size);
    Vector *mc_nuis = nuis_grad ? vec_new(nuis_grad->size) : NULL;
    double mc_lprior = 0.0, mc_migll = 0.0;

    vec_zero(mc_grad);
    if (mc_nuis) vec_zero(mc_nuis);

    /* This call:
       - samples x ~ q
       - computes E_q[log p(y|x)]
       - computes unbiased gradients wrt ALL params
    */
    double mc_ll =
      nj_elbo_montecarlo(mod, mmvn, data,
                         nminibatch,
                         mc_grad,
                         mc_nuis,
                         &mc_lprior,
                         &mc_migll);

    /* ---------------------------------------
     * 3. Cache + smooth MC correction
     * --------------------------------------- */

    /* Δ = E_q[ll + migll] − (ll(mu) + migll(mu))
       We store T ≈ 2Δ so that 0.5*T ≈ Δ
       Including migration here means the caller's separate addition
       of *migll cancels with the −*migll inside T, leaving E_q[migll].
    */
    double T = 2.0 * ((mc_ll + mc_migll) - (ll_mu + *migll));

    if (isfinite(T)) {
      if (td->iter == td->warmup) {
        td->T_cache = T;

        if (td->siggrad_cache == NULL)
          td->siggrad_cache = vec_new(sigdim);

        /* initialize variance gradient cache */
        for (int j = 0; j < sigdim; j++)
          vec_set(td->siggrad_cache, j, vec_get(mc_grad, fulld + j));
      }
      else {
        td->T_cache =
          (1.0 - td->beta) * td->T_cache + td->beta * T;
        
        blend_variance_block(td->siggrad_cache,
                             mc_grad,
                             fulld,
                             td->beta);
      }
    }

    /* on refresh iterations, use MC nuisance gradient (averaged over
       samples) instead of the single mean-point evaluation */
    if (nuis_grad != NULL && mc_nuis != NULL)
      vec_copy(nuis_grad, mc_nuis);

    /* Update ELBO bias correction.  The Taylor ELBO (ll_mu + 0.5*T)
       and MC ELBO (mc_ll + mc_migll) estimate the same quantity but
       differ due to higher-order terms and clamping.  Track the
       discrepancy so we can debias the Taylor ELBO for parameter
       selection by the caller. */
    double taylor_elbo = ll_mu + 0.5 * td->T_cache;
    double mc_elbo = mc_ll + mc_migll;
    double bias = taylor_elbo - mc_elbo;
    if (isfinite(bias)) {
      if (td->iter == td->warmup)
        td->elbo_bias = bias;
      else
        td->elbo_bias = (1.0 - td->beta) * td->elbo_bias + td->beta * bias;
    }

    vec_free(mc_grad);
    if (mc_nuis) vec_free(mc_nuis);
  }

  td->iter++;

  /* ---------------------------------------
   * 4. Assemble final gradient
   * --------------------------------------- */

  if (td->siggrad_cache != NULL)
    add_cached_variance_grad(grad, td->siggrad_cache, fulld);

  /* ---------------------------------------
   * 5. Final ELBO value (debiased)
   * --------------------------------------- */

  double elbo = ll_mu + 0.5 * td->T_cache - td->elbo_bias;

  /* ---------------------------------------
   * 6. Cleanup
   * --------------------------------------- */

  if (ll_at_mean != NULL) *ll_at_mean = ll_mu;

  vec_free(mu);
  if (mod->tree != NULL) {
    tr_free(mod->tree);
    mod->tree = NULL;
  }

  return elbo;
}
