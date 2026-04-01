/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025-2026, Cold Spring Harbor Laboratory
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */


/* handling of crispr mutation models for phylogeny reconstruction */
/* reads tab-delimited mutation data, calculates substitution probabilities, custom pruning algorithm for better efficiency */


#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <assert.h>
#include <float.h>
#include <phast/stringsplus.h>
#include <phast/hashtable.h>
#include <crispr.h>
#include <likelihoods.h>

/* for multithreading */
#ifdef _OPENMP
  #include <omp.h>
  #define NJ_OMP_GET_MAX_THREADS() omp_get_max_threads()
  #define NJ_OMP_GET_THREAD_NUM() omp_get_thread_num()
#else
  #define NJ_OMP_GET_MAX_THREADS() 1
  #define NJ_OMP_GET_THREAD_NUM()  0
#endif

/* for use in likelihood calculations to avoid underflow */
#define REL_CUTOFF 1e-300

/* auxiliary data used to keep track of restricted ancestral state
   possibilities in likelihood calculation */
static CrisprAncestralStateSets *ancsets = NULL;

CrisprMutTable *cpr_new_table() {
  CrisprMutTable *retval = malloc(sizeof(CrisprMutTable));
  retval->sitenames = lst_new_ptr(30); /* will resize as needed */
  retval->cellnames = lst_new_ptr(100);
  retval->cellmuts = lst_new_ptr(100);
  retval->nsites = 0;
  retval->ncells = 0;
  retval->nstates = 0;
  retval->sitewise_nstates = NULL;
  retval->dupnames = NULL;
  return retval;
}

CrisprMutTable *cpr_copy_table(CrisprMutTable *orig) {
  CrisprMutTable *copy = cpr_new_table();
  int i, j;

  copy->nsites = orig->nsites;
  copy->ncells = orig->ncells;
  copy->nstates = orig->nstates;
  
  for (i = 0; i < lst_size(orig->sitenames); i++) 
    lst_push_ptr(copy->sitenames, str_dup(lst_get_ptr(orig->sitenames, i)));
  for (i = 0; i < lst_size(orig->cellnames); i++) 
    lst_push_ptr(copy->cellnames, str_dup(lst_get_ptr(orig->cellnames, i)));
  for (i = 0; i < orig->ncells; i++) {
    List *muts = lst_new_int(orig->nsites);
    lst_push_ptr(copy->cellmuts, muts);
    for (j = 0; j < orig->nsites; j++) 
      lst_push_int(muts, cpr_get_mut(orig, i, j));
  }

  if (orig->dupnames != NULL) {
    copy->dupnames = malloc(orig->ncells * sizeof(List *));
    for (i = 0; i < orig->ncells; i++) {
      copy->dupnames[i] = NULL;
      if (orig->dupnames[i] != NULL) {
        copy->dupnames[i] = lst_new_ptr(lst_size(orig->dupnames[i]));
        for (j = 0; j < lst_size(orig->dupnames[i]); j++)
          lst_push_ptr(copy->dupnames[i],
                       str_dup(lst_get_ptr(orig->dupnames[i], j)));
      }
    }
  }
  
  return copy;
}

CrisprMutTable *cpr_read_table(FILE *F) {
  int i, state, lineno = 0;
  List *cols, *muts;
  String *line = str_new(STR_MED_LEN);
  CrisprMutTable *M = cpr_new_table();
  
  if (str_readline(line, F) == EOF)
    die("ERROR: cannot read from file.\n");

  cols = lst_new_ptr(M->nsites + 1);
  str_split(line, NULL, cols);
  str_free(lst_get_ptr(cols, 0)); /* blank */
  for (i = 1; i < lst_size(cols); i++) 
    lst_push_ptr(M->sitenames, lst_get_ptr(cols, i));
  M->nsites = lst_size(M->sitenames); 

  while (str_readline(line, F) != EOF) {
    lineno++;
    str_split(line, NULL, cols);
    if (lst_size(cols) != M->nsites + 1)
      die("ERROR in line %d of input file: number of mutations does not match header\n", lineno);
    lst_push_ptr(M->cellnames, lst_get_ptr(cols, 0));
    muts = lst_new_int(M->nsites);
    lst_push_ptr(M->cellmuts, muts);
    for (i = 1; i< M->nsites + 1; i++) {
      if (str_as_int(lst_get_ptr(cols, i), &state) != 0)
        die("ERROR in line %d of input file: mutation state '%s' is not an "
            "integer.\n",
            lineno, ((String *)lst_get_ptr(cols, i))->chars);
      if (state < 0 && state != -1)
        die("ERROR in line %d of input file: mutation state '%s' is not a "
          "non-negative integer or -1.\n",
          lineno, ((String *)lst_get_ptr(cols, i))->chars);
      lst_push_int(muts, state);
      if (state >= M->nstates)
        M->nstates = state + 1;
      str_free(lst_get_ptr(cols, i));
    }
  }

  M->ncells = lst_size(M->cellmuts);
  lst_free(cols);
  str_free(line);

  return M;
}

void cpr_free_table(CrisprMutTable *M) {
  int i;
  for (i = 0; i < lst_size(M->sitenames); i++)
    str_free(lst_get_ptr(M->sitenames, i));
  for (i = 0; i < lst_size(M->cellnames); i++)
    str_free(lst_get_ptr(M->cellnames, i));
  for (i = 0; i < lst_size(M->cellmuts); i++) 
    lst_free(lst_get_ptr(M->cellmuts, i));
  lst_free(M->sitenames);
  lst_free(M->cellnames);
  lst_free(M->cellmuts);

  if (M->dupnames != NULL) {
    for (i = 0; i < M->ncells; i++) {
      if (M->dupnames[i] != NULL) {
        for (int j = 0; j < lst_size(M->dupnames[i]); j++)
          str_free(lst_get_ptr(M->dupnames[i], j));
        lst_free(M->dupnames[i]);
      }
    }
    free(M->dupnames);
  }
}

void cpr_print_table(CrisprMutTable *M, FILE *F) {
  int i, j;
  fprintf(F, "cell");
  for (j = 0; j < M->nsites; j++)
    fprintf(F, "\t%s", ((String*)lst_get_ptr(M->sitenames, j))->chars);
  fprintf(F, "\n");
  for (i = 0; i < M->ncells; i++) {
    fprintf(F, "%s\t", ((String*)lst_get_ptr(M->cellnames, i))->chars);
    for (j = 0; j < M->nsites; j++) 
      fprintf(F, "%d%c", cpr_get_mut(M, i, j),
              j == M->nsites - 1 ? '\n' : '\t');
  }
}

int cpr_get_mut(CrisprMutTable *M, int cell, int site) {
  List *muts;
  
  if (cell >= M->ncells || site >= M->nsites)
    die("ERROR in cpr_get_mut: index out of range.\n");

  muts = lst_get_ptr(M->cellmuts, cell);
  return lst_get_int(muts, site);
}

void cpr_set_mut(CrisprMutTable *M, int cell, int site, int val) {
  List *muts;
  
  if (cell >= M->ncells || site >= M->nsites)
    die("ERROR in cpr_get_mut: index out of range.\n");
  if (val >= M->nstates)
    die("ERROR in cpr_get_mut: value out of range.\n");
  
  muts = lst_get_ptr(M->cellmuts, cell);
  lst_set_int(muts, site, val);
}

/* renumber states so they fall densely between 0 and nstates - 1,
   with -1 set aside to represent silent states */
void cpr_renumber_states(CrisprMutTable *M) {
  int i, j, newnstates;
  int *statemap = malloc(M->nstates * sizeof(int));
  for (i = 0; i < M->nstates; i++) statemap[i] = -1;
  newnstates = 1;  /* assume 0; ignore and don't count -1 */
  for (i = 0; i < M->ncells; i++) {
    for (j = 0; j < M->nsites; j++) {
      int stateij = cpr_get_mut(M, i, j);
      if (stateij == -1 || stateij == 0) continue;
      else if (stateij >= M->nstates)
        die("ERROR: state number out of range.\n");
      if (statemap[stateij] == -1) 
        statemap[stateij] = newnstates++;
      cpr_set_mut(M, i, j, statemap[stateij]);
    }
  }
  M->nstates = newnstates;
  free(statemap);
}

/* helper for cpr_duplicate (below) */
static void cpr_geno_str(CrisprMutTable *M, MigTable *mg, int cell, String *s) {
  str_clear(s);
  for (int j = 0; j < M->nsites; j++) {
    int state = cpr_get_mut(M, cell, j);
    if (state == -1)
      str_append_char(s, 'x');
    else
      str_append_int(s, state);
    if (j < M->nsites-1) str_append_char(s, ',');
  }
  if (mg != NULL) {  /* in case of migration model, only remove if state is the same */
    int migstate = lst_get_int(mg->states, cell);
    str_append_char(s, '|');
    str_append_int(s, migstate);
  }
}

/* remove cells with duplicate genotypes and resize the table.
   Maintain a record of the names of duplicates so they can be
   re-added if necessary */
void cpr_deduplicate(CrisprMutTable *M, struct mgtab *mg) {
  Hashtable *seen = hsh_new(M->ncells);
  String *genostr = str_new(M->nsites * 10);
  List *newcellnames = lst_new_ptr(M->ncells);
  List *newcellmuts = lst_new_ptr(M->ncells);
  List *new_mg_cellnames = (mg != NULL) ? lst_new_ptr(M->ncells) : NULL;
  List *new_mg_states = (mg != NULL) ? lst_new_int(M->ncells) : NULL;

  assert(M->dupnames == NULL); /* up to caller to ensure this is the case */
  M->dupnames = malloc(M->ncells * sizeof(List *));
  for (int i = 0; i < M->ncells; i++)      M->dupnames[i] = NULL;

  for (int i = 0; i < M->ncells; i++) {
    cpr_geno_str(M, mg, i, genostr);
    int prevcell = hsh_get_int(seen, genostr->chars);
    if (prevcell == -1) {
      hsh_put_int(seen, genostr->chars, lst_size(newcellnames));
      lst_push_ptr(newcellnames, lst_get_ptr(M->cellnames, i));
      lst_push_ptr(newcellmuts, lst_get_ptr(M->cellmuts, i));
      if (mg != NULL) {
        lst_push_ptr(new_mg_cellnames, str_dup(lst_get_ptr(mg->cellnames, i)));
        lst_push_int(new_mg_states, lst_get_int(mg->states, i));
      }
    }
    else {
      if (M->dupnames[prevcell] == NULL)
        M->dupnames[prevcell] = lst_new_ptr(10);
      String *dupname = lst_get_ptr(M->cellnames, i);
      lst_push_ptr(M->dupnames[prevcell], str_dup(dupname));
      str_free(dupname);
      lst_free(lst_get_ptr(M->cellmuts, i));
    }
  }
  lst_free(M->cellnames);
  lst_free(M->cellmuts);
  M->cellnames = newcellnames;
  M->cellmuts = newcellmuts;
  M->ncells = lst_size(M->cellnames);
  M->dupnames = realloc(M->dupnames, M->ncells * sizeof(List *));
  if (mg != NULL) {
    lst_free_strings(mg->cellnames);
    lst_free(mg->cellnames);
    lst_free(mg->states);
    mg->cellnames = new_mg_cellnames;
    mg->states = new_mg_states;
    mg->ncells = lst_size(mg->cellnames);
  }
  str_free(genostr);
  hsh_free(seen);
}

/* helper function to avoid zeros resulting from combination of
   irreversible model and very short branches */
#define CPR_PFLOOR 1.0e-200

typedef struct {
  int silst;
  double exp_t_sil;
  double one_min_exp_t_sil;
  double exp_t_sil_one_min_exp_t;
  double exp_t_one_plus_sil;
} CprBranchParams;

static inline void cpr_set_branch_params(CprBranchParams *bp, int silst,
                                         double t, double silent_rate) {
  t = CPR_T_FLOOR + (t > 0.0 ? t : 0.0);
  bp->silst = silst;
  bp->exp_t_sil = exp(-t * silent_rate);
  bp->one_min_exp_t_sil = 1.0 - bp->exp_t_sil;
  bp->exp_t_sil_one_min_exp_t = bp->exp_t_sil * (1.0 - exp(-t));
  bp->exp_t_one_plus_sil = exp(-t * (1.0 + silent_rate));
}

static inline double cpr_get_branch_prob(const CprBranchParams *bp, int i, int j,
                                         Vector *mutrates) {
  double p = 0.0;

  if (i == bp->silst)
    p = (j == bp->silst ? 1.0 : 0.0);
  else if (j == bp->silst)
    p = bp->one_min_exp_t_sil;
  else if (i == 0) {
    if (j == 0) p = bp->exp_t_one_plus_sil;
    else p = vec_get(mutrates, j) * bp->exp_t_sil_one_min_exp_t;
  }
  else
    p = (j == i ? bp->exp_t_sil : 0.0);

  return p + CPR_PFLOOR;  /* derivative still same as original value */
}

/* Compute and return the log likelihood of a tree model with respect
   to a CRISPR mutation table.  This function is derived from
   nj_compute_log_likelihood but is customized for the irreversible
   CRISPR mutation model of Seidel and Stadler (Proc. R. Soc. B
   289:20221844, 2022). If branchgrad is non-null, it will be
   populated with the gradient of the log likelihood with respect to
   the individual branches of the tree, in post-order.  */
double cpr_ll_core(CrisprMutModel *cprmod, NJDerivs *derivs,
  int *nodetypes, List *range) {

  int i, j, k, nodeidx, site, cell, state;
  int nstates;
  TreeNode *n, *sibling;
  double total_prob = 0;
  List *traversal, *pre_trav;
  double **pL = NULL, **pLbar = NULL;
  double ll = 0;
  double tmp[cprmod->nstates+1], root_eqfreqs[cprmod->nstates+1];
  Matrix *grad_mat = NULL;
  Vector *lscale, *lscale_o; /* inside and outside versions */
  List *par_states, *lchild_states, *rchild_states, *child_states, *sib_states;
  int pstate, lcstate, rcstate, cstate, sstate;
      
  /* set up "inside" probability matrices for pruning algorithm */
  pL = smalloc((cprmod->nstates+1) * sizeof(double*));
  for (j = 0; j < (cprmod->nstates+1); j++)
    pL[j] = smalloc((cprmod->mod->tree->nnodes+1) * sizeof(double));

  /* we also need to keep track of the log scale of every node for
     underflow purposes */
  lscale = vec_new(cprmod->mod->tree->nnodes+1); 
  lscale_o = vec_new(cprmod->mod->tree->nnodes+1); 
  
  if (derivs->branchgrad != NULL) {
    /* set up complementary "outside" probability matrices */
    pLbar = smalloc((cprmod->nstates+1) * sizeof(double*));
    for (j = 0; j < (cprmod->nstates+1); j++)
      pLbar[j] = smalloc((cprmod->mod->tree->nnodes + 1) * sizeof(double));
  }

  traversal = tr_postorder(cprmod->mod->tree);

  /* set up active range of sites for this thread */
  int r0 = 0, r1 = cprmod->nsites;
  if (range != NULL) {
    if (lst_size(range) != 2) die("cpr_ll_core: range must have size 2");
    r0 = lst_get_int(range, 0);
    r1 = lst_get_int(range, 1);
    if (r0 < 0) r0 = 0;
    if (r1 > cprmod->nsites) r1 = cprmod->nsites;
  }
  
  for (site = r0; site < r1; site++) {
    int silst;
    Vector *mutrates = lst_get_ptr(cprmod->sitewise_mutrates, site);
    CprBranchParams leading_bp;
    double this_deriv_sil;

    nstates = cprmod->mut->sitewise_nstates[site] + 1; /* have to allow for silent state */
    silst = nstates - 1; /* silent state will always be last */

    /* first zero out all pL values because with the smart
       algorithm, we won't visit most elements in the matrix */
    for (nodeidx = 0; nodeidx < cprmod->mod->tree->nnodes; nodeidx++) {
      nodetypes[nodeidx] = -99; /* also initialize these */
      for (i = 0; i < nstates; i++) 
        pL[i][nodeidx] = 0;
    }

    /* same for pLbar if needed */
    if (derivs->branchgrad != NULL) {
      for (nodeidx = 0; nodeidx < cprmod->mod->tree->nnodes; nodeidx++) 
        for (i = 0; i < nstates; i++)
          pLbar[i][nodeidx] = 0.0;
    }
    
    /* also reset scale */
    vec_zero(lscale); vec_zero(lscale_o);

    /* this model allows a leading branch to the root of the tree but
       forces the unedited state at the start of that branch.  We can
       simulate this behavior by setting the root eq freqs equal to
       the conditional distribution at the end of the branch given the
       unedited state at the start */
    cpr_set_branch_params(&leading_bp, silst, cprmod->mod->tree->dparent,
                          cprmod->sil_rate);
    for (i = 0; i < nstates; i++)
      root_eqfreqs[i] = cpr_get_branch_prob(&leading_bp, 0, i, mutrates);
    
    for (nodeidx = 0; nodeidx < lst_size(traversal); nodeidx++) {
      int mut;

      n = lst_get_ptr(traversal, nodeidx);

      if (n->lchild == NULL) {
        /* leaf: base case of recursion */
        cell = cprmod->mod->msa_seq_idx[n->id];
        if (cell == -1)
          die("ERROR in cpr_compute_log_likelihood: leaf '%s' not found in mutation table.\n",
              n->name);

        mut = cpr_get_mut(cprmod->mut, cell, site);
        state = (mut == -1 ? silst : mut);
        assert(state >= 0 && state <= silst);
        pL[state][n->id] = 1;

        /* also update nodetype */
        if (state < silst) /* ancestral (0) or derived edit */
          nodetypes[n->id] = state;
        else /* unrestricted if silent */
          nodetypes[n->id] = ancsets->NORESTRICT;
      }
      else {
        /* general recursive case */
        CprBranchParams lbp, rbp;
        int lchildtype, rchildtype, thistype;

        cpr_set_branch_params(&lbp, silst, n->lchild->dparent, cprmod->sil_rate);
        cpr_set_branch_params(&rbp, silst, n->rchild->dparent, cprmod->sil_rate);
        
        /* first set nodetype based on nodetypes of children */
        lchildtype = nodetypes[n->lchild->id];
        rchildtype = nodetypes[n->rchild->id];
        
        if (lchildtype == 0 || rchildtype == 0) /* if either child is
                                                   unedited, parent
                                                   must be unedited */
          thistype = 0;
        else if (lchildtype != ancsets->NORESTRICT &&
                 rchildtype != ancsets->NORESTRICT &&
                 lchildtype != rchildtype) /* if children have
                                              different edits, parent
                                              must be unedited */
          thistype = 0;
        else if (lchildtype == rchildtype && 
                 lchildtype != ancsets->NORESTRICT) /* if children have
                                                       same edits,
                                                       parent must
                                                       share it or be
                                                       unedited */
          
          thistype = lchildtype;
        else if (lchildtype != ancsets->NORESTRICT &&
                 rchildtype == ancsets->NORESTRICT) /* if one edited
                                                       child, parent
                                                       must have same
                                                       edit or be
                                                       unedited */
          thistype = lchildtype;
        
        else if (rchildtype != ancsets->NORESTRICT &&
                 lchildtype == ancsets->NORESTRICT) /* converse case */
          thistype = rchildtype;

        else /* otherwise we have to consider all possible states */
          thistype = ancsets->NORESTRICT;

        nodetypes[n->id] = thistype;

        /* now get corresponding sets of eligible states */
        par_states = cpr_get_state_set(ancsets, nodetypes, n, nstates);
        lchild_states = cpr_get_state_set(ancsets, nodetypes, n->lchild, nstates);
        rchild_states = cpr_get_state_set(ancsets, nodetypes, n->rchild, nstates);

        /* do this in a way that handles scaling.  first compute
           unnormalized inside values */
        for (i = 0; i < lst_size(par_states); i++) {
          double totl = 0.0, totr = 0.0;
          pstate = lst_get_int(par_states, i);
          for (j = 0; j < lst_size(lchild_states); j++) {
            lcstate = lst_get_int(lchild_states, j);
            totl += pL[lcstate][n->lchild->id] *
              cpr_get_branch_prob(&lbp, pstate, lcstate, mutrates);
          }
          for (k = 0; k < lst_size(rchild_states); k++) {
            rcstate = lst_get_int(rchild_states, k);
            totr += pL[rcstate][n->rchild->id] *
              cpr_get_branch_prob(&rbp, pstate, rcstate, mutrates);
          }
          
          pL[pstate][n->id] = totl * totr;
        }

        /* nodewise max-normalization across states */
        double maxv = 0.0;
        for (i = 0; i < lst_size(par_states); i++) {
          pstate = lst_get_int(par_states, i);
          if (pL[pstate][n->id] > maxv)
            maxv = pL[pstate][n->id];
        }

        /* propagate scaling from children */
        vec_set(lscale, n->id,
                vec_get(lscale, n->lchild->id) +
                vec_get(lscale, n->rchild->id));

        if (maxv > 0.0) {
          /* normalize and update scale */
          for (i = 0; i < lst_size(par_states); i++) {
            pstate = lst_get_int(par_states, i);
            pL[pstate][n->id] /= maxv;
          }

          vec_set(lscale, n->id,
                  vec_get(lscale, n->id) + log(maxv));
        }
        else
          derivs->zero_likl = TRUE;

        /* zero out tiny values to save time later */
        for (i = 0; i < lst_size(par_states); i++) {
          pstate = lst_get_int(par_states, i);
          if (pL[pstate][n->id] < REL_CUTOFF)
            pL[pstate][n->id] = 0.0;
        }
      }
    } 
  
    /* termination */
    par_states = cpr_get_state_set(ancsets, nodetypes, cprmod->mod->tree, nstates);
    total_prob = 0;
    for (i = 0; i < lst_size(par_states); i++) {
      int rstate = lst_get_int(par_states, i);
      total_prob += root_eqfreqs[rstate] * pL[rstate][cprmod->mod->tree->id];
    }
    if (!(total_prob > 0.0))  /* catches zero, negative, and NaN */
      total_prob = CPR_PFLOOR;

    ll += (log(total_prob) + vec_get(lscale, cprmod->mod->tree->id));

    /* to compute gradients efficiently, need to make a second pass
       across the tree to compute "outside" probabilities */
    if (derivs->branchgrad != NULL) {
      double expon;
      
      pre_trav = tr_preorder(cprmod->mod->tree);

      for (nodeidx = 0; nodeidx < lst_size(pre_trav); nodeidx++) {
        n = lst_get_ptr(pre_trav, nodeidx);

        if (n->parent == NULL) { /* base case */
          par_states = cpr_get_state_set(ancsets, nodetypes, n, nstates);
          double maxv = 0.0;
          for (i = 0; i < lst_size(par_states); i++) {
            pstate = lst_get_int(par_states, i);
            pLbar[pstate][n->id] = root_eqfreqs[pstate];
            if (pLbar[pstate][n->id] > maxv)
              maxv = pLbar[pstate][n->id];
          }

          /* lscale_o[root] is already zero from vec_zero */
          if (maxv > 0.0) {
            for (i = 0; i < lst_size(par_states); i++) {
              pstate = lst_get_int(par_states, i);
              pLbar[pstate][n->id] /= maxv;
            }
            vec_set(lscale_o, n->id, log(maxv));
          }
        }
        else {            /* recursive case */
          sibling = (n == n->parent->lchild ?
                     n->parent->rchild : n->parent->lchild);
          CprBranchParams par_bp, sib_bp;
          cpr_set_branch_params(&par_bp, silst, n->dparent, cprmod->sil_rate);
          cpr_set_branch_params(&sib_bp, silst, sibling->dparent, cprmod->sil_rate);

          par_states = cpr_get_state_set(ancsets, nodetypes, n->parent, nstates);
          child_states = cpr_get_state_set(ancsets, nodetypes, n, nstates);
          sib_states = cpr_get_state_set(ancsets, nodetypes, sibling, nstates);

          /* first form tmp[j] = sum_k pLbar(parent=j) * pL(sibling=k) * P_sib(j,k) */
          for (j = 0; j < lst_size(par_states); j++) {
            pstate = lst_get_int(par_states, j);
            tmp[pstate] = 0.0;
            double a = pLbar[pstate][n->parent->id];

            if (a == 0.0) continue;

            for (k = 0; k < lst_size(sib_states); k++) {
              sstate = lst_get_int(sib_states, k);
              double b = pL[sstate][sibling->id];
              if (b > 0.0)
                tmp[pstate] += a * b * (cpr_get_branch_prob(&sib_bp, pstate, sstate, mutrates));
            }
          }

          /* now propagate to child */
          for (i = 0; i < lst_size(child_states); i++) {      /* child state */
            cstate = lst_get_int(child_states, i);
            double sum = 0.0;
            for (j = 0; j < lst_size(par_states); j++) {      /* parent state */
              pstate = lst_get_int(par_states, j);
              sum += tmp[pstate] * (cpr_get_branch_prob(&par_bp, pstate, cstate, mutrates));
            }
            pLbar[cstate][n->id] = sum;
          }
          
          /* nodewise normalization of outside vector */
          double maxv = 0.0;
          for (i = 0; i < lst_size(child_states); i++) {   
            cstate = lst_get_int(child_states, i);
            if (pLbar[cstate][n->id] > maxv)
              maxv = pLbar[cstate][n->id];
          }

          /* bookkeeping for scaling */
          vec_set(lscale_o, n->id,
                  vec_get(lscale_o, n->parent->id) +
                  vec_get(lscale, sibling->id));

          if (maxv > 0.0) {
            for (i = 0; i < lst_size(child_states); i++) {   
              cstate = lst_get_int(child_states, i);
              pLbar[cstate][n->id] /= maxv;
            }
            
            vec_set(lscale_o, n->id,
                    vec_get(lscale_o, n->id) + log(maxv));
          }

          for (i = 0; i < lst_size(child_states); i++) {   
            cstate = lst_get_int(child_states, i);
            if (pLbar[cstate][n->id] < REL_CUTOFF)
              pLbar[cstate][n->id] = 0.0;
          }
        }
      }

      /* now compute branchwise derivatives in a final pass */
      grad_mat = mat_new(nstates, nstates);
      mat_zero(grad_mat);
      for (nodeidx = 0; nodeidx < lst_size(cprmod->mod->tree->nodes); nodeidx++) {
        TreeNode *par;
        double base_prob = total_prob, deriv;
        
        n = lst_get_ptr(cprmod->mod->tree->nodes, nodeidx);
        par = n->parent;
        
        if (par == NULL) 
          continue;
       
        sibling = (n == n->parent->lchild ?
                   n->parent->rchild : n->parent->lchild);

        CprBranchParams sib_bp;
        cpr_set_branch_params(&sib_bp, silst, sibling->dparent, cprmod->sil_rate);

        /* get corresponding sets of eligible states */
        par_states = cpr_get_state_set(ancsets, nodetypes, par, nstates);
        child_states = cpr_get_state_set(ancsets, nodetypes, n, nstates);
        sib_states = cpr_get_state_set(ancsets, nodetypes, sibling, nstates);
        
        /* this part is just a constant to propagate through to the
           derivative */
        for (i = 0; i < lst_size(par_states); i++) {  /* parent */
          pstate = lst_get_int(par_states, i);
          tmp[pstate] = 0;
          for (k = 0; k < lst_size(sib_states); k++) { /* sibling */
            sstate = lst_get_int(sib_states, k);
            tmp[pstate] += pL[sstate][sibling->id] * (cpr_get_branch_prob(&sib_bp, pstate, sstate, mutrates));
          }
        }

        /* adjust for all relevant scale terms; do everything in log space */
        expon = -vec_get(lscale, cprmod->mod->tree->id)
          + vec_get(lscale, sibling->id) + vec_get(lscale_o, par->id)
          + vec_get(lscale, n->id) - log(base_prob);
        /* note division by base_prob because we need deriv of log P */

        /* avoid overflow; note !isfinite also catches NaN */
        if (!isfinite(expon) || expon > 700.0) expon = 700.0;
        if (expon < -745.0) expon = -745.0;
          
        if (n != cprmod->mod->tree->rchild) { /* skip this for right branch from root because unrooted */
          /* calculate derivative analytically */
          deriv = 0;
          cpr_branch_grad(grad_mat, n->dparent, cprmod->sil_rate, mutrates);
          for (i = 0; i < lst_size(par_states); i++) {
            pstate = lst_get_int(par_states, i);
            for (j = 0; j < lst_size(child_states); j++) {
              cstate = lst_get_int(child_states, j);
              assert(isfinite(tmp[pstate]) && isfinite(pLbar[pstate][par->id]) &&
                     isfinite(pL[cstate][n->id]) &&
                     isfinite(mat_get(grad_mat, pstate, cstate)));
              deriv += tmp[pstate] * pLbar[pstate][par->id] *
                       pL[cstate][n->id] * mat_get(grad_mat, pstate, cstate);
            }
          }
          /* adjust for all relevant scale terms */
          deriv *= exp(expon);
          assert(isfinite(deriv));
          vec_set(derivs->branchgrad, n->id,
            vec_get(derivs->branchgrad, n->id) + deriv);
        } /* end skip right branch case */

        /* derivative wrt silent rate: aggregated across ALL branches
           including the right child of root (unlike branch lengths,
           the silencing rate is a global parameter affecting every branch) */
        this_deriv_sil = 0;
        cpr_silent_rate_grad(grad_mat, n->dparent, cprmod->sil_rate, mutrates);
        for (i = 0; i < lst_size(par_states); i++) {
          pstate = lst_get_int(par_states, i);
          for (j = 0; j < lst_size(child_states); j++) {
            cstate = lst_get_int(child_states, j);
            this_deriv_sil +=  tmp[pstate] * pLbar[pstate][par->id] * pL[cstate][n->id] *
              mat_get(grad_mat, pstate, cstate);
          }
        }

        /* adjust for all relevant scale terms */
        this_deriv_sil *= exp(expon);
        derivs->deriv_sil += this_deriv_sil;
      } /* end node loop */

      /* also compute gradient for leading branch */
      child_states = cpr_get_state_set(ancsets, nodetypes, cprmod->mod->tree, nstates);
      cpr_branch_grad(grad_mat, cprmod->mod->tree->dparent, cprmod->sil_rate, mutrates);
      double this_deriv_leading_t = 0;
      for (j = 0; j < lst_size(child_states); j++) {
        cstate = lst_get_int(child_states, j);
        this_deriv_leading_t += pL[cstate][cprmod->mod->tree->id]
          * mat_get(grad_mat, 0, cstate);
      }

      /* rescale; note !isfinite also catches NaN */
      expon = -log(total_prob);
      if (!isfinite(expon) || expon > 700.0) expon = 700.0;
      if (expon < -745.0) expon = -745.0;
      this_deriv_leading_t *= exp(expon);
      derivs->deriv_leading_t += this_deriv_leading_t;
      
      /* leading branch also contributes to derivative of silent rate */
      this_deriv_sil = 0;
      cpr_silent_rate_grad(grad_mat, cprmod->mod->tree->dparent, cprmod->sil_rate,
                           mutrates);
      for (j = 0; j < lst_size(child_states); j++) {
        cstate = lst_get_int(child_states, j);
        this_deriv_sil +=  pL[cstate][cprmod->mod->tree->id]
          * mat_get(grad_mat, 0, cstate);
      }
      this_deriv_sil *= exp(expon);
      derivs->deriv_sil += this_deriv_sil;

      mat_free(grad_mat);
    }
  }
  
  for (j = 0; j < cprmod->nstates+1; j++)
    sfree(pL[j]);
  sfree(pL);

  if (derivs->branchgrad != NULL) {
    for (j = 0; j < cprmod->nstates+1; j++)
      sfree(pLbar[j]);
    sfree(pLbar);
  }

  vec_free(lscale);
  vec_free(lscale_o);
  
  return ll;
}

/* compute pairwise parsimony distance between two cells under the
   irreversible CRISPR model: minimum number of mutation events to
   explain observed states, summed across sites and divided by number
   of comparable sites.  Sites with -1 in either cell are skipped.
   Unlike cpr_compute_pw_dist_nopriv, private edits are included (1
   step each).  Unlike plain Hamming, two different derived states at
   the same site contribute 2 (two independent mutations under the
   irreversible model). */
double cpr_compute_pw_dist_parsimony(CrisprMutTable *M, int i, int j) {
  int k, diff = 0, n = 0;
  for (k = 0; k < M->nsites; k++) {
    int typei = cpr_get_mut(M, i, k),
      typej = cpr_get_mut(M, j, k);
    if (typei == -1 || typej == -1)
      continue;
    n++;
    if (typei != typej)
      diff += (typei != 0 && typej != 0) ? 2 : 1;
  }
  if (n == 0)
    return 1;
  return diff * 1.0 / n;
}

/* build and return an upper triangular parsimony distance matrix
   (see cpr_compute_pw_dist_parsimony). */
Matrix *cpr_compute_dist_parsimony(CrisprMutTable *M) {
  int i, j;
  Matrix *retval = mat_new(M->ncells, M->ncells);
  mat_zero(retval);
  for (i = 0; i < M->ncells; i++)
    for (j = i+1; j < M->ncells; j++)
      mat_set(retval, i, j, cpr_compute_pw_dist_parsimony(M, i, j));
  return retval;
}

/* build and return an upper triangular distance matrix for the cells
   in a CrisprMutTable, using parsimony distance (default). */
Matrix *cpr_compute_dist(CrisprMutTable *M) {
  return cpr_compute_dist_parsimony(M);
}

/* build and return an upper triangular distance matrix using plain
   Hamming distance (see cpr_compute_pw_dist).  Kept as an alternative
   to cpr_compute_dist; use if private-edit exclusion is not desired. */
Matrix *cpr_compute_dist_hamming(CrisprMutTable *M) {
  int i, j;
  Matrix *retval = mat_new(M->ncells, M->ncells);
  mat_zero(retval);
  for (i = 0; i < M->ncells; i++)
    for (j = i+1; j < M->ncells; j++)
      mat_set(retval, i, j, cpr_compute_pw_dist(M, i, j));
  return retval;
}

/* compute pairwise distance between two cells as plain Hamming distance
   (proportion of differing sites, ignoring sites where either cell has -1).
   Private (singleton) edits are included; use cpr_compute_pw_dist_nopriv
   to exclude them. */
double cpr_compute_pw_dist(CrisprMutTable *M, int i, int j) {
  int k, diff = 0, n = 0;
  for (k = 0; k < M->nsites; k++) {
    int typei = cpr_get_mut(M, i, k),
      typej = cpr_get_mut(M, j, k);
    if (typei == -1 || typej == -1)
      continue;
    n++;
    if (typei != typej)
      diff++;
  }
  if (n == 0)
    return 1;
  return diff * 1.0 / n;
}

/* compute pairwise distance between two cells, excluding sites where
   either cell has a private (singleton) non-zero edit.  Such autapomorphies
   inflate distances uniformly against all other taxa and carry no grouping
   signal under the irreversible model.
   state_counts[k][s] = number of cells with state s at site k
   (pre-computed by cpr_compute_dist). */
double cpr_compute_pw_dist_nopriv(CrisprMutTable *M, int i, int j,
                                   int **state_counts) {
  int k, diff = 0, n = 0;
  for (k = 0; k < M->nsites; k++) {
    int typei = cpr_get_mut(M, i, k),
      typej = cpr_get_mut(M, j, k);

    if (typei == -1 || typej == -1)
      continue;
    /* skip sites where either cell has a private (singleton) non-zero edit */
    if (typei != 0 && state_counts[k][typei] == 1)
      continue;
    if (typej != 0 && state_counts[k][typej] == 1)
      continue;
    n++;
    if (typei != typej)
      /* under the irreversible model, two different derived states at
         the same site require two independent mutation events (one per
         lineage), so contribute 2 to the distance; one derived vs.
         unedited is a single event (contributes 1) */
      diff += (typei != 0 && typej != 0) ? 2 : 1;
  }

  if (n == 0)
    return 1; /* no comparable non-private sites; treat as maximally distant */

  return diff * 1.0 / n;
}

/* set P = exp(Qt) matrix for branch length t, using parameterization
   of Mai, Chu, and Raphael, doi:10.1101/2024.03.05.583638 */  
void cpr_set_branch_matrix(MarkovMatrix *P, double t, double silent_rate, Vector *mutrates) {
  int j, silst = P->size - 1; /* silent state is the last one */
  t = CPR_T_FLOOR + (t > 0.0 ? t : 0.0);
  double exp_t_sil = exp(-t * silent_rate),
    one_min_exp_t_sil = 1 - exp_t_sil,
    exp_t_sil_one_min_exp_t = exp_t_sil * (1 - exp(-t));
  assert(mutrates->size == P->size-1);

  /* this allows us to avoid resetting zeroes each time, which gets
     expensive with large matrices */
  static int silst_prev = -1;
  if (silst != silst_prev) {
    mat_zero(P->matrix);
    silst_prev = silst;
  }
  
  /* substitution probabilities from 0 (unedited) state to all edited
     (and not silent) states */
  for (j = 1; j < silst; j++)
    mm_set(P, 0, j, vec_get(mutrates, j) * exp_t_sil_one_min_exp_t);
  mm_set(P, 0, 0, exp(-t*(1+silent_rate)));
  
  /* substitution probabilities from edited states to themselves */
  for (j = 1; j < silst; j++) 
    mm_set(P, j, j, exp_t_sil);
    /* leave all others zero */
  
  /* substitution probabilities to silent state */
  for (j = 0; j < silst; j++)
    mm_set(P, j, silst, one_min_exp_t_sil);
  mm_set(P, silst, silst, 1); /* absorbing state */
}

/* compute gradients of elements of substitution matrix with respect
   to branch length */
void cpr_branch_grad(Matrix *grad, double t, double silent_rate, Vector *mutrates) {
  int j, silst = grad->nrows - 1;
  t = CPR_T_FLOOR + (t > 0.0 ? t : 0.0);

  double em1 = expm1(-t);          /* = exp(-t) - 1, accurate for small t */
  double es = exp(-t * silent_rate);
  double A  = (silent_rate * es * em1 + exp(-t * (1+silent_rate)));
  double B = silent_rate * es; 

  /* this allows us to avoid resetting zeroes each time, which gets
     expensive with large matrices */
  static int silst_prev = -1;
  if (silst != silst_prev) {
    mat_zero(grad);
    silst_prev = silst;
  }
  
  /* derivatives of substitution probabilities from 0 (unedited) state
     to all edited (and not silent) states */
  for (j = 1; j < silst; j++)
    mat_set(grad, 0, j, vec_get(mutrates, j) * A);
  mat_set(grad, 0, 0, -(1+silent_rate) * exp(-t*(1+silent_rate)));
  
  /* derivatives of substitution probabilities from edited states to
     themselves */
  for (j = 1; j < silst; j++)
    mat_set(grad, j, j, -B);

  /* derivatives of substitution probabilities to silent state */
  for (j = 0; j < silst; j++)
    mat_set(grad, j, silst, B);
}

/* compute gradients of elements of substitution matrix with respect
   to the silencing rate */
void cpr_silent_rate_grad(Matrix *grad, double t, double silent_rate, Vector *mutrates) {
  int j, silst = grad->nrows - 1;
  t = CPR_T_FLOOR + (t > 0.0 ? t : 0.0);
  double Es = -t * exp(-silent_rate * t);
  double E1 = exp(-t);
  double A = Es * (1.0-E1);

  /* this allows us to avoid resetting zeroes each time, which gets
     expensive with large matrices */
  static int silst_prev = -1;
  if (silst != silst_prev) {
    mat_zero(grad);
    silst_prev = silst;
  }
  
  /* derivatives of substitution probabilities from 0 (unedited) state
     to all edited (and not silent) states */
  for (j = 1; j < silst; j++)
    mat_set(grad, 0, j, vec_get(mutrates, j) * A);
  mat_set(grad, 0, 0, -t * exp(-t*(1.0+silent_rate)));
  
  /* derivatives of substitution probabilities from edited states to
     themselves */
  for (j = 1; j < silst; j++) 
    mat_set(grad, j, j, Es);
    /* leave all others zero */

  /* derivatives of substitution probabilities to silent state */
  for (j = 0; j < silst; j++)
    mat_set(grad, j, silst, -Es);
  mat_set(grad, silst, silst, 0);
}

/* estimate relative mutation rates based on relative frequencies in
   data set.  Returns a vector of size M->nstates.  To ensure all
   other states have nonzero values, it may be helpful to preprocess
   with cpr_renumber_states */
Vector *cpr_estim_mutrates(CrisprMutTable *M,
                            enum crispr_mutrates_type type) {
  int i, j;
  Vector *retval = vec_new(M->nstates);
  vec_zero(retval);

  if (type == UNIF) { /* uniform distrib over non-zero states */
    for (i = 1; i < M->nstates; i++)
      vec_set(retval, i, 1.0/(M->nstates-1.0));
    return retval;
  }

  /* otherwise compute empirically */
  for (i = 0; i < M->ncells; i++) {
    for (j = 0; j < M->nsites; j++) {
      int mut = cpr_get_mut(M, i, j);
      if (mut > 0)
        vec_set(retval, mut, vec_get(retval, mut) + 1.0);
    }
  }
  pv_normalize(retval);
  return retval;
}

/* estimate relative mutation rates based on relative frequencies in
   data set, but separately for each site.  Returns a list of
   vectors. Assumes mutation matrix has been re-indexed using
   cpr_new_sitewise_table */
List *cpr_estim_sitewise_mutrates(CrisprMutTable *M,
                                  enum crispr_mutrates_type type) {
  int i, j, mut;
  List *sitewise_mutrates = lst_new_ptr(M->nsites);
  for (j = 0; j < M->nsites; j++) {
    Vector *f = vec_new(M->sitewise_nstates[j]);
    vec_zero(f);

    if (type == UNIF) {
      for (mut = 1; mut < M->sitewise_nstates[j]; mut++)
      vec_set(f, mut, 1.0/(M->sitewise_nstates[j]-1.0));
    }

    else {
      for (i = 0; i < M->ncells; i++) {
        int mut = cpr_get_mut(M, i, j);
        if (mut > 0)
          vec_set(f, mut, vec_get(f, mut) + 1.0);
      }
      pv_normalize(f);
    }
    
    lst_push_ptr(sitewise_mutrates, f);
  }
  
  return sitewise_mutrates;
}

/*  Build index of leaf ids to cell indices based on matching names.
    Leaves not present in the alignment will be ignored.  Also, it's
    not required that there's a leaf for every sequence in the
    alignment. This is a version of tm_build_seq_idx adapted to work
    with a CrisprMutTable */
void cpr_build_seq_idx(TreeModel *mod, CrisprMutTable *M) {
  int i, idx;  
  mod->msa_seq_idx = smalloc(mod->tree->nnodes * sizeof(int));
  /* let's just reuse this even though it's misnamed for the purpose */
  
  for (i = 0; i < mod->tree->nnodes; i++) {
    TreeNode *n = lst_get_ptr(mod->tree->nodes, i);
    mod->msa_seq_idx[i] = -1;
    if (n->lchild == NULL && n->rchild == NULL) {
      String *namestr = str_new_charstr(n->name);
      if (str_in_list_idx(namestr, M->cellnames, &idx) == 1)
        mod->msa_seq_idx[i] = idx;
      str_free(namestr);
    }
  }
}

CrisprAncestralStateSets *cpr_new_state_sets(int nnodes) {
  CrisprAncestralStateSets *retval = malloc(sizeof(CrisprAncestralStateSets));
  retval->nnodes = nnodes;
  retval->NORESTRICT = -1;
  retval->unr_lists = lst_new_ptr(100); /* starting size; will realloc */
  retval->restr_lists = lst_new_ptr(100); 
  retval->sil_lists = lst_new_ptr(100); 
  return retval;
}

void cpr_state_sets_resize(CrisprAncestralStateSets *sets, int newnnodes) {
  sets->nnodes = newnnodes;
}

/* efficiently retrieve ancestral state set for a given node type and
   total number of states, using caching.  Specifically, if nodetype
   is NORESTRICT, returns a list of integers from 0 to nstates - 1
   inclusive.  If nodetype has another (restricted) value, returns a
   list of two integers consisting of 0 and that value.  If nodetype
   is 0, returns a list containing zero only. */
List *cpr_get_state_set(CrisprAncestralStateSets *set, int *nodetypes,
                        TreeNode *n, int nstates) {
  int i, j;

  int nodetype = nodetypes[n->id];
  
  if (nodetype == set->NORESTRICT) {

    /* special case: node is a leaf but has NORESTRICT type; can
       only arise when node is observed to have silent state, in
       which case that is the only state we need to consider */
    if (n->lchild == NULL) {
        if (nstates >= lst_size(set->sil_lists)) {
          /* fill in all of the ones up to the requested size and cache them */
          for (i = lst_size(set->sil_lists); i <= nstates; i++) {
            List *l = NULL;
            if (i > 1) {
              l = lst_new_int(1);
              lst_push_int(l, i-1);
            }
            lst_push_ptr(set->sil_lists, l); /* the ith element of
                                                sil_lists will be a list
                                                consisting of just i-1 (or
                                                NULL if i <= 1) */
          }
        }
        return lst_get_ptr(set->sil_lists, nstates);
    }

    /* otherwise return the appropriate unrestricted list */
    if (nstates >= lst_size(set->unr_lists)) {
      /* fill in all of the ones up to the requested size and cache them */
      for (i = lst_size(set->unr_lists); i <= nstates; i++) {
        List *l = lst_new_int(i);
        for (j = 0; j < i; j++)
          lst_push_int(l, j);
        lst_push_ptr(set->unr_lists, l); /* the ith element of
                                            unr_lists will be a list
                                            of integers from 0 to i,
                                            inclusive */
      }
    }
    return lst_get_ptr(set->unr_lists, nstates);
  }
  else { /* restricted list; includes the 0 case */
    if (nodetype >= lst_size(set->restr_lists)) {
      /* fill in all of the ones up to the requested size and cache them */
      for (i = lst_size(set->restr_lists); i <= nodetype; i++) {
        List *l = lst_new_int(2);
        lst_push_int(l, 0);
        if (i > 0) 
          lst_push_int(l, i);
        lst_push_ptr(set->restr_lists, l); /* the ith element of
                                              restr_lists will be a list
                                              consisting of 0 and i (or
                                              just 0 if i == 0) */
      }
    }
    return lst_get_ptr(set->restr_lists, nodetype);
  }
}

/* Pre-populate all cached state-set lists so that cpr_get_state_set
   is purely read-only (safe for concurrent use by multiple threads).
   Must be called before any parallel invocation of cpr_ll_core. */
void cpr_prepopulate_state_sets(CrisprAncestralStateSets *sets,
                                int max_nstates) {
  int i, j;

  /* unr_lists: element i is list of integers 0..i-1 */
  for (i = lst_size(sets->unr_lists); i <= max_nstates; i++) {
    List *l = lst_new_int(i);
    for (j = 0; j < i; j++)
      lst_push_int(l, j);
    lst_push_ptr(sets->unr_lists, l);
  }

  /* restr_lists: element i is {0, i} (or just {0} if i==0) */
  for (i = lst_size(sets->restr_lists); i <= max_nstates; i++) {
    List *l = lst_new_int(2);
    lst_push_int(l, 0);
    if (i > 0)
      lst_push_int(l, i);
    lst_push_ptr(sets->restr_lists, l);
  }

  /* sil_lists: element i is {i-1} for i>1, NULL otherwise */
  for (i = lst_size(sets->sil_lists); i <= max_nstates; i++) {
    List *l = NULL;
    if (i > 1) {
      l = lst_new_int(1);
      lst_push_int(l, i - 1);
    }
    lst_push_ptr(sets->sil_lists, l);
  }
}

void cpr_free_state_sets(CrisprAncestralStateSets *sets) {
  int i;
  List *l;
  for (i = 0; i < lst_size(sets->unr_lists); i++) {
    l = lst_get_ptr(sets->unr_lists, i);
    if (l == NULL) break; /* non-NULL must be contiguous */
    lst_free(l);
  }
  for (i = 0; i < lst_size(sets->restr_lists); i++) {
    l = lst_get_ptr(sets->restr_lists, i);
    if (l == NULL) break;
    lst_free(l);
  }
  for (i = 0; i < lst_size(sets->sil_lists); i++) {
    l = lst_get_ptr(sets->restr_lists, i);
    if (l != NULL) 
      lst_free(l);
  }
  free(sets);
}

/* renumber mutation states so they are dense for each site; needed
   for sitewise mutation matrices */
CrisprMutTable *cpr_new_sitewise_table(CrisprMutTable *origM) {
  int i, j, k, state, newstate;
  CrisprMutTable *M = cpr_copy_table(origM);
  int map[origM->nstates]; 

  M->sitewise_nstates = smalloc(origM->nsites * sizeof(int));
  
  for (j = 0; j < origM->nsites; j++) {
    /* create mapping for col j */
    for (k = 0; k < origM->nstates; k++) map[k] = -1;
    M->sitewise_nstates[j] = 1;
    
    for (i = 0; i < origM->ncells; i++) {
      state = cpr_get_mut(origM, i, j);
      assert(state < origM->nstates);  
      
      if (state == -1 || state == 0)
        newstate = state;
      else {
        if (map[state] == -1)
          map[state] = M->sitewise_nstates[j]++;
        
        newstate = map[state];
      }
      cpr_set_mut(M, i, j, newstate);
    }
  }
  return M;
}

/* create and return a new model based on a given mutation matrix and tree model */
CrisprMutModel *cpr_new_model(CrisprMutTable *M, TreeModel *mod,
                              enum crispr_model_type modtype,
                              enum crispr_mutrates_type mrtype) {
  CrisprMutModel *retval = smalloc(sizeof(CrisprMutModel));
  retval->model_type = modtype;
  retval->mod = mod;
  retval->mut = M;
  retval->nsites = M->nsites;
  retval->ncells = M->ncells;
  retval->nstates = M->nstates;
  retval->sil_rate = CPR_SIL_RATE_INIT;
  retval->leading_t = 0.05;
  retval->mutrates = NULL;
  retval->sitewise_mutrates = NULL;
  retval->mutrates_type = mrtype;
  retval->nthreads = 1;
  return retval;
}
  
/* preprocessing steps for likelihood calculations -- compute
   equilibrium frequencies, initialize substitution models, etc. This
   function allocates new memory and should be called only once prior
   to repeated likelihood calculations */
void cpr_prep_model(CrisprMutModel *cprmod) {
  int j;
  
  if (cprmod->model_type == SITEWISE) {
    cprmod->mut = cpr_new_sitewise_table(cprmod->mut); /* replace with pointer to new table */
    
    /* compute equilibrium frequencies */
    cprmod->sitewise_mutrates = cpr_estim_sitewise_mutrates(cprmod->mut,
                                                            cprmod->mutrates_type);  
  }
  else {
    /* no need to alter the mutation table in this case; just build
       global eq freqs */
    /* however, we do need a dummy sitewise_nstates array */
    cprmod->mut->sitewise_nstates = smalloc(cprmod->nsites * sizeof(int));
    for (j = 0; j < cprmod->nsites; j++)
      cprmod->mut->sitewise_nstates[j] = cprmod->nstates;
    
    cprmod->mutrates = cpr_estim_mutrates(cprmod->mut,
                                          cprmod->mutrates_type);

    /* make all sitewise eqfreqs point to the global eqfreqs */
    cprmod->sitewise_mutrates = lst_new_ptr(cprmod->nsites);
    for (j = 0; j < cprmod->nsites; j++) 
      lst_push_ptr(cprmod->sitewise_mutrates, cprmod->mutrates);
  }
}

/* dump a CrisprMutModel object to a file.  For debugging */
void cpr_print_model(CrisprMutModel *cprmod, FILE *F) {
  int j;
  fprintf(F, "CrisprMutModel:\nmodel_type = %s\nnsites = %d\nncells = %d\nnstates = %d\nsilencing_rate = %f\n",
          (cprmod->model_type == SITEWISE ? "SITEWISE" : "GLOBAL"), cprmod->nsites,
          cprmod->ncells, cprmod->nstates, cprmod->sil_rate);
  
  for (j = 0; j < cprmod->nsites; j++) {
    Vector *mutrates = lst_get_ptr(cprmod->sitewise_mutrates, j);
    fprintf(F, "Model for site %d:\n", j);
    fprintf(F, "Mutation rates:\n");
    vec_print(mutrates, F);
  }
}

/* free memory allocated by cpr_prep_model */
void cpr_free_model(CrisprMutModel *cprmod) {
  int j;
  if (cprmod->model_type == SITEWISE && cprmod->sitewise_mutrates != NULL) {
    for (j = 0; j < cprmod->nsites; j++)
      vec_free(lst_get_ptr(cprmod->sitewise_mutrates, j));
  }

  if (cprmod->mutrates != NULL)
    vec_free(cprmod->mutrates);
  if (cprmod->sitewise_mutrates != NULL)
    lst_free(cprmod->sitewise_mutrates);
}

/* --- code for multithreading of likelihood calculations --- */

/* The main likelihood function (same interface as previously) is now
   a wrapper that handles setup, subsampling, and multithreading.  The
   actual likelihood calculation is done in cpr_ll_core, which is
   called by each thread separately. */
double cpr_compute_log_likelihood(CrisprMutModel *cprmod, Vector *branchgrad) {
  double ll;
    
  /* ---- one-time model setup (not thread-safe if repeated) ---- */
  if (cprmod->mod->msa_seq_idx == NULL)
    cpr_build_seq_idx(cprmod->mod, cprmod->mut);

  /* update length of leading branch */
  cprmod->mod->tree->dparent = cprmod->leading_t;

  /* also set up ancestral state sets if not already available */
  if (ancsets == NULL)
    ancsets = cpr_new_state_sets(cprmod->mod->tree->nnodes);
  if (cprmod->mod->tree->nnodes > ancsets->nnodes) /* in case number of nodes grows */
    cpr_state_sets_resize(ancsets, cprmod->mod->tree->nnodes);

  /* pre-populate cached state-set lists so they are read-only during
     parallel execution of cpr_ll_core */
  cpr_prepopulate_state_sets(ancsets, cprmod->nstates + 1);

  tr_postorder(cprmod->mod->tree); /* ensure these are cached */
  tr_preorder(cprmod->mod->tree);
  
  /* ---- set up gradient cache ---- */
  if (branchgrad != NULL && branchgrad->size != cprmod->mod->tree->nnodes - 1)
    die("ERROR in cpr_compute_log_likelihood: size of branchgrad must be 2n-2\n");

  /* if multiple threads are requested but OpenMP is not active, catch it here */
#ifndef _OPENMP
  if (cprmod->nthreads > 1)   /* FIXME: put this in cprmod also? */
    die("ERROR: Multithreading requested but OpenMP is not enabled.\n");
#endif
  
  /* ---- sequential path ---- */
  if (cprmod->nthreads == 1) 
    ll = cpr_ll_parallel(cprmod, branchgrad, 1);

  /* ---- parallel path ---- */
  else
    ll = cpr_ll_parallel(cprmod, branchgrad, cprmod->nthreads);

  return ll;
}

double cpr_ll_parallel(CrisprMutModel *cprmod, Vector *branchgrad,
  int nthreads_requested) {

  int nsites = cprmod->nsites;

  int maxthreads = NJ_OMP_GET_MAX_THREADS();
  int nthreads = (nthreads_requested <= maxthreads ? nthreads_requested : maxthreads);

  double ll_total = 0.0;

  /* ---- allocate per-thread scratch arrays ---- */
  int nnodes = cprmod->mod->tree->nnodes;
  int **thread_nodetypes = malloc(nthreads * sizeof(int *));
  for (int t = 0; t < nthreads; t++)
    thread_nodetypes[t] = malloc(nnodes * sizeof(int));

  NJDerivs **thread_derivs = malloc(nthreads * sizeof(NJDerivs *));
  double *thread_ll = calloc(nthreads, sizeof(double));

  for (int t = 0; t < nthreads; t++) {
    thread_derivs[t] = malloc(sizeof(NJDerivs));
    thread_derivs[t]->zero_likl = FALSE;

    if (branchgrad != NULL) {
      thread_derivs[t]->branchgrad = vec_new(branchgrad->size);
      vec_zero(thread_derivs[t]->branchgrad);
      thread_derivs[t]->deriv_leading_t = 0.0;
      thread_derivs[t]->deriv_sil = 0.0;
    }
    else {
      thread_derivs[t]->branchgrad = NULL;
    }
  }
  
  /* ---- parallel likelihood computation ---- */
#pragma omp parallel num_threads(nthreads)
  {
    int tid = NJ_OMP_GET_THREAD_NUM();

    int start = (tid * nsites) / nthreads;
    int end = ((tid + 1) * nsites) / nthreads;

    List *range = lst_new_int(2);
    lst_push_int(range, start);
    lst_push_int(range, end);

    thread_ll[tid] = cpr_ll_core(cprmod, thread_derivs[tid],
               thread_nodetypes[tid], range);
    
    lst_free(range);
  }

  /* ---- reduction ---- */
  for (int t = 0; t < nthreads; t++)
    ll_total += thread_ll[t];

  cprmod->zero_likl = FALSE;
  for (int t = 0; t < nthreads; t++) {
    if (thread_derivs[t]->zero_likl)
      cprmod->zero_likl = TRUE;
  }

  if (branchgrad != NULL) {
    vec_zero(branchgrad);
    cprmod->deriv_leading_t = 0.0;
    cprmod->deriv_sil = 0.0;
    for (int t = 0; t < nthreads; t++) {
      vec_plus_eq(branchgrad, thread_derivs[t]->branchgrad);
      cprmod->deriv_leading_t += thread_derivs[t]->deriv_leading_t;
      cprmod->deriv_sil += thread_derivs[t]->deriv_sil;
    }
  }
  
  /* ---- cleanup ---- */
  for (int t = 0; t < nthreads; t++)
    free(thread_nodetypes[t]);
  free(thread_nodetypes);

  free(thread_ll);

  for (int t = 0; t < nthreads; t++) {
    if (thread_derivs[t]->branchgrad != NULL)
      vec_free(thread_derivs[t]->branchgrad);
    free(thread_derivs[t]);
  }
  free(thread_derivs);

  return ll_total;
}

/* Add back duplicate leaves that were collapsed by cpr_deduplicate.
   For each leaf whose name matches a cell with duplicates, replace it
   with a binary caterpillar subtree containing the original leaf plus
   all duplicates, connected by zero-length branches (a polytomy
   within the binary tree constraint).  NOTE: this function only
   modifies the tree structure; call cpr_expand_tables_for_dups()
   once beforehand to add duplicate entries to M and mg. */
void cpr_add_dup_leaves(TreeNode *tree, CrisprMutTable *M) {
  int i, j, idx;
  TreeNode *leaf, *newint, *newleaf, *subtree;

  if (M->dupnames == NULL) return;

  /* collect leaves to expand; can't modify tree during traversal */
  List *trav = tr_postorder(tree);
  int ntrav = lst_size(trav);
  TreeNode **to_expand = smalloc(ntrav * sizeof(TreeNode *));
  int *cell_idx = smalloc(ntrav * sizeof(int));
  int nexpand = 0;
  int total_added = 0;

  for (i = 0; i < ntrav; i++) {
    leaf = lst_get_ptr(trav, i);
    if (leaf->lchild != NULL || leaf->rchild != NULL) continue;
    String *namestr = str_new_charstr(leaf->name);
    if (str_in_list_idx(namestr, M->cellnames, &idx) == 1 &&
        idx < M->ncells && M->dupnames[idx] != NULL) {
      to_expand[nexpand] = leaf;
      cell_idx[nexpand] = idx;
      total_added += 2 * lst_size(M->dupnames[idx]);
      nexpand++;
    }
    str_free(namestr);
  }

  if (nexpand == 0) {
    sfree(to_expand);
    sfree(cell_idx);
    return;
  }

  /* expand each leaf into a caterpillar subtree */
  for (i = 0; i < nexpand; i++) {
    leaf = to_expand[i];
    List *dups = M->dupnames[cell_idx[i]];
    int ndups = lst_size(dups);
    TreeNode *parent = leaf->parent;
    double orig_dparent = leaf->dparent;

    /* start with original leaf and first dup as siblings */
    leaf->dparent = 0;

    newleaf = tr_new_node();
    strcpy(newleaf->name, ((String*)lst_get_ptr(dups, 0))->chars);
    newleaf->dparent = 0;

    subtree = tr_new_node();
    subtree->lchild = leaf;
    subtree->rchild = newleaf;
    leaf->parent = subtree;
    newleaf->parent = subtree;
    subtree->dparent = 0;

    /* wrap in additional internal nodes for remaining dups */
    for (j = 1; j < ndups; j++) {
      newleaf = tr_new_node();
      strcpy(newleaf->name, ((String*)lst_get_ptr(dups, j))->chars);
      newleaf->dparent = 0;

      newint = tr_new_node();
      newint->lchild = subtree;
      newint->rchild = newleaf;
      subtree->parent = newint;
      newleaf->parent = newint;
      newint->dparent = 0;

      subtree = newint;
    }

    /* attach subtree in place of original leaf */
    subtree->dparent = orig_dparent;
    subtree->parent = parent;
    if (parent->lchild == leaf)
      parent->lchild = subtree;
    else
      parent->rchild = subtree;
  }

  sfree(to_expand);
  sfree(cell_idx);

  /* rebuild tree metadata */
  tree->nnodes += total_added;
  tr_reset_nnodes(tree);
}

/* Add duplicate cell entries to M->cellnames, mg->cellnames, and
   mg->states.  Call this ONCE before the first cpr_add_dup_leaves()
   so that cpr_build_seq_idx can resolve duplicate leaf names. */
void cpr_expand_tables_for_dups(CrisprMutTable *M, struct mgtab *mg) {
  int i, j;

  if (M->dupnames == NULL) return;

  for (i = 0; i < M->ncells; i++) {
    if (M->dupnames[i] == NULL) continue;
    List *dups = M->dupnames[i];
    int ndups = lst_size(dups);
    int rep_state = (mg != NULL) ? lst_get_int(mg->states, i) : -1;

    for (j = 0; j < ndups; j++) {
      lst_push_ptr(M->cellnames, str_dup(lst_get_ptr(dups, j)));
      if (mg != NULL) {
        lst_push_ptr(mg->cellnames, str_dup(lst_get_ptr(dups, j)));
        lst_push_int(mg->states, rep_state);
      }
    }
  }

  if (mg != NULL)
    mg->ncells = lst_size(mg->cellnames);
}

/* Diagnostic function to verify consistency of mutation and migration
   tables through the dedup/restore cycle.  Gated by CHECK_DEDUP env
   var.  The 'stage' string is printed for context.  Checks:
   1. M->cellnames and mg->cellnames have same size and same entries
   2. mg->states values are valid (0..mg->nstates-1)
   3. If after expansion, duplicate cells have the same tissue state
      as their representative */
void cpr_check_dedup_tables(CrisprMutTable *M, MigTable *mg,
                            const char *stage) {
  int i, j;
  if (getenv("CHECK_DEDUP") == NULL) return;

  fprintf(stderr, "[CHECK_DEDUP] %s\n", stage);

  /* check 1: sizes match */
  int msize = lst_size(M->cellnames);
  fprintf(stderr, "  M->ncells=%d  lst_size(M->cellnames)=%d", M->ncells, msize);
  if (mg != NULL) {
    int mgsize = lst_size(mg->cellnames);
    int mgstsize = lst_size(mg->states);
    fprintf(stderr, "  mg->ncells=%d  lst_size(mg->cellnames)=%d  lst_size(mg->states)=%d",
            mg->ncells, mgsize, mgstsize);
    if (mgsize != msize)
      fprintf(stderr, "\n  ** MISMATCH: M->cellnames size %d != mg->cellnames size %d", msize, mgsize);
    if (mgstsize != mgsize)
      fprintf(stderr, "\n  ** MISMATCH: mg->cellnames size %d != mg->states size %d", mgsize, mgstsize);
  }
  fprintf(stderr, "\n");

  /* check 2: cellnames match entry by entry */
  if (mg != NULL) {
    int n = msize < lst_size(mg->cellnames) ? msize : lst_size(mg->cellnames);
    int mismatches = 0;
    for (i = 0; i < n; i++) {
      String *mname = lst_get_ptr(M->cellnames, i);
      String *mgname = lst_get_ptr(mg->cellnames, i);
      if (str_compare(mname, mgname) != 0) {
        if (mismatches < 5)
          fprintf(stderr, "  ** NAME MISMATCH at index %d: M='%s' mg='%s'\n",
                  i, mname->chars, mgname->chars);
        mismatches++;
      }
    }
    if (mismatches > 5)
      fprintf(stderr, "  ** ... %d total name mismatches\n", mismatches);
    else if (mismatches == 0)
      fprintf(stderr, "  cellnames match OK (%d entries)\n", n);
  }

  /* check 3: mg->states values are valid */
  if (mg != NULL) {
    int bad = 0;
    for (i = 0; i < lst_size(mg->states); i++) {
      int st = lst_get_int(mg->states, i);
      if (st < 0 || st >= mg->nstates) {
        if (bad < 3)
          fprintf(stderr, "  ** INVALID state %d at index %d (nstates=%d)\n",
                  st, i, mg->nstates);
        bad++;
      }
    }
    if (bad == 0)
      fprintf(stderr, "  mg->states all valid (0..%d)\n", mg->nstates - 1);
  }

  /* check 4: if dupnames exist, verify dup tissue states match representative.
     After dedup but before expansion, dups will not be in mg->cellnames
     (expected).  After expansion, they should be present with matching states. */
  if (M->dupnames != NULL && mg != NULL) {
    Hashtable *namehash = hsh_new(lst_size(mg->cellnames) * 2);
    for (i = 0; i < lst_size(mg->cellnames); i++) {
      String *s = lst_get_ptr(mg->cellnames, i);
      hsh_put_int(namehash, s->chars, i);
    }
    int checked = 0, bad = 0, not_found = 0;
    for (i = 0; i < M->ncells; i++) {
      if (M->dupnames[i] == NULL) continue;
      int rep_state = lst_get_int(mg->states, i);
      List *dups = M->dupnames[i];
      for (j = 0; j < lst_size(dups); j++) {
        String *dname = lst_get_ptr(dups, j);
        int idx = hsh_get_int(namehash, dname->chars);
        if (idx == -1) {
          not_found++;
        }
        else {
          int dup_state = lst_get_int(mg->states, idx);
          if (dup_state != rep_state) {
            fprintf(stderr, "  ** DUP STATE MISMATCH: rep '%s' state=%d, dup '%s' state=%d\n",
                    ((String*)lst_get_ptr(M->cellnames, i))->chars, rep_state,
                    dname->chars, dup_state);
            bad++;
          }
        }
        checked++;
      }
    }
    if (not_found > 0)
      fprintf(stderr, "  %d dup name(s) not yet in mg->cellnames (expected before expansion)\n", not_found);
    if (bad == 0 && checked > 0 && not_found == 0)
      fprintf(stderr, "  dup tissue states match representatives OK (%d dups checked)\n", checked);
    else if (checked == 0 && not_found == 0)
      fprintf(stderr, "  no duplicates to check\n");
    hsh_free(namehash);
  }
}
