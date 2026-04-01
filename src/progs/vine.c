/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025-2026, Cold Spring Harbor Laboratory
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */

#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <time.h>
#include <sys/utsname.h>
#include <phast/misc.h>
#include <phast/msa.h>
#include <phast/maf.h>
#include <nj.h>
#include <likelihoods.h>
#include <variational.h>
#include <geometry.h>
#include <upgma.h>
#include <phast/tree_model.h>
#include <phast/dgamma.h>
#include <phast/subst_mods.h>
#include <phast/sufficient_stats.h>
#include <mvn.h>
#include <mcmc.h>
#include <tree_prior.h>
#include <migration.h>
#include <multiDAG.h>
#include <version.h>
#include "vine.help"

#define DEFAULT_NSAMPLES 100
#define DEFAULT_BATCHSIZE 10
#define DEFAULT_LEARNRATE 0.05
#define DEFAULT_NITER_CONV 50
#define DEFAULT_MIN_ITER 200
#define DEFAULT_KAPPA 4
#define DEFAULT_RANK 3
#define DEFAULT_MCMC_THIN 10

/* default dimensionality is a linear function of log number of taxa */
#define DEFAULT_DIM_INTERCEPT 3.25
#define DEFAULT_DIM_SLOPE 0.92

/* helper to write log file header with version and arguments */
static inline void write_log_header(FILE *LOGF, int argc, char *argv[]) {
  struct utsname u;
  time_t now = time(NULL);
  fprintf(LOGF, "# logfile for VINE (version %s), %s", VINE_VERSION, ctime(&now));
  if (uname(&u) == 0) 
    fprintf(LOGF, "# Host: %s [%s, release %s, %s]\n", u.nodename, u.sysname, u.release, u.machine);
  fprintf(LOGF, "# Command line: ");
  int nchars = 0;
  for (int i = 0; i < argc; i++) {
    nchars += strlen(argv[i]) + 1;
    if (nchars > 80) {
      fprintf(LOGF, "\\\n#   ");
      nchars = strlen(argv[i]) + 4;
    }
    fprintf(LOGF, "%s ", argv[i]);
  }
  fprintf(LOGF, "\n#\n");
}

static void print_embedding(multi_MVN *mmvn, char **names, int n, int d, FILE *F) {
  int i, j;
  Vector *mu_full = vec_new(n * d);
  mmvn_save_mu(mmvn, mu_full);
  for (i = 0; i < n; i++) {
    fprintf(F, "%s", names[i]);
    for (j = 0; j < d; j++)
      fprintf(F, "\t%f", vec_get(mu_full, i*d + j));
    fprintf(F, "\n");
  }
  vec_free(mu_full);
}

int main(int argc, char *argv[]) {
  signed char c;
  int opt_idx, i, ntips = 0, nsamples = DEFAULT_NSAMPLES, dim = -1,
                  batchsize = DEFAULT_BATCHSIZE,
                  niter_conv = DEFAULT_NITER_CONV, min_iter = DEFAULT_MIN_ITER,
                  rank = DEFAULT_RANK, nthreads = 1, dgamma_cats = 1,
                  mcmc_thin = DEFAULT_MCMC_THIN;
  unsigned int nj_only = FALSE, random_start = FALSE, hyperbolic = FALSE,
               dist_embedding = FALSE, mcmc = FALSE,
               natural_grad = FALSE, is_crispr = FALSE,
               ultrametric = FALSE, radial_flow = FALSE, planar_flow = FALSE,
               use_taylor = TRUE, had_dups = FALSE, silent = FALSE,
               log_all = FALSE;
  MSA *msa = NULL;
  enum covar_type covar_param = CONST;
  char *alphabet = "ACGT";
  char **names = NULL;
  msa_format_type format = UNKNOWN_FORMAT;
  FILE *infile = NULL, *indistfile = NULL, *outdistfile = NULL, *logfile = NULL,
    *postmeanfile = NULL, *graphsfile = NULL, *nexusfile = NULL, *embeddingfile = NULL;
  Matrix *D = NULL;
  TreeNode *tree;
  List *namestr, *trees;
  subst_mod_type subst_mod = JC69;
  TreeModel *mod = NULL;
  double learnrate = DEFAULT_LEARNRATE,
    negcurvature = 1.0, var_reg = 1.0, kld_upweight = 1.0;
  MarkovMatrix *rmat = NULL;
  multi_MVN *mmvn = NULL;
  TreeNode *init_tree = NULL;
  CovarData *covar_data = NULL;
  CrisprMutTable *crispr_muts = NULL;
  CrisprMutModel *crispr_mod = NULL;
  enum crispr_model_type crispr_modtype = SITEWISE;
  enum crispr_mutrates_type crispr_muttype = UNIF;
  TreePrior *tprior = NULL;
  enum tree_prior_type tp_type = NONE;
  unsigned int relclock = FALSE;
  MigTable *migtable = NULL;
  List *migstates_lst = NULL;
  char *primary_state = NULL;
  
  struct option long_opts[] = {
    {"format", 1, 0, 'i'},
    {"batchsize", 1, 0, 'b'},
    {"niterconv", 1, 0, 'c'},
    {"dimensionality", 1, 0, 'D'},
    {"distances", 1, 0, 'd'},
    {"dist-embedding", 0, 0, 'e'},
    {"hky85", 0, 0, 'k'}, 
    {"gtr", 0, 0, 'g'}, 
    {"hyperbolic", 0, 0, 'H'},
    {"mcmc", 0, 0, 'J'},
    {"parallel", 1, 0, 'j'},
    {"logfile", 1, 0, 'l'},
    {"mean", 1, 0, 'm'},
    {"miniter", 1, 0, 'M'},
    {"names", 1, 0, 'n'},
    {"negcurvature", 1, 0, 'w'},
    {"nj-only", 0, 0, '0'},
    {"natural-grad", 0, 0, 'N'},
    {"out-dists", 1, 0, 'o'},
    {"sample-graphs", 1, 0, 'O'},
    {"labeled-trees", 1, 0, 'B'},
    {"nsamples", 1, 0, 's'},
    {"learnrate", 1, 0, 'r'},
    {"random-start", 0, 0, 'R'},
    {"var-reg", 1, 0, 'v'},
    {"covar", 1, 0, 'S'},
    {"tree", 1, 0, 't'},
    {"treemodel", 1, 0, 'T'},
    {"thin", 1, 0, 'Q'},
    {"upweight-kld", 1, 0, 'U'},
    {"ultrametric", 0, 0, 'C'},
    {"embedding", 1, 0, 'V'},
    {"rank", 1, 0, 'W'},
    {"radial-flow", 0, 0, 'F'}, 
    {"planar-flow", 0, 0, 'Z'}, 
    {"crispr-modtype", 1, 0, 'Y'},
    {"crispr-mutprior", 1, 0, 'p'},
    {"treeprior", 1, 0, 'P'},
    {"relclock", 0, 0, 'L'},
    {"migration", 1, 0, 'G'},
    {"primary", 1, 0, '1'},
    {"dgamma", 1, 0, 'K'},
    {"montecarlo", 0, 0, 'y'},
    {"log-all", 0, 0, 'a'},
    {"version", 0, 0, 'x'},
    {"silent", 0, 0, 'X'},
    {"help", 0, 0, 'h'},
    {0, 0, 0, 0}
  };

  while ((c = getopt_long(argc, argv, "0:1:ab:B:c:d:D:egG:hHi:FZj:JkK:l:L:m:M:n:No:v:r:Rt:T:Vw:W:S:s:CY:yPp:Xx", long_opts, &opt_idx)) != -1) {
    switch (c) {
    case 'b':
      batchsize = atoi(optarg);
      if (batchsize <= 0)
        die("ERROR: --batchsize must be positive\n");
      break;
    case 'B':
      nexusfile = phast_fopen(optarg, "w");
      break;
    case 'c':
      niter_conv = atoi(optarg);
      if (niter_conv <= 0)
        die("ERROR: --niterconv must be positive\n");
      break;
    case 'd':
      indistfile = phast_fopen(optarg, "r");
      break;
    case 'D':
      dim = atoi(optarg);
      if (dim <= 0)
        die("ERROR: --dimensionality must be positive\n");
      break;
    case 'e':
      dist_embedding = TRUE;
      break;
    case 'H':
      hyperbolic = TRUE;
      break;
    case 'F':
      radial_flow = TRUE;
      break;
    case 'i':
      if (!strcmp(optarg, "CRISPR"))
        is_crispr = TRUE;
      else
        format = msa_str_to_format(optarg);
      break;
    case 'j':
      nthreads = atoi(optarg);
      if (nthreads <= 0)
        die("ERROR: --nthreads must be positive\n");
      break;
    case 'g':
      subst_mod = REV;
      break;
    case 'G':
      migtable = mig_read_table(phast_fopen(optarg, "r"));
      break;
    case 'J':
      mcmc = TRUE;
      break;
    case 'Q':
      mcmc_thin = atoi(optarg);
      if (mcmc_thin <= 0)
        die("ERROR: --mcmc-thin must be positive\n");
      break;
    case '0':
      nj_only = TRUE;
      break;
    case 'k':
      subst_mod = HKY85;
      break;
    case 'K':
      dgamma_cats = atoi(optarg);
      if (dgamma_cats <= 0)
        die("ERROR: --dgamma <cats> must be positive\n");
      break;
    case 'w':
      negcurvature = atof(optarg);
      if (negcurvature < 0)
        die("ERROR: --negcurvature must be nonnegative\n");
      break;
    case 'U':
      kld_upweight = atof(optarg);
      if (kld_upweight < 1)
        die("ERROR: --upweight-kld must be at least 1.0\n");
      break;
    case 'l':
      logfile = phast_fopen(optarg, "w");
      write_log_header(logfile, argc, argv);
      break;
    case 'L':
      relclock = TRUE;
      break;
    case 'm':
      postmeanfile = phast_fopen(optarg, "w");
      break;
    case 'M':
      min_iter = atoi(optarg);
      if (min_iter <= 0)
        die("ERROR: --miniter must be positive\n");
      break;
    case 'n':
      namestr = get_arg_list(optarg);
      ntips = lst_size(namestr);
      names = smalloc(ntips * sizeof(char*));
      for (i = 0; i < ntips; i++)
        names[i] = ((String*)lst_get_ptr(namestr, i))->chars;
      break;
    case 'N':
      natural_grad = TRUE;
      break;
    case 'o':
      outdistfile = phast_fopen(optarg, "w");
      break;
    case 'O':
      graphsfile = phast_fopen(optarg, "w");
      break;
    case 'v':
      var_reg = atof(optarg);
      if (var_reg < 0)
        die("ERROR: --var-reg must be non-negative\n");
      break;
    case 'r':
      learnrate = atof(optarg);
      if (learnrate <= 0)
        die("ERROR: --learnrate must be positive\n");
      break;
    case 'R':
      random_start = TRUE;
      break;
    case 's':
      nsamples = atoi(optarg);
      if (nsamples <= 0)
        die("ERROR: --nsamples must be > 0\n");
      break;
    case 'S':
      if (!strcmp(optarg, "DIAG"))
        covar_param = DIAG;
      else if (!strcmp(optarg, "CONST"))
        covar_param = CONST;
      else if (!strcmp(optarg, "DIST"))
        covar_param = DIST;
      else if (!strcmp(optarg, "LOWR"))
        covar_param = LOWR;
      else die("ERROR: bad argument to --covar (-S).\n");
      break;
    case 't':
      init_tree = tr_new_from_file(phast_fopen(optarg, "r"));
      break;
    case 'T':
      mod = tm_new_from_file(phast_fopen(optarg, "r"), 1);
      init_tree = mod->tree;
      break;
    case 'V':
      embeddingfile = phast_fopen(optarg, "w");
      break;
    case 'C':
      ultrametric = TRUE;
      break;
    case 'W':
      rank = atoi(optarg);
      if (rank <= 0)
        die("ERROR: --rank must be positive\n");
      break;
    case 'Y':
      if (!strcmp(optarg, "SITEWISE"))
        crispr_modtype = SITEWISE;
      else if (!strcmp(optarg, "GLOBAL"))
        crispr_modtype = GLOBAL;
      else die("ERROR: bad argument to --crispr-modtype (-Y).\n");
      break;
    case 'Z':
      planar_flow = TRUE;
      break;
    case 'p':
      if (!strcmp(optarg, "UNIF"))
        crispr_muttype = UNIF;
      else if (!strcmp(optarg, "EMPIRICAL"))
        crispr_muttype = EMPIRICAL;
      else die("ERROR: bad argument to --crispr-mutprior (-p).\n");
      break;
    case 'P':
      if (!strcmp(optarg, "GAMMA"))
        tp_type = GAMMA;
      else if (!strcmp(optarg, "YULE"))
        tp_type = YULE;
      else die("ERROR: bad argument to --treeprior (-P).\n");
      break;
    case '1':
      primary_state = optarg;
      break;
    case 'y':
      use_taylor = FALSE;
      break;
    case 'X':
      silent = TRUE;
      break;
    case 'a':
      log_all = TRUE;
      break;
    case 'x':
      printf("VINE version %s\n", VINE_VERSION);
      exit(0);
    case 'h':
      printf("%s", HELP); 
      exit(0);
    case '?':
      die("Bad argument.  Try 'vine -h'.\n");
    }
  }

  if (init_tree != NULL && indistfile != NULL)
    die("Cannot specify both --tree/-treemod and --distances\n");

  if (hyperbolic == TRUE && (radial_flow == TRUE || planar_flow == TRUE))
    die("Cannot use --radial-flow or --planar-flow with --hyperbolic.\n");
  
  if (hyperbolic == TRUE && negcurvature == 0) 
    hyperbolic = FALSE;
  /* for convenience in scripting; nonhyperbolic considered special case of hyperbolic */

  /* set up tree prior if selected */
  if (tp_type != NONE || relclock == TRUE)
    tprior = tp_new(tp_type, relclock);
  
  if (tprior != NULL && (is_crispr == TRUE || ultrametric == TRUE))
    die("Tree prior cannot be used with CRISPR mutation model or ultrametric trees.\n");
  
  if (rank != DEFAULT_RANK && covar_param != LOWR && !silent)
    fprintf(stderr, "WARNING: --rank ignored when --covar is not LOWR\n");

  if (migtable != NULL && is_crispr == FALSE)
      die("--migration requires -i CRISPR\n");

  if (graphsfile != NULL && migtable == NULL)
      die("--sample-graphs requires --migration\n");

  if (primary_state != NULL && migtable == NULL)
      die("--primary requires --migration\n");
  
  if (nexusfile != NULL && migtable == NULL)
      die("--labeled-trees requires --migration\n");

  if (use_taylor && batchsize != DEFAULT_BATCHSIZE && !silent)
    fprintf(stderr,
            "WARNING: --batchsize ignored when using Taylor approximation.\n");

  if (is_crispr == TRUE && dgamma_cats != 1)
    die("--dgamma-cats cannot be used with CRISPR.\n");
  
  if ((nj_only || dist_embedding) &&
      (indistfile != NULL || init_tree != NULL)) {
    if (optind != argc) 
      die("ERROR: No alignment needed in this case.  Too many arguments.  Try 'vine -h'.\n");
  }
  else { /* handle alignment file or crispr mutation table */
    if (optind != argc - 1)
      die("ERROR: alignment/mutation file required.\n");

    if (!silent) fprintf(stderr, "Reading genotype data from %s...\n", argv[optind]);
    infile = phast_fopen(argv[optind], "r");
    if (is_crispr) { /* CRISPR mutation table */
      crispr_muts = cpr_read_table(infile);
      ntips = crispr_muts->ncells;
      if (!silent) fprintf(stderr, "Read mutation matrix with %d cells and %d sites...\n", crispr_muts->ncells, crispr_muts->nsites);

      if (migtable != NULL)  /* do this before deduplication */
        mig_check_table(migtable, crispr_muts); /* ensure same cell names */

      cpr_check_dedup_tables(crispr_muts, migtable, "after mig_check_table (pre-dedup)");

      cpr_deduplicate(crispr_muts, migtable); /* collapse identical genotypes; modifies
                                       crispr_muts in place */
      cpr_check_dedup_tables(crispr_muts, migtable, "after cpr_deduplicate");

      if (crispr_muts->ncells < ntips) {
        had_dups = TRUE;
        if (!silent) fprintf(stderr, "Collapsed %d duplicates; %d unique genotypes remain...\n",
          ntips - crispr_muts->ncells, crispr_muts->ncells);
        ntips = crispr_muts->ncells;
      }
      names = smalloc(ntips * sizeof(char*));
      for (i = 0; i < ntips; i++)
        names[i] = ((String*)lst_get_ptr(crispr_muts->cellnames, i))->chars;
      ultrametric = TRUE;
      crispr_mod = cpr_new_model(crispr_muts, NULL, crispr_modtype, crispr_muttype);
      /* leave tree model null for now; fill in later */

    }
    else { /* standard alignment file */
      if (format == UNKNOWN_FORMAT)
        format = msa_format_for_content(infile, 1);
      if (format == MAF) 
        msa = maf_read(infile, NULL, 1, alphabet,
                       NULL, NULL, -1, TRUE, NULL, NO_STRIP, FALSE);
      else
        msa = msa_new_from_file_define_format(infile, format, alphabet);
      
      if (msa->ss == NULL)
        ss_from_msas(msa, 1, TRUE, NULL, NULL, NULL, -1, 0);

      if (!silent) fprintf(stderr, "Read DNA alignment with %d sequences and %d sites (%d distinct tuples)...\n", msa->nseqs, msa->length, msa->ss->ntuples);
      
      names = msa->names;
      ntips = msa->nseqs;
    }
  }

  /* case where we have a tree only, no alignment or mutation table */
  if (msa == NULL && crispr_muts == NULL && names == NULL && init_tree) {
    List *namelst = tr_leaf_names(init_tree); /* have to convert to char arrays */
    ntips = lst_size(namelst);
    names = smalloc(sizeof(char*)*ntips);
    for (i = 0; i < ntips; i++) {
      String *str = lst_get_ptr(namelst, i);
      names[i] = smalloc(sizeof(char) * (str->length+1));
      strcpy(names[i], str->chars);
    }
  }
    
  /* get a distance matrix */
  if (init_tree != NULL)
    D = nj_tree_to_distances(init_tree, names, ntips);  
  else if (indistfile != NULL) {
    D = mat_new_from_file_square(indistfile);
    ntips = D->nrows;
    /* in this case we may still be missing names; just assign numbers */
    if (names == NULL) {
      names = smalloc(sizeof(char*)*ntips);
      for (i = 0; i < ntips; i++) {
        names[i] = smalloc(STR_SHORT_LEN * sizeof(char));
        snprintf(names[i], STR_SHORT_LEN, "leaf_%d", i);
      }
    }
  }
  else if (msa != NULL)
    D = nj_compute_JC_matr(msa);
  else if (crispr_muts != NULL)
    D = cpr_compute_dist(crispr_muts);
  else
    die("ERROR: no distance matrix available\n");

  /* make sure we have at least three taxa */
  if (ntips < 3)
    die("ERROR: at least three taxa/cells are required.\n");
  
  /* at this point, names and ntips must be defined even if we don't have an alignment */
  /* We must also have a distance matrix now; make sure valid */
  nj_test_D(D);  
  
  /* set default dimensionality if not specified */
  if (dim == -1) {
    assert(DEFAULT_DIM_INTERCEPT >= 2);
    dim = round(DEFAULT_DIM_INTERCEPT + DEFAULT_DIM_SLOPE * log((double)ntips));
    if (!silent) fprintf(stderr, "Setting dimensionality to default of %d based on %d taxa...\n", dim, ntips);
  }
  
  covar_data = nj_new_covar_data(covar_param, D, dim, msa, crispr_mod, names,
                                 natural_grad, kld_upweight, rank, var_reg,
                                 hyperbolic, negcurvature, ultrametric,
                                 radial_flow, planar_flow, tprior, migtable,
                                 use_taylor);
  if (is_crispr)
    covar_data->no_zero_br = TRUE;
  if (primary_state != NULL)
    mig_set_primary_state(migtable, primary_state);

  if (dist_embedding == TRUE) {
    /* in this case, embed the distances now */
    if (outdistfile == NULL)
      die("ERROR: must use --out-dists with --dist-embedding\n");

    mmvn = mmvn_new(ntips, dim, covar_data->mvn_type);
    nj_estimate_mmvn_from_distances(covar_data, mmvn);
  }

  else {
    /* we'll need a starting tree for either variational inference
       or NJ-only */
    tree = nj_inf(D, names, NULL, NULL, covar_data);

    if (nj_only == TRUE) { /* just print in this case */
      if (had_dups == TRUE) {
        cpr_expand_tables_for_dups(crispr_muts, migtable);
        cpr_check_dedup_tables(crispr_muts, migtable, "after cpr_expand_tables_for_dups (NJ path)");
        cpr_add_dup_leaves(tree, crispr_muts);
      }
      if (!silent) fprintf(stderr, "Outputting NJ tree...\n");
      tr_print(stdout, tree, TRUE);

      if (embeddingfile != NULL) {  /* set up initial embedding for output below */
        mmvn = mmvn_new(ntips, dim, covar_data->mvn_type);
        nj_estimate_mmvn_from_distances(covar_data, mmvn);
      }
    }

    else {  /* full variational inference */
      if (msa == NULL && crispr_muts == NULL)
        die("ERROR: Alignment/mutations required for variational inference\n");

      /* set up a tree model if necessary */
      if (mod == NULL) {
        /* note: this model is just a dummy in the crispr case; tree
           will be used but subst model will be ignored */
        rmat = mm_new(strlen(DEFAULT_ALPHABET), DEFAULT_ALPHABET, CONTINUOUS);
        mod = tm_new(tree, rmat, NULL, subst_mod, DEFAULT_ALPHABET,
                     dgamma_cats, 1, NULL, -1);
        if (msa != NULL)
          tm_init_backgd(mod, msa, -1);

        if (is_crispr) {
          if (!silent) fprintf(stderr, "Using CRISPR mutation model...\n");
          crispr_mod->mod = mod;
          cpr_prep_model(crispr_mod);
          if (!silent && migtable != NULL)
            fprintf(stderr, "Using migration model with %d states...\n", migtable->nstates);
        }
        else if (subst_mod == JC69) {
          if (!silent) fprintf(stderr, "Using JC69 substitution model...\n");
          tm_set_JC69_matrix(mod);
        }
        else if (subst_mod == HKY85) {
          if (!silent) fprintf(stderr, "Using HKY85 substitution model...\n");
          covar_data->hky_kappa = DEFAULT_KAPPA;
          tm_set_HKY_matrix(mod, covar_data->hky_kappa, -1);
        }
        else if (subst_mod == REV) {
          if (!silent) fprintf(stderr, "Using GTR substitution model...\n");
          covar_data->gtr_params = vec_new(GTR_NPARAMS);
          covar_data->deriv_gtr = vec_new(GTR_NPARAMS);
          vec_set_random(covar_data->gtr_params, 1.0, 0.1);
          nj_init_gtr_mapping(mod);
          tm_set_rate_matrix(mod, covar_data->gtr_params, 0);
        }
        else
          (assert(0)); /* should not get here */

        if (dgamma_cats > 1) {
          if (!silent) fprintf(stderr, "Using %d discrete gamma rate categories...\n", dgamma_cats);
          covar_data->dgamma_cats = dgamma_cats;
          DiscreteGamma(mod->freqK, mod->rK, mod->alpha, mod->alpha, 
            mod->nratecats, 0); 
        }
      }

      /* initialize parameters of multivariate normal */
      mmvn = mmvn_new(ntips, dim, covar_data->mvn_type);
      if (random_start == TRUE) {
        mvn_sample_std(mmvn->mvn->mu);
        vec_scale(mmvn->mvn->mu, 0.1);
      }
      else 
        nj_estimate_mmvn_from_distances(covar_data, mmvn); 

      if (use_taylor && !silent)
        fprintf(stderr, "Using Taylor approximation for ELBO...\n");
      else if (!silent)
        fprintf(stderr, "Using Monte Carlo estimation of ELBO...\n");

      if (nthreads > 1) {
        if (is_crispr)
          crispr_mod->nthreads = nthreads;
        else
          covar_data->nthreads = nthreads;
        if (!silent) fprintf(stderr, "Using %d threads for likelihood calculations...\n", nthreads);
      }
      else
        if (!silent) fprintf(stderr, "Multithreading is OFF (see -j)...\n");  

      if (!silent) fprintf(stderr, "Starting variational inference...\n");

      nj_variational_inf(mod, mmvn, batchsize, learnrate,
                         niter_conv, min_iter, 
                         covar_data, logfile, silent, log_all);

      if (had_dups == TRUE && !silent)
        fprintf(stderr, "Sampling trees and re-adding duplicate cells...\n");
      else if (!silent)
        fprintf(stderr, "Sampling trees...\n");

      /* expand mutation and migration tables once for duplicate names */
      if (had_dups == TRUE) {
        cpr_expand_tables_for_dups(crispr_mod->mut, migtable);
        cpr_check_dedup_tables(crispr_mod->mut, migtable, "after cpr_expand_tables_for_dups (VI path)");
      }

      if (mcmc == TRUE) {
        if (!silent)
          fprintf(stderr, "Refining samples by MCMC with thinning interval of %d...\n", mcmc_thin);
        covar_data->subsample = FALSE;  /* MCMC always needs exact likelihood */
        trees = nj_var_sample_mcmc(nsamples, mcmc_thin, mmvn, covar_data, mod,
                                   logfile, silent);
      }

      else /* otherwise just sample directly from approx posterior */
        trees = nj_var_sample(nsamples, mmvn, covar_data, names, NULL);

      /* expand all sampled trees; msa_seq_idx must be rebuilt per tree
         because caterpillar node IDs vary across trees (the global
         idcounter advances with each expansion and tr_set_nnodes
         renumbers, so the same dup leaf may get a different ID in
         different sampled trees) */
      if (had_dups == TRUE) {
        for (i = 0; i < nsamples; i++)
          cpr_add_dup_leaves(lst_get_ptr(trees, i), crispr_mod->mut);
        /* invalidate msa_seq_idx so it is rebuilt per-tree below */
        if (crispr_mod->mod->msa_seq_idx != NULL) {
          sfree(crispr_mod->mod->msa_seq_idx);
          crispr_mod->mod->msa_seq_idx = NULL;
        }
      }

      for (i = 0; i < nsamples; i++) {
        TreeNode *t = (TreeNode *)lst_get_ptr(trees, i);

        /* rebuild msa_seq_idx for this tree's expanded leaf IDs */
        if (had_dups == TRUE) {
          if (crispr_mod->mod->msa_seq_idx != NULL) {
            sfree(crispr_mod->mod->msa_seq_idx);
            crispr_mod->mod->msa_seq_idx = NULL;
          }
          TreeNode *saved_tree = crispr_mod->mod->tree;
          crispr_mod->mod->tree = t;
          cpr_build_seq_idx(crispr_mod->mod, crispr_mod->mut);
          crispr_mod->mod->tree = saved_tree;
        }

        tr_print(stdout, t, TRUE);

        /* in these cases we need to sample cell states for each tree */
        if (graphsfile != NULL || nexusfile != NULL) {
          if (i == 0 && !silent)
            fprintf(stderr, "Sampling cell states...\n");

          if (migstates_lst == NULL) migstates_lst = lst_new_ptr(nsamples);
          List *states = lst_new_ptr(t->nnodes);
          mig_sample_states(t, migtable, crispr_mod, states);
          lst_push_ptr(migstates_lst, states); /* mark end of sample */
        }
      }

      /* output sampled cell states if needed */
      if (graphsfile != NULL) {
        if (!silent) fprintf(stderr, "Writing migration graphs...\n");
        mig_print_set_dot(trees, graphsfile, migtable, migstates_lst);
      }
      if (nexusfile != NULL) {
        if (!silent) fprintf(stderr, "Writing cell-state-labeled trees...\n");
        mig_print_set_labeled_nexus(trees, nexusfile, migtable, migstates_lst);
      }

      if (postmeanfile != NULL) {
        if (!silent) fprintf(stderr, "Writing posterior mean tree...\n");
        Vector *mu_full = vec_new(mmvn->d * mmvn->n);
        mmvn_save_mu(mmvn, mu_full);
        TreeNode *t = nj_mean(mu_full, names, covar_data);
        if (had_dups == TRUE)
          cpr_add_dup_leaves(t, crispr_mod->mut); /* add back in duplicate leaves if needed */
        tr_print(postmeanfile, t, TRUE);
        vec_free(mu_full);
      }
    }
  }

  if (outdistfile != NULL) {
    if (dist_embedding == TRUE || nj_only == FALSE)
      /* in this case need to reset D */
      nj_mmvn_to_distances(mmvn, covar_data);

    mat_print(D, outdistfile);
  }

  if (embeddingfile != NULL) {
    if (!silent) fprintf(stderr, "Dumping embedding...\n");
    print_embedding(mmvn, names, covar_data->nseqs, covar_data->dim, embeddingfile);
  }
  
  /* free everything */
  if (msa != NULL)
    msa_free(msa);
  if (crispr_muts != NULL)
    cpr_free_table(crispr_muts);
  if (mod != NULL)
    tm_free(mod);
  if (covar_data != NULL)
    nj_free_covar_data(covar_data);
  if (mmvn != NULL)
    mmvn_free(mmvn);
  if (logfile != NULL)
    fclose(logfile);
  if (outdistfile != NULL)
    fclose(outdistfile);
  if (postmeanfile != NULL)
    fclose(postmeanfile);
  if (graphsfile != NULL)
    fclose(graphsfile);
  if (nexusfile != NULL)
    fclose(nexusfile);

  if (!silent) fprintf(stderr, "Done.\n");
  
  return (0);
}
