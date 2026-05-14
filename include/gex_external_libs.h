#ifndef _GEX_EXTERNAL_LIBS_
#define _GEX_EXTERNAL_LIBS_

#include <stdio.h>
#include <stdlib.h>

#ifdef VINE_HAS_OPENMP
#include <omp.h>
#endif

#ifdef VINE_HAS_OPENBLAS_THREAD_CONTROL
#include <openblas/cblas.h>
#endif

#ifdef VINE_HAS_MKL_THREAD_CONTROL
#include <mkl.h>
#endif

typedef int    LAPACK_INT;
typedef double LAPACK_DOUBLE;

void dgemm_(const char *transa, const char *transb,
            const LAPACK_INT *m, const LAPACK_INT *n, const LAPACK_INT *k,
            const LAPACK_DOUBLE *alpha,
            const LAPACK_DOUBLE *a, const LAPACK_INT *lda,
            const LAPACK_DOUBLE *b, const LAPACK_INT *ldb,
            const LAPACK_DOUBLE *beta,
            LAPACK_DOUBLE *c, const LAPACK_INT *ldc);

void dtrtrs_(const char *uplo, const char *trans, const char *diag,
            const LAPACK_INT *n, const LAPACK_INT *nrhs,
            const LAPACK_DOUBLE *a, const LAPACK_INT *lda,
            LAPACK_DOUBLE *b, const LAPACK_INT *ldb,
            LAPACK_INT *info);

void dgesdd_(const char *jobz,
             const LAPACK_INT *m, const LAPACK_INT *n,
             LAPACK_DOUBLE *a, const LAPACK_INT *lda,
             LAPACK_DOUBLE *s,
             LAPACK_DOUBLE *u, const LAPACK_INT *ldu,
             LAPACK_DOUBLE *vt, const LAPACK_INT *ldvt,
             LAPACK_DOUBLE *work, const LAPACK_INT *lwork,
             LAPACK_INT *iwork,
             LAPACK_INT *info);

static inline int has_blas_thread_control(void) {
#ifdef VINE_HAS_OPENBLAS_THREAD_CONTROL
    return openblas_get_parallel() != 0;
#elif defined(VINE_HAS_MKL_THREAD_CONTROL)
    return 1;
#else
    return 0;
#endif
}

static inline int has_thread_control(void) {
#if defined(VINE_HAS_OPENMP)
    return 1;
#else
    return has_blas_thread_control();
#endif
}

static inline void set_openmp_num_threads(int nthreads, int verbose) {
    (void)nthreads;
    (void)verbose;

#ifdef VINE_HAS_OPENMP
    omp_set_num_threads(nthreads);
    if (verbose)
        printf("Using %d thread(s) for OpenMP-enabled calculations.\n", nthreads);
#endif
}

static inline void set_blas_num_threads(int nthreads, int verbose) {
    (void)nthreads;
    (void)verbose;

#ifdef VINE_HAS_OPENBLAS_THREAD_CONTROL
    openblas_set_num_threads(nthreads);
    if (verbose) {
        int actual_nthreads = openblas_get_num_threads();
        if (actual_nthreads == nthreads) {
            printf("Using %d thread(s) for OpenBLAS matrix multiplications.\n", nthreads);
        }
        else {
            printf("Requested %d OpenBLAS thread(s), but OpenBLAS reports %d thread(s) (%s).\n",
                   nthreads, actual_nthreads, openblas_get_config());
        }
    }
#endif

#ifdef VINE_HAS_MKL_THREAD_CONTROL
    mkl_set_num_threads(nthreads);
    if (verbose)
        printf("Using %d thread(s) for MKL matrix multiplications.\n", mkl_get_max_threads());
#endif
}

static inline void set_num_threads(int nthreads, int verbose) {
    set_openmp_num_threads(nthreads, verbose);
    set_blas_num_threads(nthreads, verbose);
}

#endif
