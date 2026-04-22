#ifndef _GEX_EXTERNAL_LIBS_
#define _GEX_EXTERNAL_LIBS_

#include <stdlib.h>

typedef int    LAPACK_INT;
typedef double LAPACK_DOUBLE;

void dgemm_(const char *transa, const char *transb,
            const LAPACK_INT *m, const LAPACK_INT *n, const LAPACK_INT *k,
            const LAPACK_DOUBLE *alpha,
            const LAPACK_DOUBLE *a, const LAPACK_INT *lda,
            const LAPACK_DOUBLE *b, const LAPACK_INT *ldb,
            const LAPACK_DOUBLE *beta,
            LAPACK_DOUBLE *c, const LAPACK_INT *ldc);

#endif