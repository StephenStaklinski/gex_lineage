#ifndef GEX_LINEAGE_MVN_H
#define GEX_LINEAGE_MVN_H

#include <phast/matrix.h>

enum mvn_type {MVN_STD, MVN_IDENTITY, MVN_DIAG, MVN_GEN};

typedef struct {
    int dim;
    Vector *mu;
    Matrix *sigma;
    Matrix *cholL;
    Vector *evals;
    Matrix *evecs;
    enum mvn_type type;
} MVN;

MVN *mvn_new(int dim, Vector *mu, Matrix *sigma);

void mvn_free(MVN *mvn);

void mvn_preprocess(MVN *mvn, unsigned int force_eigen);

void mvn_sample(MVN *mvn, Vector *retval);

#endif
