#include "latentflow.h"

#include "gexmatrix.h"
#include "misc.h"
#include "model.h"
#include "pca.h"

#include <phast/matrix.h>
#include <phast/misc.h>
#include <phast/trees.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *node_output_name(TreeNode *node, char *buf, size_t buf_size) {
    if (node->name != NULL && node->name[0] != '\0' && strcmp(node->name, ";") != 0)
        return node->name;
    snprintf(buf, buf_size, "node_%d", node->id);
    return buf;
}

static int find_cell_index(char **cell_names, int n_cells, const char *name) {
    int i;

    if (cell_names == NULL || name == NULL)
        return -1;

    for (i = 0; i < n_cells; i++) {
        if (cell_names[i] != NULL && strcmp(cell_names[i], name) == 0)
            return i;
    }
    return -1;
}

static void project_latent_state(Matrix *states,
                                 PCA *latent_pca,
                                 double *tip_factor_means,
                                 int node_id,
                                 double *pc1,
                                 double *pc2) {
    int d;

    *pc1 = 0.0;
    *pc2 = 0.0;
    if (latent_pca == NULL || latent_pca->components == NULL)
        return;

    for (d = 0; d < states->ncols; d++) {
        double centered = mat_get(states, node_id, d) - tip_factor_means[d];
        if (latent_pca->K >= 1)
            *pc1 += centered * mat_get(latent_pca->components, 0, d);
        if (latent_pca->K >= 2)
            *pc2 += centered * mat_get(latent_pca->components, 1, d);
    }
}

static void project_factor_values(double *values,
                                  int k,
                                  PCA *latent_pca,
                                  double *tip_factor_means,
                                  double *pc1,
                                  double *pc2) {
    int d;

    *pc1 = 0.0;
    *pc2 = 0.0;
    if (values == NULL || latent_pca == NULL || latent_pca->components == NULL)
        return;

    for (d = 0; d < k; d++) {
        double centered = values[d] - tip_factor_means[d];
        if (latent_pca->K >= 1)
            *pc1 += centered * mat_get(latent_pca->components, 0, d);
        if (latent_pca->K >= 2)
            *pc2 += centered * mat_get(latent_pca->components, 1, d);
    }
}

static int write_latent_flow_tsv(const char *outprefix,
                                 TreeNode *tree,
                                 Matrix *states,
                                 PCA *latent_pca,
                                 double *tip_factor_means,
                                 Matrix *F,
                                 char **cell_names) {
    int i, d;
    char path[4096];
    FILE *fh = NULL;
    List *preorder = NULL;

    tr_set_nnodes(tree);
    preorder = tr_preorder(tree);

    snprintf(path, sizeof(path), "%s.latent_flow.tsv", outprefix);
    fh = fopen(path, "w");
    if (fh == NULL) {
        fprintf(stderr, "ERROR: could not open %s for writing.\n", path);
        return -1;
    }

    fprintf(fh, "node_id\tparent_id\tnode_name\tparent_name\tis_tip\ttree_depth\tbranch_length\tpc1\tpc2");
    for (d = 0; d < states->ncols; d++)
        fprintf(fh, "\tfactor_%d", d + 1);
    fprintf(fh, "\n");

    for (i = 0; i < lst_size(preorder); i++) {
        TreeNode *node = lst_get_ptr(preorder, i);
        char name_buf[64];
        char parent_name_buf[64];
        const char *name;
        const char *parent_name = "NA";
        int parent_id = -1;
        int is_tip;
        int cell_index = -1;
        double pc1, pc2;
        double *tip_values = NULL;

        if (node == NULL)
            continue;

        name = node_output_name(node, name_buf, sizeof(name_buf));
        if (node->parent != NULL) {
            parent_id = node->parent->id;
            parent_name = node_output_name(node->parent, parent_name_buf,
                                           sizeof(parent_name_buf));
        }
        is_tip = (node->lchild == NULL && node->rchild == NULL);

        if (is_tip) {
            cell_index = find_cell_index(cell_names, F->nrows, name);
            if (cell_index < 0) {
                fclose(fh);
                fprintf(stderr, "ERROR: could not find tip '%s' in expression data while writing latent flow.\n",
                        name);
                return -1;
            }
            tip_values = F->data[cell_index];
            project_factor_values(tip_values, F->ncols, latent_pca,
                                  tip_factor_means, &pc1, &pc2);
        }
        else {
            project_latent_state(states, latent_pca, tip_factor_means,
                                 node->id, &pc1, &pc2);
        }

        fprintf(fh, "%d\t%d\t%s\t%s\t%d\t%.17g\t%.17g\t%.17g\t%.17g",
                node->id,
                parent_id,
                name,
                parent_name,
                is_tip ? 1 : 0,
                tr_distance_to_root(node),
                node->parent == NULL ? 0.0 : node->dparent,
                pc1,
                pc2);
        for (d = 0; d < states->ncols; d++) {
            double value = is_tip ? tip_values[d] : mat_get(states, node->id, d);
            fprintf(fh, "\t%.17g", value);
        }
        fprintf(fh, "\n");
    }

    fclose(fh);
    return 0;
}

int gex_write_latent_flow_outputs(const char *outprefix,
                                  TreeNode *tree,
                                  GexMatrix *gex,
                                  GexLatentBrownianModel *model) {
    int i, d;
    int n_pcs;
    Matrix *F = NULL;
    double *log_sigma2_latent = NULL;
    char **cell_names = NULL;
    double *tip_factor_means = NULL;
    PCA *latent_pca = NULL;
    Matrix *states = NULL;
    int status = 0;

    if (outprefix == NULL || tree == NULL || gex == NULL || model == NULL ||
        model->F == NULL || model->log_sigma2_latent == NULL ||
        gex->cell_names == NULL)
        return -1;

    F = model->F;
    log_sigma2_latent = model->log_sigma2_latent;
    cell_names = gex->cell_names;

    n_pcs = F->ncols < 2 ? F->ncols : 2;
    if (n_pcs <= 0)
        return -1;

    latent_pca = compute_pca(F, n_pcs);
    tip_factor_means = scalloc(F->ncols, sizeof(double));

    for (d = 0; d < F->ncols; d++) {
        for (i = 0; i < F->nrows; i++)
            tip_factor_means[d] += mat_get(F, i, d);
        tip_factor_means[d] /= (double)F->nrows;
    }

    states = gex_reconstruct_latent_tree_states(tree, F,
                                                log_sigma2_latent,
                                                cell_names);
    if (states == NULL) {
        fprintf(stderr, "ERROR: failed to reconstruct latent states.\n");
        status = -1;
    }
    else {
        if (write_latent_flow_tsv(outprefix, tree, states, latent_pca,
                                  tip_factor_means, F, cell_names) != 0)
            status = -1;
        mat_free(states);
    }

    if (latent_pca != NULL)
        free_pca(latent_pca);
    if (tip_factor_means != NULL)
        free(tip_factor_means);

    return status;
}
