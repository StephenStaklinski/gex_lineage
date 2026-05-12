#include "gexlatentflow.h"

#include "gexmatrix.h"
#include "gexmisc.h"
#include "gexmodel.h"
#include "gexpca.h"

#include <phast/matrix.h>
#include <phast/misc.h>
#include <phast/trees.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void collect_tree_nodes_and_depths(TreeNode *node,
                                          TreeNode **nodes_by_id,
                                          double *depth_by_id,
                                          double depth) {
    if (node == NULL)
        return;

    nodes_by_id[node->id] = node;
    depth_by_id[node->id] = depth;

    if (node->lchild != NULL)
        collect_tree_nodes_and_depths(node->lchild, nodes_by_id, depth_by_id,
                                      depth + node->lchild->dparent);
    if (node->rchild != NULL)
        collect_tree_nodes_and_depths(node->rchild, nodes_by_id, depth_by_id,
                                      depth + node->rchild->dparent);
}

static TreeNode *find_tip_by_name(TreeNode *node, const char *name) {
    TreeNode *found = NULL;

    if (node == NULL || name == NULL)
        return NULL;

    if (node->lchild == NULL && node->rchild == NULL &&
        node->name != NULL && strcmp(node->name, name) == 0)
        return node;

    found = find_tip_by_name(node->lchild, name);
    if (found != NULL)
        return found;
    return find_tip_by_name(node->rchild, name);
}

static const char *node_output_name(TreeNode *node, char *buf, size_t buf_size) {
    if (node->name != NULL && node->name[0] != '\0' && strcmp(node->name, ";") != 0)
        return node->name;
    snprintf(buf, buf_size, "node_%d", node->id);
    return buf;
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

static void print_factor_header(FILE *fh, int k) {
    int d;

    for (d = 0; d < k; d++)
        fprintf(fh, "\tfactor_%d", d + 1);
    fprintf(fh, "\n");
}

static int write_latent_nodes_pca_tsv(const char *path,
                                      TreeNode *tree,
                                      TreeNode **nodes_by_id,
                                      double *depth_by_id,
                                      Matrix *states,
                                      PCA *latent_pca,
                                      double *tip_factor_means) {
    int id, d;
    FILE *fh = fopen(path, "w");

    if (fh == NULL) {
        fprintf(stderr, "ERROR: could not open %s for writing.\n", path);
        return -1;
    }

    fprintf(fh, "node_id\tnode_name\tis_tip\ttree_depth\tpc1\tpc2");
    print_factor_header(fh, states->ncols);

    for (id = 0; id < tree->nnodes; id++) {
        TreeNode *node = nodes_by_id[id];
        char name_buf[64];
        double pc1, pc2;

        if (node == NULL)
            continue;

        project_latent_state(states, latent_pca, tip_factor_means, id, &pc1, &pc2);
        fprintf(fh, "%d\t%s\t%d\t%.17g\t%.17g\t%.17g",
                id,
                node_output_name(node, name_buf, sizeof(name_buf)),
                (node->lchild == NULL && node->rchild == NULL) ? 1 : 0,
                depth_by_id[id],
                pc1,
                pc2);
        for (d = 0; d < states->ncols; d++)
            fprintf(fh, "\t%.17g", mat_get(states, id, d));
        fprintf(fh, "\n");
    }

    fclose(fh);
    return 0;
}

static int write_latent_flow_edge(FILE *fh,
                                  TreeNode *parent,
                                  TreeNode *child,
                                  double *depth_by_id,
                                  Matrix *states,
                                  PCA *latent_pca,
                                  double *tip_factor_means) {
    int d;
    char parent_name_buf[64];
    char child_name_buf[64];
    double parent_pc1, parent_pc2, child_pc1, child_pc2;
    double latent_delta_norm = 0.0;

    if (parent == NULL || child == NULL)
        return 0;

    project_latent_state(states, latent_pca, tip_factor_means,
                         parent->id, &parent_pc1, &parent_pc2);
    project_latent_state(states, latent_pca, tip_factor_means,
                         child->id, &child_pc1, &child_pc2);

    for (d = 0; d < states->ncols; d++) {
        double diff = mat_get(states, child->id, d) - mat_get(states, parent->id, d);
        latent_delta_norm += diff * diff;
    }
    latent_delta_norm = sqrt(latent_delta_norm);

    fprintf(fh, "%d\t%d\t%s\t%s\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\n",
            parent->id,
            child->id,
            node_output_name(parent, parent_name_buf, sizeof(parent_name_buf)),
            node_output_name(child, child_name_buf, sizeof(child_name_buf)),
            parent_pc1,
            parent_pc2,
            child_pc1,
            child_pc2,
            child_pc1 - parent_pc1,
            child_pc2 - parent_pc2,
            child->dparent,
            latent_delta_norm,
            depth_by_id[parent->id],
            depth_by_id[child->id]);

    return 0;
}

static int write_latent_flow_edges_recursive(FILE *fh,
                                             TreeNode *node,
                                             double *depth_by_id,
                                             Matrix *states,
                                             PCA *latent_pca,
                                             double *tip_factor_means) {
    if (node == NULL)
        return 0;

    /* Brownian motion reconstructs latent states along the tree, while flow
       direction comes from the rooted parent-to-child tree orientation. */
    write_latent_flow_edge(fh, node, node->lchild, depth_by_id, states,
                           latent_pca, tip_factor_means);
    write_latent_flow_edge(fh, node, node->rchild, depth_by_id, states,
                           latent_pca, tip_factor_means);

    write_latent_flow_edges_recursive(fh, node->lchild, depth_by_id, states,
                                      latent_pca, tip_factor_means);
    write_latent_flow_edges_recursive(fh, node->rchild, depth_by_id, states,
                                      latent_pca, tip_factor_means);
    return 0;
}

static int write_latent_flow_edges_pca_tsv(const char *path,
                                           TreeNode *tree,
                                           double *depth_by_id,
                                           Matrix *states,
                                           PCA *latent_pca,
                                           double *tip_factor_means) {
    FILE *fh = fopen(path, "w");

    if (fh == NULL) {
        fprintf(stderr, "ERROR: could not open %s for writing.\n", path);
        return -1;
    }

    fprintf(fh, "parent_node_id\tchild_node_id\tparent_name\tchild_name\tparent_pc1\tparent_pc2\tchild_pc1\tchild_pc2\tdelta_pc1\tdelta_pc2\tbranch_length\tlatent_delta_norm\ttree_depth_parent\ttree_depth_child\n");
    write_latent_flow_edges_recursive(fh, tree, depth_by_id, states,
                                      latent_pca, tip_factor_means);

    fclose(fh);
    return 0;
}

static int write_developmental_scores_tsv(const char *path,
                                          TreeNode *tree,
                                          double *depth_by_id,
                                          Matrix *states,
                                          PCA *latent_pca,
                                          double *tip_factor_means,
                                          char **cell_names,
                                          int n_cells) {
    int i, d;
    FILE *fh = fopen(path, "w");
    double root_pc1, root_pc2;

    if (fh == NULL) {
        fprintf(stderr, "ERROR: could not open %s for writing.\n", path);
        return -1;
    }

    project_latent_state(states, latent_pca, tip_factor_means, tree->id,
                         &root_pc1, &root_pc2);

    fprintf(fh, "cell\ttree_depth\tlatent_distance_from_root\tprojected_distance_from_root\tpc1\tpc2");
    print_factor_header(fh, states->ncols);

    for (i = 0; i < n_cells; i++) {
        TreeNode *tip = find_tip_by_name(tree, cell_names[i]);
        double pc1, pc2;
        double latent_distance = 0.0;
        double projected_distance;

        if (tip == NULL) {
            fclose(fh);
            fprintf(stderr, "ERROR: could not find cell '%s' in tree while writing developmental scores.\n",
                    cell_names[i]);
            return -1;
        }

        project_latent_state(states, latent_pca, tip_factor_means, tip->id, &pc1, &pc2);
        for (d = 0; d < states->ncols; d++) {
            double diff = mat_get(states, tip->id, d) - mat_get(states, tree->id, d);
            latent_distance += diff * diff;
        }
        latent_distance = sqrt(latent_distance);
        projected_distance = sqrt((pc1 - root_pc1) * (pc1 - root_pc1) +
                                  (pc2 - root_pc2) * (pc2 - root_pc2));

        fprintf(fh, "%s\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g",
                cell_names[i],
                depth_by_id[tip->id],
                latent_distance,
                projected_distance,
                pc1,
                pc2);
        for (d = 0; d < states->ncols; d++)
            fprintf(fh, "\t%.17g", mat_get(states, tip->id, d));
        fprintf(fh, "\n");
    }

    fclose(fh);
    return 0;
}

static int write_latent_flow_for_tree(const char *outprefix,
                                      int tree_index,
                                      int write_aliases,
                                      TreeNode *tree,
                                      Matrix *states,
                                      PCA *latent_pca,
                                      double *tip_factor_means,
                                      char **cell_names,
                                      int n_cells) {
    char path[4096];
    TreeNode **nodes_by_id = NULL;
    double *depth_by_id = NULL;
    int status = 0;

    tr_set_nnodes(tree);
    nodes_by_id = scalloc(tree->nnodes, sizeof(TreeNode *));
    depth_by_id = scalloc(tree->nnodes, sizeof(double));
    collect_tree_nodes_and_depths(tree, nodes_by_id, depth_by_id, 0.0);

    snprintf(path, sizeof(path), "%s.tree%d.latent_nodes.pca.tsv", outprefix, tree_index + 1);
    status |= write_latent_nodes_pca_tsv(path, tree, nodes_by_id, depth_by_id,
                                         states, latent_pca, tip_factor_means);

    snprintf(path, sizeof(path), "%s.tree%d.latent_flow_edges.pca.tsv", outprefix, tree_index + 1);
    status |= write_latent_flow_edges_pca_tsv(path, tree, depth_by_id,
                                              states, latent_pca, tip_factor_means);

    snprintf(path, sizeof(path), "%s.tree%d.developmental_scores.tsv", outprefix, tree_index + 1);
    status |= write_developmental_scores_tsv(path, tree, depth_by_id,
                                             states, latent_pca, tip_factor_means,
                                             cell_names, n_cells);

    if (write_aliases) {
        snprintf(path, sizeof(path), "%s.latent_nodes.pca.tsv", outprefix);
        status |= write_latent_nodes_pca_tsv(path, tree, nodes_by_id, depth_by_id,
                                             states, latent_pca, tip_factor_means);

        snprintf(path, sizeof(path), "%s.latent_flow_edges.pca.tsv", outprefix);
        status |= write_latent_flow_edges_pca_tsv(path, tree, depth_by_id,
                                                  states, latent_pca, tip_factor_means);

        snprintf(path, sizeof(path), "%s.developmental_scores.tsv", outprefix);
        status |= write_developmental_scores_tsv(path, tree, depth_by_id,
                                                 states, latent_pca, tip_factor_means,
                                                 cell_names, n_cells);
    }

    if (nodes_by_id != NULL)
        free(nodes_by_id);
    if (depth_by_id != NULL)
        free(depth_by_id);

    return status == 0 ? 0 : -1;
}

int gex_write_latent_flow_outputs(const char *outprefix,
                                  TreeNode **trees,
                                  int n_trees,
                                  GexMatrix *gex,
                                  GexLatentBrownianModel *model) {
    int t, i, d;
    int n_pcs;
    double *tip_factor_means = NULL;
    PCA *latent_pca = NULL;
    int status = 0;

    if (outprefix == NULL || trees == NULL || n_trees <= 0 ||
        gex == NULL || model == NULL || model->F == NULL)
        return -1;

    n_pcs = model->F->ncols < 2 ? model->F->ncols : 2;
    if (n_pcs <= 0)
        return -1;

    latent_pca = compute_pca(model->F, n_pcs, 1.0);
    tip_factor_means = scalloc(model->F->ncols, sizeof(double));

    for (d = 0; d < model->F->ncols; d++) {
        for (i = 0; i < model->F->nrows; i++)
            tip_factor_means[d] += mat_get(model->F, i, d);
        tip_factor_means[d] /= (double)model->F->nrows;
    }

    for (t = 0; t < n_trees; t++) {
        Matrix *states = gex_reconstruct_latent_tree_states(trees[t],
                                                            model->F,
                                                            model->log_sigma2_latent,
                                                            gex->cell_names);
        if (states == NULL) {
            fprintf(stderr, "ERROR: failed to reconstruct latent states for tree %d.\n", t + 1);
            status = -1;
            break;
        }

        if (write_latent_flow_for_tree(outprefix, t, n_trees == 1, trees[t],
                                       states, latent_pca, tip_factor_means,
                                       gex->cell_names, gex->X->nrows) != 0)
            status = -1;

        mat_free(states);
        if (status != 0)
            break;
    }

    if (latent_pca != NULL)
        free_pca(latent_pca);
    if (tip_factor_means != NULL)
        free(tip_factor_means);

    return status;
}
