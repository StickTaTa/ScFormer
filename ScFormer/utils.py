import anndata as ad
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter
from tqdm import tqdm
import math
import scanpy as sc
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import normalized_mutual_info_score


def subgraph(graph, seed, n_neighbors, node_sele_prob):
    """
    Generates a subgraph starting from a seed node.

    Parameters:
    - graph (scipy.sparse matrix): Adjacency matrix of the graph.
    - seed (int): Seed node ID.
    - n_neighbors (list of int): Number of neighbors to select at each layer.
    - node_sele_prob (np.array): Selection probabilities for nodes.

    Returns:
    - List of selected node indices excluding the seed.
    """
    picked_nodes = {seed}
    last_layer_nodes = {seed}

    # Iteratively select neighbors for each layer
    for layer, n_neighbors_current in enumerate(n_neighbors):
        # Find all neighbors of the last layer nodes
        neighbors = graph[list(last_layer_nodes), :].nonzero()[1]
        neighbors = np.unique(neighbors)

        # If there are no neighbors, terminate early
        if len(neighbors) == 0:
            break

        # Get selection probabilities for the neighbors
        neighbors_prob = node_sele_prob[neighbors]
        neighbors_prob = softmax_stable(neighbors_prob)  # Normalize probabilities

        # Determine the number of neighbors to pick
        to_pick = n_neighbors_current
        n_neighbors_real = min(to_pick, len(neighbors))

        # Select neighbors without replacement based on probabilities
        selected_neighbors = np.random.choice(
            neighbors,
            size=n_neighbors_real,
            replace=False,
            p=neighbors_prob
        )

        # Update the sets of picked and last layer nodes
        last_layer_nodes = set(selected_neighbors)
        picked_nodes.update(last_layer_nodes)

    # Exclude the seed node from the final indices
    indices = sorted(picked_nodes - {seed})
    return indices


def batch_select_whole(RNA_matrix, neighbor=[20], cell_size=30):
    """
    Partitions the RNA matrix into batches and selects subgraphs for each cell.

    Parameters:
    - RNA_matrix (scipy.sparse matrix): Gene expression matrix (genes x cells).
    - neighbor (list of int): Number of neighbors to select at each layer.
    - cell_size (int): Number of cells per batch.

    Returns:
    - indices_ss (list of dict): List containing gene and cell indices for each batch.
    - node_ids (np.array): Array of shuffled cell IDs.
    - dic (dict): Dictionary mapping each cell to its selected gene indices.
    """
    print('Partitioning the data into batches. Please wait...')

    # Shuffle cell IDs
    node_ids = np.random.choice(RNA_matrix.shape[1], size=RNA_matrix.shape[1], replace=False)
    n_batch = math.ceil(node_ids.shape[0] / cell_size)
    indices_ss = []

    RNA_matrix = RNA_matrix.tocsr()  # Ensure efficient row slicing
    dic = {}

    for i in tqdm(range(n_batch), desc="Processing Batches"):
        gene_indices_all = []
        cell_indices_all = []

        # Determine the range for the current batch
        start_idx = i * cell_size
        end_idx = min((i + 1) * cell_size, node_ids.shape[0])
        batch_node_ids = node_ids[start_idx:end_idx]

        for node in batch_node_ids:
            # Extract gene expression for the current cell
            rna_expression = RNA_matrix[:, node].toarray().flatten()
            rna_expression[rna_expression < 5] = 0  # Thresholding

            # Compute selection probabilities
            selection_prob = np.log(rna_expression + 1)
            selection_prob = np.squeeze(selection_prob)

            # Generate subgraph for the current cell
            gene_indices = subgraph(RNA_matrix.transpose(), node, neighbor, selection_prob)

            # Update dictionaries and lists
            dic[node] = {'g': gene_indices}
            gene_indices_all.extend(gene_indices)

        # Remove duplicate gene indices
        gene_indices_all = sorted(set(gene_indices_all))

        # Prepare the batch dictionary
        batch_dict = {
            'gene_index': gene_indices_all,
            'cell_index': list(batch_node_ids)
        }
        indices_ss.append(batch_dict)

    return indices_ss, node_ids, dic


# def batch_select_whole(RNA_matrix, ATAC_matrix, neighbor=[20], cell_size=30):
#     print('We are currently in the process of partitioning the data into batches. Kindly wait for a moment, please.')
#     node_ids = np.random.choice(RNA_matrix.shape[1], size=RNA_matrix.shape[1], replace=False)
#     n_batch = math.ceil(node_ids.shape[0] / cell_size)
#     indices_ss = []
#
#     RNA_matrix1 = RNA_matrix
#     dic = {}
#     for i in tqdm(range(n_batch)):
#         gene_indices_all = []
#         peak_indices_all = []
#         if i < n_batch:
#             node_range = node_ids[i * cell_size:(i + 1) * cell_size]
#         else:
#             node_range = node_ids[i * cell_size:]
#
#         for index, node in enumerate(node_range):
#             rna_ = RNA_matrix1[:, node].todense()
#             rna_[rna_ < 5] = 0
#
#             # Unified gene_indices computation
#             gene_indices = subgraph(RNA_matrix.transpose(), node, neighbor, np.squeeze(np.array(np.log(rna_ + 1))))
#
#             peak_indices = subgraph(ATAC_matrix.transpose(), node, neighbor,
#                                     np.squeeze(np.array(np.log(ATAC_matrix[:, node].todense() + 1))))
#             dic[node] = {'g': gene_indices, 'p': peak_indices}
#             gene_indices_all = gene_indices_all + gene_indices
#             peak_indices_all = peak_indices_all + peak_indices
#
#         node_indices_all = node_range
#         gene_indices_all = list(set(gene_indices_all))
#         peak_indices_all = list(set(peak_indices_all))
#
#         h = {
#             'gene_index': gene_indices_all,
#             'peak_index': peak_indices_all,
#             'cell_index': node_indices_all
#         }
#
#         indices_ss.append(h)
#
#     return indices_ss, node_ids, dic


# def softmax(x):
#     return (np.exp(x) / np.exp(x).sum())

def softmax_stable(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def softmax_simple(x):
    return np.exp(x) / np.exp(x).sum()


class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.0):
        """Constructor for LabelSmoothing module.
        :param smoothing: Label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def initial_clustering(RNA_matrix, custom_n_neighbors=None, n_pcs=40, custom_resolution=None, use_rep=None):
    print(
        '\tWhen the number of cells is less than or equal to 500, it is recommended to set the resolution value to 0.2.')
    print('\tWhen the number of cells is within the range of 500 to 5000, the resolution value should be set to 0.5.')
    print('\tWhen the number of cells is greater than 5000, the resolution value should be set to 0.8.')

    def segment_function(x):
        if x <= 500:
            return 0.2, 5
        elif x <= 5000:
            return 0.5, 10
        else:
            return 0.8, 15

    adata = ad.AnnData(RNA_matrix.transpose(), dtype='int32')

    # If the user did not provide a custom resolution or n_neighbors value, use the values calculated by segment_function
    if custom_resolution is None or custom_n_neighbors is None:
        resolution, n_neighbors = segment_function(adata.shape[0])
    else:
        resolution = custom_resolution
        n_neighbors = custom_n_neighbors

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Use the user-provided embedding if available, otherwise use n_pcs
    if use_rep is not None:
        adata.obsm['use_rep'] = use_rep
        sc.pp.neighbors(adata, use_rep='use_rep', n_neighbors=n_neighbors)
    else:
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)

    sc.tl.leiden(adata, resolution)
    return adata.obs['leiden']


def purity_score(y_true, y_pred):
    """Purity score

    Args:
        y_true (np.ndarray): n*1 matrix, true labels
        y_pred (np.ndarray): n*1 matrix, predicted clusters

    Returns:
        float: Purity score
    """
    # Create a matrix to store the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)

    # Sort the labels
    # Some labels might be missing, e.g., a set {0,2} where 1 is missing
    # First, find the unique labels and then map them to an ordered set
    # E.g., {0,2} should be mapped to {0,1}
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    y_true = np.array(y_true, dtype='int64')

    # Update the unique labels
    labels = np.unique(y_true)

    # Set the number of bins to n_classes + 2 so that we can compute the actual
    # class occurrences between two consecutive bins
    # The larger bin is excluded: [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels) + 1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        # Find the most frequent label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    y_true = np.array(y_true, dtype='int8')
    y_voted_labels = np.array(y_voted_labels, dtype='int8')
    return accuracy_score(y_true, y_voted_labels), y_true


def Entropy(pred_label, true_label):
    e = 0
    for k in set(pred_label):
        en = 0
        pred_k = Counter(pred_label)[k]
        index_pred_k = pred_label == k
        for j in set(true_label):
            true_j = Counter(true_label)[j]
            intersection_kj = (true_label[index_pred_k] == j).sum()
            p = np.array(intersection_kj) / np.array(pred_k)
            if p != 0:
                en += np.log(p) * p
        e = e + en * pred_k / true_label.shape[0]
    return abs(e)
