import anndata as ad
import torch
import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm
import math
import scanpy as sc
from scipy.sparse import issparse, csr_matrix, diags  # ✨ 增加 diags
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed, cpu_count
import pickle
import os

# ====== 全局缓存：预计算量（细胞文库深度、size factor、基因强度、idf 等） ======
_GRAPH_CACHE = None

# 默认超参（如需调整，可直接改这里，外部调用保持不变）
THETA = 100.0                    # 解析 Pearson 残差 NB 参数
MIN_LOG_EXPR = 1.0               # log1p(TP10K) 的最小表达阈
WEIGHTS = (0.5, 0.3, 0.2)        # (残差, TF-IDF, 强度) 的权重

def _build_graph_cache(RNA_matrix: csr_matrix):
    """
    预计算用于“高表达且特异”打分的全局量并缓存：
    - lib: 每个细胞的文库深度
    - s:   细胞 size factor（以中位数归一）
    - g:   基因全局强度（用于残差的期望）
    - idf: TF-IDF 的 idf
    """
    global _GRAPH_CACHE
    assert isinstance(RNA_matrix, csr_matrix)
    G, N = RNA_matrix.shape

    lib = RNA_matrix.sum(axis=0).A1 + 1e-8                 # 每个细胞的总 UMI
    s = lib / np.median(lib)                                # 细胞 size factor
    df = (RNA_matrix > 0).sum(axis=1).A1                    # 基因出现于多少细胞
    idf = np.log1p(N / (1.0 + df))                          # idf_j
    g = RNA_matrix.sum(axis=1).A1 / s.sum()                 # 基因强度（全局）

    _GRAPH_CACHE = {
        "lib": lib,
        "s": s,
        "idf": idf,
        "g": g,
        "theta": THETA,
        "min_log_expr": MIN_LOG_EXPR,
        "weights": WEIGHTS,
        "n_cells": N,
    }

# ==========（保留以兼容，但本实现不会再用到随机扩张）==========
def subgraph(graph, seed, n_neighbors, node_sele_prob):
    """
    保留原函数以兼容旧代码；当前流程不再依赖该随机采样方式。
    """
    picked_nodes = {seed}
    last_layer_nodes = {seed}
    for layer, n_neighbors_current in enumerate(n_neighbors):
        neighbors = graph[list(last_layer_nodes), :].nonzero()[1]
        neighbors = np.unique(neighbors)
        if len(neighbors) == 0:
            break
        neighbors_prob = node_sele_prob[neighbors]
        neighbors_prob = softmax_stable(neighbors_prob)
        to_pick = n_neighbors_current
        n_neighbors_real = min(to_pick, len(neighbors))
        selected_neighbors = np.random.choice(
            neighbors, size=n_neighbors_real, replace=False, p=neighbors_prob
        )
        last_layer_nodes = set(selected_neighbors)
        picked_nodes.update(last_layer_nodes)
    indices = sorted(picked_nodes - {seed})
    return indices

# ========== 新的整体批处理入口：外部调用保持不变 ==========
def batch_select_whole(
    RNA_matrix, neighbor=[20], cell_size=30, save_path="processed_data_subset"
):
    """
    根据“高表达且特异”的确定性 Top-K 规则，构建异质图所需索引。
    返回：
      - indices_ss: 按批的 { "gene_index": 本批所有细胞TopK基因去重集合, "cell_index": 本批细胞ID列表 }
      - Node_Ids:   打乱后的细胞ID顺序
      - dic:        每个细胞的 TopK 基因映射，dic[cell_id] = {"g": [gene_idx, ...]}
    """
    indices_ss_file = os.path.join(save_path, "indices_ss.pkl")
    Node_Ids_file   = os.path.join(save_path, "Node_Ids.pkl")
    dic_file        = os.path.join(save_path, "dic.pkl")

    if (
        os.path.exists(indices_ss_file)
        and os.path.exists(Node_Ids_file)
        and os.path.exists(dic_file)
    ):
        print("正在从磁盘加载处理后的数据...")
        with open(indices_ss_file, "rb") as f:
            indices_ss = pickle.load(f)
        with open(Node_Ids_file, "rb") as f:
            Node_Ids = pickle.load(f)
        with open(dic_file, "rb") as f:
            dic = pickle.load(f)
    else:
        print("正在将数据划分为批次并进行并行处理。请稍候...")

        # ---- 形状与稀疏格式保障 ----
        if issparse(RNA_matrix):
            RNA_matrix = RNA_matrix.tocsr()
        else:
            RNA_matrix = csr_matrix(RNA_matrix)

        # ---- 预计算缓存（一次性）----
        _build_graph_cache(RNA_matrix)

        # 打乱细胞 ID
        Node_Ids = np.random.choice(
            RNA_matrix.shape[1], size=RNA_matrix.shape[1], replace=False
        )
        n_batch = math.ceil(Node_Ids.shape[0] / cell_size)
        indices_ss = []
        dic = {}

        for i in tqdm(range(n_batch), desc="处理批次"):
            gene_indices_all = []

            # 当前批次的细胞范围
            start_idx = i * cell_size
            end_idx   = min((i + 1) * cell_size, Node_Ids.shape[0])
            batch_node_ids = Node_Ids[start_idx:end_idx]

            # 并行处理每个细胞（如需：n_jobs=max(1, cpu_count()//2)）
            results = Parallel(n_jobs=1)(
                delayed(process_node)(node, RNA_matrix, neighbor)
                for node in batch_node_ids
            )

            for node, gene_indices in results:
                dic[node] = {"g": gene_indices}
                gene_indices_all.extend(gene_indices)

            # 批次内基因去重
            gene_indices_all = sorted(set(gene_indices_all))

            indices_ss.append({
                "gene_index": gene_indices_all,
                "cell_index": list(batch_node_ids),
            })

        # 持久化
        os.makedirs(save_path, exist_ok=True)
        with open(indices_ss_file, "wb") as f:
            pickle.dump(indices_ss, f)
        with open(Node_Ids_file, "wb") as f:
            pickle.dump(Node_Ids, f)
        with open(dic_file, "wb") as f:
            pickle.dump(dic, f)

    return indices_ss, Node_Ids, dic

# ========== 单细胞 Top-K 选基因：高表达 + 特异 ==========
def process_node(node, RNA_matrix, neighbor):
    """
    新逻辑：对细胞 node 的非零基因，计算组合分数（残差 + TF-IDF + 强度），确定性 Top-K。
    K 使用 neighbor[0]（兼容原接口）。
    """
    cache = _GRAPH_CACHE
    assert cache is not None, "Graph cache is not built. Call batch_select_whole first."

    # 兼容传参：neighbor 可以是 int 或 list/tuple
    if isinstance(neighbor, (list, tuple, np.ndarray)):
        K = int(neighbor[0]) if len(neighbor) > 0 else 20
    else:
        K = int(neighbor)

    # 该细胞非零基因索引与原始计数
    col = RNA_matrix[:, node]
    idx = col.indices
    if idx.size == 0:
        return node, []

    x_cnt = col.data.astype(np.float64)
    lib_i = cache["lib"][node]

    # --- 强度分：log1p(TP10K) ---
    x_log = np.log1p((x_cnt / lib_i) * 1e4)

    # --- TF-IDF（只取该细胞的非零）---
    tfidf = (x_cnt / lib_i) * cache["idf"][idx]

    # --- 解析 Pearson 残差 ---
    mu = cache["g"][idx] * cache["s"][node] + 1e-8
    var = mu + (mu**2) / cache["theta"]
    r = (x_cnt - mu) / np.sqrt(var)
    r = np.clip(r, 0.0, np.sqrt(cache["n_cells"]))   # 只取正向并裁剪

    # --- 过滤极低表达 ---
    keep = x_log >= cache["min_log_expr"]
    if not np.any(keep):  # 若全部过低，则不做阈值过滤
        keep = np.ones_like(x_log, dtype=bool)

    idx_k = idx[keep]
    r_k   = r[keep]
    tf_k  = tfidf[keep]
    x_k   = x_log[keep]

    # --- 细胞内 0-1 归一化并线性加权 ---
    def norm01(v):
        vmin, vmax = float(v.min()), float(v.max())
        return (v - vmin) / (vmax - vmin + 1e-8)

    wr, wt, wx = cache["weights"]
    score = wr * norm01(r_k) + wt * norm01(tf_k) + wx * norm01(x_k)

    # --- 确定性 Top-K ---
    order = np.argsort(score)[::-1][:min(K, score.size)]
    genes = idx_k[order].astype(np.int64).tolist()
    return node, genes

# ====== 下面为原有的通用函数（未改动） ======
def softmax_stable(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def softmax_simple(x):
    return np.exp(x) / np.exp(x).sum()

class LabelSmoothing(torch.nn.Module):
    """NLL loss with label smoothing."""
    def __init__(self, smoothing=0.0, num_classes=10):
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

def initial_clustering(
    RNA_matrix, custom_n_neighbors=None, n_pcs=40, custom_resolution=None, use_rep=None
):
    print("\tWhen the number of cells is less than or equal to 500, it is recommended to set the resolution value to 0.2.")
    print("\tWhen the number of cells is within the range of 500 to 5000, the resolution value should be set to 0.5.")
    print("\tWhen the number of cells is greater than 5000, the resolution value should be set to 0.8.")

    def segment_function(x):
        if x <= 500:
            return 0.2, 5
        elif x <= 5000:
            return 0.5, 10
        else:
            return 0.8, 15

    adata = ad.AnnData(RNA_matrix.transpose())
    if custom_resolution is None or custom_n_neighbors is None:
        resolution, n_neighbors = segment_function(adata.shape[0])
    else:
        resolution = custom_resolution
        n_neighbors = custom_n_neighbors

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    sc.tl.pca(adata, svd_solver="arpack")

    if use_rep is not None:
        adata.obsm["use_rep"] = use_rep
        sc.pp.neighbors(adata, use_rep="use_rep", n_neighbors=n_neighbors)
    else:
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)

    sc.tl.leiden(adata, resolution)
    return adata.obs["leiden"]

def purity_score(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    y_true = np.array(y_true, dtype="int64")
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels) + 1]), axis=0)
    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner
    y_true = np.array(y_true, dtype="int8")
    y_voted_labels = np.array(y_voted_labels, dtype="int8")
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

def compute_lisi(data, labels, k=30):
    if isinstance(data, pd.DataFrame):
        data = data.values
    labels = np.array(labels)
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(data)
    distances, indices = nbrs.kneighbors(data)
    lisi_scores = []
    for i in range(len(data)):
        neighbor_labels = labels[indices[i][1:]]
        label_counts = Counter(neighbor_labels)
        total_count = sum(label_counts.values())
        simpson_index = sum((count / total_count) ** 2 for count in label_counts.values())
        lisi = 1.0 / simpson_index
        lisi_scores.append(lisi)
    return lisi_scores

def compute_ilisi(data, labels, k=30):
    lisi_scores = compute_lisi(data, labels, k)
    ilisi_score = np.mean(lisi_scores)
    return ilisi_score

def compute_nmi(true_labels, predicted_labels):
    return normalized_mutual_info_score(true_labels, predicted_labels)

def compute_ari(true_labels, predicted_labels):
    return adjusted_rand_score(true_labels, predicted_labels)

def compute_nn_entropy(data, labels, k=30):
    if isinstance(data, pd.DataFrame):
        data = data.values
    labels = np.array(labels)
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(data)
    distances, indices = nbrs.kneighbors(data)
    nn_entropy_scores = []
    for i in range(len(data)):
        neighbor_labels = labels[indices[i][1:]]
        label_counts = Counter(neighbor_labels)
        total_count = sum(label_counts.values())
        entropy = -sum(
            (count / total_count) * math.log(count / total_count)
            for count in label_counts.values()
            if count > 0
        )
        nn_entropy_scores.append(entropy)
    mean_nn_entropy = np.mean(nn_entropy_scores)
    return mean_nn_entropy

def compute_silhouette_score(data, labels):
    if isinstance(data, pd.DataFrame):
        data = data.values
    silhouette = silhouette_score(data, labels)
    return silhouette

def compute_davies_bouldin_score(data, labels):
    if isinstance(data, pd.DataFrame):
        data = data.values
    db_index = davies_bouldin_score(data, labels)
    return db_index

def compute_calinski_harabasz_score(data, labels):
    if isinstance(data, pd.DataFrame):
        data = data.values
    ch_index = calinski_harabasz_score(data, labels)
    return ch_index
