import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import *
from .utils import *
from .egrn import *


class GNN_from_raw(nn.Module):
    def __init__(self, in_dim, n_hid, num_types, num_relations, n_heads, n_layers, dropout=0.2, conv_name='hgt',
                 prev_norm=True, last_norm=True):
        super(GNN_from_raw, self).__init__()
        self.gcs = nn.ModuleList()
        self.num_types = num_types
        self.in_dim = in_dim
        self.n_hid = n_hid
        self.adapt_ws = nn.ModuleList()
        self.drop = nn.Dropout(dropout)
        self.embedding1 = nn.ModuleList()

        # Initialize MLP weight matrices
        for ti in range(num_types):
            self.embedding1.append(nn.Linear(in_dim[ti], 256))

        for t in range(num_types):
            self.adapt_ws.append(nn.Linear(256, n_hid))

        # Initialize graph convolution layers
        for l in range(n_layers - 1):
            self.gcs.append(
                GeneralConv(conv_name, n_hid, n_hid, num_types, num_relations, n_heads, dropout, use_norm=prev_norm))
        self.gcs.append(
            GeneralConv(conv_name, n_hid, n_hid, num_types, num_relations, n_heads, dropout, use_norm=last_norm))

    def encode(self, x, t_id):
        h1 = F.relu(self.embedding1[t_id](x))
        return h1

    def forward(self, node_feature, node_type, edge_index, edge_type):
        node_embedding = []
        for t_id in range(self.num_types):
            node_embedding += list(self.encode(node_feature[t_id], t_id))

        node_embedding = torch.stack(node_embedding)
        # Initialize result matrix
        res = torch.zeros(node_embedding.size(0), self.n_hid).to(node_feature[0].device)

        # Process each node type
        for t_id in range(self.num_types):
            idx = (node_type == int(t_id))
            if idx.sum() == 0:
                continue
            # Update result matrix
            res[idx] = torch.tanh(self.adapt_ws[t_id](node_embedding[idx]))

        # Apply dropout to the result matrix
        meta_xs = self.drop(res)
        del res

        # Iterate through graph convolution layers and update result matrix
        for gc in self.gcs:
            meta_xs = gc(meta_xs, node_type, edge_index, edge_type)

        return meta_xs


class Net(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return x


# class NodeDimensionReduction(nn.Module):
#     def __init__(self, RNA_matrix, ATAC_matrix, indices, ini_p1, n_hid, n_heads,
#                  n_layers, labsm, lr, wd, device, num_types=3, num_relations=2, epochs=1):
#         super(NodeDimensionReduction, self).__init__()
#         self.RNA_matrix = RNA_matrix
#         self.ATAC_matrix = ATAC_matrix
#         self.indices = indices
#         self.ini_p1 = ini_p1
#         self.in_dim = [RNA_matrix.shape[0], RNA_matrix.shape[1], ATAC_matrix.shape[1]]
#         self.n_hid = n_hid
#         self.num_types = num_types
#         self.num_relations = num_relations
#         self.n_heads = n_heads
#         self.n_layers = n_layers
#         self.labsm = labsm
#         self.lr = lr
#         self.wd = wd
#         self.device = device
#         self.epochs = epochs
#
#         self.LabSm = LabelSmoothing(self.labsm)
#
#         self.gnn = GNN_from_raw(in_dim=self.in_dim,
#                                 n_hid=self.n_hid,
#                                 num_types=self.num_types,
#                                 num_relations=self.num_relations,
#                                 n_heads=self.n_heads,
#                                 n_layers=self.n_layers,
#                                 dropout=0.3).to(self.device)
#
#         self.optimizer = torch.optim.AdamW(self.gnn.parameters(), lr=self.lr, weight_decay=self.wd)
#         self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=5,
#                                                                     verbose=True)
#
#     def train_model(self, n_batch):
#         print('The training process for the NodeDimensionReduction model has started. Please wait.')
#         for epoch in tqdm(range(self.epochs)):
#             for batch_id in np.arange(n_batch):
#                 gene_index = self.indices[batch_id]['gene_index']
#                 cell_index = self.indices[batch_id]['cell_index']
#                 peak_index = self.indices[batch_id]['peak_index']
#                 gene_feature = self.RNA_matrix[list(gene_index),]
#                 cell_feature = self.RNA_matrix[:, list(cell_index)].T
#                 peak_feature = self.ATAC_matrix[list(peak_index),]
#                 gene_feature = torch.tensor(np.array(gene_feature.todense()), dtype=torch.float32).to(self.device)
#                 cell_feature = torch.tensor(np.array(cell_feature.todense()), dtype=torch.float32).to(self.device)
#                 peak_feature = torch.tensor(np.array(peak_feature.todense()), dtype=torch.float32).to(self.device)
#
#                 node_feature = [cell_feature, gene_feature, peak_feature]
#                 gene_cell_sub = self.RNA_matrix[list(gene_index),][:, list(cell_index)]
#                 peak_cell_sub = self.ATAC_matrix[list(peak_index),][:, list(cell_index)]
#                 # gene_cell_edge_index = torch.LongTensor([np.nonzero(gene_cell_sub)[0]+gene_cell_sub.shape[1],np.nonzero(gene_cell_sub)[1]]).to(device)
#                 # peak_cell_edge_index = torch.LongTensor([np.nonzero(peak_cell_sub)[0]+gene_cell_sub.shape[0]+gene_cell_sub.shape[1],np.nonzero(peak_cell_sub)[1]]).to(device)
#                 gene_cell_edge_index1 = list(np.nonzero(gene_cell_sub)[0] + gene_cell_sub.shape[1]) + list(
#                     np.nonzero(gene_cell_sub)[1])
#                 gene_cell_edge_index2 = list(np.nonzero(gene_cell_sub)[1]) + list(
#                     np.nonzero(gene_cell_sub)[0] + gene_cell_sub.shape[1])
#                 gene_cell_edge_index = torch.LongTensor([gene_cell_edge_index1, gene_cell_edge_index2]).to(self.device)
#                 peak_cell_edge_index1 = list(
#                     np.nonzero(peak_cell_sub)[0] + gene_cell_sub.shape[0] + gene_cell_sub.shape[1]) + list(
#                     np.nonzero(peak_cell_sub)[1])
#                 peak_cell_edge_index2 = list(np.nonzero(peak_cell_sub)[1]) + list(
#                     np.nonzero(peak_cell_sub)[0] + gene_cell_sub.shape[0] + gene_cell_sub.shape[1])
#                 peak_cell_edge_index = torch.LongTensor([peak_cell_edge_index1, peak_cell_edge_index2]).to(self.device)
#
#                 edge_index = torch.cat((gene_cell_edge_index, peak_cell_edge_index), dim=1)
#                 node_type = torch.LongTensor(np.array(
#                     list(np.zeros(len(cell_index))) + list(np.ones(len(gene_index))) + list(
#                         np.ones(len(peak_index)) * 2))).to(self.device)
#                 # edge_type = torch.LongTensor(np.array(list(np.zeros(gene_cell_edge_index.shape[1]))+list(np.ones(peak_cell_edge_index.shape[1]) ))).to(device)
#                 edge_type = torch.LongTensor(np.array(list(np.zeros(np.nonzero(gene_cell_sub)[0].shape[0])) + list(
#                     np.ones(np.nonzero(gene_cell_sub)[1].shape[0])) + list(
#                     2 * np.ones(np.nonzero(peak_cell_sub)[0].shape[0])) + list(
#                     3 * np.ones(np.nonzero(peak_cell_sub)[1].shape[0])))).to(self.device)
#                 l = torch.LongTensor(np.array(self.ini_p1)[[cell_index]]).to(self.device)
#                 node_rep = self.gnn.forward(node_feature, node_type,
#                                             edge_index,
#                                             edge_type).to(self.device)
#                 cell_emb = node_rep[node_type == 0]
#                 gene_emb = node_rep[node_type == 1]
#                 peak_emb = node_rep[node_type == 2]
#
#                 decoder1 = torch.mm(gene_emb, cell_emb.t())
#                 decoder2 = torch.mm(peak_emb, cell_emb.t())
#                 gene_cell_sub = torch.tensor(np.array(gene_cell_sub.todense()), dtype=torch.float32).to(self.device)
#                 peak_cell_sub = torch.tensor(np.array(peak_cell_sub.todense()), dtype=torch.float32).to(self.device)
#
#                 logp_x1 = F.log_softmax(decoder1, dim=-1)
#                 p_y1 = F.softmax(gene_cell_sub, dim=-1)
#
#                 loss_kl1 = F.kl_div(logp_x1, p_y1, reduction='mean')
#
#                 logp_x2 = F.log_softmax(decoder2, dim=-1)
#                 p_y2 = F.softmax(peak_cell_sub, dim=-1)
#
#                 loss_kl2 = F.kl_div(logp_x2, p_y2, reduction='mean')
#                 loss_kl = loss_kl1 + loss_kl2
#                 loss_cluster = self.LabSm(cell_emb, l)
#                 lll = 0
#                 g = [int(i) for i in l]
#                 for i in set([int(k) for k in l]):
#                     h = cell_emb[[True if i == j else False for j in g]]
#                     ll = F.cosine_similarity(h[list(range(h.shape[0])) * h.shape[0],],
#                                              h[[v for v in range(h.shape[0]) for i in range(h.shape[0])]]).mean()
#                     lll = ll + lll
#                 loss = loss_cluster - lll
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()
#         print('The training for the NodeDimensionReduction model has been completed.')
#         return self.gnn, cell_emb, gene_emb, peak_emb, h

class NodeDimensionReduction(nn.Module):
    def __init__(self, RNA_matrix, indices, ini_p1, n_hid, n_heads,
                 n_layers, labsm, lr, wd, device, num_types=2, num_relations=2, epochs=1):
        super(NodeDimensionReduction, self).__init__()
        self.RNA_matrix = RNA_matrix
        self.indices = indices
        self.ini_p1 = ini_p1
        self.in_dim = [RNA_matrix.shape[0], RNA_matrix.shape[1]]  # 仅包含基因和细胞的维度
        self.n_hid = n_hid
        self.num_types = num_types  # 2 种类型：细胞和基因
        self.num_relations = num_relations  # 2 种关系：基因到细胞和细胞到基因
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.labsm = labsm
        self.lr = lr
        self.wd = wd
        self.device = device
        self.epochs = epochs

        # 标签平滑
        self.LabSm = LabelSmoothing(self.labsm)

        # GNN 模型
        self.gnn = GNN_from_raw(in_dim=self.in_dim,
                                n_hid=self.n_hid,
                                num_types=self.num_types,
                                num_relations=self.num_relations,
                                n_heads=self.n_heads,
                                n_layers=self.n_layers,
                                dropout=0.3).to(self.device)

        # 优化器和学习率调度器
        self.optimizer = torch.optim.AdamW(self.gnn.parameters(), lr=self.lr, weight_decay=self.wd)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=5,
                                                                    verbose=True)

    def train_model(self, n_batch):
        print('The training process for the NodeDimensionReduction model has started. Please wait.')
        h_final = None  # 用于存储最后一个 h
        for epoch in tqdm(range(self.epochs)):
            for batch_id in np.arange(n_batch):
                # 获取当前批次的基因和细胞索引
                gene_index = self.indices[batch_id]['gene_index']
                cell_index = self.indices[batch_id]['cell_index']

                # 提取基因和细胞的特征
                gene_feature = self.RNA_matrix[list(gene_index), :]
                cell_feature = self.RNA_matrix[:, list(cell_index)].T

                # 转换为张量并移动到设备上
                gene_feature = torch.tensor(gene_feature.todense(), dtype=torch.float32).to(self.device)
                cell_feature = torch.tensor(cell_feature.todense(), dtype=torch.float32).to(self.device)

                # 构建节点特征列表（细胞和基因）
                node_feature = [cell_feature, gene_feature]

                # 构建基因-细胞子图的邻接矩阵
                gene_cell_sub = self.RNA_matrix[list(gene_index), :][:, list(cell_index)]

                # 构建基因到细胞的边索引
                gene_to_cell_src = list(np.nonzero(gene_cell_sub)[0] + len(cell_index))  # 基因索引从 len(cell_index) 开始
                gene_to_cell_dst = list(np.nonzero(gene_cell_sub)[1])

                # 构建细胞到基因的边索引
                cell_to_gene_src = list(np.nonzero(gene_cell_sub)[1])
                cell_to_gene_dst = list(np.nonzero(gene_cell_sub)[0] + len(cell_index))  # 基因索引

                # 合并边索引
                edge_index = torch.LongTensor([
                    gene_to_cell_src + cell_to_gene_src,  # 源节点
                    gene_to_cell_dst + cell_to_gene_dst  # 目标节点
                ]).to(self.device)

                # 定义节点类型：0 代表细胞，1 代表基因
                node_type = torch.LongTensor(
                    np.array(
                        list(np.zeros(len(cell_index))) + list(np.ones(len(gene_index)))
                    )
                ).to(self.device)

                # 定义边类型：0 代表基因到细胞，1 代表细胞到基因
                edge_type_gene_to_cell = np.zeros(len(gene_to_cell_src), dtype=int)
                edge_type_cell_to_gene = np.ones(len(cell_to_gene_src), dtype=int)
                edge_type = torch.LongTensor(
                    np.concatenate([edge_type_gene_to_cell, edge_type_cell_to_gene])
                ).to(self.device)

                # 获取标签
                l = torch.LongTensor(np.array(self.ini_p1)[cell_index]).to(self.device)

                # 前向传播
                node_rep = self.gnn.forward(node_feature, node_type, edge_index, edge_type).to(self.device)
                cell_emb = node_rep[node_type == 0]
                gene_emb = node_rep[node_type == 1]

                # 解码器：基因与细胞的关系
                decoder_gene_to_cell = torch.mm(gene_emb, cell_emb.t())
                decoder_cell_to_gene = torch.mm(cell_emb, gene_emb.t())

                # 构建目标矩阵
                gene_cell_sub_tensor = torch.tensor(gene_cell_sub.todense(), dtype=torch.float32).to(self.device)

                # 计算基因到细胞的 KL 散度损失
                logp_x1 = F.log_softmax(decoder_gene_to_cell, dim=-1)
                p_y1 = F.softmax(gene_cell_sub_tensor, dim=-1)
                loss_kl1 = F.kl_div(logp_x1, p_y1, reduction='mean')

                # 计算细胞到基因的 KL 散度损失
                logp_x2 = F.log_softmax(decoder_cell_to_gene, dim=-1)
                p_y2 = F.softmax(gene_cell_sub_tensor.t(), dim=-1)
                loss_kl2 = F.kl_div(logp_x2, p_y2, reduction='mean')

                # 总的 KL 散度损失
                loss_kl = loss_kl1 + loss_kl2

                # 聚类损失
                loss_cluster = self.LabSm(cell_emb, l)

                # 余弦相似度损失
                lll = 0
                g = l.tolist()
                unique_labels = set(g)
                for label in unique_labels:
                    mask = np.array(g) == label
                    h = cell_emb[mask]
                    if h.size(0) > 1:
                        # 计算细胞间的余弦相似度
                        similarity = F.cosine_similarity(h.unsqueeze(1), h.unsqueeze(0), dim=-1)
                        lll += similarity.mean()
                        h_final = h  # 更新 h_final 为当前标签的 h

                # 总损失
                loss = loss_cluster - lll

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        print('The training for the NodeDimensionReduction model has been completed.')
        return self.gnn, cell_emb, gene_emb, h_final  # 返回 h_final


# class MarsGT(nn.Module):
#     def __init__(self, gnn, h, labsm, n_hid, n_batch, device, lr, wd, num_epochs=1):
#         super(MarsGT, self).__init__()
#         self.lr = lr
#         self.wd = wd
#         self.gnn = gnn
#         self.h = h
#         self.n_hid = n_hid
#         self.n_batch = n_batch
#         self.device = device
#         self.num_epochs = num_epochs
#         self.net = Net(2 * self.n_hid, self.n_hid).to(self.device)
#         self.gnn_optimizer = torch.optim.AdamW(self.gnn.parameters(), lr=self.lr, weight_decay=self.wd)
#         self.net_optimizer = torch.optim.AdamW(self.net.parameters(), lr=1e-2)
#         self.labsm = labsm
#         self.LabSm = LabelSmoothing(self.labsm)
#
#     def forward(self, indices, RNA_matrix, ATAC_matrix, Gene_Peak, ini_p1):
#         cluster_l = list()
#         cluster_kl_l = list()
#         sim_l = list()
#         nmi_l = list()
#         ini_p1 = np.array(ini_p1)
#
#         for epoch in range(self.num_epochs):
#             for batch_id in tqdm(np.arange(self.n_batch)):
#                 gene_index = indices[batch_id]['gene_index']
#                 cell_index = indices[batch_id]['cell_index']
#                 peak_index = indices[batch_id]['peak_index']
#                 gene_feature = RNA_matrix[list(gene_index),]
#                 cell_feature = RNA_matrix[:, list(cell_index)].T
#                 peak_feature = ATAC_matrix[list(peak_index),]
#                 gene_feature = torch.tensor(np.array(gene_feature.todense()), dtype=torch.float32).to(self.device)
#                 cell_feature = torch.tensor(np.array(cell_feature.todense()), dtype=torch.float32).to(self.device)
#                 peak_feature = torch.tensor(np.array(peak_feature.todense()), dtype=torch.float32).to(self.device)
#                 node_feature = [cell_feature, gene_feature, peak_feature]
#                 gene_cell_sub = RNA_matrix[list(gene_index),][:, list(cell_index)]
#                 peak_cell_sub = ATAC_matrix[list(peak_index),][:, list(cell_index)]
#                 gene_cell_edge_index1 = list(np.nonzero(gene_cell_sub)[0] + gene_cell_sub.shape[1]) + list(
#                     np.nonzero(gene_cell_sub)[1])
#                 gene_cell_edge_index2 = list(np.nonzero(gene_cell_sub)[1]) + list(
#                     np.nonzero(gene_cell_sub)[0] + gene_cell_sub.shape[1])
#                 gene_cell_edge_index = torch.LongTensor([gene_cell_edge_index1, gene_cell_edge_index2]).to(self.device)
#                 peak_cell_edge_index1 = list(
#                     np.nonzero(peak_cell_sub)[0] + gene_cell_sub.shape[0] + gene_cell_sub.shape[1]) + list(
#                     np.nonzero(peak_cell_sub)[1])
#                 peak_cell_edge_index2 = list(np.nonzero(peak_cell_sub)[1]) + list(
#                     np.nonzero(peak_cell_sub)[0] + gene_cell_sub.shape[0] + gene_cell_sub.shape[1])
#                 peak_cell_edge_index = torch.LongTensor([peak_cell_edge_index1, peak_cell_edge_index2]).to(self.device)
#                 # gene_cell_edge_index = torch.LongTensor([np.nonzero(gene_cell_sub)[0]+gene_cell_sub.shape[1],np.nonzero(gene_cell_sub)[1]]).to(device)
#                 # peak_cell_edge_index = torch.LongTensor([np.nonzero(peak_cell_sub)[0]+gene_cell_sub.shape[0]+gene_cell_sub.shape[1],np.nonzero(peak_cell_sub)[1]]).to(device)
#                 edge_index = torch.cat((gene_cell_edge_index, peak_cell_edge_index), dim=1)
#                 node_type = torch.LongTensor(np.array(
#                     list(np.zeros(len(cell_index))) + list(np.ones(len(gene_index))) + list(
#                         np.ones(len(peak_index)) * 2))).to(self.device)
#                 edge_type = torch.LongTensor(np.array(list(np.zeros(np.nonzero(gene_cell_sub)[0].shape[0])) + list(
#                     np.ones(np.nonzero(gene_cell_sub)[1].shape[0])) + list(
#                     2 * np.ones(np.nonzero(peak_cell_sub)[0].shape[0])) + list(
#                     3 * np.ones(np.nonzero(peak_cell_sub)[1].shape[0])))).to(self.device)
#
#                 # edge_type = torch.LongTensor(np.array(list(np.zeros(gene_cell_edge_index.shape[1]))+list(np.ones(peak_cell_edge_index.shape[1]) ))).to(device)
#                 l = torch.LongTensor(np.array(ini_p1)[[cell_index]]).to(self.device)
#                 # l2 = torch.LongTensor(label[[cell_index]])
#
#                 # l = torch.LongTensor(ini_p1)[[cell_index]].to(device)
#
#                 node_rep = self.gnn.forward(node_feature, node_type,
#                                             edge_index,
#                                             edge_type).to(self.device)
#                 cell_emb = node_rep[node_type == 0]
#                 gene_emb = node_rep[node_type == 1]
#                 peak_emb = node_rep[node_type == 2]
#
#                 decoder1 = torch.mm(gene_emb, cell_emb.t())
#                 decoder2 = torch.mm(peak_emb, cell_emb.t())
#                 gene_cell_sub = torch.tensor(np.array(gene_cell_sub.todense()), dtype=torch.float32).to(self.device)
#                 peak_cell_sub = torch.tensor(np.array(peak_cell_sub.todense()), dtype=torch.float32).to(self.device)
#
#                 logp_x1 = F.log_softmax(decoder1, dim=-1)
#                 p_y1 = F.softmax(gene_cell_sub, dim=-1)
#
#                 loss_kl1 = F.kl_div(logp_x1, p_y1, reduction='mean')
#
#                 logp_x2 = F.log_softmax(decoder2, dim=-1)
#                 p_y2 = F.softmax(peak_cell_sub, dim=-1)
#
#                 loss_kl2 = F.kl_div(logp_x2, p_y2, reduction='mean')
#
#                 loss_kl = loss_kl1 + loss_kl2
#
#                 lll2 = 0
#                 g = [int(i) for i in l]
#                 for i in set([int(k) for k in l]):
#                     ll2 = F.cosine_similarity(self.h[list(range(self.h.shape[0])) * self.h.shape[0],], self.h[
#                         [v for v in range(self.h.shape[0]) for i in range(self.h.shape[0])]]).mean()
#                     lll2 = ll2 + lll2
#
#                 loss_cluster = self.LabSm(cell_emb, l)
#
#                 m = range(peak_emb.shape[0])
#                 gene_emb_enh = gene_emb[list(range(gene_emb.shape[0])) * peak_emb.shape[0]]
#                 peak_emb_enh = peak_emb[[v for v in m for i in range(gene_emb.shape[0])]]
#
#                 net_input = torch.cat((gene_emb_enh, peak_emb_enh), 1)
#                 gene_peak_cluster_pre = self.net(net_input)
#                 m = range(peak_feature.shape[0])
#                 # gene_feature_ori = gene_feature[list(range (gene_feature.shape[0]))*peak_emb.shape[0]]
#                 # peak_feature_ori = peak_feature[[v for v in m for i in range(gene_feature.shape[0])]]
#                 peak_feature_ori = peak_Sparse(peak_feature[:, cell_index], gene_feature[:, cell_index], self.device)
#                 gene_feature_ori = gene_Sparse(peak_feature[:, cell_index], gene_feature[:, cell_index], self.device)
#                 # g1p1 g2p1 g3p1
#                 gene_peak_ori = gene_feature_ori.mul(peak_feature_ori)
#                 gene_peak_1 = Gene_Peak[list(gene_index),][:, list(peak_index)].reshape(
#                     len(gene_index) * len(peak_index), 1)
#                 row_ind = torch.Tensor(list(gene_peak_1.tocoo().row) * len(cell_index))
#                 col_ind = torch.Tensor([v for v in range(len(cell_index)) for i in range(len(gene_peak_1.tocoo().row))])
#                 data = torch.Tensor(list(gene_peak_1.tocoo().data) * len(cell_index))
#                 a = torch.sparse.FloatTensor(torch.vstack((row_ind, col_ind)).long(), data,
#                                              (gene_peak_1.shape[0], len(cell_index))).to(self.device)
#                 gene_peak_cell = gene_peak_ori.mul(a)
#                 # the original in cell level (peak * gene * peak_gene)
#                 gene_peak_cell = gene_peak_ori.mul(a)
#                 # the original in cell cluster level (peak * gene * peak_gene)
#                 gene_peak_cell_cluster = torch.mm(gene_peak_ori.mul(a), cell_emb)
#
#                 logp_x3 = F.log_softmax(gene_peak_cluster_pre, dim=-1)
#                 p_y3 = F.softmax(gene_peak_cell_cluster, dim=-1)
#
#                 loss_net = F.kl_div(logp_x3, p_y3, reduction='mean')
#
#                 loss = loss_net + loss_cluster + loss_kl  # - lll2
#                 # loss = loss_cluster  + loss_kl - lll + loss_net  #+ loss_S_R#+ loss_net
#                 self.gnn_optimizer.zero_grad()
#                 self.net_optimizer.zero_grad()
#                 loss.backward()
#                 self.gnn_optimizer.step()
#                 self.net_optimizer.step()
#         return self.gnn
#
#     def train_model(self, indices, RNA_matrix, ATAC_matrix, Gene_Peak, ini_p1):
#         self.train()
#         print('The training process for the MarsGT model has started. Please wait.')
#         Mars_gnn = self.forward(indices, RNA_matrix, ATAC_matrix, Gene_Peak, ini_p1)
#         print('The training for the MarsGT model has been completed.')
#         return Mars_gnn

class MarsGT(nn.Module):
    def __init__(self, gnn, h, labsm, n_hid, n_batch, device, lr, wd, num_epochs=1):
        super(MarsGT, self).__init__()
        self.lr = lr
        self.wd = wd
        self.gnn = gnn
        self.h = h  # 假设 h 是一个张量或嵌入向量，在初始化时传入
        self.n_hid = n_hid
        self.n_batch = n_batch
        self.device = device
        self.num_epochs = num_epochs

        # 移除 Net 相关代码，因为不再处理峰值
        # self.net = Net(2 * self.n_hid, self.n_hid).to(self.device)

        # 优化器
        self.gnn_optimizer = torch.optim.AdamW(self.gnn.parameters(), lr=self.lr, weight_decay=self.wd)
        # self.net_optimizer = torch.optim.AdamW(self.net.parameters(), lr=1e-2)  # 移除 Net 优化器
        self.labsm = labsm
        self.LabSm = LabelSmoothing(self.labsm)

    def forward(self, indices, RNA_matrix, ini_p1):
        print('Forward pass started.')
        h_final = None  # 用于存储最后一个 h
        for epoch in tqdm(range(self.num_epochs), desc="Epochs"):
            for batch_id in tqdm(range(self.n_batch), desc="Batches", leave=False):
                # 获取当前批次的基因和细胞索引
                gene_index = indices[batch_id]['gene_index']
                cell_index = indices[batch_id]['cell_index']

                # 提取基因和细胞的特征
                gene_feature = RNA_matrix[list(gene_index), :]
                cell_feature = RNA_matrix[:, list(cell_index)].T

                # 转换为张量并移动到设备上
                gene_feature = torch.tensor(gene_feature.todense(), dtype=torch.float32).to(self.device)
                cell_feature = torch.tensor(cell_feature.todense(), dtype=torch.float32).to(self.device)

                # 构建节点特征列表（细胞和基因）
                node_feature = [cell_feature, gene_feature]

                # 构建基因-细胞子图的邻接矩阵
                gene_cell_sub = RNA_matrix[list(gene_index), :][:, list(cell_index)]

                # 构建基因到细胞的边索引
                gene_to_cell_src = list(np.nonzero(gene_cell_sub)[0] + len(cell_index))
                gene_to_cell_dst = list(np.nonzero(gene_cell_sub)[1])

                # 构建细胞到基因的边索引
                cell_to_gene_src = list(np.nonzero(gene_cell_sub)[1])
                cell_to_gene_dst = list(np.nonzero(gene_cell_sub)[0] + len(cell_index))

                # 合并边索引
                edge_index = torch.LongTensor([
                    gene_to_cell_src + cell_to_gene_src,
                    gene_to_cell_dst + cell_to_gene_dst
                ]).to(self.device)

                # 定义节点类型：0 代表细胞，1 代表基因
                node_type = torch.LongTensor(
                    np.array(
                        list(np.zeros(len(cell_index))) + list(np.ones(len(gene_index)))
                    )
                ).to(self.device)

                # 定义边类型：0 代表基因到细胞，1 代表细胞到基因
                edge_type_gene_to_cell = np.zeros(len(gene_to_cell_src), dtype=int)
                edge_type_cell_to_gene = np.ones(len(cell_to_gene_src), dtype=int)
                edge_type = torch.LongTensor(
                    np.concatenate([edge_type_gene_to_cell, edge_type_cell_to_gene])
                ).to(self.device)

                # 获取标签
                l = torch.LongTensor(np.array(ini_p1)[cell_index]).to(self.device)

                # 前向传播
                node_rep = self.gnn.forward(node_feature, node_type, edge_index, edge_type).to(self.device)
                cell_emb = node_rep[node_type == 0]
                gene_emb = node_rep[node_type == 1]
                # peak_emb = node_rep[node_type == 2]  # 移除峰值嵌入

                # 解码器：基因与细胞的关系
                decoder_gene_to_cell = torch.mm(gene_emb, cell_emb.t())

                # 构建目标矩阵
                gene_cell_sub_tensor = torch.tensor(gene_cell_sub.todense(), dtype=torch.float32).to(self.device)

                # 计算基因到细胞的 KL 散度损失
                logp_x1 = F.log_softmax(decoder_gene_to_cell, dim=-1)
                p_y1 = F.softmax(gene_cell_sub_tensor, dim=-1)
                loss_kl1 = F.kl_div(logp_x1, p_y1, reduction='mean')

                # 总的 KL 散度损失 (只包括基因-细胞)
                loss_kl = loss_kl1

                # 聚类损失
                loss_cluster = self.LabSm(cell_emb, l)

                # 余弦相似度损失
                lll = 0
                g = l.tolist()
                unique_labels = set(g)
                for label in unique_labels:
                    mask = np.array(g) == label
                    h = cell_emb[mask]
                    if h.size(0) > 1:
                        # 计算细胞间的余弦相似度
                        similarity = F.cosine_similarity(h.unsqueeze(1), h.unsqueeze(0), dim=-1)
                        lll += similarity.mean()
                        h_final = h  # 保留最后一个 h

                # 总损失
                loss = loss_cluster + loss_kl - lll

                # 反向传播和优化
                self.gnn_optimizer.zero_grad()
                # self.net_optimizer.zero_grad()  # 移除 Net 优化器
                loss.backward()
                self.gnn_optimizer.step()
                # self.net_optimizer.step()  # 移除 Net 优化器

        print('The training for the MarsGT model has been completed.')
        return self.gnn, cell_emb, gene_emb, h_final  # 返回 h_final

    def train_model(self, indices, RNA_matrix, ini_p1):
        self.train()
        print('The training process for the MarsGT model has started. Please wait.')
        Mars_gnn, cell_emb, gene_emb, h_final = self.forward(indices, RNA_matrix, ini_p1)
        print('The training for the MarsGT model has been completed.')
        return Mars_gnn, cell_emb, gene_emb, h_final


# def MarsGT_pred(RNA_matrix, ATAC_matrix, RP_matrix, egrn, MarsGT_gnn, indices, nodes_id, cell_size, device, gene_names,
#                 peak_names):
#     n_batch = math.ceil(nodes_id.shape[0] / cell_size)
#     embedding = []
#     l_pre = []
#     MarsGT_result = {}
#     with torch.no_grad():
#         for batch_id in tqdm(range(n_batch)):
#             gene_index = indices[batch_id]['gene_index']
#             cell_index = indices[batch_id]['cell_index']
#             peak_index = indices[batch_id]['peak_index']
#             gene_feature = RNA_matrix[list(gene_index),]
#             cell_feature = RNA_matrix[:, list(cell_index)].T
#             peak_feature = ATAC_matrix[list(peak_index),]
#             gene_feature = torch.tensor(np.array(gene_feature.todense()), dtype=torch.float32).to(device)
#             cell_feature = torch.tensor(np.array(cell_feature.todense()), dtype=torch.float32).to(device)
#             peak_feature = torch.tensor(np.array(peak_feature.todense()), dtype=torch.float32).to(device)
#             node_feature = [cell_feature, gene_feature, peak_feature]
#             gene_cell_sub = RNA_matrix[list(gene_index),][:, list(cell_index)]
#             peak_cell_sub = ATAC_matrix[list(peak_index),][:, list(cell_index)]
#             gene_cell_edge_index = torch.LongTensor(
#                 [np.nonzero(gene_cell_sub)[0] + gene_cell_sub.shape[1], np.nonzero(gene_cell_sub)[1]]).to(device)
#             peak_cell_edge_index = torch.LongTensor(
#                 [np.nonzero(peak_cell_sub)[0] + gene_cell_sub.shape[0] + gene_cell_sub.shape[1],
#                  np.nonzero(peak_cell_sub)[1]]).to(device)
#             edge_index = torch.cat((gene_cell_edge_index, peak_cell_edge_index), dim=1)
#             node_type = torch.LongTensor(np.array(
#                 list(np.zeros(len(cell_index))) + list(np.ones(len(gene_index))) + list(
#                     np.ones(len(peak_index)) * 2))).to(device)
#             edge_type = torch.LongTensor(np.array(
#                 list(np.zeros(gene_cell_edge_index.shape[1])) + list(np.ones(peak_cell_edge_index.shape[1])))).to(
#                 device)
#             node_rep = MarsGT_gnn.forward(node_feature, node_type,
#                                           edge_index,
#                                           edge_type).to(device)
#             cell_emb = node_rep[node_type == 0]
#             gene_emb = node_rep[node_type == 1]
#             peak_emb = node_rep[node_type == 2]
#
#             # If the device is CUDA, copy the tensor to CPU memory
#             if device.type == "cuda":
#                 cell_emb = cell_emb.cpu()
#             # It is now safe to convert the tensor to a NumPy array
#             embedding.append(cell_emb.detach().numpy())
#
#             cell_pre = list(cell_emb.argmax(dim=1).detach().numpy())
#             l_pre.extend(cell_pre)
#
#     cell_embedding = np.vstack(embedding)
#     cell_clu = np.array(l_pre)
#
#     if egrn:
#         final_egrn_df = egrn_calculate(cell_clu, nodes_id, RNA_matrix, ATAC_matrix, RP_matrix, gene_names, peak_names)
#         MarsGT_result = {'pred_label': cell_clu, 'cell_embedding': cell_embedding, 'egrn': final_egrn_df}
#         return MarsGT_result
#     else:
#         MarsGT_result = {'pred_label': cell_clu, 'cell_embedding': cell_embedding}
#         return MarsGT_result

def MarsGT_pred(RNA_matrix, MarsGT_gnn, indices, nodes_id, device, cell_size=30):
    """
    使用训练好的 MarsGT 模型进行预测。

    :param RNA_matrix: 基因表达矩阵 (scipy.sparse 或类似格式)
    :param egrn: 是否计算 EGRN (布尔值)
    :param MarsGT_gnn: 训练好的 MarsGT 模型
    :param indices: 批次索引列表，每个元素包含 'gene_index' 和 'cell_index'
    :param nodes_id: 节点 ID 数组或列表
    :param cell_size: 每个批次处理的细胞数量
    :param device: 计算设备 (e.g., torch.device('cuda') 或 torch.device('cpu'))
    :param gene_names: 基因名称列表
    :return: 包含预测标签、细胞嵌入以及（如果需要）EGRN 结果的字典
    """
    n_batch = math.ceil(nodes_id.shape[0] / cell_size)
    embedding = []
    l_pre = []
    MarsGT_result = {}
    with torch.no_grad():
        for batch_id in tqdm(range(n_batch), desc="Prediction Batches"):
            gene_index = indices[batch_id]['gene_index']
            cell_index = indices[batch_id]['cell_index']
            # peak_index = indices[batch_id]['peak_index']  # 移除峰值索引

            # 提取基因和细胞的特征
            gene_feature = RNA_matrix[list(gene_index), :]
            cell_feature = RNA_matrix[:, list(cell_index)].T

            # 转换为张量并移动到设备上
            gene_feature = torch.tensor(gene_feature.todense(), dtype=torch.float32).to(device)
            cell_feature = torch.tensor(cell_feature.todense(), dtype=torch.float32).to(device)

            # 构建节点特征列表（细胞和基因）
            node_feature = [cell_feature, gene_feature]

            # 构建基因-细胞子图的邻接矩阵
            gene_cell_sub = RNA_matrix[list(gene_index), :][:, list(cell_index)]

            # 构建基因到细胞的边索引
            gene_to_cell_src = list(np.nonzero(gene_cell_sub)[0] + len(cell_index))
            gene_to_cell_dst = list(np.nonzero(gene_cell_sub)[1])

            # 构建细胞到基因的边索引
            cell_to_gene_src = list(np.nonzero(gene_cell_sub)[1])
            cell_to_gene_dst = list(np.nonzero(gene_cell_sub)[0] + len(cell_index))

            # 合并边索引
            edge_index = torch.LongTensor([
                gene_to_cell_src + cell_to_gene_src,
                gene_to_cell_dst + cell_to_gene_dst
            ]).to(device)

            # 定义节点类型：0 代表细胞，1 代表基因
            node_type = torch.LongTensor(
                np.array(
                    list(np.zeros(len(cell_index))) + list(np.ones(len(gene_index)))
                )
            ).to(device)

            # 定义边类型：0 代表基因到细胞，1 代表细胞到基因
            edge_type_gene_to_cell = np.zeros(len(gene_to_cell_src), dtype=int)
            edge_type_cell_to_gene = np.ones(len(cell_to_gene_src), dtype=int)
            edge_type = torch.LongTensor(
                np.concatenate([edge_type_gene_to_cell, edge_type_cell_to_gene])
            ).to(device)

            # 获取标签
            l = torch.LongTensor(np.array(nodes_id)[cell_index]).to(device)

            # 前向传播
            node_rep = MarsGT_gnn.forward(node_feature, node_type, edge_index, edge_type).to(device)
            cell_emb = node_rep[node_type == 0]
            gene_emb = node_rep[node_type == 1]
            # peak_emb = node_rep[node_type == 2]  # 移除峰值嵌入

            # 解码器：基因与细胞的关系
            decoder_gene_to_cell = torch.mm(gene_emb, cell_emb.t())

            # 构建目标矩阵
            gene_cell_sub_tensor = torch.tensor(gene_cell_sub.todense(), dtype=torch.float32).to(device)

            # 计算基因到细胞的 KL 散度损失
            logp_x1 = F.log_softmax(decoder_gene_to_cell, dim=-1)
            p_y1 = F.softmax(gene_cell_sub_tensor, dim=-1)
            loss_kl1 = F.kl_div(logp_x1, p_y1, reduction='mean')

            # 总的 KL 散度损失 (只包括基因-细胞)
            loss_kl = loss_kl1

            # 聚类损失（如果需要）
            # 由于在预测阶段通常不需要计算损失，可以选择移除或保留根据需求
            # 这里只保留嵌入和预测标签

            # 取 cell_emb 的嵌入作为输出
            embedding.append(cell_emb.cpu().numpy())

            # 预测标签为 cell_emb 的某个维度的最大值 (假设分类任务)
            cell_pre = list(cell_emb.argmax(dim=1).detach().cpu().numpy())
            l_pre.extend(cell_pre)

    cell_embedding = np.vstack(embedding)
    cell_clu = np.array(l_pre)

    # if egrn:
    #     # 假设 egrn_calculate 已经根据新的输入调整
    #     final_egrn_df = egrn_calculate(cell_clu, nodes_id, RNA_matrix, egrn, gene_names)
    #     MarsGT_result = {'pred_label': cell_clu, 'cell_embedding': cell_embedding, 'egrn': final_egrn_df}
    #     return MarsGT_result
    # else:
    MarsGT_result = {'pred_label': cell_clu, 'cell_embedding': cell_embedding}
    return MarsGT_result
