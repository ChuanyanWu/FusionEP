import os
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch_geometric.nn import GATv2Conv
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    matthews_corrcoef, precision_score, recall_score, confusion_matrix
)
from typing import Dict, List, Tuple, Optional, Set
import warnings
import networkx as nx
from collections import defaultdict

warnings.filterwarnings('ignore')


# ============================================================
# 1. 随机种子
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 2. 数据集 (支持 centrality 开关)
# ============================================================
class ProteinDataset(Dataset):
    """多模态蛋白质数据集 - 支持中心性特征开关"""

    def __init__(self, gene_data, sub_data, protein_data, adj_matrix,
                 labels, protein_names=None, use_centrality=True):
        super().__init__()
        self.gene_data = torch.FloatTensor(gene_data)
        self.sub_data = torch.FloatTensor(sub_data)
        self.labels = torch.LongTensor(labels)
        self.protein_names = protein_names or [f"P{i}" for i in range(len(labels))]

        # 类别权重
        self.class_counts = np.bincount(labels)
        class_weights = 1.0 / self.class_counts
        class_weights = class_weights * len(self.class_counts) / class_weights.sum()
        self.sample_weights = torch.FloatTensor([class_weights[l] for l in labels])

        # 中心性特征开关
        if use_centrality:
            centrality = self._compute_centrality(adj_matrix)
            self.protein_data = torch.FloatTensor(
                np.concatenate([protein_data, centrality], axis=1)
            )
        else:
            self.protein_data = torch.FloatTensor(protein_data)

        self.adj_matrix = adj_matrix
        self.n_nodes = len(labels)

        # 边索引
        rows, cols = np.where(adj_matrix > 0)
        mask = rows < cols
        rows, cols = rows[mask], cols[mask]
        weights = adj_matrix[rows, cols]
        self.edge_index = torch.LongTensor(np.array([
            np.concatenate([rows, cols]),
            np.concatenate([cols, rows])
        ]))
        self.edge_weight = torch.FloatTensor(np.concatenate([weights, weights]))
        self.neighbors = self._build_neighbor_index()

        self.gene_dim = gene_data.shape[1]
        self.sub_dim = sub_data.shape[1]
        self.graph_dim = self.protein_data.shape[1]

    def get_stratified_sampler(self, train_idx):
        train_labels = self.labels[train_idx].numpy()
        train_counts = np.bincount(train_labels)
        class_weights = 1.0 / train_counts
        class_weights = class_weights * len(train_counts) / class_weights.sum()
        sample_weights = torch.FloatTensor([class_weights[l] for l in train_labels])
        return WeightedRandomSampler(weights=sample_weights,
                                     num_samples=len(train_idx),
                                     replacement=True)

    def _compute_centrality(self, adj_matrix):
        print("  [Centrality] Computing centrality features...")
        G = nx.from_numpy_array(adj_matrix)
        n = len(G)
        features = np.zeros((n, 4))
        features[:, 0] = [nx.degree_centrality(G)[i] for i in range(n)]
        features[:, 1] = [nx.betweenness_centrality(G, k=min(n, 1000))[i] for i in range(n)]
        features[:, 2] = [nx.closeness_centrality(G)[i] for i in range(n)]
        try:
            eig = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-3)
            features[:, 3] = [eig[i] for i in range(n)]
        except Exception:
            features[:, 3] = features[:, 0]
        return features

    def _build_neighbor_index(self):
        neighbors = defaultdict(set)
        for i in range(self.edge_index.size(1)):
            src, dst = self.edge_index[0, i].item(), self.edge_index[1, i].item()
            neighbors[src].add(dst)
            neighbors[dst].add(src)
        return neighbors

    def get_subgraph(self, node_indices, num_hops=2):
        node_indices = node_indices.long().cpu()
        target_set = set(node_indices.tolist())
        all_nodes = target_set.copy()
        current_nodes = target_set.copy()
        for _ in range(num_hops):
            new_nodes = set()
            for node in current_nodes:
                new_nodes.update(self.neighbors.get(node, set()))
            current_nodes = new_nodes - all_nodes
            all_nodes.update(current_nodes)
        all_nodes = sorted(list(all_nodes))
        node_map = {old: new for new, old in enumerate(all_nodes)}
        edge_list, weight_list = [], []
        for i in range(self.edge_index.size(1)):
            src = self.edge_index[0, i].item()
            dst = self.edge_index[1, i].item()
            if src in node_map and dst in node_map:
                edge_list.append([node_map[src], node_map[dst]])
                weight_list.append(self.edge_weight[i].item())
        if edge_list:
            sub_edge_index = torch.LongTensor(edge_list).t()
            sub_edge_weight = torch.FloatTensor(weight_list)
        else:
            sub_edge_index = torch.zeros(2, 0, dtype=torch.long)
            sub_edge_weight = torch.zeros(0)
        sub_x = self.protein_data[all_nodes]
        target_indices = torch.tensor([node_map[idx.item()] for idx in node_indices],
                                      dtype=torch.long)
        return sub_x, sub_edge_index, sub_edge_weight, target_indices

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'gene': self.gene_data[idx],
            'sub': self.sub_data[idx],
            'label': self.labels[idx],
            'idx': torch.tensor(idx, dtype=torch.long),
            'name': self.protein_names[idx]
        }


# ============================================================
# 3. Collator
# ============================================================
class GraphCollator:
    def __init__(self, dataset, num_hops=2):
        self.dataset = dataset
        self.num_hops = num_hops

    def __call__(self, batch_list):
        gene = torch.stack([b['gene'] for b in batch_list])
        sub = torch.stack([b['sub'] for b in batch_list])
        labels = torch.stack([b['label'] for b in batch_list])
        indices = torch.stack([b['idx'] for b in batch_list])
        sub_x, edge_index, edge_weight, target_indices = \
            self.dataset.get_subgraph(indices, num_hops=self.num_hops)
        return {
            'gene': gene, 'sub': sub, 'protein': sub_x,
            'label': labels, 'idx': target_indices,
            'edge_index': edge_index, 'edge_weight': edge_weight,
            'batch_size': len(batch_list), 'num_sub_nodes': sub_x.size(0)
        }


# ============================================================
# 4. 数据加载
# ============================================================
class DataLoaderHelper:
    @staticmethod
    def load_from_paths(use_centrality=True):
        paths = {
            'label_path': './Data/Label_Data/BioGRID_label.csv',
            'gene_path': './Data/Gene_Data/BioGRID_gene.xlsx',
            'sub_path': './Data/Subcellular_Data/BioGRID/Top_1024_normalized.csv',
            'protein_path': './Data/CentralityData/BioGRID_centrality.csv',
            'ppi_path': './Data/Protein_data/BioGRID.xlsx'
        }
        labels_df = pd.read_csv(paths['label_path'])
        labels = labels_df.iloc[:, 1:].to_numpy().astype('int32').flatten()
        gene_df = pd.read_excel(paths['gene_path'])
        gene_data = gene_df.iloc[:, 1:].to_numpy().astype('float32')
        sub_df = pd.read_csv(paths['sub_path'])
        sub_data = sub_df.iloc[:, 1:].to_numpy().astype('float32')
        protein_df = pd.read_csv(paths['protein_path'])
        protein_data = protein_df.iloc[:, 1:].to_numpy().astype('float32')
        protein_names = protein_df.iloc[:, 0].tolist()
        adj_matrix = DataLoaderHelper._build_adj(paths['ppi_path'], len(labels), protein_names)
        return ProteinDataset(gene_data, sub_data, protein_data, adj_matrix,
                              labels, protein_names, use_centrality=use_centrality)

    @staticmethod
    def _build_adj(ppi_path, n_proteins, protein_names):
        adj = np.zeros((n_proteins, n_proteins), dtype='float32')
        name_to_idx = {name: idx for idx, name in enumerate(protein_names)}
        try:
            ppi_df = pd.read_excel(ppi_path) if ppi_path.endswith(('.xlsx', '.xls')) \
                else pd.read_csv(ppi_path)
            for _, row in ppi_df.iterrows():
                id1, id2 = str(row.iloc[0]), str(row.iloc[1])
                idx1 = name_to_idx.get(id1)
                idx2 = name_to_idx.get(id2)
                if idx1 is None and id1.isdigit():
                    idx1 = int(id1)
                if idx2 is None and id2.isdigit():
                    idx2 = int(id2)
                if idx1 is not None and idx2 is not None \
                        and 0 <= idx1 < n_proteins and 0 <= idx2 < n_proteins:
                    w = float(row.iloc[2]) if len(row) > 2 and pd.notna(row.iloc[2]) else 1.0
                    adj[idx1, idx2] = w
                    adj[idx2, idx1] = w
        except Exception as e:
            print(f"  Warning: PPI load failed ({e}), using random sparse matrix.")
            adj = np.random.rand(n_proteins, n_proteins)
            adj = (adj + adj.T) / 2
            np.fill_diagonal(adj, 0)
            adj = (adj > 0.9).astype('float32')
        return adj


# ============================================================
# 5. 数据增强 (支持开关)
# ============================================================
class DataAugmentation:
    @staticmethod
    def augment_gene(x, noise_ratio=0.05):
        noise = torch.randn_like(x) * noise_ratio * x.std(dim=1, keepdim=True)
        return x + noise

    @staticmethod
    def drop_sub_features(x, drop_ratio=0.1):
        mask = torch.rand_like(x) > drop_ratio
        return x * mask

    @staticmethod
    def drop_edges(edge_index, edge_weight, drop_ratio=0.15):
        if edge_index.numel() == 0:
            return edge_index, edge_weight
        keep = torch.rand(edge_index.size(1)) > drop_ratio
        return edge_index[:, keep], edge_weight[keep]

    @staticmethod
    def augment_batch(batch, enabled=True):
        if not enabled:
            return batch
        batch = batch.copy()
        if torch.rand(1) < 0.5:
            batch['gene'] = DataAugmentation.augment_gene(batch['gene'])
        if torch.rand(1) < 0.3:
            batch['sub'] = DataAugmentation.drop_sub_features(batch['sub'])
        if batch.get('edge_index') is not None and batch['edge_index'].numel() > 0:
            if torch.rand(1) < 0.4:
                batch['edge_index'], batch['edge_weight'] = \
                    DataAugmentation.drop_edges(batch['edge_index'], batch['edge_weight'])
        return batch


# ============================================================
# 6. 编码器
# ============================================================
class GeneEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=768, depth=6, num_heads=12, dropout=0.15):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim), nn.LayerNorm(embed_dim),
            nn.GELU(), nn.Dropout(dropout)
        )
        self.pos_encoding = nn.Parameter(torch.randn(1, 32, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim),
            nn.GELU(), nn.Dropout(dropout)
        )
        self.contrast_proj = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 256),
            nn.ReLU(), nn.Linear(256, 128)
        )

    def forward(self, x):
        B = x.size(0)
        x = self.input_proj(x).unsqueeze(1).expand(-1, 32, -1)
        x = x + self.pos_encoding[:, :32, :]
        x = self.transformer(x)
        features = self.output_proj(x.mean(dim=1))
        proj = F.normalize(self.contrast_proj(features), dim=1)
        return features, proj


class SubcellularEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=768, num_scales=4, dropout=0.25):
        super().__init__()
        self.convs = nn.ModuleList()
        for k in [3, 5, 7, 15][:num_scales]:
            self.convs.append(nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(64), nn.ReLU(),
                nn.Conv1d(64, 64, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2)
            ))
        self.seq_len = input_dim // 2
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(64 * num_scales, 64 * num_scales // 4), nn.ReLU(),
            nn.Linear(64 * num_scales // 4, 64 * num_scales), nn.Sigmoid()
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv1d(64 * num_scales, 1, kernel_size=7, padding=3), nn.Sigmoid()
        )
        self.fusion = nn.Sequential(
            nn.Linear(64 * num_scales * self.seq_len, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim),
            nn.ReLU(), nn.Dropout(dropout / 2)
        )
        self.contrast_proj = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.ReLU(), nn.Linear(256, 128)
        )

    def forward(self, x):
        B = x.size(0)
        x = x.unsqueeze(1)
        feats = [conv(x) for conv in self.convs]
        x = torch.cat(feats, dim=1)
        x = x * self.channel_attn(x).view(B, -1, 1)
        x = x * self.spatial_attn(x)
        features = self.fusion(x.view(B, -1))
        proj = F.normalize(self.contrast_proj(features), dim=1)
        return features, proj


class GraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=384, out_dim=768, num_layers=4, dropout=0.25):
        super().__init__()
        self.node_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()
        )
        self.gat_layers = nn.ModuleList()
        self.residual_projs = nn.ModuleList()
        for i in range(num_layers):
            in_ch = hidden_dim if i == 0 else hidden_dim * 4
            out_ch = hidden_dim * 4
            self.gat_layers.append(GATv2Conv(
                in_channels=in_ch, out_channels=hidden_dim, heads=4,
                concat=True, edge_dim=1, dropout=dropout, add_self_loops=False
            ))
            self.residual_projs.append(
                nn.Linear(in_ch, out_ch) if in_ch != out_ch else None
            )
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 4, out_dim), nn.LayerNorm(out_dim),
            nn.ReLU(), nn.Dropout(dropout)
        )
        self.contrast_proj = nn.Sequential(
            nn.Linear(out_dim, 256), nn.ReLU(), nn.Linear(256, 128)
        )

    def forward(self, x, edge_index, edge_weight, target_indices):
        num_nodes = x.size(0)
        if edge_index.numel() > 0:
            loops = torch.arange(num_nodes, device=x.device).unsqueeze(0).repeat(2, 1)
            edge_index = torch.cat([edge_index, loops], dim=1)
            edge_weight = torch.cat([edge_weight, torch.ones(num_nodes, device=x.device)])
        x = self.node_proj(x)
        for gat, res in zip(self.gat_layers, self.residual_projs):
            h = gat(x, edge_index, edge_attr=edge_weight.unsqueeze(-1))
            h = F.elu(h)
            h = F.dropout(h, p=0.3, training=self.training)
            if res is not None:
                x = res(x)
            x = h + x
        target_features = x[target_indices]
        features = self.readout(target_features)
        proj = F.normalize(self.contrast_proj(features), dim=1)
        return features, proj, torch.tensor(0.0, device=x.device)


# ============================================================
# 7. 对比学习损失 (支持开关)
# ============================================================
class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature
        self.alignment_proj = nn.ModuleDict({
            'gene_sub': nn.Sequential(nn.Linear(128, 128), nn.LayerNorm(128),
                                      nn.ReLU(), nn.Linear(128, 64)),
            'gene_graph': nn.Sequential(nn.Linear(128, 128), nn.LayerNorm(128),
                                        nn.ReLU(), nn.Linear(128, 64)),
            'sub_graph': nn.Sequential(nn.Linear(128, 128), nn.LayerNorm(128),
                                       nn.ReLU(), nn.Linear(128, 64)),
        })
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))

    def forward(self, z_gene, z_sub, z_graph, labels):
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        temp = 1.0 / logit_scale
        pairs = [('gene', z_gene), ('sub', z_sub), ('graph', z_graph)]
        total_loss, valid_pairs = 0, 0
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                name_i, z_i = pairs[i]
                name_j, z_j = pairs[j]
                B = min(z_i.size(0), z_j.size(0))
                z_i, z_j = z_i[:B], z_j[:B]
                lab = labels[:B]
                if B <= 1:
                    continue
                zi = self.alignment_proj[f'{name_i}_{name_j}'](z_i)
                zj = self.alignment_proj[f'{name_i}_{name_j}'](z_j)
                sim = torch.mm(zi, zj.t()) * temp
                pos_mask = (lab.unsqueeze(1) == lab.unsqueeze(0)).float()
                neg_mask = 1 - pos_mask
                diag = 1 - torch.eye(B, device=lab.device)
                pos_mask, neg_mask = pos_mask * diag, neg_mask * diag
                pos_sim = torch.clamp((sim.exp() * pos_mask).sum(dim=1), min=1e-8)
                neg_sim = (sim.exp() * neg_mask).sum(dim=1)
                loss = -torch.log(pos_sim / (pos_sim + neg_sim + 1e-8))
                loss = loss[loss > -1e8].mean()
                if not torch.isnan(loss):
                    total_loss += loss
                    valid_pairs += 1
        return total_loss / valid_pairs if valid_pairs > 0 \
            else torch.tensor(0.0, device=z_gene.device)


# ============================================================
# 8. 自适应融合门控 (支持 uncertainty 开关)
# ============================================================
class AdaptiveFusionGate(nn.Module):
    def __init__(self, dim=768, dropout=0.25, use_uncertainty=True):
        super().__init__()
        self.use_uncertainty = use_uncertainty
        self.gene_score = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(),
                                        nn.Dropout(dropout / 2), nn.Linear(256, 1))
        self.sub_score = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(),
                                       nn.Dropout(dropout / 2), nn.Linear(256, 1))
        self.graph_score = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(),
                                         nn.Dropout(dropout / 2), nn.Linear(256, 1))
        if use_uncertainty:
            self.gene_unc = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(),
                                          nn.Linear(256, 1), nn.Sigmoid())
            self.sub_unc = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(),
                                         nn.Linear(256, 1), nn.Sigmoid())
            self.graph_unc = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(),
                                           nn.Linear(256, 1), nn.Sigmoid())
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=8,
                                                dropout=dropout, batch_first=True)
        self.fusion = nn.Sequential(
            nn.Linear(dim * 3, dim * 2), nn.LayerNorm(dim * 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim), nn.ReLU(), nn.Dropout(dropout / 2)
        )

    def forward(self, gene_f, sub_f, graph_f):
        B = gene_f.size(0)
        s_g = self.gene_score(gene_f).view(B, 1)
        s_s = self.sub_score(sub_f).view(B, 1)
        s_gr = self.graph_score(graph_f).view(B, 1)
        scores = torch.cat([s_g, s_s, s_gr], dim=1)

        if self.use_uncertainty:
            u_g = self.gene_unc(gene_f).view(B, 1)
            u_s = self.sub_unc(sub_f).view(B, 1)
            u_gr = self.graph_unc(graph_f).view(B, 1)
            uncertainties = torch.cat([u_g, u_s, u_gr], dim=1)
            adjusted = scores * (1 - uncertainties)
        else:
            uncertainties = torch.zeros_like(scores)
            adjusted = scores

        weights = F.softmax(adjusted, dim=1)
        stack = torch.stack([gene_f, sub_f, graph_f], dim=1)
        attn_out, _ = self.cross_attn(stack, stack, stack)
        attn_out = attn_out.mean(dim=1)
        fused = self.fusion(torch.cat([gene_f, sub_f, graph_f], dim=1))
        output = fused + attn_out
        return output, weights, uncertainties


# ============================================================
# 9. 预测头 (支持 prototype 正则化开关)
# ============================================================
class PredictionHead(nn.Module):
    def __init__(self, in_dim=768, num_classes=2, dropout=0.35,
                 use_prototype=True):
        super().__init__()
        self.use_prototype = use_prototype
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout / 2),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(dropout / 4),
            nn.Linear(128, num_classes)
        )
        if use_prototype:
            self.register_buffer('prototypes', torch.zeros(num_classes, 128))
            self.register_buffer('prototype_counts', torch.zeros(num_classes))
            self.prototype_initialized = False
        self.register_buffer('focal_alpha', torch.tensor([0.25, 0.75]))
        self.focal_gamma = 2.0
        self.label_smoothing = 0.1

    def forward(self, x, labels=None, mode='train'):
        logits = self.classifier(x)
        if mode == 'train' and labels is not None:
            features = self.classifier[:-1](x)
            if self.use_prototype:
                with torch.no_grad():
                    for c in range(2):
                        mask = (labels == c)
                        if mask.any():
                            cf = features[mask].mean(dim=0)
                            if self.prototype_counts[c] == 0:
                                self.prototypes[c] = cf
                            else:
                                self.prototypes[c] = 0.9 * self.prototypes[c] + 0.1 * cf
                            self.prototype_counts[c] += mask.sum().item()
                    self.prototype_initialized = True

            num_classes = logits.size(-1)
            smoothed = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), 1.0)
            smoothed = smoothed * (1 - self.label_smoothing) + self.label_smoothing / num_classes
            log_probs = F.log_softmax(logits, dim=1)
            ce_loss = -(smoothed * log_probs).sum(dim=1).mean()
            probs = torch.exp(-ce_loss)
            focal_weight = (1 - probs) ** self.focal_gamma
            alpha = self.focal_alpha[labels]
            focal_loss = (alpha * focal_weight * ce_loss).mean()

            proto_loss = self._proto_loss(features, labels) if self.use_prototype \
                else torch.tensor(0.0, device=x.device)
            total_loss = focal_loss + 0.05 * proto_loss
            return logits, total_loss, focal_loss, proto_loss

        zero = torch.tensor(0.0, device=x.device)
        return logits, zero, zero, zero

    def _proto_loss(self, features, labels):
        if not self.prototype_initialized:
            return torch.tensor(0.0, device=features.device)
        pos = F.cosine_similarity(features, self.prototypes[labels], dim=1)
        neg_sims = []
        for c in range(len(self.prototypes)):
            neg_sims.append(F.cosine_similarity(
                features, self.prototypes[c].unsqueeze(0), dim=1
            ).unsqueeze(1))
        all_sims = torch.cat(neg_sims, dim=1)
        neg_mask = torch.ones_like(all_sims, dtype=torch.bool)
        neg_mask.scatter_(1, labels.unsqueeze(1), False)
        neg = (all_sims * neg_mask.float()).max(dim=1)[0]
        return -torch.log(
            torch.exp(pos) / (torch.exp(pos) + torch.exp(neg) + 1e-8) + 1e-8
        ).mean()


# ============================================================
# FusionEP 主模型
# ============================================================
class FusionEP(nn.Module):
    """
    ablation 开关字典:
      use_centrality   : 中心性增强子图采样 (影响 graph_in_dim)
      use_contrastive  : 跨模态对比学习
      use_uncertainty  : 自适应不确定性感知融合
      use_prototype    : 原型正则化
      use_augmentation : 数据增强
    """

    def __init__(self, config, ablation):
        super().__init__()
        self.config = config
        self.ablation = ablation

        gene_dim = config.get('gene_dim', 36)
        sub_dim = config.get('sub_dim', 1024)
        graph_dim = config.get('graph_in_dim', 9)
        embed_dim = config.get('embed_dim', 768)

        self.gene_encoder = GeneEncoder(
            gene_dim, embed_dim, depth=config.get('gene_depth', 6),
            num_heads=config.get('num_heads', 12), dropout=config.get('dropout', 0.15)
        )
        self.sub_encoder = SubcellularEncoder(
            sub_dim, embed_dim, num_scales=4, dropout=config.get('dropout', 0.25)
        )
        self.graph_encoder = GraphEncoder(
            graph_dim, hidden_dim=config.get('graph_hidden', 384),
            out_dim=embed_dim, num_layers=config.get('graph_layers', 4),
            dropout=config.get('dropout', 0.25)
        )
        self.contrast_loss = SupervisedContrastiveLoss(
            temperature=config.get('temperature', 0.05)
        )
        self.fusion_gate = AdaptiveFusionGate(
            dim=embed_dim, dropout=config.get('dropout', 0.25),
            use_uncertainty=ablation.get('use_uncertainty', True)
        )
        self.predictor = PredictionHead(
            in_dim=embed_dim, num_classes=config.get('num_classes', 2),
            dropout=config.get('dropout', 0.35),
            use_prototype=ablation.get('use_prototype', True)
        )
        self.lambda_contrast = config.get('lambda_contrast', 1.0)
        self.lambda_entropy = config.get('lambda_entropy', 0.001)

    def forward(self, batch, mode='train'):
        if mode == 'train':
            batch = DataAugmentation.augment_batch(
                batch, enabled=self.ablation.get('use_augmentation', True)
            )

        gene_feat, gene_proj = self.gene_encoder(batch['gene'])
        sub_feat, sub_proj = self.sub_encoder(batch['sub'])

        ei = batch.get('edge_index')
        if ei is not None and ei.numel() > 0:
            graph_feat, graph_proj, entropy = self.graph_encoder(
                batch['protein'], ei, batch['edge_weight'], batch['idx']
            )
        else:
            graph_feat = torch.zeros_like(gene_feat)
            graph_proj = torch.zeros_like(gene_proj)
            entropy = torch.tensor(0.0, device=gene_feat.device)

        B = gene_feat.size(0)
        sub_feat, sub_proj = sub_feat[:B], sub_proj[:B]
        graph_feat, graph_proj = graph_feat[:B], graph_proj[:B]

        # 对比学习开关
        if self.ablation.get('use_contrastive', True):
            contrast = self.contrast_loss(gene_proj, sub_proj, graph_proj, batch['label'])
        else:
            contrast = torch.tensor(0.0, device=gene_feat.device)

        fused, weights, unc = self.fusion_gate(gene_feat, sub_feat, graph_feat)
        logits, pred_loss, focal, proto = self.predictor(fused, batch['label'], mode)

        if mode == 'train':
            total = pred_loss + self.lambda_contrast * contrast + self.lambda_entropy * entropy
        else:
            total = torch.tensor(0.0, device=logits.device)

        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        return {
            'logits': logits, 'probs': probs, 'preds': preds,
            'fused_features': fused, 'fusion_weights': weights,
            'uncertainties': unc,
            'losses': {
                'total': total, 'prediction': pred_loss,
                'contrastive': contrast, 'entropy': entropy,
                'focal': focal, 'prototype': proto
            }
        }


# ============================================================
# 11. 训练器
# ============================================================
class Trainer:
    def __init__(self, model, config, device='cpu'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.best_state = None
        self.best_metric = 0

    def train_epoch(self, loader, optimizer):
        self.model.train()
        total_loss, n = 0, 0
        for batch in loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            optimizer.zero_grad()
            out = self.model(batch, mode='train')
            loss = out['losses']['total']
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1
        return total_loss / max(n, 1)

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        probs, labels, preds = [], [], []
        for batch in loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            out = self.model(batch, mode='eval')
            probs.extend(out['probs'][:, 1].cpu().numpy())
            labels.extend(batch['label'].cpu().numpy())
            preds.extend(out['preds'].cpu().numpy())
        labels, probs, preds = np.array(labels), np.array(probs), np.array(preds)
        try:
            auc = roc_auc_score(labels, probs)
        except ValueError:
            auc = 0.5
        try:
            aupr = average_precision_score(labels, probs)
        except ValueError:
            aupr = 0.0
        return {
            'auc_roc': auc,
            'auc_pr': aupr,
            'f1': f1_score(labels, preds, zero_division=0),
            'mcc': matthews_corrcoef(labels, preds),
            'precision': precision_score(labels, preds, zero_division=0),
            'recall': recall_score(labels, preds, zero_division=0),
            'accuracy': (preds == labels).mean(),
            'confusion_matrix': confusion_matrix(labels, preds)
        }

    def fit(self, train_loader, val_loader, test_loader=None):
        groups = [
            {'params': self.model.gene_encoder.parameters(), 'lr': 5e-5, 'weight_decay': 0.01},
            {'params': self.model.sub_encoder.parameters(), 'lr': 1e-4, 'weight_decay': 0.01},
            {'params': self.model.graph_encoder.parameters(), 'lr': 1e-4, 'weight_decay': 0.01},
            {'params': self.model.fusion_gate.parameters(), 'lr': 2e-4, 'weight_decay': 0.001},
            {'params': self.model.predictor.parameters(), 'lr': 3e-4, 'weight_decay': 0.001},
            {'params': self.model.contrast_loss.parameters(), 'lr': 2e-4, 'weight_decay': 0.001},
        ]
        optimizer = torch.optim.AdamW(groups)
        epochs = self.config.get('epochs', 10)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        patience = self.config.get('patience', 30)
        counter = 0
        for epoch in range(epochs):
            loss = self.train_epoch(train_loader, optimizer)
            val = self.evaluate(val_loader)
            metric = val['auc_roc'] * 0.4 + val['auc_pr'] * 0.35 + val['f1'] * 0.25
            if metric > self.best_metric:
                self.best_metric = metric
                self.best_state = copy.deepcopy(self.model.state_dict())
                counter = 0
            else:
                counter += 1
            if epoch % 5 == 0:
                cm = val['confusion_matrix']
                print(f"  Epoch {epoch}/{epochs} | Loss {loss:.4f} | "
                      f"AUC {val['auc_roc']:.4f} | AUPR {val['auc_pr']:.4f} | "
                      f"F1 {val['f1']:.4f} | MCC {val['mcc']:.4f} | "
                      f"TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")
            scheduler.step()
            if counter >= patience:
                print(f"  Early stop at epoch {epoch}")
                break
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        return self


# ============================================================
# 12. 消融实验配置
# ============================================================
ABLATION_VARIANTS = [
    {
        'name': 'FusionEP (Full Model)',
        'removed': '— (完整模型)',
        'ablation': {
            'use_centrality': True,
            'use_contrastive': True,
            'use_uncertainty': True,
            'use_prototype': True,
            'use_augmentation': True,
        }
    },
    {
        'name': 'w/o Centrality-augmented subgraph sampling',
        'removed': 'Centrality-augmented subgraph sampling',
        'ablation': {
            'use_centrality': False,
            'use_contrastive': True,
            'use_uncertainty': True,
            'use_prototype': True,
            'use_augmentation': True,
        }
    },
    {
        'name': 'w/o Cross-modal contrastive learning',
        'removed': 'Cross-modal contrastive learning',
        'ablation': {
            'use_centrality': True,
            'use_contrastive': False,
            'use_uncertainty': True,
            'use_prototype': True,
            'use_augmentation': True,
        }
    },
    {
        'name': 'w/o Adaptive uncertainty-aware fusion',
        'removed': 'Adaptive uncertainty-aware fusion (static weighting)',
        'ablation': {
            'use_centrality': True,
            'use_contrastive': True,
            'use_uncertainty': False,
            'use_prototype': True,
            'use_augmentation': True,
        }
    },
    {
        'name': 'w/o Prototype regularization',
        'removed': 'Prototype regularization',
        'ablation': {
            'use_centrality': True,
            'use_contrastive': True,
            'use_uncertainty': True,
            'use_prototype': False,
            'use_augmentation': True,
        }
    },
    {
        'name': 'w/o Data augmentation',
        'removed': 'Data augmentation',
        'ablation': {
            'use_centrality': True,
            'use_contrastive': True,
            'use_uncertainty': True,
            'use_prototype': True,
            'use_augmentation': False,
        }
    },
]


# ============================================================
# 13. 消融实验主函数
# ============================================================
def run_ablation():
    set_seed(42)
    device = 'cpu'
    print(f"Device: {device}")
    print("=" * 70)

    base_config = {
        'gene_dim': 36,
        'sub_dim': 1024,
        'graph_in_dim': 9,        # 不含中心性时的原始维度
        'graph_in_dim_centrality': 13,  # 含中心性时 9 + 4 = 13
        'embed_dim': 768,
        'gene_depth': 6,
        'graph_layers': 4,
        'num_heads': 12,
        'dropout': 0.2,
        'temperature': 0.05,
        'lambda_contrast': 1.0,
        'lambda_entropy': 0.001,
        'epochs': 10,
        'batch_size': 64,
        'patience': 30,
    }

    # 固定数据划分 (所有变体共用同一份 train/val/test)
    print("Loading data (with centrality for full model)...")
    dataset_full = DataLoaderHelper.load_from_paths(use_centrality=True)
    print("Loading data (without centrality for ablation)...")
    dataset_no_cent = DataLoaderHelper.load_from_paths(use_centrality=False)

    n_total = len(dataset_full)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    indices = np.random.permutation(n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    results = []

    for variant in ABLATION_VARIANTS:
        print("\n" + "=" * 70)
        print(f"[Ablation] {variant['name']}")
        print(f"  Removed: {variant['removed']}")
        print(f"  Settings: {variant['ablation']}")
        print("-" * 70)

        ablation = variant['ablation']

        # 根据 centrality 开关选择数据集
        if ablation['use_centrality']:
            dataset = dataset_full
            config = {**base_config, 'graph_in_dim': base_config['graph_in_dim_centrality']}
        else:
            dataset = dataset_no_cent
            config = {**base_config, 'graph_in_dim': base_config['graph_in_dim']}

        collator = GraphCollator(dataset, num_hops=2)
        train_sampler = dataset.get_stratified_sampler(train_idx)

        train_loader = DataLoader(
            torch.utils.data.Subset(dataset, train_idx),
            batch_size=config['batch_size'], sampler=train_sampler,
            shuffle=False, num_workers=0, collate_fn=collator
        )
        val_loader = DataLoader(
            torch.utils.data.Subset(dataset, val_idx),
            batch_size=config['batch_size'] * 2, shuffle=False,
            num_workers=0, collate_fn=collator
        )
        test_loader = DataLoader(
            torch.utils.data.Subset(dataset, test_idx),
            batch_size=config['batch_size'] * 2, shuffle=False,
            num_workers=0, collate_fn=collator
        )

        set_seed(42)
        model = FusionEP(config, ablation)
        trainer = Trainer(model, config, device)
        trainer.fit(train_loader, val_loader, test_loader)

        test_metrics = trainer.evaluate(test_loader)
        results.append({
            'variant': variant['name'],
            'removed': variant['removed'],
            **{k: v for k, v in test_metrics.items() if k != 'confusion_matrix'},
            'cm': test_metrics['confusion_matrix']
        })

        print(f"\n  >> Test Results:")
        print(f"     AUC-ROC : {test_metrics['auc_roc']:.4f}")
        print(f"     AUC-PR  : {test_metrics['auc_pr']:.4f}")
        print(f"     F1      : {test_metrics['f1']:.4f}")
        print(f"     MCC     : {test_metrics['mcc']:.4f}")
        print(f"     Precision: {test_metrics['precision']:.4f}")
        print(f"     Recall  : {test_metrics['recall']:.4f}")
        cm = test_metrics['confusion_matrix']
        print(f"     CM      : TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")

    # ============================================================
    # 14. 汇总表格输出
    # ============================================================
    print("\n" + "=" * 70)
    print("ABLATION STUDY SUMMARY")
    print("=" * 70)
    header = f"{'Variant':<55} {'AUC':>7} {'AUPR':>7} {'F1':>7} {'MCC':>7}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['variant']:<55} "
              f"{r['auc_roc']:>7.4f} {r['auc_pr']:>7.4f} "
              f"{r['f1']:>7.4f} {r['mcc']:>7.4f}")

    # 保存为 CSV
    df = pd.DataFrame([{
        'Variant': r['variant'],
        'Removed Component': r['removed'],
        'AUC-ROC': round(r['auc_roc'], 4),
        'AUC-PR': round(r['auc_pr'], 4),
        'F1': round(r['f1'], 4),
        'MCC': round(r['mcc'], 4),
        'Precision': round(r['precision'], 4),
        'Recall': round(r['recall'], 4),
        'Accuracy': round(r['accuracy'], 4),
        'TN': int(r['cm'][0, 0]),
        'FP': int(r['cm'][0, 1]),
        'FN': int(r['cm'][1, 0]),
        'TP': int(r['cm'][1, 1]),
    } for r in results])
    df.to_csv('ablation_results.csv', index=False, encoding='utf-8-sig')
    print(f"\nResults saved to ablation_results.csv")
    print("Done!")
    return results


if __name__ == "__main__":
    run_ablation()
