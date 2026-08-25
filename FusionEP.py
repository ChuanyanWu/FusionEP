import os
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler, WeightedRandomSampler
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data as GeoData
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
from typing import Dict, List, Tuple, Optional, Set
import warnings
import networkx as nx
from collections import defaultdict
import math
from sklearn.metrics import confusion_matrix, matthews_corrcoef
warnings.filterwarnings('ignore')


# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# set_seed(42)




class ProteinDataset(Dataset):
    """多模态蛋白质数据集 - 支持子图采样和加权采样"""

    def __init__(self,
                 gene_data: np.ndarray,
                 sub_data: np.ndarray,
                 protein_data: np.ndarray,
                 adj_matrix: np.ndarray,
                 labels: np.ndarray,
                 protein_names: Optional[List[str]] = None,
                 compute_centrality: bool = True):
        super().__init__()

        self.gene_data = torch.FloatTensor(gene_data)
        self.sub_data = torch.FloatTensor(sub_data)
        self.labels = torch.LongTensor(labels)
        self.protein_names = protein_names or [f"P{i}" for i in range(len(labels))]

        # 计算类别分布并创建采样权重
        self.class_counts = np.bincount(labels)
        print(f"类别分布: {self.class_counts} (负类:正类 = {self.class_counts[0]}:{self.class_counts[1]})")
        
        # 计算每个样本的权重（逆频率加权）
        # 策略：少数类样本获得更高权重
        class_weights = 1.0 / self.class_counts
        # 归一化类别权重，使平均权重为1
        class_weights = class_weights * len(self.class_counts) / class_weights.sum()
        self.sample_weights = torch.FloatTensor([class_weights[label] for label in labels])
        print(f"类别权重: {class_weights}")
        print(f"采样权重范围: [{self.sample_weights.min():.3f}, {self.sample_weights.max():.3f}]")

        # 计算中心性特征
        if compute_centrality:
            centrality = self._compute_centrality_features(adj_matrix)
            self.protein_data = torch.FloatTensor(
                np.concatenate([protein_data, centrality], axis=1)
            )
        else:
            self.protein_data = torch.FloatTensor(protein_data)

        self.adj_matrix = adj_matrix
        self.n_nodes = len(labels)

        # 构建边索引（无向图，只存储一次）
        rows, cols = np.where(adj_matrix > 0)
        # 只保留上三角部分避免重复
        mask = rows < cols
        rows, cols = rows[mask], cols[mask]
        weights = adj_matrix[rows, cols]

        # 创建双向边
        self.edge_index = torch.LongTensor(np.array([
            np.concatenate([rows, cols]),
            np.concatenate([cols, rows])
        ]))
        self.edge_weight = torch.FloatTensor(np.concatenate([weights, weights]))

        # 构建邻居索引加速子图采样
        self.neighbors = self._build_neighbor_index()

        self.gene_dim = gene_data.shape[1]
        self.sub_dim = sub_data.shape[1]
        self.graph_dim = self.protein_data.shape[1]

    def get_weighted_sampler(self, replacement: bool = True) -> WeightedRandomSampler:
        """
        创建加权随机采样器，平衡类别分布
        
        Args:
            replacement: 是否允许重复采样（建议True以充分学习少数类）
        
        Returns:
            WeightedRandomSampler实例
        """
        return WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=len(self),
            replacement=replacement
        )

    def get_stratified_sampler(self, train_idx: np.ndarray) -> WeightedRandomSampler:
        """
        为特定训练子集创建分层加权采样器
        
        Args:
            train_idx: 训练集索引
        
        Returns:
            针对训练集的WeightedRandomSampler
        """
        train_labels = self.labels[train_idx].numpy()
        train_counts = np.bincount(train_labels)
        
        # 计算训练集内的类别权重
        class_weights = 1.0 / train_counts
        class_weights = class_weights * len(train_counts) / class_weights.sum()
        
        # 为训练样本分配权重
        sample_weights = torch.FloatTensor([class_weights[label] for label in train_labels])
        
        print(f"训练集类别分布: {train_counts}")
        print(f"训练集类别权重: {class_weights}")
        
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_idx),
            replacement=True
        )

    def _compute_centrality_features(self, adj_matrix: np.ndarray) -> np.ndarray:
        """计算网络中心性特征"""
        print("Computing centrality features...")
        G = nx.from_numpy_array(adj_matrix)

        n = len(G)
        features = np.zeros((n, 4))

        # 度中心性
        deg_cent = nx.degree_centrality(G)
        features[:, 0] = [deg_cent[i] for i in range(n)]

        # 介数中心性（采样加速）
        bet_cent = nx.betweenness_centrality(G, k=min(n, 1000))
        features[:, 1] = [bet_cent[i] for i in range(n)]

        # 接近中心性
        clo_cent = nx.closeness_centrality(G)
        features[:, 2] = [clo_cent[i] for i in range(n)]

        # 特征向量中心性
        try:
            eig_cent = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-3)
            features[:, 3] = [eig_cent[i] for i in range(n)]
        except:
            features[:, 3] = features[:, 0]  # 回退到度中心性

        print(f"  Centrality features computed: {features.shape}")
        return features

    def _build_neighbor_index(self) -> Dict[int, Set[int]]:
        """构建邻居索引字典"""
        neighbors = defaultdict(set)
        for i in range(self.edge_index.size(1)):
            src, dst = self.edge_index[0, i].item(), self.edge_index[1, i].item()
            neighbors[src].add(dst)
            neighbors[dst].add(src)
        return neighbors

    def get_subgraph(self, node_indices: torch.Tensor, num_hops: int = 2) -> Tuple:
        """
        获取k-hop子图，并重新编号节点
        Returns:
            sub_x: 子图节点特征 (N_sub, graph_dim)
            sub_edge_index: 重新编号的边索引 (2, E_sub)
            sub_edge_weight: 边权重 (E_sub,)
            target_indices: 目标节点在子图中的索引 (B,)
        """
        node_indices = node_indices.long().cpu()
        target_set = set(node_indices.tolist())

        # 收集k-hop邻居
        all_nodes = target_set.copy()
        current_nodes = target_set.copy()

        for _ in range(num_hops):
            new_nodes = set()
            for node in current_nodes:
                new_nodes.update(self.neighbors.get(node, set()))
            current_nodes = new_nodes - all_nodes
            all_nodes.update(current_nodes)

        # 创建节点映射
        all_nodes = sorted(list(all_nodes))
        node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(all_nodes)}

        # 筛选边（两个端点都在子图中）
        edge_list = []
        weight_list = []
        for i in range(self.edge_index.size(1)):
            src = self.edge_index[0, i].item()
            dst = self.edge_index[1, i].item()
            if src in node_map and dst in node_map:
                edge_list.append([node_map[src], node_map[dst]])
                weight_list.append(self.edge_weight[i].item())

        if len(edge_list) > 0:
            sub_edge_index = torch.LongTensor(edge_list).t()
            sub_edge_weight = torch.FloatTensor(weight_list)
        else:
            sub_edge_index = torch.zeros(2, 0, dtype=torch.long)
            sub_edge_weight = torch.zeros(0)

        # 获取子图特征
        sub_x = self.protein_data[all_nodes]

        # 目标节点在新子图中的索引
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


class GraphCollator:
    """支持子图采样的DataLoader收集器"""

    def __init__(self, dataset: ProteinDataset, num_hops: int = 2):
        self.dataset = dataset
        self.num_hops = num_hops

    def __call__(self, batch_list: List[Dict]) -> Dict:
        # 收集基本数据
        gene = torch.stack([b['gene'] for b in batch_list])
        sub = torch.stack([b['sub'] for b in batch_list])
        labels = torch.stack([b['label'] for b in batch_list])
        indices = torch.stack([b['idx'] for b in batch_list])

        # 获取子图
        sub_x, edge_index, edge_weight, target_indices =self.dataset.get_subgraph(indices, num_hops=self.num_hops)

        return {
            'gene': gene,
            'sub': sub,
            'protein': sub_x,
            'label': labels,
            'idx': target_indices,
            'edge_index': edge_index,
            'edge_weight': edge_weight,
            'batch_size': len(batch_list),
            'num_sub_nodes': sub_x.size(0)
        }


class DataLoaderHelper:
    @staticmethod
    def load_from_paths(config: Dict) -> ProteinDataset:
        paths = {
            'label_path': './Data/Label_Data/BioGRID_label.csv',
            'gene_path': './Data/Gene_Data/BioGRID_gene.xlsx',
            'sub_path': './Data/Subcellular_Data/BioGRID/Top_1024_normalized.csv',
            'protein_path': './Data/CentralityData/BioGRID_centrality.csv',
            'ppi_path': './Data/Protein_data/BioGRID.xlsx'
        }

        # 加载标签
        labels_df = pd.read_csv(paths['label_path'])
        labels = labels_df.iloc[:, 1:].to_numpy().astype('int32').flatten()

        # 加载基因数据
        gene_df = pd.read_excel(paths['gene_path'])
        gene_data = gene_df.iloc[:, 1:].to_numpy().astype('float32')

        # 加载亚细胞数据
        sub_df = pd.read_csv(paths['sub_path'])
        sub_data = sub_df.iloc[:, 1:].to_numpy().astype('float32')

        # 加载蛋白质网络数据
        protein_df = pd.read_csv(paths['protein_path'])
        protein_data = protein_df.iloc[:, 1:].to_numpy().astype('float32')
        protein_names = protein_df.iloc[:, 0].tolist()

        # 构建邻接矩阵
        adj_matrix = DataLoaderHelper._build_adjacency_matrix(
            paths['ppi_path'], len(labels), protein_names
        )

        print(f"数据加载完成:")
        print(f"  基因数据: {gene_data.shape}")
        print(f"  亚细胞数据: {sub_data.shape}")
        print(f"  蛋白质网络数据: {protein_data.shape}")
        print(f"  邻接矩阵: {adj_matrix.shape}")

        return ProteinDataset(gene_data, sub_data, protein_data, adj_matrix,
                              labels, protein_names, compute_centrality=True)

    @staticmethod
    def _build_adjacency_matrix(ppi_path: str, n_proteins: int,
                                protein_names: List[str]) -> np.ndarray:
        adj = np.zeros((n_proteins, n_proteins), dtype='float32')
        name_to_idx = {name: idx for idx, name in enumerate(protein_names)}

        try:
            if ppi_path.endswith('.xlsx') or ppi_path.endswith('.xls'):
                ppi_df = pd.read_excel(ppi_path)
            else:
                ppi_df = pd.read_csv(ppi_path)

            print(f"PPI文件列名: {ppi_df.columns.tolist()}")

            for _, row in ppi_df.iterrows():
                id1, id2 = str(row.iloc[0]), str(row.iloc[1])

                idx1 = name_to_idx.get(id1)
                idx2 = name_to_idx.get(id2)

                if idx1 is None and id1.isdigit():
                    idx1 = int(id1)
                if idx2 is None and id2.isdigit():
                    idx2 = int(id2)

                if idx1 is not None and idx2 is not None and 0 <= idx1 < n_proteins and 0 <= idx2 < n_proteins:
                    weight = float(row.iloc[2]) if len(row) > 2 and pd.notna(row.iloc[2]) else 1.0
                    adj[idx1, idx2] = weight
                    adj[idx2, idx1] = weight

        except Exception as e:
            print(f"Warning: Could not load PPI file: {e}")
            print("Creating random sparse matrix for testing...")
            adj = np.random.rand(n_proteins, n_proteins)
            adj = (adj + adj.T) / 2
            np.fill_diagonal(adj, 0)
            adj = (adj > 0.9).astype('float32')

        return adj




class DataAugmentation:


    @staticmethod
    def augment_gene(x: torch.Tensor, noise_ratio: float = 0.05) -> torch.Tensor:

        noise = torch.randn_like(x) * noise_ratio * x.std(dim=1, keepdim=True)
        return x + noise

    @staticmethod
    def drop_sub_features(x: torch.Tensor, drop_ratio: float = 0.1) -> torch.Tensor:

        mask = torch.rand_like(x) > drop_ratio
        return x * mask

    @staticmethod
    def drop_edges(edge_index: torch.Tensor, edge_weight: torch.Tensor,
                   drop_ratio: float = 0.15) -> Tuple[torch.Tensor, torch.Tensor]:

        if edge_index.numel() == 0:
            return edge_index, edge_weight

        num_edges = edge_index.size(1)
        keep_mask = torch.rand(num_edges) > drop_ratio
        return edge_index[:, keep_mask], edge_weight[keep_mask]

    @staticmethod
    def augment_batch(batch: Dict, training: bool = True) -> Dict:

        if not training:
            return batch

        batch = batch.copy()


        if torch.rand(1) < 0.5:
            batch['gene'] = DataAugmentation.augment_gene(batch['gene'])


        if torch.rand(1) < 0.3:
            batch['sub'] = DataAugmentation.drop_sub_features(batch['sub'])


        if 'edge_index' in batch and batch['edge_index'].numel() > 0:
            if torch.rand(1) < 0.4:
                batch['edge_index'], batch['edge_weight'] =  DataAugmentation.drop_edges(batch['edge_index'], batch['edge_weight'])

        return batch




class GeneEncoder(nn.Module):


    def __init__(self, input_dim: int, embed_dim: int = 768,
                 depth: int = 6, num_heads: int = 12, dropout: float = 0.15):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )


        self.pos_encoding = nn.Parameter(torch.randn(1, 32, embed_dim) * 0.02)


        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 对比学习投影头
        self.contrast_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.size(0)

        x = self.input_proj(x)

        # 扩展为序列
        x = x.unsqueeze(1).expand(-1, 32, -1)
        x = x + self.pos_encoding[:, :32, :]

        # Transformer编码
        x = self.transformer(x)

        # 全局平均池化
        features = x.mean(dim=1)
        features = self.output_proj(features)

        # 对比投影
        proj = F.normalize(self.contrast_proj(features), dim=1)

        return features, proj


class SubcellularEncoder(nn.Module):
    """亚细胞编码器 - 使用更深的CNN"""

    def __init__(self, input_dim: int, embed_dim: int = 768,
                 num_scales: int = 4, dropout: float = 0.25):
        super().__init__()
        self.input_dim = input_dim

        # 多尺度卷积（更多尺度）
        self.convs = nn.ModuleList()
        kernel_sizes = [3, 5, 7, 15]

        for k in kernel_sizes[:num_scales]:
            self.convs.append(nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, 64, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(2)
            ))

        # 计算池化后的序列长度
        self.seq_len = input_dim // 2

        # 通道注意力
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64 * num_scales, 64 * num_scales // 4),
            nn.ReLU(),
            nn.Linear(64 * num_scales // 4, 64 * num_scales),
            nn.Sigmoid()
        )

        # 空间注意力
        self.spatial_attn = nn.Sequential(
            nn.Conv1d(64 * num_scales, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        # 融合层
        fusion_input_dim = 64 * num_scales * self.seq_len
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout / 2)
        )

        # 对比投影
        self.contrast_proj = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.size(0)
        x = x.unsqueeze(1)

        # 多尺度特征提取
        multi_scale_feats = []
        for conv in self.convs:
            feat = conv(x)
            multi_scale_feats.append(feat)

        x = torch.cat(multi_scale_feats, dim=1)

        # 注意力
        channel_weights = self.channel_attn(x).view(B, -1, 1)
        x = x * channel_weights

        spatial_weights = self.spatial_attn(x)
        x = x * spatial_weights

        # 融合
        x = x.view(B, -1)
        features = self.fusion(x)
        proj = F.normalize(self.contrast_proj(features), dim=1)

        return features, proj


class GraphEncoder(nn.Module):
    """图编码器 - 使用更深的GAT和残差连接"""

    def __init__(self, in_dim: int, hidden_dim: int = 384,
                 out_dim: int = 768, num_layers: int = 4, dropout: float = 0.25):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 输入投影
        self.node_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # GAT层（带残差）
        self.gat_layers = nn.ModuleList()
        self.residual_projs = nn.ModuleList()

        for i in range(num_layers):
            in_ch = hidden_dim if i == 0 else hidden_dim * 4
            out_ch = hidden_dim * 4

            self.gat_layers.append(
                GATv2Conv(
                    in_channels=in_ch,
                    out_channels=hidden_dim,
                    heads=4,
                    concat=True,
                    edge_dim=1,
                    dropout=dropout,
                    add_self_loops=False
                )
            )

            # 残差投影
            if in_ch != out_ch:
                self.residual_projs.append(nn.Linear(in_ch, out_ch))
            else:
                self.residual_projs.append(None)

        # 读出层
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 4, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 对比投影
        self.contrast_proj = nn.Sequential(
            nn.Linear(out_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor, target_indices: torch.Tensor) -> Tuple:
        """
        Args:
            x: (N, in_dim) 子图所有节点特征
            edge_index: (2, E) 子图边
            edge_weight: (E,) 边权重
            target_indices: (B,) 目标节点在子图中的索引
        """
        # 添加自环
        num_nodes = x.size(0)
        if edge_index.numel() > 0:
            self_loops = torch.arange(num_nodes, device=x.device).unsqueeze(0).repeat(2, 1)
            edge_index = torch.cat([edge_index, self_loops], dim=1)
            edge_weight = torch.cat([edge_weight, torch.ones(num_nodes, device=x.device)])

        # 输入投影
        x = self.node_proj(x)

        # 多层GAT（带残差）
        for gat_layer, res_proj in zip(self.gat_layers, self.residual_projs):
            h = gat_layer(x, edge_index, edge_attr=edge_weight.unsqueeze(-1))
            h = F.elu(h)
            h = F.dropout(h, p=0.3, training=self.training)

            # 残差连接
            if res_proj is not None:
                x = res_proj(x)

            x = h + x

        # 提取目标节点特征
        target_features = x[target_indices]

        # 读出
        features = self.readout(target_features)
        proj = F.normalize(self.contrast_proj(features), dim=1)

        entropy_loss = torch.tensor(0.0, device=x.device)

        return features, proj, entropy_loss




class SupervisedContrastiveLoss(nn.Module):
    """监督对比学习损失 - 同类拉近，异类推开"""

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

        # 模态对齐投影
        self.alignment_proj = nn.ModuleDict({
            'gene_sub': nn.Sequential(
                nn.Linear(128, 128), nn.LayerNorm(128), nn.ReLU(), nn.Linear(128, 64)
            ),
            'gene_graph': nn.Sequential(
                nn.Linear(128, 128), nn.LayerNorm(128), nn.ReLU(), nn.Linear(128, 64)
            ),
            'sub_graph': nn.Sequential(
                nn.Linear(128, 128), nn.LayerNorm(128), nn.ReLU(), nn.Linear(128, 64)
            ),
        })

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))

    def forward(self, z_gene: torch.Tensor, z_sub: torch.Tensor,
                z_graph: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, Dict]:

        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        temp = 1.0 / logit_scale

        losses = {}
        pairs = [('gene', z_gene), ('sub', z_sub), ('graph', z_graph)]
        total_loss = 0
        valid_pairs = 0

        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                name_i, z_i = pairs[i]
                name_j, z_j = pairs[j]
                pair_key = f'{name_i}_{name_j}'

                B = min(z_i.size(0), z_j.size(0))
                z_i = z_i[:B]
                z_j = z_j[:B]
                labels_pair = labels[:B]

                if B <= 1:
                    continue

                z_i_aligned = self.alignment_proj[pair_key](z_i)
                z_j_aligned = self.alignment_proj[pair_key](z_j)

                sim_matrix = torch.mm(z_i_aligned, z_j_aligned.t()) * temp

                labels_i = labels_pair.unsqueeze(1)
                labels_j = labels_pair.unsqueeze(0)

                pos_mask = (labels_i == labels_j).float()
                neg_mask = 1 - pos_mask
                diag_mask = 1 - torch.eye(B, device=labels_pair.device)
                pos_mask = pos_mask * diag_mask
                neg_mask = neg_mask * diag_mask

                pos_sim = (sim_matrix.exp() * pos_mask).sum(dim=1)
                neg_sim = (sim_matrix.exp() * neg_mask).sum(dim=1)
                pos_sim = torch.clamp(pos_sim, min=1e-8)

                loss = -torch.log(pos_sim / (pos_sim + neg_sim + 1e-8))
                loss = loss[loss > -1e8].mean()

                if not torch.isnan(loss):
                    losses[pair_key] = loss
                    total_loss += loss
                    valid_pairs += 1

        avg_loss = total_loss / valid_pairs if valid_pairs > 0 else torch.tensor(0.0, device=z_gene.device)
        return avg_loss, losses




class AdaptiveFusionGate(nn.Module):
    """自适应融合门控 - 改进版"""

    def __init__(self, gene_dim: int = 768, sub_dim: int = 768,
                 graph_dim: int = 768, out_dim: int = 768, dropout: float = 0.25):
        super().__init__()

        self.gene_score = nn.Sequential(
            nn.Linear(gene_dim, 256), nn.ReLU(), nn.Dropout(dropout / 2),
            nn.Linear(256, 1)
        )
        self.sub_score = nn.Sequential(
            nn.Linear(sub_dim, 256), nn.ReLU(), nn.Dropout(dropout / 2),
            nn.Linear(256, 1)
        )
        self.graph_score = nn.Sequential(
            nn.Linear(graph_dim, 256), nn.ReLU(), nn.Dropout(dropout / 2),
            nn.Linear(256, 1)
        )

        self.gene_unc = nn.Sequential(
            nn.Linear(gene_dim, 256), nn.ReLU(), nn.Linear(256, 1), nn.Sigmoid()
        )
        self.sub_unc = nn.Sequential(
            nn.Linear(sub_dim, 256), nn.ReLU(), nn.Linear(256, 1), nn.Sigmoid()
        )
        self.graph_unc = nn.Sequential(
            nn.Linear(graph_dim, 256), nn.ReLU(), nn.Linear(256, 1), nn.Sigmoid()
        )

        self.cross_attn = nn.MultiheadAttention(out_dim, num_heads=8,
                                                dropout=dropout, batch_first=True)

        self.fusion = nn.Sequential(
            nn.Linear(gene_dim + sub_dim + graph_dim, out_dim * 2),
            nn.LayerNorm(out_dim * 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim), nn.ReLU(), nn.Dropout(dropout / 2)
        )

    def forward(self, gene_f: torch.Tensor, sub_f: torch.Tensor,
                graph_f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = gene_f.size(0)

        s_gene = self.gene_score(gene_f).view(B, 1)
        s_sub = self.sub_score(sub_f).view(B, 1)
        s_graph = self.graph_score(graph_f).view(B, 1)

        u_gene = self.gene_unc(gene_f).view(B, 1)
        u_sub = self.sub_unc(sub_f).view(B, 1)
        u_graph = self.graph_unc(graph_f).view(B, 1)

        scores = torch.cat([s_gene, s_sub, s_graph], dim=1)
        uncertainties = torch.cat([u_gene, u_sub, u_graph], dim=1)
        adjusted = scores * (1 - uncertainties)
        weights = F.softmax(adjusted, dim=1)

        modal_stack = torch.stack([gene_f, sub_f, graph_f], dim=1)
        attn_out, _ = self.cross_attn(modal_stack, modal_stack, modal_stack)
        attn_out = attn_out.mean(dim=1)

        concat = torch.cat([gene_f, sub_f, graph_f], dim=1)
        fused = self.fusion(concat)
        output = fused + attn_out

        return output, weights, uncertainties


class PredictionHead(nn.Module):
    """预测头 - 使用标签平滑和Focal Loss"""

    def __init__(self, in_dim: int = 768, num_classes: int = 2, dropout: float = 0.35):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout / 2),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(dropout / 4),
            nn.Linear(128, num_classes)
        )

        self.register_buffer('prototypes', torch.zeros(num_classes, 128))
        self.register_buffer('prototype_counts', torch.zeros(num_classes))
        self.prototype_initialized = False

        self.register_buffer('focal_alpha', torch.tensor([0.25, 0.75]))
        self.focal_gamma = 2.0
        self.label_smoothing = 0.1

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None,
                mode: str = 'train') -> Tuple:

        logits = self.classifier(x)

        if mode == 'train' and labels is not None:
            features = self.classifier[:-1](x)

            with torch.no_grad():
                for c in range(2):
                    mask = (labels == c)
                    if mask.any():
                        class_feats = features[mask].mean(dim=0)
                        if self.prototype_counts[c] == 0:
                            self.prototypes[c] = class_feats
                        else:
                            momentum = 0.9
                            self.prototypes[c] = (
                                    momentum * self.prototypes[c] +
                                    (1 - momentum) * class_feats
                            )
                        self.prototype_counts[c] += mask.sum().item()
                self.prototype_initialized = True

            num_classes = logits.size(-1)
            smoothed_labels = torch.zeros_like(logits).scatter_(
                1, labels.unsqueeze(1), 1.0
            )
            smoothed_labels = smoothed_labels * (1 - self.label_smoothing) +  self.label_smoothing / num_classes

            log_probs = F.log_softmax(logits, dim=1)
            ce_loss = -(smoothed_labels * log_probs).sum(dim=1).mean()

            probs = torch.exp(-ce_loss)
            focal_weight = (1 - probs) ** self.focal_gamma
            alpha = self.focal_alpha[labels]
            focal_loss = (alpha * focal_weight * ce_loss).mean()

            proto_loss = self.prototype_contrastive_loss(features, labels)
            total_loss = focal_loss + 0.05 * proto_loss

            return logits, total_loss, focal_loss, proto_loss

        return logits, torch.tensor(0.0, device=x.device), torch.tensor(0.0, device=x.device), torch.tensor(0.0, device=x.device)

    def prototype_contrastive_loss(self, features: torch.Tensor,
                                   labels: torch.Tensor) -> torch.Tensor:
        if not self.prototype_initialized:
            return torch.tensor(0.0, device=features.device)

        pos_protos = self.prototypes[labels]
        pos_sim = F.cosine_similarity(features, pos_protos, dim=1)

        neg_sims = []
        for c in range(len(self.prototypes)):
            neg_sim = F.cosine_similarity(
                features, self.prototypes[c].unsqueeze(0), dim=1
            )
            neg_sims.append(neg_sim.unsqueeze(1))

        all_sims = torch.cat(neg_sims, dim=1)
        neg_mask = torch.ones_like(all_sims, dtype=torch.bool)
        neg_mask.scatter_(1, labels.unsqueeze(1), False)
        neg_sim = (all_sims * neg_mask.float()).max(dim=1)[0]

        loss = -torch.log(
            torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim) + 1e-8) + 1e-8
        ).mean()

        return loss




class FusionEP(nn.Module):


    def __init__(self, config: Dict):
        super().__init__()

        self.config = config

        self.gene_dim = config.get('gene_dim', 36)
        self.sub_dim = config.get('sub_dim', 1024)
        self.graph_dim = config.get('graph_in_dim', 9)
        self.embed_dim = config.get('embed_dim', 768)

        self.gene_encoder = GeneEncoder(
            input_dim=self.gene_dim,
            embed_dim=self.embed_dim,
            depth=config.get('gene_depth', 6),
            num_heads=config.get('num_heads', 12),
            dropout=config.get('dropout', 0.15)
        )

        self.sub_encoder = SubcellularEncoder(
            input_dim=self.sub_dim,
            embed_dim=self.embed_dim,
            num_scales=4,
            dropout=config.get('dropout', 0.25)
        )

        self.graph_encoder = GraphEncoder(
            in_dim=self.graph_dim,
            hidden_dim=config.get('graph_hidden', 384),
            out_dim=self.embed_dim,
            num_layers=config.get('graph_layers', 4),
            dropout=config.get('dropout', 0.25)
        )

        self.contrast_loss = SupervisedContrastiveLoss(
            temperature=config.get('temperature', 0.05)
        )

        self.fusion_gate = AdaptiveFusionGate(
            gene_dim=self.embed_dim,
            sub_dim=self.embed_dim,
            graph_dim=self.embed_dim,
            out_dim=self.embed_dim,
            dropout=config.get('dropout', 0.25)
        )

        self.predictor = PredictionHead(
            in_dim=self.embed_dim,
            num_classes=config.get('num_classes', 2),
            dropout=config.get('dropout', 0.35)
        )

        self.lambda_contrast = config.get('lambda_contrast', 1.0)
        self.lambda_entropy = config.get('lambda_entropy', 0.001)

    def forward(self, batch: Dict, mode: str = 'train') -> Dict:
        if mode == 'train':
            batch = DataAugmentation.augment_batch(batch, training=True)

        gene_input = batch['gene']
        sub_input = batch['sub']
        protein_input = batch['protein']
        labels = batch['label']

        edge_index = batch.get('edge_index')
        edge_weight = batch.get('edge_weight')
        target_indices = batch.get('idx')

        gene_feat, gene_proj = self.gene_encoder(gene_input)
        sub_feat, sub_proj = self.sub_encoder(sub_input)

        if edge_index is not None and edge_index.numel() > 0:
            graph_feat, graph_proj, entropy_loss = self.graph_encoder(
                protein_input, edge_index, edge_weight, target_indices
            )
        else:
            graph_feat = torch.zeros_like(gene_feat)
            graph_proj = torch.zeros_like(gene_proj)
            entropy_loss = torch.tensor(0.0, device=gene_feat.device)

        B = gene_feat.size(0)
        if sub_feat.size(0) != B:
            sub_feat = sub_feat[:B]
            sub_proj = sub_proj[:B]
        if graph_feat.size(0) != B:
            graph_feat = graph_feat[:B]
            graph_proj = graph_proj[:B]

        contrast_loss, contrast_details = self.contrast_loss(
            gene_proj, sub_proj, graph_proj, labels
        )

        fused_feat, fusion_weights, uncertainties = self.fusion_gate(
            gene_feat, sub_feat, graph_feat
        )

        logits, pred_loss, focal_loss, proto_loss = self.predictor(
            fused_feat, labels, mode
        )

        if mode == 'train':
            total_loss = (
                    pred_loss +
                    self.lambda_contrast * contrast_loss +
                    self.lambda_entropy * entropy_loss
            )
        else:
            total_loss = torch.tensor(0.0, device=logits.device)

        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        return {
            'logits': logits,
            'probs': probs,
            'preds': preds,
            'fused_features': fused_feat,
            'fusion_weights': fusion_weights,
            'uncertainties': uncertainties,
            'losses': {
                'total': total_loss,
                'prediction': pred_loss,
                'contrastive': contrast_loss,
                'entropy': entropy_loss,
                'focal': focal_loss,
                'prototype': proto_loss
            }
        }




class FusionEPTrainerV2:


    def __init__(self, model: FusionEP, config: Dict, device: str = 'cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.history = {
            'train_loss': [], 'val_auc': [], 'val_f1': [], 'val_aupr': [],
            'test_auc': [], 'test_f1': [], 'test_aupr': [],
            'val_precision': [], 'val_recall': []  # 新增
        }
        self.best_model_state = None
        self.best_metric = 0

    def train_epoch(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> float:
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # 监控训练批次中的类别分布
        batch_class_counts = []

        for batch in dataloader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            
            # 统计类别分布
            labels = batch['label'].cpu().numpy()
            batch_class_counts.append(np.bincount(labels, minlength=2))

            optimizer.zero_grad()
            outputs = self.model(batch, mode='train')

            loss = outputs['losses']['total']

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        # 打印平均类别分布
        avg_dist = np.mean(batch_class_counts, axis=0)
        print(f"  Train batch class dist: {avg_dist.astype(int)} (ratio: {avg_dist[1]/avg_dist[0]:.2f})")
        
        return total_loss / max(num_batches, 1)

    def evaluate(self, dataloader: DataLoader) -> Dict:
        self.model.eval()
        all_probs = []
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                outputs = self.model(batch, mode='eval')

                all_probs.extend(outputs['probs'][:, 1].cpu().numpy())
                all_labels.extend(batch['label'].cpu().numpy())
                all_preds.extend(outputs['preds'].cpu().numpy())

        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)

        try:
            auc_roc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc_roc = 0.5

        try:
            auc_pr = average_precision_score(all_labels, all_probs)
        except ValueError:
            auc_pr = 0.0

        f1 = f1_score(all_labels, all_preds, zero_division=0)
        mcc = matthews_corrcoef(all_labels, all_preds)
        
        # 计算精确率和召回率
        from sklearn.metrics import precision_score, recall_score, confusion_matrix
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)

        return {
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'f1': f1,
            'mcc': mcc,
            'accuracy': (all_preds == all_labels).mean(),
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm
        }

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            test_loader: Optional[DataLoader] = None) -> Dict:

        param_groups = [
            {'params': self.model.gene_encoder.parameters(), 'lr': 5e-5, 'weight_decay': 0.01},
            {'params': self.model.sub_encoder.parameters(), 'lr': 1e-4, 'weight_decay': 0.01},
            {'params': self.model.graph_encoder.parameters(), 'lr': 1e-4, 'weight_decay': 0.01},
            {'params': self.model.fusion_gate.parameters(), 'lr': 2e-4, 'weight_decay': 0.001},
            {'params': self.model.predictor.parameters(), 'lr': 3e-4, 'weight_decay': 0.001},
            {'params': self.model.contrast_loss.parameters(), 'lr': 2e-4, 'weight_decay': 0.001},
        ]

        optimizer = torch.optim.AdamW(param_groups)

        num_epochs = self.config.get('epochs', 100)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )

        patience = self.config.get('patience', 30)
        patience_counter = 0

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader, optimizer)
            val_metrics = self.evaluate(val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['val_auc'].append(val_metrics['auc_roc'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['val_aupr'].append(val_metrics['auc_pr'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])

            # 综合指标：更关注F1和AUPR（对不平衡数据更敏感）
            current_metric = val_metrics['auc_roc'] * 0.4 + val_metrics['auc_pr'] * 0.35 + val_metrics['f1'] * 0.25

            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0

                if test_loader is not None:
                    test_metrics = self.evaluate(test_loader)
                    self.history['test_auc'].append(test_metrics['auc_roc'])
                    self.history['test_f1'].append(test_metrics['f1'])
                    self.history['test_aupr'].append(test_metrics['auc_pr'])
            else:
                patience_counter += 1

            if epoch % 5 == 0:
                print(f"Epoch {epoch}/{num_epochs} | "
                      f"Loss: {train_loss:.4f} | "
                      f"Val AUC: {val_metrics['auc_roc']:.4f} | "
                      f"Val AUPR: {val_metrics['auc_pr']:.4f} | "
                      f"Val F1: {val_metrics['f1']:.4f} | "
                      f"Val Recall: {val_metrics['recall']:.4f}")
                print(f"  Confusion Matrix: TN={val_metrics['confusion_matrix'][0,0]}, "
                      f"FP={val_metrics['confusion_matrix'][0,1]}, "
                      f"FN={val_metrics['confusion_matrix'][1,0]}, "
                      f"TP={val_metrics['confusion_matrix'][1,1]}")

            scheduler.step()

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return self.history




class ModelEnsemble:
    """模型集成"""

    def __init__(self, models: List[FusionEP], device: str = 'cuda'):
        self.models = [m.to(device).eval() for m in models]
        self.device = device

    def predict(self, dataloader: DataLoader) -> np.ndarray:
        all_probs = []

        with torch.no_grad():
            for model in self.models:
                model_probs = []
                for batch in dataloader:
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}
                    outputs = model(batch, mode='eval')
                    model_probs.append(outputs['probs'][:, 1].cpu())
                all_probs.append(torch.cat(model_probs).numpy())

        ensemble_probs = np.mean(all_probs, axis=0)
        return ensemble_probs


# ==================== 9. 主函数（关键修改：使用WeightedRandomSampler） ====================

def main():
    config = {
        'gene_dim': 36,
        'sub_dim': 1024,
        'graph_in_dim': 9,
        'embed_dim': 768,
        'gene_depth': 6,
        'graph_layers': 4,
        'num_heads': 12,
        'dropout': 0.2,
        'temperature': 0.05,
        'lambda_contrast': 1.0,
        'lambda_entropy': 0.001,
        'epochs': 10,#100
        'batch_size': 64,
        'patience': 30,
    }

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu' #if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")


    print("Loading real data...")
    dataset = DataLoaderHelper.load_from_paths(config)

    print(f"\\nModel config:")
    print(f"  gene_dim: {dataset.gene_dim}")
    print(f"  sub_dim: {dataset.sub_dim}")
    print(f"  graph_dim: {dataset.graph_dim}")
    print(f"  embed_dim: {config['embed_dim']}")

    # 划分数据
    n_total = len(dataset)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)

    indices = np.random.permutation(n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    print(f"\\n数据划分:")
    print(f"  训练集: {len(train_idx)} ({len(train_idx)/n_total:.1%})")
    print(f"  验证集: {len(val_idx)} ({len(val_idx)/n_total:.1%})")
    print(f"  测试集: {len(test_idx)} ({len(test_idx)/n_total:.1%})")

    # 创建DataLoader
    collator = GraphCollator(dataset, num_hops=2)



    
    # 为训练集创建加权采样器
    train_sampler = dataset.get_stratified_sampler(train_idx)
    
    # 训练集使用WeightedRandomSampler（注意：shuffle必须为False）
    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_idx),
        batch_size=config['batch_size'],
        sampler=train_sampler,
        shuffle=False,
        num_workers=0,
        collate_fn=collator
    )
    
    # 验证集和测试集保持原有方式
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, val_idx),
        batch_size=config['batch_size'] * 2,
        shuffle=False,
        num_workers=0,
        collate_fn=collator
    )
    test_loader = DataLoader(
        torch.utils.data.Subset(dataset, test_idx),
        batch_size=config['batch_size'] * 2,
        shuffle=False,
        num_workers=0,
        collate_fn=collator
    )
    
    print(f"\\n训练集采样器创建完成，每轮迭代 {len(train_loader)} 个batch")
    print("="*60 + "\\n")

    # 训练
    print("Creating model...")
    model = FusionEP(config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\\nStarting training with balanced sampling...")
    trainer = FusionEPTrainerV2(model, config, device)
    history = trainer.fit(train_loader, val_loader, test_loader)

    # 最终评估
    print("\\n" + "=" * 50)
    print("Final evaluation on test set:")
    test_metrics = trainer.evaluate(test_loader)
    for k, v in test_metrics.items():
        if k != 'confusion_matrix':
            print(f"  {k}: {v:.4f}")
    print(f"  Confusion Matrix:\n{test_metrics['confusion_matrix']}")

    # 训练多个模型进行集成
    print("\\n" + "=" * 50)
    print("Training ensemble models with balanced sampling...")

    ensemble_models = [model]
    for i in range(2):
        print(f"\\nTraining ensemble model {i + 2}/3...")
        set_seed(42 + i + 1)

        new_model = FusionEP(config)
        new_trainer = FusionEPTrainerV2(new_model, config, device)
        new_trainer.fit(train_loader, val_loader, None)
        ensemble_models.append(new_model)

    # 集成预测
    ensemble = ModelEnsemble(ensemble_models, device)
    ensemble_probs = ensemble.predict(test_loader)
    ensemble_preds = (ensemble_probs > 0.5).astype(int)

    # 计算集成指标
    test_labels = dataset.labels[test_idx].numpy()
    ensemble_auc = roc_auc_score(test_labels, ensemble_probs)
    ensemble_aupr = average_precision_score(test_labels, ensemble_probs)
    ensemble_f1 = f1_score(test_labels, ensemble_preds)
    
    # 计算混淆矩阵
    from sklearn.metrics import confusion_matrix
    ensemble_cm = confusion_matrix(test_labels, ensemble_preds)
    ensemble_mcc = matthews_corrcoef(test_labels, ensemble_preds)

    print(f"\\nEnsemble Results:")
    print(f"  AUC-ROC: {ensemble_auc:.4f}")
    print(f"  AUC-PR: {ensemble_aupr:.4f}")
    print(f"  F1: {ensemble_f1:.4f}")
    print(f"  MCC: {ensemble_mcc:.4f}")
    print(f"  Confusion Matrix:\n{ensemble_cm}")


    print("\\nDone!")
    return trainer, ensemble, history


if __name__ == "__main__":
    trainer, ensemble, history = main()

