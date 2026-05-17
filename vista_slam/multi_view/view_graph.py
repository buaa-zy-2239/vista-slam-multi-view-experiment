import torch
import torch.nn as nn
import torch.nn.functional as F


class ViewRelationGraph:
    def __init__(self, edge_threshold=0.3):
        self.edge_threshold = edge_threshold
        self.nodes = {}
        self.edges = {}
        self.edge_weights = {}

    def build(self, features, similarity_matrix=None):
        self.nodes.clear()
        self.edges.clear()
        self.edge_weights.clear()

        num_views = len(features)
        for i in range(num_views):
            self.nodes[i] = features[i]

        if similarity_matrix is None:
            similarity_matrix = self._compute_similarity(features)

        for i in range(num_views):
            for j in range(i+1, num_views):
                sim = similarity_matrix[i, j]
                if sim > self.edge_threshold:
                    self.edges[(i, j)] = sim
                    self.edge_weights[(i, j)] = sim

    def _compute_similarity(self, features):
        num_views = len(features)
        sim = torch.zeros(num_views, num_views)
        for i in range(num_views):
            for j in range(num_views):
                fi = features[i].mean(dim=0) if features[i].ndim > 1 else features[i]
                fj = features[j].mean(dim=0) if features[j].ndim > 1 else features[j]
                sim[i, j] = F.cosine_similarity(fi.unsqueeze(0), fj.unsqueeze(0)).item()
        return sim

    def get_top_k_neighbors(self, view_idx, k=3):
        scores = []
        for (i, j), w in self.edge_weights.items():
            if i == view_idx:
                scores.append((j, w))
            elif j == view_idx:
                scores.append((i, w))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, w in scores[:k]]


class ViewGraphAggregator(nn.Module):
    def __init__(self, feature_dim=768, hidden_dim=512, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            GraphAttentionLayer(
                in_dim=feature_dim if i == 0 else hidden_dim,
                out_dim=hidden_dim if i < num_layers - 1 else feature_dim,
                num_heads=4,
                concat=(i < num_layers - 1)
            )
            for i in range(num_layers)
        ])

    def forward(self, node_features, edge_index):
        x = node_features
        for layer in self.layers:
            x = x + layer(x, edge_index)
        return x


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, concat=True):
        super().__init__()
        self.num_heads = num_heads
        self.concat = concat
        self.W = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.a = nn.Parameter(torch.zeros(2 * out_dim, num_heads))
        nn.init.xavier_uniform_(self.a)

    def forward(self, x, edge_index):
        h = self.W(x).view(x.shape[0], -1, self.num_heads)
        src, dst = edge_index[0], edge_index[1]
        edge_feat = torch.cat([h[src], h[dst]], dim=-1).permute(1, 0, 2)
        attn = torch.matmul(edge_feat.transpose(-1, -2), self.a).squeeze(-1).T
        attn = F.softmax(attn, dim=1)
        msgs = h[dst] * attn.unsqueeze(-1)
        out = torch.zeros_like(h)
        out.index_add_(0, src, msgs)
        if self.concat:
            return out.view(x.shape[0], -1)
        return out.mean(dim=-1)
