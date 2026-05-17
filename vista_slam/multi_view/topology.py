import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TopologySelector:
    def __init__(self, config=None):
        cfg = config or {}
        self.available_gpu_memory = cfg.get('gpu_memory_gb', 8.0)
        self.target_fps = cfg.get('target_fps', 15)
        self.force_topology = cfg.get('force_topology', None)

    def select(self, num_views, scene_complexity=0.5):
        if self.force_topology is not None:
            return self.force_topology

        if self.available_gpu_memory < 4 or self.target_fps > 30:
            return 'baseline'
        if num_views <= 3:
            return 'star'
        if num_views <= 5 and self.available_gpu_memory >= 8:
            return 'hierarchical' if scene_complexity > 0.6 else 'sparse_hierarchical'
        if num_views > 5 and self.available_gpu_memory >= 16:
            return 'fully_connected'
        return 'hierarchical'


class SparseMultiViewAttention(nn.Module):
    def __init__(self, max_pairs=8):
        super().__init__()
        self.max_pairs = max_pairs

    def forward(self, features, positions):
        num_views = len(features)
        if num_views <= 2:
            return list(zip(features, features))

        essential = [(i, i+1) for i in range(num_views-1)]
        candidates = []
        for i in range(num_views):
            for j in range(i+2, num_views):
                score = self._pair_score(features[i], features[j])
                candidates.append((i, j, score))
        candidates.sort(key=lambda x: x[2], reverse=True)

        n_extra = max(0, self.max_pairs - len(essential))
        selected = essential[:]
        for i, j, s in candidates[:n_extra]:
            selected.append((i, j))

        results = {}
        for i, j in selected:
            key = (min(i, j), max(i, j))
            if key not in results:
                results[key] = (features[i], features[j])
        return list(results.values())

    def _pair_score(self, feat_i, feat_j):
        fi = feat_i.mean(dim=1) if feat_i.ndim == 3 else feat_i
        fj = feat_j.mean(dim=1) if feat_j.ndim == 3 else feat_j
        sim = F.cosine_similarity(fi, fj, dim=-1).mean().item()
        return 1.0 - sim
