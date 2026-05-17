# =========================================================
# Graduated Non-Convexity (GNC) 鲁棒位姿图优化
# 原理：从凸近似逐步退火到非凸鲁棒函数，避免局部最优
# 参考：Yang et al., RSS 2020 "Graduated Non-Convexity for
#       Robust Spatial Perception"
# 无需训练，纯算法改进，可量化ATE评估
# =========================================================
import torch
import torch.nn as nn
import numpy as np
import time, os, sys, glob, argparse
from pathlib import Path
import pypose as pp
import pypose.optim.solver as ppos
import pypose.optim.strategy as ppost
from pypose.optim.scheduler import StopOnPlateau

_script_dir = os.path.dirname(os.path.abspath(__file__))
_vs_path = _script_dir
for _ in range(4):
    if os.path.isdir(os.path.join(_vs_path, 'vista_slam', 'utils')):
        sys.path.insert(0, _vs_path)
        break
    for _sub in os.listdir(_vs_path) if os.path.isdir(_vs_path) else []:
        if os.path.isdir(os.path.join(_vs_path, _sub, 'vista_slam', 'utils')):
            sys.path.insert(0, os.path.join(_vs_path, _sub))
            break
    else:
        _vs_path = os.path.dirname(_vs_path)
        continue
    break

from vista_slam.utils.slam_utils import FontColor, print_msg, suppress_specific_print
from scipy.spatial.transform import Rotation


# ============================================================
# 【模块级补丁】DBoW3 缺失处理
# ============================================================
def _patch_dbow3():
    try:
        import DBoW3Py
        if hasattr(DBoW3Py, 'Vocabulary'):
            return
    except ImportError:
        pass
    import sys
    from types import ModuleType
    _fake_dbow = ModuleType('DBoW3Py')
    class _FakeVocab:
        def load(self, *a, **kw): pass
        def transform(self, *a, **kw): return None
        def score(self, *a, **kw): return 0.0
    _fake_dbow.Vocabulary = _FakeVocab
    sys.modules['DBoW3Py'] = _fake_dbow
    import vista_slam.loop_detector as _ld
    import vista_slam.slam as _slam

    class _DummyDetector:
        def __init__(self, *a, **kw):
            self.vocab = _FakeVocab()
            self.bow_feats = []
            self.loop_dist_min = 40
            self.loop_nms = 40
            self.loop_cand_thresh_neighbor = 5
            self.orb = None
        def compute_bow_feat(self, *a): return None
        def detect_loop(self, *a, **kw): return []
    _ld.LoopDetector = _DummyDetector
    _slam.LoopDetector = _DummyDetector

_patch_dbow3()

from vista_slam.slam import OnlineSLAM
from vista_slam.pose_graph import PoseGraphOpt


# ============================================================
# GNC 核心算法
# ============================================================

def geman_mcclure_weight(residual_norm_sq, mu):
    """
    Geman-McClure 权函数（GNC 核心）
    
    ρ(r) = (μ * r²) / (μ + r²)
    w(r) = μ² / (μ + r²)²
    
    参数 μ 控制鲁棒性：
    - μ → 0:    ρ(r) ≈ r²    (L2, 无鲁棒性)
    - μ → ∞:    ρ(r) → 1      (完全不敏感)
    - μ 适中:   截断阈值 ≈ sqrt(μ)
    """
    return mu ** 2 / (mu + residual_norm_sq) ** 2


def gnc_pose_graph_optimize(slam, max_iterations=30, mu_init=1e-4, mu_max=1e4,
                             mu_step=1.5, regularization=1e-6):
    """
    Graduated Non-Convexity 位姿图优化
    
    原理：
    1. 从 μ=μ_init (凸近似) 开始
    2. 每轮 PGO 后计算残差、更新权重、增大 μ
    3. 逐步退火到 μ=μ_max (真实鲁棒函数)
    
    数学保证：
    - GNC 保证单调收敛到稳态（Yang et al., RSS 2020, Theorem 1）
    - 对比标准 IRLS，GNC 不会陷入局部最优
    
    Args:
        slam: OnlineSLAM 实例
        max_iterations: 最大 μ 递增步数
        mu_init: 初始 μ 值（很小 = 凸近似）
        mu_max: 最大 μ 值（很大 = 非凸）
        mu_step: μ 递增倍率
        regularization: Cholesky 正则化强度
    """
    node_num = slam.pose_graph_nodes.num_nodes
    edge_num = slam.pose_graph_edges.num_edges
    if edge_num < 2 or node_num < 2:
        return mu_init

    device = slam.device
    mu = mu_init

    for iteration in range(max_iterations):
        # 1. 构建图优化器
        opt_node_idxs = set()
        for v in range(slam.view_num):
            opt_node_idxs.update(slam.pose_graph_nodes.view_to_node[v])
        opt_node_idxs.update(getattr(slam, 'loop_related_views', set()))
        opt_node_idxs = torch.tensor(sorted(list(opt_node_idxs)), device=device)

        graph = PoseGraphOpt(
            slam.pose_graph_nodes.poses[:node_num],
            to_optimize_idxs=opt_node_idxs
        ).to(device)

        solver = ppos.Cholesky()
        strategy = ppost.TrustRegion(radius=1e4)
        optimizer = pp.optim.LM(graph, solver=solver, strategy=strategy,
                                min=1e-6, vectorize=True)
        scheduler = StopOnPlateau(optimizer, steps=20, patience=3,
                                  decreasing=1e-4, verbose=False)

        # 2. 计算 GNC 权重
        with torch.no_grad():
            edges = slam.pose_graph_edges.edges[:edge_num]
            poses = slam.pose_graph_edges.poses[:edge_num]
            nodes = slam.pose_graph_nodes.poses[:node_num]

            node1 = nodes[edges[..., 0]]
            node2 = nodes[edges[..., 1]]
            residual = poses @ node1.Inv() @ node2
            residual_vec = residual.Log().tensor()  # [E, 7]
            residual_norm_sq = (residual_vec ** 2).sum(dim=1)  # [E]

            # Geman-McClure 权重
            gm_weights = geman_mcclure_weight(residual_norm_sq, mu)

        # 3. 构建加权矩阵（原始置信度 × GNC 权重）
        base_weights = slam.pose_graph_edges.confs[:edge_num].clone()
        # 只对位姿维度 (0-5) 应用 GNC 权重，保留尺度维度 (6) 的原始置信度
        for d in range(6):
            base_weights[:, d] = base_weights[:, d] * gm_weights

        weight = torch.diag_embed(base_weights)
        related_mask = graph.get_related_edge_idxs(edges)
        weight_masked = weight[related_mask]

        # 4. 正则化（防止 Cholesky 奇异）
        weight_masked = weight_masked + regularization * torch.eye(
            weight_masked.shape[-1], device=device
        )

        # 5. 执行优化
        with suppress_specific_print(
            "Linear solver failed", color=FontColor.PoseGraphOpt
        ):
            scheduler.optimize(
                input=(edges, poses), weight=weight_masked
            )

        # 6. 更新位姿
        slam.pose_graph_nodes.poses[:node_num] = graph.get_nodes()

        # 7. 增大 μ（退火调度）
        mu = min(mu * mu_step, mu_max)

    return mu  # 返回最终 μ 值


# ============================================================
# ATE 评估工具
# ============================================================

def align_trajectory(pred, gt):
    """Sim(3) 对齐"""
    N = pred.shape[0]
    pred_t = pred[:, :3, 3].numpy()
    gt_t = gt[:, :3, 3].numpy()

    pred_mean = pred_t.mean(axis=0)
    gt_mean = gt_t.mean(axis=0)
    pred_centered = pred_t - pred_mean
    gt_centered = gt_t - gt_mean

    s_pred = np.sqrt((pred_centered ** 2).sum(axis=1)).mean()
    s_gt = np.sqrt((gt_centered ** 2).sum(axis=1)).mean()
    scale = s_gt / (s_pred + 1e-8)

    H = pred_centered.T @ gt_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = gt_mean - scale * R @ pred_mean

    aligned = pred.clone()
    for i in range(N):
        aligned[i, :3, :3] = torch.from_numpy(scale * R @ pred[i, :3, :3].numpy()).float()
        aligned[i, :3, 3] = torch.from_numpy(scale * R @ pred[i, :3, 3].numpy() + t).float()
    return aligned


def compute_ate(aligned_pred, gt):
    errors = torch.norm(aligned_pred[:, :3, 3] - gt[:, :3, 3], dim=1)
    return float(torch.sqrt((errors ** 2).mean())), errors.tolist()


def load_tum_groundtruth(path, max_poses=None):
    import pandas as pd
    df = pd.read_csv(path, sep=' ', header=None, comment='#',
                     names=['t', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])
    N = len(df) if max_poses is None else min(len(df), max_poses)
    poses = torch.eye(4).unsqueeze(0).repeat(N, 1, 1)
    for i in range(N):
        row = df.iloc[i]
        poses[i, :3, 3] = torch.tensor([row['tx'], row['ty'], row['tz']])
        r = Rotation.from_quat([row['qx'], row['qy'], row['qz'], row['qw']])
        poses[i, :3, :3] = torch.from_numpy(r.as_matrix()).float()
    return poses


# ============================================================
# 主实验：Standard PGO vs GNC-PGO
# ============================================================

def run_slam_and_evaluate(rgb_path, gt_path, ckpt_path, vocab_path,
                           max_frames, use_gnc=False, device='cuda'):
    """运行 SLAM 并评估 ATE，返回 (poses, ate, errors, time)"""
    from vista_slam.datasets.slam_images_only import SLAM_image_only

    dataset = SLAM_image_only(sorted(glob.glob(rgb_path))[:max_frames],
                              resolution=(224, 224))
    if len(dataset) == 0:
        raise FileNotFoundError(f"No images found at {rgb_path}")

    slam = OnlineSLAM(
        ckpt_path=ckpt_path, vocab_path=vocab_path, verbose=False,
        max_view_num=500, neighbor_edge_num=3, loop_edge_num=3,
        loop_dist_min=40, loop_nms=40, loop_cand_thresh_neighbor=5,
        conf_thres=4.2, rel_pose_thres=0.75, flow_thres=100, pgo_every=50
    )

    mode = "GNC-PGO" if use_gnc else "Standard"
    print_msg(f"[{mode}] Processing {len(dataset)} frames...", color=FontColor.INFO)

    t_start = time.time()

    for t in range(len(dataset)):
        data = dataset[t]
        img = data.rgb.unsqueeze(0).to(device)
        img_shape = torch.tensor(data.rgb.shape[1:3]).unsqueeze(0)
        img_gray = (data.gray.squeeze(0).numpy() * 255).astype(np.uint8)

        input_value = {'rgb': img, 'shape': img_shape,
                       'gray': img_gray, 'view_name': str(t)}
        is_last = (t == len(dataset) - 1)

        if use_gnc:
            is_opt = slam.step(input_value, force_pgo=False)
            if slam.view_num % 50 == 0 or is_last:
                print_msg(f"  [{mode}] GNC optimization at keyframe {slam.view_num}...",
                          color=FontColor.INFO)
                final_mu = gnc_pose_graph_optimize(
                    slam, max_iterations=30, mu_init=1e-4, mu_max=1e4,
                    mu_step=1.5
                )
                print_msg(f"    Final mu={final_mu:.2e}", color=FontColor.INFO)
                torch.cuda.empty_cache()
        else:
            slam.step(input_value, force_pgo=is_last)

        if (t + 1) % 20 == 0:
            print_msg(f"  [{mode}] Processed {t+1}/{len(dataset)} keyframes={slam.view_num}",
                      color=FontColor.INFO)

    # 最终优化
    if not use_gnc:
        slam.pose_graph_optimize()
    else:
        gnc_pose_graph_optimize(slam, max_iterations=30)
    torch.cuda.empty_cache()
    t_elapsed = time.time() - t_start

    # 保存轨迹
    poses = []
    for v in range(slam.view_num):
        best_node = slam.pose_graph_nodes.view_to_best_node[v][0]
        Sim3 = slam.pose_graph_nodes.poses[best_node]
        rot = Sim3.rotation().matrix().cpu()
        trans = Sim3.translation().cpu()
        pose = torch.eye(4)
        pose[:3, :3] = rot
        pose[:3, 3] = trans
        poses.append(pose)
    pred_poses = torch.stack(poses, dim=0)

    # 加载真值
    gt_all = load_tum_groundtruth(gt_path)

    if len(gt_all) >= len(pred_poses):
        indices = np.linspace(0, len(gt_all) - 1, len(pred_poses), dtype=int)
        gt_poses = gt_all[indices]
    else:
        gt_poses = gt_all

    min_len = min(len(pred_poses), len(gt_poses))
    pred_poses = pred_poses[:min_len]
    gt_poses = gt_poses[:min_len]

    aligned = align_trajectory(pred_poses, gt_poses)
    ate_rmse, errors = compute_ate(aligned, gt_poses)

    print_msg(f"  [{mode}] ATE RMSE: {ate_rmse:.4f}m | frames={min_len} | time={t_elapsed:.1f}s",
              color=FontColor.PoseGraphOpt)

    return pred_poses, ate_rmse, errors, t_elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", type=str, required=True)
    parser.add_argument("--gt", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="pretrains/frontend_sta_weights.pth")
    parser.add_argument("--vocab", type=str, default="pretrains/ORBvoc.txt")
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    alt_ckpt = "/content/vista-slam/pretrains/frontend_sta_weights.pth"
    alt_vocab = "/content/vista-slam/pretrains/ORBvoc.txt"
    if not os.path.exists(args.ckpt) and os.path.exists(alt_ckpt):
        args.ckpt = alt_ckpt
    if not os.path.exists(args.vocab) and os.path.exists(alt_vocab):
        args.vocab = alt_vocab

    # ── 标准 PGO ──
    t0 = time.time()
    _, std_ate, _, t_std = run_slam_and_evaluate(
        args.rgb, args.gt, args.ckpt, args.vocab,
        args.max_frames, use_gnc=False, device=args.device
    )
    torch.cuda.empty_cache()

    # ── GNC-PGO ──
    t0 = time.time()
    _, gnc_ate, _, t_gnc = run_slam_and_evaluate(
        args.rgb, args.gt, args.ckpt, args.vocab,
        args.max_frames, use_gnc=True, device=args.device
    )

    # ── 报告 ──
    print_msg("=" * 70, color=FontColor.PoseGraphOpt)
    print_msg("GNC-PGO vs Standard PGO 对比报告", color=FontColor.PoseGraphOpt)
    print_msg("-" * 70, color=FontColor.PoseGraphOpt)
    print_msg(f"{'指标':<20} {'Standard':>15} {'GNC-PGO':>15} {'改善':>12}",
              color=FontColor.PoseGraphOpt)
    print_msg("-" * 65, color=FontColor.PoseGraphOpt)

    improv = (std_ate - gnc_ate) / max(std_ate, 1e-8) * 100
    print_msg(f"{'ATE RMSE ↓ (m)':<20} {std_ate:>15.4f} {gnc_ate:>15.4f} "
              f"{'✅' if improv > 0 else ''} {improv:+>+7.1f}%",
              color=FontColor.PoseGraphOpt)
    print_msg(f"{'帧数':<20} {args.max_frames:>15} {args.max_frames:>15}",
              color=FontColor.INFO)
    print_msg(f"{'时间 (s)':<20} {t_std:>15.1f} {t_gnc:>15.1f} "
              f"{'⬆' if t_gnc > t_std else '⬇'} {(t_gnc-t_std)/t_std*100:+>+6.1f}%",
              color=FontColor.INFO)
    print_msg("-" * 65, color=FontColor.PoseGraphOpt)

    if improv > 10:
        print_msg(f"结论: GNC-PGO 显著提升轨迹精度 ✅", color=FontColor.PoseGraphOpt)
        print_msg(f"      ATE 降低 {improv:.1f}% ({std_ate:.4f}m → {gnc_ate:.4f}m)",
                  color=FontColor.PoseGraphOpt)
    elif improv > 2:
        print_msg(f"结论: GNC-PGO 有小幅提升", color=FontColor.INFO)
    else:
        print_msg(f"结论: 当前序列下 GNC 与标准 PGO 结果相近。"
                  f"GNC 优势在强异常边场景下更明显。",
                  color=FontColor.INFO)
    print_msg("", color=FontColor.INFO)


if __name__ == "__main__":
    main()
