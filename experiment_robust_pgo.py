# =========================================================
# 迭代式鲁棒位姿图优化 (IR-PGO)
# 原理：多次迭代优化 + 残差重加权（M-估计）
# 无需训练，纯算法改进，可量化ATE评估
# =========================================================
import torch
import torch.nn as nn
import numpy as np
import time, os, sys, glob, argparse
from pathlib import Path

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

from vista_slam.slam import OnlineSLAM
from vista_slam.utils.slam_utils import FontColor, print_msg
import pypose as pp
from scipy.spatial.transform import Rotation


# ============================================================
# 核心改进：迭代重加权位姿图优化
# ============================================================

def tukey_weight(residual_norm, c=4.6851):
    """
    Tukey's bisquare 权函数
    对残差小的边给予权重1，残差大的边逐渐降到0
    
    c=4.6851: 95% 渐近效率（相对最小二乘）
    当残差 > c*sigma 时，权重为0（完全剔除）
    """
    r = residual_norm / (c * 1.4826)  # 1.4826 = 高斯分布的MAD标定
    mask = r <= 1.0
    w = torch.zeros_like(r)
    w[mask] = (1 - r[mask]**2)**2
    return w


def huber_weight(residual_norm, delta=1.345):
    """
    Huber 权函数
    小残差用L2，大残差用L1
    delta=1.345: 95% 渐近效率
    """
    r = residual_norm / delta
    w = torch.ones_like(r)
    w[r > 1] = 1.0 / r[r > 1]
    return w


def robust_pose_graph_optimize(slam, num_iterations=5, kernel='tukey'):
    """
    迭代式鲁棒位姿图优化
    
    替代 slam.pose_graph_optimize()
    
    流程：
    for iter in range(num_iterations):
        1. 标准 PGO
        2. 计算每条边的残差范数
        3. 用鲁棒核函数计算新权重
        4. 下一轮使用新权重重新优化
    """
    node_num = slam.pose_graph_nodes.num_nodes
    edge_num = slam.pose_graph_edges.num_edges
    if edge_num == 0:
        return

    device = slam.device
    kernel_fn = tukey_weight if kernel == 'tukey' else huber_weight

    for iteration in range(num_iterations):
        # 1. 构建图优化问题
        opt_node_idxs = set()
        for v in range(slam.view_num):
            opt_node_idxs.update(slam.pose_graph_nodes.view_to_node[v])
        opt_node_idxs.update(getattr(slam, 'loop_related_views', set()))
        opt_node_idxs = torch.tensor(sorted(list(opt_node_idxs)), device=device)

        graph = PoseGraphOptIR(
            slam.pose_graph_nodes.poses[:node_num],
            to_optimize_idxs=opt_node_idxs
        ).to(device)

        solver = pp.optim.solver.Cholesky()
        strategy = pp.optim.strategy.TrustRegion(radius=1e4)
        optimizer = pp.optim.LM(graph, solver=solver, strategy=strategy,
                                min=1e-6, vectorize=True)
        scheduler = pp.optim.scheduler.StopOnPlateau(
            optimizer, steps=20, patience=3, decreasing=1e-4, verbose=False
        )

        # 2. 获取当前权重（首次=原始权重，后续=鲁棒权重）
        if iteration == 0:
            weight = torch.diag_embed(slam.pose_graph_edges.confs[:edge_num])
        else:
            # 用新权重
            weight = torch.diag_embed(new_weights)

        related_mask = graph.get_related_edge_idxs(slam.pose_graph_edges.edges[:edge_num])
        weight_masked = weight[related_mask]

        # 3. 执行优化
        scheduler.optimize(
            input=(slam.pose_graph_edges.edges[:edge_num],
                   slam.pose_graph_edges.poses[:edge_num]),
            weight=weight_masked
        )

        # 4. 更新节点位姿
        slam.pose_graph_nodes.poses[:node_num] = graph.get_nodes()

        # 5. 计算每条边的残差范数（用于下一轮重加权）
        if iteration < num_iterations - 1:
            with torch.no_grad():
                edges = slam.pose_graph_edges.edges[:edge_num]
                poses = slam.pose_graph_edges.poses[:edge_num]
                nodes = slam.pose_graph_nodes.poses[:node_num]

                node1 = nodes[edges[..., 0]]
                node2 = nodes[edges[..., 1]]
                error = poses @ node1.Inv() @ node2
                residuals = error.Log().tensor()  # [E, 7]
                residual_norms = torch.norm(residuals, dim=1)  # [E]

                # 重加权
                robust_weights = kernel_fn(residual_norms)
                new_weights = slam.pose_graph_edges.confs[:edge_num].clone()
                # 将原始权重的第7列（尺度权重）用鲁棒权重替换
                new_weights[:, 6] = new_weights[:, 6] * robust_weights
                # 位姿权重也替换
                for d in range(6):
                    new_weights[:, d] = new_weights[:, d] * robust_weights


class PoseGraphOptIR(nn.Module):
    """可重复使用的位姿图优化器（与PoseGraphOpt逻辑相同，但不处理fixed节点缓存问题）"""
    def __init__(self, nodes, to_optimize_idxs="all"):
        super().__init__()
        device = nodes.device
        if to_optimize_idxs == "all":
            to_optimize_idxs = torch.arange(nodes.shape[0], device=device)

        self.idxs_opt = to_optimize_idxs.long().to(device)
        all_idxs = torch.arange(nodes.shape[0], device=device)
        self.idxs_fix = all_idxs[~torch.isin(all_idxs, self.idxs_opt)]

        self.register_buffer("opt_map", -torch.ones(nodes.shape[0], dtype=torch.long, device=device))
        self.register_buffer("fix_map", -torch.ones(nodes.shape[0], dtype=torch.long, device=device))
        self.opt_map[self.idxs_opt] = torch.arange(len(self.idxs_opt), device=device)
        self.fix_map[self.idxs_fix] = torch.arange(len(self.idxs_fix), device=device)

        self.nodes_opt = pp.Parameter(nodes[self.idxs_opt]).to(device)
        self.nodes_fixed = nodes[self.idxs_fix]

    def get_nodes(self):
        n_all = self.idxs_opt.shape[0] + self.idxs_fix.shape[0]
        nodes = pp.identity_Sim3(n_all).to(self.nodes_opt.device)
        nodes[self.idxs_opt] = self.nodes_opt.detach().clone()
        nodes[self.idxs_fix] = self.nodes_fixed.detach().clone()
        return nodes

    def forward(self, edges, poses):
        in_opt = torch.isin(edges, self.idxs_opt)
        both_opt   = in_opt[:, 0] & in_opt[:, 1]
        first_opt  = in_opt[:, 0] & ~in_opt[:, 1]
        second_opt = ~in_opt[:, 0] & in_opt[:, 1]

        error_parts = []
        if both_opt.any():
            i = self.opt_map[edges[both_opt][:, 0]]
            j = self.opt_map[edges[both_opt][:, 1]]
            error_parts.append(poses[both_opt] @ self.nodes_opt[i].Inv() @ self.nodes_opt[j])
        if first_opt.any():
            i = self.opt_map[edges[first_opt][:, 0]]
            j = self.fix_map[edges[first_opt][:, 1]]
            error_parts.append(poses[first_opt] @ self.nodes_opt[i].Inv() @ self.nodes_fixed[j])
        if second_opt.any():
            i = self.fix_map[edges[second_opt][:, 0]]
            j = self.opt_map[edges[second_opt][:, 1]]
            error_parts.append(poses[second_opt] @ self.nodes_fixed[i].Inv() @ self.nodes_opt[j])

        if error_parts:
            error = torch.cat(error_parts, dim=0)
        else:
            B = poses.shape[0]
            error = pp.identity_Sim3(B).to(self.nodes_opt.device)
        return error.Log().tensor()

    def get_related_edge_idxs(self, edges):
        in_opt = torch.isin(edges, self.idxs_opt)
        return in_opt[:, 0] | in_opt[:, 1]


# ============================================================
# ATE 评估工具
# ============================================================
def align_trajectory(pred, gt):
    """
    Sim(3) 对齐：将预测轨迹对齐到真值坐标系
    pred, gt: [N, 4, 4]
    返回：对齐后的预测轨迹 [N, 4, 4]
    """
    N = pred.shape[0]
    pred_t = pred[:, :3, 3].numpy()
    gt_t = gt[:, :3, 3].numpy()

    # Procrustes 对齐（相似变换）
    pred_mean = pred_t.mean(axis=0)
    gt_mean = gt_t.mean(axis=0)

    pred_centered = pred_t - pred_mean
    gt_centered = gt_t - gt_mean

    # 尺度
    s_pred = np.sqrt((pred_centered ** 2).sum(axis=1)).mean()
    s_gt = np.sqrt((gt_centered ** 2).sum(axis=1)).mean()
    scale = s_gt / (s_pred + 1e-8)

    # 旋转
    H = pred_centered.T @ gt_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T

    # 平移
    t = gt_mean - scale * R @ pred_mean

    # 应用
    aligned = pred.clone()
    for i in range(N):
        aligned[i, :3, :3] = torch.from_numpy(scale * R @ pred[i, :3, :3].numpy()).float()
        aligned[i, :3, 3] = torch.from_numpy(scale * R @ pred[i, :3, 3].numpy() + t).float()
    return aligned


def compute_ate(aligned_pred, gt):
    """计算ATE RMSE（米）"""
    errors = torch.norm(aligned_pred[:, :3, 3] - gt[:, :3, 3], dim=1)
    return float(torch.sqrt((errors ** 2).mean())), errors.tolist()


def load_tum_groundtruth(path, max_poses=None):
    """
    加载 TUM 真值轨迹（groundtruth.txt）
    格式: timestamp tx ty tz qx qy qz qw
    返回: [N, 4, 4] 位姿矩阵
    """
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


def load_tum_timestamps(path):
    """加载 TUM rgb.txt 的时间戳列表"""
    import pandas as pd
    df = pd.read_csv(path, sep=' ', header=None, comment='#', names=['t', 'file'])
    return df['t'].values


# ============================================================
# 主实验
# ============================================================

def run_slam_and_evaluate(rgb_path, gt_path, timestamps_path,
                           ckpt_path, vocab_path, max_frames,
                           use_robust=False, robust_iters=5, device='cuda'):
    """
    运行 SLAM 并评估 ATE

    返回: (poses, ate, per_frame_errors)
    """
    from vista_slam.datasets.slam_images_only import SLAM_image_only

    # 【兼容】如果 DBoW3 不可用，打补丁绕过回环检测
    try:
        import DBoW3Py
        _dbow_ok = True
    except ImportError:
        _dbow_ok = False
    if not _dbow_ok:
        import vista_slam.loop_detector as _ld
        class _DummyVocab:
            def load(self, *a, **kw): pass
            def transform(self, *a, **kw): return None
            def score(self, *a, **kw): return 0.0
        class _DummyDetector:
            def __init__(self, *a, **kw):
                self.vocab = _DummyVocab()
                self.bow_feats = []
                self.loop_dist_min = 40
                self.loop_nms = 40
                self.loop_cand_thresh_neighbor = 5
                self.orb = None
            def compute_bow_feat(self, *a): return None
            def detect_loop(self, *a, **kw): return []
        _ld.LoopDetector = _DummyDetector

    dataset = SLAM_image_only(sorted(glob.glob(rgb_path))[:max_frames],
                              resolution=(224, 224))

    slam = OnlineSLAM(
        ckpt_path=ckpt_path, vocab_path=vocab_path, verbose=False,
        max_view_num=100, neighbor_edge_num=3, loop_edge_num=3,
        loop_dist_min=40, loop_nms=40, loop_cand_thresh_neighbor=5,
        conf_thres=4.2, rel_pose_thres=0.75, flow_thres=100, pgo_every=50
    )

    mode = "Robust-PGO" if use_robust else "Standard"
    print_msg(f"[{mode}] Processing {len(dataset)} frames...", color=FontColor.INFO)

    for t in range(len(dataset)):
        data = dataset[t]
        img = data.rgb.unsqueeze(0).to(device)
        img_shape = torch.tensor(data.rgb.shape[1:3]).unsqueeze(0)
        img_gray = (data.gray.squeeze(0).numpy() * 255).astype(np.uint8)

        input_value = {
            'rgb': img, 'shape': img_shape,
            'gray': img_gray, 'view_name': str(t)
        }

        is_last = (t == len(dataset) - 1)

        if use_robust:
            # 替换为鲁棒PGO
            is_opt = slam.step(input_value, force_pgo=False)
            if slam.view_num % 50 == 0 or is_last:
                print_msg(f"  [{mode}] Running robust PGO at keyframe {slam.view_num}...",
                          color=FontColor.INFO)
                robust_pose_graph_optimize(slam, num_iterations=robust_iters)
                torch.cuda.empty_cache()
        else:
            slam.step(input_value, force_pgo=is_last)

        if (t + 1) % 10 == 0:
            print_msg(f"  [{mode}] Processed {t+1}/{len(dataset)} keyframes={slam.view_num}",
                      color=FontColor.INFO)

    # 最终优化
    if not use_robust:
        slam.pose_graph_optimize()
    else:
        robust_pose_graph_optimize(slam, num_iterations=robust_iters)

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

    # 加载真值并按时间戳对齐帧
    gt_all = load_tum_groundtruth(gt_path)

    # 稀疏采样真值到与估计轨迹相同的帧数
    # 简单方法：均匀采样
    if len(gt_all) >= len(pred_poses):
        indices = np.linspace(0, len(gt_all) - 1, len(pred_poses), dtype=int)
        gt_poses = gt_all[indices]
    else:
        gt_poses = gt_all

    # 对齐
    min_len = min(len(pred_poses), len(gt_poses))
    pred_poses = pred_poses[:min_len]
    gt_poses = gt_poses[:min_len]

    aligned = align_trajectory(pred_poses, gt_poses)
    ate_rmse, errors = compute_ate(aligned, gt_poses)

    print_msg(f"  [{mode}] ATE RMSE: {ate_rmse:.4f}m | frames={min_len}",
              color=FontColor.PoseGraphOpt)

    return pred_poses, ate_rmse, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", type=str, required=True)
    parser.add_argument("--gt", type=str, required=True,
                        help="Path to groundtruth.txt")
    parser.add_argument("--ckpt", type=str, default="pretrains/frontend_sta_weights.pth")
    parser.add_argument("--vocab", type=str, default="pretrains/ORBvoc.txt")
    parser.add_argument("--max-frames", type=int, default=50)
    parser.add_argument("--robust-iters", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    alt_ckpt = "/content/vista-slam/pretrains/frontend_sta_weights.pth"
    alt_vocab = "/content/vista-slam/pretrains/ORBvoc.txt"
    if not os.path.exists(args.ckpt) and os.path.exists(alt_ckpt):
        args.ckpt = alt_ckpt
    if not os.path.exists(args.vocab) and os.path.exists(alt_vocab):
        args.vocab = alt_vocab

    # ── 标准PGO ──
    t0 = time.time()
    _, std_ate, _ = run_slam_and_evaluate(
        args.rgb, args.gt, None, args.ckpt, args.vocab,
        args.max_frames, use_robust=False, device=args.device
    )
    t_std = time.time() - t0
    torch.cuda.empty_cache()

    # ── 鲁棒PGO ──
    t0 = time.time()
    _, robust_ate, _ = run_slam_and_evaluate(
        args.rgb, args.gt, None, args.ckpt, args.vocab,
        args.max_frames, use_robust=True, robust_iters=args.robust_iters,
        device=args.device
    )
    t_robust = time.time() - t0

    # ── 报告 ──
    print_msg("=" * 70, color=FontColor.PoseGraphOpt)
    print_msg("IR-PGO vs Standard PGO 对比报告", color=FontColor.PoseGraphOpt)
    print_msg("-" * 70, color=FontColor.PoseGraphOpt)
    print_msg(f"{'指标':<20} {'Standard':>15} {'Robust-PGO':>15} {'改善':>12}",
              color=FontColor.PoseGraphOpt)
    print_msg("-" * 65, color=FontColor.PoseGraphOpt)
    improv = (std_ate - robust_ate) / max(std_ate, 1e-8) * 100
    print_msg(f"{'ATE RMSE ↓ (m)':<20} {std_ate:>15.4f} {robust_ate:>15.4f} "
              f"{'✅' if improv > 0 else ''} {improv:+>+7.1f}%",
              color=FontColor.PoseGraphOpt)
    print_msg(f"{'帧数':<20} {args.max_frames:>15} {args.max_frames:>15}",
              color=FontColor.INFO)
    print_msg(f"{'时间 (s)':<20} {t_std:>15.1f} {t_robust:>15.1f} "
              f"{'⬆' if t_robust > t_std else '⬇'} {(t_robust-t_std)/t_std*100:+>+6.1f}%",
              color=FontColor.INFO)
    print_msg("-" * 65, color=FontColor.PoseGraphOpt)

    if improv > 5:
        print_msg(f"结论: 迭代式鲁棒PGO显著提升轨迹精度 ✅", color=FontColor.PoseGraphOpt)
        print_msg(f"      ATE 降低 {improv:.1f}% ({std_ate:.4f}m → {robust_ate:.4f}m)",
                  color=FontColor.PoseGraphOpt)
    elif improv > 0:
        print_msg(f"结论: 迭代式鲁棒PGO有小幅提升", color=FontColor.INFO)
    else:
        print_msg(f"结论: 当前条件下鲁棒PGO未超越标准PGO", color=FontColor.WARNING)

    print_msg("", color=FontColor.INFO)


if __name__ == "__main__":
    main()
