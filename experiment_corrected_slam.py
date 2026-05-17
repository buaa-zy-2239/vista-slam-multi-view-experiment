"""
修正版 SLAM 实验框架 —— 正确复现 run.py 的完整关键帧选择流程

核心改进（对比 experiment_gnc_pgo.py）：
  1. 复现 evaluation_tumrgbd.py 的 stride 关键帧选择
  2. 支持 SLAM_TUMRGBD（含内参/深度/GT）和 SLAM_image_only（纯图像+外部GT）
  3. Standard PGO 与 GNC-PGO 使用完全相同的输入（公平对比）
  4. 使用 evo 库的 full_traj_eval 进行标准 ATE 评估
  5. 详细的计时和对比报告

用法：
  # TUM RGB-D 数据集模式（推荐，包含完整真值）
  python experiment_corrected_slam.py \
      --dataset-folder /data/tum/rgbd_dataset_freiburg2_desk \
      --config configs/tumrgbd.yaml \
      --output output/corrected_slam

  # 纯图像模式（需提供TUM格式的groundtruth.txt）
  python experiment_corrected_slam.py \
      --images "/data/tum/rgbd_dataset_freiburg2_desk/rgb/*.png" \
      --gt /data/tum/rgbd_dataset_freiburg2_desk/groundtruth.txt \
      --config configs/tumrgbd.yaml \
      --output output/corrected_slam_image_only
"""
import torch
import numpy as np
import time, os, sys, glob, argparse, yaml, munch, json
from pathlib import Path
import torch.backends.cudnn as cudnn

# 为 Colab 环境添加路径（向上搜索父目录中的 vista_slam 包）
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

from vista_slam.utils.slam_utils import FontColor, print_msg
from vista_slam.slam import OnlineSLAM
from vista_slam.eval.eval_traj import full_traj_eval


# ============================================================
# GNC 鲁棒位姿图优化（从 experiment_gnc_pgo.py 移植）
# ============================================================

from vista_slam.pose_graph import PoseGraphOpt
import pypose as pp
import pypose.optim.solver as ppos
import pypose.optim.strategy as ppost
from pypose.optim.scheduler import StopOnPlateau
from vista_slam.utils.slam_utils import suppress_specific_print


def geman_mcclure_weight(residual_norm_sq, mu):
    return mu ** 2 / (mu + residual_norm_sq) ** 2


def gnc_pose_graph_optimize(slam, max_iterations=30, mu_init=1e-4, mu_max=1e4,
                             mu_step=1.5, regularization=1e-6):
    node_num = slam.pose_graph_nodes.num_nodes
    edge_num = slam.pose_graph_edges.num_edges
    if edge_num < 2 or node_num < 2:
        return mu_init

    device = slam.device
    mu = mu_init

    for iteration in range(max_iterations):
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

        with torch.no_grad():
            edges = slam.pose_graph_edges.edges[:edge_num]
            poses = slam.pose_graph_edges.poses[:edge_num]
            nodes = slam.pose_graph_nodes.poses[:node_num]

            node1 = nodes[edges[..., 0]]
            node2 = nodes[edges[..., 1]]
            residual = poses @ node1.Inv() @ node2
            residual_vec = residual.Log().tensor()
            residual_norm_sq = (residual_vec ** 2).sum(dim=1)

            gm_weights = geman_mcclure_weight(residual_norm_sq, mu)

        base_weights = slam.pose_graph_edges.confs[:edge_num].clone()
        for d in range(6):
            base_weights[:, d] = base_weights[:, d] * gm_weights

        weight = torch.diag_embed(base_weights)
        related_mask = graph.get_related_edge_idxs(edges)
        weight_masked = weight[related_mask]
        weight_masked = weight_masked + regularization * torch.eye(
            weight_masked.shape[-1], device=device
        )

        with suppress_specific_print(
            "Linear solver failed", color=FontColor.PoseGraphOpt
        ):
            scheduler.optimize(
                input=(edges, poses), weight=weight_masked
            )

        slam.pose_graph_nodes.poses[:node_num] = graph.get_nodes()
        mu = min(mu * mu_step, mu_max)

    return mu


# ============================================================
# TUM 格式 GT 加载（用于 SLAM_image_only 模式）
# ============================================================

def load_tum_groundtruth(path, max_poses=None):
    import pandas as pd
    from scipy.spatial.transform import Rotation
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
# 核心 SLAM 运行函数（支持 Standard / GNC-PGO）
# ============================================================

def run_slam(cfg, image_paths=None, gt_path=None, dataset_folder=None,
             use_gnc=False, device='cuda'):
    """
    运行 SLAM 并返回位姿轨迹和真值。

    参数
    ----------
    cfg : munch.Munch
        配置对象（来自 YAML 文件）
    image_paths : list or None
        图像路径列表（SLAM_image_only 模式）
    gt_path : str or None
        真值文件路径（SLAM_image_only 模式）
    dataset_folder : str or None
        TUM RGB-D 数据集文件夹（SLAM_TUMRGBD 模式）
    use_gnc : bool
        是否使用 GNC-PGO 替代标准 PGO
    device : str
        计算设备

    返回
    -------
    pred_poses : np.ndarray [N, 4, 4]
        估计位姿
    gt_poses : np.ndarray [N, 4, 4]
        真值位姿
    time_elapsed : float
        总耗时（秒）
    view_num : int
        关键帧数量
    """
    mode = "GNC-PGO" if use_gnc else "Standard"

    # ---------- 数据集初始化 ----------
    stride = cfg.stride

    if dataset_folder is not None:
        from vista_slam.datasets.slam_tumrgbd import SLAM_TUMRGBD
        dataset = SLAM_TUMRGBD(dataset_folder, resolution=(224, 224))
        print_msg(f"[{mode}] 数据集: TUM RGB-D ({Path(dataset_folder).name}) "
                  f"共 {len(dataset)} 帧, stride={stride}",
                  color=FontColor.INFO)
    else:
        from vista_slam.datasets.slam_images_only import SLAM_image_only
        dataset = SLAM_image_only(sorted(image_paths), resolution=(224, 224))
        print_msg(f"[{mode}] 数据集: SLAM_image_only 共 {len(dataset)} 帧, stride={stride}",
                  color=FontColor.INFO)

    # ---------- 关键帧选择（核心修复！） ----------
    last = len(dataset)
    keyframe_indices = list(range(1, last, stride))
    if len(keyframe_indices) > cfg.max_view_num:
        original_count = len(keyframe_indices)
        keyframe_indices = list(torch.linspace(0, last - 1, steps=cfg.max_view_num).long())
        keyframe_indices = [int(i) for i in keyframe_indices]
        print_msg(f"[{mode}] 关键帧数 {original_count} 超过 max_view_num({cfg.max_view_num})，"
                  f"均匀采样至 {len(keyframe_indices)} 帧", color=FontColor.WARNING)

    print_msg(f"[{mode}] 关键帧: {len(keyframe_indices)} 个 (from {last} 原始帧, stride={stride})",
              color=FontColor.INFO)

    # ---------- SLAM 初始化 ----------
    torch.manual_seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    slam = OnlineSLAM(
        ckpt_path=cfg.STA_pretrain_path,
        vocab_path=cfg.vocab_path,
        verbose=cfg.verbose,
        max_view_num=cfg.max_view_num,
        neighbor_edge_num=cfg.neighbor_edge_num,
        loop_edge_num=cfg.loop_edge_num,
        loop_dist_min=cfg.loop_dist_min,
        loop_nms=cfg.loop_nms,
        loop_cand_thresh_neighbor=cfg.loop_cand_thresh_neighbor,
        conf_thres=cfg.point_conf_thres,
        rel_pose_thres=cfg.rel_pose_thres,
        flow_thres=cfg.flow_thres,
        pgo_every=cfg.pgo_every
    )

    # ---------- 主体循环 ----------
    t_start = time.time()
    gt_poses_list = []
    first = True

    for idx in range(len(keyframe_indices)):
        t = keyframe_indices[idx]
        data = dataset[t]

        img_gray = (data.gray.squeeze(0).numpy() * 255).astype(np.uint8)
        img_shape = torch.tensor(data.rgb.shape[1:3]).unsqueeze(0)
        img = data.rgb.unsqueeze(0).to(device)

        input_value = {
            'rgb': img,
            'shape': img_shape,
            'gray': img_gray,
            'view_name': data.img_name if hasattr(data, 'img_name') else f"frame_{t:06d}",
        }

        # 收集真值
        if dataset_folder is not None:
            gt_poses_list.append(data.camera_pose)

        is_last = (idx == len(keyframe_indices) - 1)

        if use_gnc:
            # GNC 模式：禁止 step() 内部的标准 PGO
            slam.step(input_value, force_pgo=False)
            if is_last:
                print_msg(f"  [{mode}] 最终 GNC 优化 (keyframe {slam.view_num})...",
                          color=FontColor.INFO)
                gnc_pose_graph_optimize(slam, max_iterations=30)
                torch.cuda.empty_cache()
        else:
            # Standard 模式：与 evaluation_tumrgbd.py 完全一致
            slam.step(input_value, force_pgo=is_last)

        if first:
            first = False

        if (idx + 1) % 20 == 0 or is_last:
            print_msg(f"  [{mode}] 已处理 {idx+1}/{len(keyframe_indices)} 关键帧 "
                      f"(view_num={slam.view_num})", color=FontColor.INFO)

    # 确保最终优化
    if use_gnc and slam.view_num > 1:
        print_msg(f"  [{mode}] 最终 GNC 优化...", color=FontColor.INFO)
        gnc_pose_graph_optimize(slam, max_iterations=30)
        torch.cuda.empty_cache()

    t_elapsed = time.time() - t_start

    # ---------- 提取估计位姿 ----------
    pred_poses_list = []
    for v in range(slam.view_num):
        view = slam.get_view(v, return_pose=True, return_depth=False, return_intri=False)
        pred_poses_list.append(view.pose.cpu().numpy())
    pred_poses = np.stack(pred_poses_list, axis=0)

    # ---------- 真值位姿处理 ----------
    if dataset_folder is not None:
        gt_poses = np.stack(gt_poses_list, axis=0)
    else:
        gt_all = load_tum_groundtruth(gt_path).numpy()
        if len(gt_all) >= len(pred_poses):
            indices = np.linspace(0, len(gt_all) - 1, len(pred_poses), dtype=int)
            gt_poses = gt_all[indices]
        else:
            gt_poses = gt_all

    # 确保数量一致
    min_len = min(len(pred_poses), len(gt_poses))
    pred_poses = pred_poses[:min_len]
    gt_poses = gt_poses[:min_len]

    print_msg(f"[{mode}] 完成: {slam.view_num} 关键帧, "
              f"评估 {min_len} 个位姿, 耗时 {t_elapsed:.1f}s",
              color=FontColor.INFO)

    return pred_poses, gt_poses, t_elapsed, slam.view_num


# ============================================================
# ATE 评估（基于 evo 库的 full_traj_eval）
# ============================================================

def evaluate_ate(pred_poses, gt_poses, output_dir, label=""):
    """
    使用 evo 库进行标准 ATE 评估
    """
    plot_dir = os.path.join(output_dir, f"plots{label}")
    os.makedirs(plot_dir, exist_ok=True)

    try:
        _, _, r_a, t_a, s, ape_stats = full_traj_eval(
            [np.asarray(p) for p in pred_poses],
            [np.asarray(p) for p in gt_poses],
            plot_dir, "trajectory"
        )
        ate_rmse = ape_stats['rmse']
        print_msg(f"  评估结果: ATE RMSE={ate_rmse:.4f}m, "
                  f"scale={s:.4f}, rot={np.degrees(r_a):.2f}deg",
                  color=FontColor.PoseGraphOpt)
        return ate_rmse, ape_stats
    except Exception as e:
        print_msg(f"  evo 评估失败 ({e})，使用简易评估...", color=FontColor.WARNING)
        return _simple_ate(pred_poses, gt_poses)


def _simple_ate(pred_poses, gt_poses):
    """简易 ATE 评估（Sim(3) 对齐 + RMSE）"""
    from scipy.spatial.transform import Rotation

    pred_t = pred_poses[:, :3, 3]
    gt_t = gt_poses[:, :3, 3]

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

    aligned_t = scale * pred_t @ R.T + t
    errors = np.linalg.norm(aligned_t - gt_t, axis=1)
    ate_rmse = float(np.sqrt((errors ** 2).mean()))
    print_msg(f"  简易评估: ATE RMSE={ate_rmse:.4f}m, scale={scale:.4f}",
              color=FontColor.PoseGraphOpt)
    return ate_rmse, {'rmse': ate_rmse, 'mean': float(errors.mean()),
                      'std': float(errors.std()), 'min': float(errors.min()),
                      'max': float(errors.max())}


# ============================================================
# 配置加载
# ============================================================

def load_config(config_path, dataset_folder=None, output_dir=None):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    if output_dir is not None:
        cfg['output_dir'] = output_dir
    cfg = munch.Munch(cfg)

    # 检查预训练权重和词袋文件（支持 Colab 子目录/父目录等常见布局）
    for key in ['STA_pretrain_path', 'vocab_path']:
        path = cfg.get(key, '')
        if not os.path.exists(path):
            alt_paths = [
                os.path.join(_script_dir, path),
                os.path.join(os.path.dirname(_script_dir), path),
                os.path.join(os.path.dirname(_script_dir), 'pretrains', os.path.basename(path)),
            ]
            found = False
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    cfg[key] = alt_path
                    print_msg(f"  使用替代路径: {alt_path}", color=FontColor.INFO)
                    found = True
                    break
            if not found:
                print_msg(f"  警告: 未找到 {key} ({path})，请确认文件路径", color=FontColor.WARNING)

    return cfg


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="修正版 SLAM 实验框架 —— 正确复现 run.py 关键帧选择"
    )
    # 输入模式（二选一）
    parser.add_argument("--dataset-folder", type=str, default=None,
                        help="TUM RGB-D 数据集文件夹路径（推荐）")
    parser.add_argument("--images", type=str, default=None,
                        help="RGB 图像 glob 模式（纯图像模式）")
    parser.add_argument("--gt", type=str, default=None,
                        help="TUM 格式真值文件（纯图像模式）")
    # 配置
    parser.add_argument("--config", type=str, default="configs/tumrgbd.yaml",
                        help="YAML 配置文件路径")
    parser.add_argument("--output", type=str, default="output/corrected_slam",
                        help="输出目录")
    parser.add_argument("--stride", type=int, default=None,
                        help="关键帧步长（覆盖配置文件）")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="最大处理帧数")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-trajectory", action="store_true", default=True,
                        help="保存轨迹文件")
    args = parser.parse_args()

    # ---------- 参数验证 ----------
    if args.dataset_folder is None and args.images is None:
        print_msg("错误: 必须指定 --dataset-folder 或 --images",
                  color=FontColor.Error)
        sys.exit(1)
    if args.images is not None and args.gt is None:
        print_msg("错误: --images 模式需要 --gt 提供真值文件",
                  color=FontColor.Error)
        sys.exit(1)

    # ---------- 加载配置 ----------
    cfg = load_config(args.config, args.dataset_folder, args.output)
    if args.stride is not None:
        cfg.stride = args.stride
        print_msg(f"  使用命令行 stride={args.stride}（覆盖配置）", color=FontColor.INFO)

    # 输出目录
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 保存使用的配置
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(dict(cfg), f)

    # ---------- 准备图像路径 ----------
    image_paths = None
    if args.images is not None:
        image_paths = sorted(glob.glob(args.images))
        if len(image_paths) == 0:
            print_msg(f"错误: 未找到匹配的图像 ({args.images})",
                      color=FontColor.Error)
            sys.exit(1)

    # 限制帧数
    if args.max_frames is not None and image_paths is not None:
        image_paths = image_paths[:args.max_frames]

    # ---------- 显示实验配置 ----------
    dataset_name = (Path(args.dataset_folder).name if args.dataset_folder
                    else "image_only")
    print_msg("=" * 70, color=FontColor.PoseGraphOpt)
    print_msg(f"修正版 SLAM 实验: {dataset_name}", color=FontColor.PoseGraphOpt)
    print_msg(f"  配置: {os.path.basename(args.config)}", color=FontColor.INFO)
    print_msg(f"  Stride: {cfg.stride}", color=FontColor.INFO)
    print_msg(f"  Max views: {cfg.max_view_num}", color=FontColor.INFO)
    print_msg(f"  PGO every: {cfg.pgo_every}", color=FontColor.INFO)
    print_msg(f"  设备: {args.device}", color=FontColor.INFO)
    print_msg(f"  输出: {output_dir}", color=FontColor.INFO)
    print_msg("=" * 70, color=FontColor.PoseGraphOpt)

    # ========== 实验 1: Standard PGO ==========
    print_msg("\n" + "─" * 70, color=FontColor.INFO)
    print_msg("【实验 1】Standard PGO（基线）", color=FontColor.PoseGraphOpt)
    print_msg("─" * 70, color=FontColor.INFO)

    pred_std, gt_std, t_std, n_views_std = run_slam(
        cfg,
        image_paths=image_paths,
        gt_path=args.gt,
        dataset_folder=args.dataset_folder,
        use_gnc=False,
        device=args.device,
    )

    ate_std, stats_std = evaluate_ate(
        pred_std, gt_std, output_dir, label="_standard"
    )

    if args.save_trajectory:
        np.save(os.path.join(output_dir, 'trajectory_standard.npy'), pred_std)
        np.save(os.path.join(output_dir, 'gt_poses.npy'), gt_std)

    # 清理显存
    torch.cuda.empty_cache()

    # ========== 实验 2: GNC-PGO ==========
    print_msg("\n" + "─" * 70, color=FontColor.INFO)
    print_msg("【实验 2】GNC-PGO（鲁棒位姿图优化）", color=FontColor.PoseGraphOpt)
    print_msg("─" * 70, color=FontColor.INFO)

    pred_gnc, gt_gnc, t_gnc, n_views_gnc = run_slam(
        cfg,
        image_paths=image_paths,
        gt_path=args.gt,
        dataset_folder=args.dataset_folder,
        use_gnc=True,
        device=args.device,
    )

    ate_gnc, stats_gnc = evaluate_ate(
        pred_gnc, gt_gnc, output_dir, label="_gnc"
    )

    if args.save_trajectory:
        np.save(os.path.join(output_dir, 'trajectory_gnc.npy'), pred_gnc)

    # ========== 对比报告 ==========
    print_msg("\n" + "=" * 70, color=FontColor.PoseGraphOpt)
    print_msg("Standard PGO vs GNC-PGO 对比报告", color=FontColor.PoseGraphOpt)
    print_msg("=" * 70, color=FontColor.PoseGraphOpt)

    ate_improvement = (ate_std - ate_gnc) / max(ate_std, 1e-8) * 100
    time_overhead = (t_gnc - t_std) / max(t_std, 1e-8) * 100

    header = f"{'指标':<25} {'Standard':>14} {'GNC-PGO':>14} {'变化':>12}"
    print_msg(header, color=FontColor.PoseGraphOpt)
    print_msg("─" * 68, color=FontColor.PoseGraphOpt)

    report_lines = [
        ("ATE RMSE ↓ (m)", f"{ate_std:.4f}", f"{ate_gnc:.4f}",
         f"{'✅' if ate_improvement > 0 else ''} {ate_improvement:+>+7.1f}%"),
        ("关键帧数", str(n_views_std), str(n_views_gnc), ""),
        ("耗时 (s)", f"{t_std:.1f}", f"{t_gnc:.1f}",
         f"{'⬆' if time_overhead > 0 else '⬇'} {time_overhead:+>+6.1f}%"),
    ]

    if 'mean' in stats_std:
        report_lines.append(("ATE 均值 (m)", f"{stats_std['mean']:.4f}",
                             f"{stats_gnc['mean']:.4f}", ""))
    if 'std' in stats_std:
        report_lines.append(("ATE 标准差 (m)", f"{stats_std['std']:.4f}",
                             f"{stats_gnc['std']:.4f}", ""))

    for name, std_v, gnc_v, trend in report_lines:
        print_msg(f"{name:<25} {std_v:>14} {gnc_v:>14} {trend:>12}",
                  color=FontColor.PoseGraphOpt)

    print_msg("─" * 68, color=FontColor.PoseGraphOpt)

    # 结论
    if ate_improvement > 10:
        conclusion = (f"GNC-PGO 显著提升轨迹精度\n"
                      f"  ATE 降低 {ate_improvement:.1f}% ({ate_std:.4f}m → {ate_gnc:.4f}m)\n"
                      f"  耗时增加 {time_overhead:.1f}%")
        print_msg(f"结论: {conclusion}", color=FontColor.PoseGraphOpt)
    elif ate_improvement > 2:
        conclusion = (f"GNC-PGO 有小幅提升\n"
                      f"  ATE 降低 {ate_improvement:.1f}%")
        print_msg(f"结论: {conclusion}", color=FontColor.INFO)
    else:
        print_msg("结论: 当前序列下 GNC 与标准 PGO 结果相近。"
                  "GNC 优势在强异常边场景下更明显。",
                  color=FontColor.INFO)

    # 保存报告
    report = {
        'dataset': dataset_name,
        'config': dict(cfg),
        'standard': {
            'ate_rmse': ate_std,
            'time': t_std,
            'keyframes': n_views_std,
            'stats': {k: float(v) if isinstance(v, (np.floating,)) else v
                      for k, v in stats_std.items()},
        },
        'gnc_pgo': {
            'ate_rmse': ate_gnc,
            'time': t_gnc,
            'keyframes': n_views_gnc,
            'stats': {k: float(v) if isinstance(v, (np.floating,)) else v
                      for k, v in stats_gnc.items()},
        },
        'improvement': {
            'ate_percent': ate_improvement,
            'time_overhead_percent': time_overhead,
        },
    }
    with open(os.path.join(output_dir, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    print_msg(f"\n完整报告已保存到: {output_dir}/", color=FontColor.INFO)
    print_msg(f"  配置文件: config.yaml", color=FontColor.INFO)
    print_msg(f"  报告: report.json", color=FontColor.INFO)
    print_msg(f"  轨迹(Standard): trajectory_standard.npy", color=FontColor.INFO)
    print_msg(f"  轨迹(GNC): trajectory_gnc.npy", color=FontColor.INFO)
    print_msg(f"  真值: gt_poses.npy", color=FontColor.INFO)
    print_msg(f"  评估图: plots_standard/, plots_gnc/", color=FontColor.INFO)

    print_msg("\n实验完成 ✅", color=FontColor.PoseGraphOpt)


if __name__ == "__main__":
    main()
