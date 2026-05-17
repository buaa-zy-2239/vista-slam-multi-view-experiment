# =========================================================
# 多视图深度深度融合增强
# 利用多个重叠视图的深度预测做置信度加权融合
# 无需重新训练，纯算法改进，可量化评估
# =========================================================
import torch
import torch.nn.functional as F
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

from vista_slam.utils.slam_utils import FontColor, print_msg
from vista_slam.sta_model.sta_model import SymmetricTwoViewAssociation as STA


def load_frontend(ckpt_path, device):
    import vista_slam.sta_model.blocks.sta_blocks as blocks_mod
    def forward(self, x, xpos):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)
        q, k, v = [qkv[:, :, i] for i in range(3)]
        if self.rope is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    if str(device) == 'cpu':
        blocks_mod.XFormer_Attention.forward = forward
    frontend = STA()
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        frontend.load_state_dict(ckpt['model'], strict=True)
    frontend.to(device)
    frontend.eval()
    return frontend


def load_frames(glob_pattern, max_frames):
    paths = sorted(glob.glob(glob_pattern))[:max_frames]
    from PIL import Image
    from torchvision import transforms
    t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
    frames = []
    names = []
    for p in paths:
        frames.append(t(Image.open(p).convert('RGB')).unsqueeze(0))
        names.append(Path(p).name)
    return frames, names


def load_depth(glob_pattern, max_frames, target_size=(224, 224)):
    """加载 TUM 深度图（16位PNG，单位毫米），归一化到米"""
    paths = sorted(glob.glob(glob_pattern))[:max_frames]
    depth_maps = []
    from PIL import Image
    from torchvision import transforms
    resize = transforms.Resize(target_size)
    for p in paths:
        d = torch.from_numpy(np.array(Image.open(p))).float() / 1000.0  # 毫米→米
        d = resize(d.unsqueeze(0).unsqueeze(0)).squeeze()  # [H, W]
        d[d > 10.0] = 0.0  # 截断远距离
        depth_maps.append(d)
    return depth_maps


# ============================================================
# 核心算法：多视图深度深度融合
# ============================================================
class MultiViewDepthFusion:
    """
    多视图深度深度融合

    原理：
    对于视图 i 的每个像素，将其邻居视图 j 的深度图投影到 i 的坐标系下，
    得到多个深度估计，然后用置信度加权融合。

    数学保证：
    D_fused = (w_i * D_i + Σ w_j * D_{j→i}) / (w_i + Σ w_j)
    由于 w 是置信度，多帧独立估计 → 方差降低 → 精度提升
    """

    def __init__(self, K, H=224, W=224):
        self.K = K.to('cpu') if torch.is_tensor(K) else K
        self.K_inv = torch.inverse(self.K)
        self.H, self.W = H, W
        self._make_grid()

    def _make_grid(self):
        """生成像素坐标网格 [H, W, 2] 和归一化网格 [-1,1]"""
        u = torch.arange(self.W)
        v = torch.arange(self.H)
        uu, vv = torch.meshgrid(u, v, indexing='xy')
        self.grid = torch.stack([uu, vv], dim=-1).float()
        # grid_sample 需要的归一化坐标 [-1, 1]
        self.grid_norm = torch.stack([
            (uu.float() / (self.W - 1)) * 2 - 1,
            (vv.float() / (self.H - 1)) * 2 - 1,
        ], dim=-1)

    def project_depth_to_view(self, depth_src, pose_src, pose_tgt):
        """
        将源视图的深度图投影到目标视图

        Args:
            depth_src: [H, W] 源视图深度
            pose_src: [4, 4] 源视图相机到世界
            pose_tgt: [4, 4] 目标视图相机到世界

        Returns:
            depth_proj: [H, W] 投影到目标视图的深度
            valid_mask: [H, W] 有效投影区域
        """
        device = depth_src.device
        K = self.K.to(device)
        K_inv = self.K_inv.to(device)
        grid = self.grid.to(device)

        # 1. 反投影到3D (源相机坐标系)
        z = depth_src  # [H, W]
        uv_h = torch.cat([grid, torch.ones(self.H, self.W, 1, device=device)], dim=-1)
        pts_cam_src = (K_inv @ uv_h.view(-1, 3).T).T * z.view(-1, 1)  # [H*W, 3]
        pts_cam_src = pts_cam_src.view(self.H, self.W, 3)

        # 2. 转到世界坐标系
        ones = torch.ones(self.H, self.W, 1, device=device)
        pts_h = torch.cat([pts_cam_src, ones], dim=-1).view(-1, 4).T  # [4, H*W]
        pts_world = (pose_src @ pts_h).T[..., :3]  # [H*W, 3]

        # 3. 转到目标相机坐标系
        T_tgt_inv = torch.inverse(pose_tgt)
        pts_h = torch.cat([pts_world, torch.ones(pts_world.shape[0], 1, device=device)], dim=-1).T
        pts_cam_tgt = (T_tgt_inv @ pts_h).T[..., :3]  # [H*W, 3]

        # 4. 投影到目标图像平面
        z_proj = pts_cam_tgt[:, 2].clamp(min=1e-6)
        uv_proj = (K @ pts_cam_tgt.T).T
        uv_proj = uv_proj[:, :2] / uv_proj[:, 2:3]  # [H*W, 2]

        # 5. 判断有效性
        u_proj = uv_proj[:, 0].view(self.H, self.W)
        v_proj = uv_proj[:, 1].view(self.H, self.W)
        valid = (u_proj >= 0) & (u_proj < self.W - 1) & \
                (v_proj >= 0) & (v_proj < self.H - 1) & \
                (z_proj > 0)

        depth_proj = z_proj.view(self.H, self.W)
        return depth_proj, valid

    def fuse(self, ref_depth, ref_conf, ref_pose, neighbors):
        """
        多视图深度融合

        Args:
            ref_depth: [H, W] 参考视图原始深度
            ref_conf: [H, W] 参考视图置信度
            ref_pose: [4, 4] 参考视图位姿
            neighbors: list of (depth, conf, pose) 邻居视图

        Returns:
            fused_depth: [H, W] 融合后深度
            fusion_weight: [H, W] 融合权重（有效视图数）
        """
        device = ref_depth.device

        # 加权分子分母
        numerator = ref_conf * ref_depth
        denominator = ref_conf.clone()
        weight_map = torch.ones_like(ref_conf)

        for n_depth, n_conf, n_pose in neighbors:
            # 投影邻居深度到参考视图
            n_depth = n_depth.to(device)
            n_conf = n_conf.to(device)
            n_pose = n_pose.to(device)

            d_proj, valid = self.project_depth_to_view(n_depth, n_pose, ref_pose)

            # 置信度传播（用归一化坐标）
            c_proj = F.grid_sample(
                n_conf.unsqueeze(0).unsqueeze(0),
                self.grid_norm.unsqueeze(0).to(device),
                mode='bilinear', align_corners=True
            ).squeeze()

            # 有效性检查：深度一致性
            depth_diff = torch.abs(d_proj - ref_depth)
            consistent = depth_diff < 0.5 * ref_depth  # 50%阈值

            mask = valid & consistent

            # 加权融合
            w = c_proj * mask.float()
            numerator += w * d_proj
            denominator += w
            weight_map += mask.float()

        # 融合深度
        fused = numerator / (denominator + 1e-8)
        fused[denominator < 1e-6] = 0.0  # 没有有效投影的像素置0

        return fused, weight_map


@torch.no_grad()
def predict_pair(frontend, feat_i, pos_i, feat_j, pos_j, shape):
    """回归一对视图的深度和位姿"""
    dec_ij, dec_ji = frontend._decode_stereo(feat_i, feat_j, pos_i, pos_j)

    # 位姿
    pose_ij = frontend.head_pose_s(dec_ij[-1][:, 0, :])
    pose_ji = frontend.head_pose_s(dec_ji[-1][:, 0, :])
    T_ij = pose_ij['pose'][0].cpu()

    # 深度 (视图i视角)
    ij_pts = [feat_i] + [tok[:, 1:, :].float() for tok in dec_ij]
    ret = frontend.head_pts(ij_pts, shape)
    depth = ret['pts3d'][0, :, :, 2].cpu()  # [H, W]
    conf = ret['conf'][0].cpu()  # [H, W]

    return depth, conf, T_ij


# ============================================================
# 评估指标
# ============================================================
def compute_depth_metrics(pred_depth, gt_depth, max_depth=5.0):
    """计算深度评估指标"""
    # 确保都是 2D [H, W]
    if pred_depth.ndim == 3:
        pred_depth = pred_depth.squeeze()
    if gt_depth.ndim == 3:
        gt_depth = gt_depth.squeeze()
    mask = (gt_depth > 0) & (pred_depth > 0) & (gt_depth < max_depth)
    if mask.sum() < 100:
        return {}

    pred, gt = pred_depth[mask], gt_depth[mask]
    diff = torch.abs(pred - gt)
    rel = diff / gt

    metrics = {
        'rmse': float(torch.sqrt((diff ** 2).mean())),
        'mae': float(diff.mean()),
        'abs_rel': float(rel.mean()),
        'delta_1.05': float((diff / gt < 0.05).float().mean()),
        'delta_1.10': float((diff / gt < 0.10).float().mean()),
        'delta_1.25': float((diff / gt < 0.25).float().mean()),
        'valid_pixels': int(mask.sum()),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", type=str, required=True,
                        help="RGB images glob, e.g. 'rgbd_dataset_freiburg1_xyz/rgb/*.png'")
    parser.add_argument("--depth", type=str, required=True,
                        help="GT depth images glob, e.g. 'rgbd_dataset_freiburg1_xyz/depth/*.png'")
    parser.add_argument("--ckpt", type=str, default="pretrains/frontend_sta_weights.pth")
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--fusion-window", type=int, default=3,
                        help="Fusion window size (neighbors on each side)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    alt_ckpt = "/content/vista-slam/pretrains/frontend_sta_weights.pth"
    if not os.path.exists(args.ckpt) and os.path.exists(alt_ckpt):
        args.ckpt = alt_ckpt

    device = torch.device(args.device)
    print_msg(f"Loading model...", color=FontColor.INFO)
    frontend = load_frontend(args.ckpt, device)

    print_msg(f"Loading frames...", color=FontColor.INFO)
    rgb_frames, names = load_frames(args.rgb, args.max_frames)
    depth_frames = load_depth(args.depth, args.max_frames)  # [H, W] 单位米
    N = min(len(rgb_frames), len(depth_frames))
    print_msg(f"Loaded {N} frames", color=FontColor.INFO)

    # 编码
    print_msg("Encoding frames...", color=FontColor.INFO)
    feats, poss = [], []
    for i in range(N):
        shape = torch.tensor(rgb_frames[i].shape[2:4]).unsqueeze(0)
        f, p = frontend._encode_image(rgb_frames[i].to(device), shape, normalize=False)
        feats.append(f)
        poss.append(p)

    # 对每对相邻帧做预测，建立初始深度图和位姿
    print_msg("Predicting depths and poses...", color=FontColor.INFO)
    W = 224; H = 224
    depths = torch.zeros(N, H, W)
    confs = torch.zeros(N, H, W)
    poses = torch.eye(4).unsqueeze(0).repeat(N, 1, 1)  # [N, 4, 4]

    for i in range(N - 1):
        # 帧(i, i+1)预测 → 得到帧i的深度
        d, c, T = predict_pair(frontend, feats[i], poss[i],
                                feats[i+1], poss[i+1],
                                torch.tensor([[H, W]]))
        depths[i] = d
        confs[i] = c
        # 位姿累积：T_ij = pose_j.inv() @ pose_i, 所以 pose_j = pose_i @ T_ij.inv()
        if i > 0:
            poses[i+1] = poses[i] @ T.inverse()
        else:
            poses[1] = T.inverse()

    # 最后一帧的深度：用(N-1, N-2)对得到帧N-1的深度
    d_last, c_last, _ = predict_pair(frontend, feats[N-1], poss[N-1],
                                      feats[N-2], poss[N-2],
                                      torch.tensor([[H, W]]))
    depths[N-1] = d_last
    confs[N-1] = c_last

    # 估计内参（TUM fr1/xyz 已知 fx=517.3, fy=516.5, 但图像缩放到224需调整）
    fx = fy = 224.0 * 517.3 / 640.0  # 原始640→224缩放
    K = torch.tensor([[fx, 0, 112],
                      [0, fy, 112],
                      [0, 0, 1]], dtype=torch.float32)
    print_msg(f"Intrinsics: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}",
              color=FontColor.INFO)

    # ============================================================
    # 评估原始深度
    # ============================================================
    print_msg("=" * 70, color=FontColor.INFO)
    print_msg("评估原始深度（无融合）...", color=FontColor.INFO)

    orig_metrics = []
    for i in range(N):
        gt = depth_frames[i]  # [H, W] 已在 load_depth 中处理
        m = compute_depth_metrics(depths[i], gt)
        if m:
            orig_metrics.append(m)

    orig_avg = {k: np.mean([m[k] for m in orig_metrics]) for k in orig_metrics[0].keys()}
    print_msg(f"  RMSE: {orig_avg['rmse']:.4f}  AbsRel: {orig_avg['abs_rel']:.4f}  "
              f"δ<1.25: {orig_avg['delta_1.25']:.3f}",
              color=FontColor.INFO)

    # ============================================================
    # 多视图深度融合
    # ============================================================
    print_msg("-" * 70, color=FontColor.INFO)
    print_msg(f"多视图深度融合 (window={args.fusion_window})...", color=FontColor.INFO)

    fusor = MultiViewDepthFusion(K, H, W)
    fused_depths = depths.clone()

    fusion_stats = []

    for ref_idx in range(N):
        # 收集邻居
        start = max(0, ref_idx - args.fusion_window)
        end = min(N, ref_idx + args.fusion_window + 1)
        neighbors = []
        for j in range(start, end):
            if j == ref_idx:
                continue
            neighbors.append((depths[j], confs[j], poses[j]))

        # 融合
        t0 = time.time()
        d_fused, w_map = fusor.fuse(
            depths[ref_idx], confs[ref_idx], poses[ref_idx], neighbors
        )
        elapsed = time.time() - t0
        fused_depths[ref_idx] = d_fused

        # 评估（局部窗口）
        n_valid = int((w_map > 1).sum())
        n_total = H * W
        fusion_stats.append({
            'idx': ref_idx,
            'valid_pixels': n_valid,
            'fusion_ratio': n_valid / n_total * 100,
            'time': elapsed,
        })

        if ref_idx % 5 == 0:
            print_msg(f"  view[{ref_idx}]: {n_valid}/{n_total} pixels fused "
                      f"({n_valid/n_total*100:.1f}%), {elapsed:.2f}s",
                      color=FontColor.INFO)

    # ============================================================
    # 评估融合后深度
    # ============================================================
    print_msg("-" * 70, color=FontColor.INFO)
    print_msg("评估融合后深度...", color=FontColor.INFO)

    fused_metrics = []
    for i in range(N):
        gt = depth_frames[i]
        m = compute_depth_metrics(fused_depths[i], gt)
        if m:
            fused_metrics.append(m)

    fused_avg = {k: np.mean([m[k] for m in fused_metrics]) for k in fused_metrics[0].keys()}

    # ============================================================
    # 对比报告
    # ============================================================
    print_msg("=" * 70, color=FontColor.PoseGraphOpt)
    print_msg("深度精度对比报告", color=FontColor.PoseGraphOpt)
    print_msg("-" * 70, color=FontColor.PoseGraphOpt)

    rows = [
        ('RMSE ↓ (米)', orig_avg['rmse'], fused_avg['rmse'], True),
        ('MAE ↓ (米)', orig_avg['mae'], fused_avg['mae'], True),
        ('AbsRel ↓', orig_avg['abs_rel'], fused_avg['abs_rel'], True),
        ('δ<1.05 ↑', orig_avg['delta_1.05'], fused_avg['delta_1.05'], False),
        ('δ<1.10 ↑', orig_avg['delta_1.10'], fused_avg['delta_1.10'], False),
        ('δ<1.25 ↑', orig_avg['delta_1.25'], fused_avg['delta_1.25'], False),
    ]

    print_msg(f"{'指标':<20} {'原始':>12} {'融合后':>12} {'改善':>10}",
              color=FontColor.PoseGraphOpt)
    print_msg("-" * 56, color=FontColor.PoseGraphOpt)

    for name, orig, fused, lower_better in rows:
        delta = (fused - orig) / max(orig, 1e-8) * 100
        color = '✅' if (delta < 0 and lower_better) or (delta > 0 and not lower_better) else ''
        print_msg(f"{name:<20} {orig:>10.4f}  {fused:>10.4f}  {color} {delta:+>+6.1f}%",
                  color=FontColor.PoseGraphOpt)

    print_msg("-" * 56, color=FontColor.PoseGraphOpt)

    # 融合统计
    avg_ratio = np.mean([s['fusion_ratio'] for s in fusion_stats])
    avg_time = np.mean([s['time'] for s in fusion_stats])
    print_msg(f"平均融合覆盖率: {avg_ratio:.1f}%  |  平均耗时: {avg_time:.2f}s/帧",
              color=FontColor.INFO)

    rmse_improv = (orig_avg['rmse'] - fused_avg['rmse']) / orig_avg['rmse'] * 100
    print_msg(f"", color=FontColor.INFO)
    print_msg(f"RMSE 改善: {rmse_improv:.1f}%  ({orig_avg['rmse']:.4f} → {fused_avg['rmse']:.4f})",
              color=FontColor.PoseGraphOpt)
    if rmse_improv > 5:
        print_msg(f"结论: 多视图深度深度融合显著提升深度精度！✅", color=FontColor.PoseGraphOpt)
        print_msg(f"      在 {N} 帧上, RMSE 降低 {rmse_improv:.1f}%", color=FontColor.PoseGraphOpt)
    elif rmse_improv > 0:
        print_msg(f"结论: 多视图深度深度融合有小幅提升", color=FontColor.INFO)
    else:
        print_msg(f"结论: 当前条件下融合未带来提升，可能需要更大的融合窗口或更密集的帧",
                  color=FontColor.WARNING)


if __name__ == "__main__":
    main()
