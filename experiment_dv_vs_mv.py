# =========================================================
# DV vs MV 对比实验 —— 完全自包含版
# 纯 CPU 可运行，仅需3-5帧
# 指标：对称性误差 / 置信度 / 深度尺度一致性
# =========================================================
import torch
import numpy as np
import time, os, sys, glob, argparse
from pathlib import Path

# 【路径修复】向上查找包含 vista_slam/utils/ 的父目录
_script_dir = os.path.dirname(os.path.abspath(__file__))
_vs_path = _script_dir
for _ in range(4):
    # 检查当前目录是否直接包含 vista_slam/utils/
    if os.path.isdir(os.path.join(_vs_path, 'vista_slam', 'utils')):
        sys.path.insert(0, _vs_path)
        break
    # 检查当前目录下是否有子目录包含 vista_slam/utils/
    # 处理 git clone 目录名为 vista-slam 的情况
    for _sub in os.listdir(_vs_path) if os.path.isdir(_vs_path) else []:
        if os.path.isdir(os.path.join(_vs_path, _sub, 'vista_slam', 'utils')):
            sys.path.insert(0, os.path.join(_vs_path, _sub))
            break
    else:
        _vs_path = os.path.dirname(_vs_path)
        continue
    break

# ------ 直接用 SLAM 模型，跳过数据集依赖 ------
from vista_slam.utils.slam_utils import (
    FontColor, print_msg, estimate_scale_with_depth_and_confidence
)


def load_images_simple(glob_pattern, max_frames=5, target_size=(224, 224)):
    """
    自包含图片加载器：跳过 cv2，用 torchvision 或 PIL
    """
    paths = sorted(glob.glob(glob_pattern))
    if len(paths) == 0:
        raise FileNotFoundError(f"No images at {glob_pattern}")
    paths = paths[:max_frames]

    try:
        from torchvision.io import decode_image
        use_tv = True
    except ImportError:
        use_tv = False

    from PIL import Image
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),  # 与 SLAM 一致 [-1,1]
    ])
    to_gray = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Grayscale(),
    ])

    frames = []
    for p in paths:
        img = Image.open(p).convert('RGB')
        rgb = transform(img).unsqueeze(0)                # [1,3,H,W]
        gray = to_gray(img) * 2 - 1                      # [-1,1] 归一化
        frames.append({
            'rgb': rgb,
            'gray': gray,
            'path': p,
            'name': Path(p).stem,
        })
    return frames


def load_frontend(ckpt_path, device):
    """直接加载 STA 模型，绕过 OnlineSLAM 的 CUDA 硬编码"""
    # 【CPU兼容补丁】替换 xformers attention 为 PyTorch 原生实现
    import vista_slam.sta_model.blocks.sta_blocks as blocks_mod
    original_forward = blocks_mod.XFormer_Attention.forward

    def cpu_compat_forward(self, x, xpos):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)
        q, k, v = [qkv[:, :, i] for i in range(3)]
        if self.rope is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)
        # 手动实现 scaled dot-product attention（CPU 兼容）
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    if str(device) == 'cpu':
        blocks_mod.XFormer_Attention.forward = cpu_compat_forward

    from vista_slam.sta_model.sta_model import SymmetricTwoViewAssociation as STA

    frontend = STA()
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        frontend.load_state_dict(ckpt['model'], strict=True)
        print_msg(f"  Loaded pretrained weights from {ckpt_path}", color=FontColor.INFO)
    else:
        print_msg(f"  [WARNING] {ckpt_path} not found, using untrained model",
                  color=FontColor.WARNING)

    frontend.to(device)
    frontend.eval()
    return frontend


def run_single_experiment(ckpt_path, frames, device='cpu', neighbor_count=1):
    """
    通用回归函数：在 frames 上按 neighbor_count 连接边，
    收集每条边的对称性误差、置信度、深度一致性、推理时间
    """
    frontend = load_frontend(ckpt_path, device)
    N = len(frames)
    enc_feats, enc_pos, img_shapes = [], [], []

    for t in range(N):
        img = frames[t]['rgb'].to(device)
        shape = torch.tensor(img.shape[2:4]).unsqueeze(0)
        with torch.no_grad():
            feat, pos = frontend._encode_image(img, shape, normalize=False)
        enc_feats.append(feat)
        enc_pos.append(pos)
        img_shapes.append(shape)

    errors, confs, times, scales = [], [], [], []

    for i in range(N):
        start_j = max(0, i - neighbor_count)
        for j in range(start_j, i):
            t0 = time.time()
            with torch.no_grad():
                dec_ij, dec_ji = frontend._decode_stereo(
                    enc_feats[i], enc_feats[j], enc_pos[i], enc_pos[j]
                )
                pose_ij = frontend.head_pose_s(dec_ij[-1][:, 0, :])
                pose_ji = frontend.head_pose_s(dec_ji[-1][:, 0, :])
            elapsed = time.time() - t0

            # pose_ij['pose'] 本身已经是 4x4 矩阵
            T_ij = pose_ij['pose'][0].cpu()
            T_ji = pose_ji['pose'][0].cpu()
            I4 = torch.eye(4)

            sym_err = torch.norm(T_ij @ T_ji - I4).item()
            conf = (pose_ij['conf'].item() + pose_ji['conf'].item()) / 2.0

            scale_err = 0.0
            try:
                ji_pts = [enc_feats[j]] + [t[:, 1:, :].float() for t in dec_ji]
                ij_pts = [enc_feats[i]] + [t[:, 1:, :].float() for t in dec_ij]
                ji_r = frontend.head_pts(ji_pts, img_shapes[j])
                ij_r = frontend.head_pts(ij_pts, img_shapes[i])
                d_i, conf_i = ij_r['pts3d'][..., 2], ij_r['conf']
                d_j, conf_j = ji_r['pts3d'][..., 2], ji_r['conf']
                s = estimate_scale_with_depth_and_confidence(d_i, d_j, conf_i, conf_j)
                scale_err = abs(s.item() - 1.0)
            except Exception:
                pass

            errors.append(sym_err)
            confs.append(conf)
            times.append(elapsed)
            scales.append(scale_err)

            print_msg(f"  edge({i},{j}): sym_err={sym_err:.4f}  conf={conf:.4f}  t={elapsed:.1f}s",
                      color=FontColor.INFO)

    return {
        'errors': errors, 'confs': confs, 'times': times, 'scales': scales,
        'num_edges': len(errors), 'num_frames': N, 'neighbor_count': neighbor_count,
    }


def main():
    parser = argparse.ArgumentParser(description="DV vs MV comparison (self-contained)")
    parser.add_argument("--images", type=str,
                        default="/home/zhang/ORB_SLAM_Learning/datasets/tum/rgbd_dataset_freiburg1_xyz/rgb/*.png",
                        help="Glob to RGB images")
    parser.add_argument("--ckpt", type=str, default="pretrains/frontend_sta_weights.pth")
    parser.add_argument("--max-frames", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # 如果默认 pretrains 路径不存在，尝试其他常见位置
    alt_ckpt = "/home/zhang/vista-slam/pretrains/frontend_sta_weights.pth"
    if not os.path.exists(args.ckpt) and os.path.exists(alt_ckpt):
        args.ckpt = alt_ckpt

    if not os.path.exists(args.ckpt):
        print_msg(f"[WARNING] Weights not found at {args.ckpt}. "
                  "Results will be from untrained model.",
                  color=FontColor.WARNING)

    frames = load_images_simple(args.images, max_frames=args.max_frames)
    print_msg(f"Loaded {len(frames)} frames on {args.device}", color=FontColor.INFO)
    for f in frames:
        print_msg(f"  └ {f['name']}", color=FontColor.INFO)

    # ── DV 模式 ──
    print_msg("\n" + "=" * 60, color=FontColor.INFO)
    print_msg("[DV] connecting only adjacent edges (i,i+1)...", color=FontColor.INFO)
    t0 = time.time()
    dv = run_single_experiment(args.ckpt, frames, args.device, neighbor_count=1)
    t_dv = time.time() - t0

    # ── MV 模式 ──
    mv_window = min(args.max_frames - 1, 3)
    print_msg("\n" + "=" * 60, color=FontColor.INFO)
    print_msg(f"[MV] connecting multi-view edges (window={mv_window})...",
              color=FontColor.INFO)
    t0 = time.time()
    mv = run_single_experiment(args.ckpt, frames, args.device, neighbor_count=mv_window)
    t_mv = time.time() - t0

    # ── 报告 ──
    print_msg("\n" + "=" * 70, color=FontColor.PoseGraphOpt)
    print_msg("DV vs MV 对比报告", color=FontColor.PoseGraphOpt)
    print_msg("-" * 70, color=FontColor.PoseGraphOpt)

    def stat(arr, name):
        a = np.array(arr)
        return {
            f'{name}_cnt': len(a),
            f'{name}_mean': float(a.mean()),
            f'{name}_std': float(a.std()),
            f'{name}_min': float(a.min()),
            f'{name}_max': float(a.max()),
        }

    dv_s = stat(dv['errors'], 'sym')
    dv_c = stat(dv['confs'], 'conf')
    dv_t = stat(dv['times'], 'time')

    mv_s = stat(mv['errors'], 'sym')
    mv_c = stat(mv['confs'], 'conf')
    mv_t = stat(mv['times'], 'time')

    rows = [
        ("边数量",        str(dv_s['sym_cnt']),  str(mv_s['sym_cnt'])),
        ("对称误差均值 ↓", f"{dv_s['sym_mean']:.4f}", f"{mv_s['sym_mean']:.4f}"),
        ("对称误差中位 ↓", f"{dv_s['sym_min']:.4f}", f"{mv_s['sym_min']:.4f}"),
        ("对称误差最大 ↓", f"{dv_s['sym_max']:.4f}", f"{mv_s['sym_max']:.4f}"),
        ("置信度均值 ↑",   f"{dv_c['conf_mean']:.4f}", f"{mv_c['conf_mean']:.4f}"),
        ("置信度最小值 ↑", f"{dv_c['conf_min']:.4f}", f"{mv_c['conf_min']:.4f}"),
        ("平均推理(s)",    f"{dv_t['time_mean']:.1f}", f"{mv_t['time_mean']:.1f}"),
        ("总时间(s)",      f"{t_dv:.1f}",             f"{t_mv:.1f}"),
    ]

    print_msg(f"{'指标':<20} │ {'DV(双视图)':>14} │ {'MV(多视图)':>14} │ {'趋势':>12}",
              color=FontColor.PoseGraphOpt)
    print_msg("─" * 65, color=FontColor.PoseGraphOpt)
    for name, dv_v, mv_v in rows:
        dv_f = float(dv_v) if dv_v.replace('.', '', 1).lstrip('-').isdigit() else None
        mv_f = float(mv_v) if mv_v.replace('.', '', 1).lstrip('-').isdigit() else None
        trend = ""
        if dv_f is not None and mv_f is not None and dv_f != 0:
            d = (mv_f - dv_f) / abs(dv_f) * 100
            # 方向：边/置信度越大越好，误差/时间越小越好
            better = (name.endswith("↑") and d > 0) or (name.endswith("↓") and d < 0)
            trend = f"{'✅' if better else ''} {d:+.1f}%"
        print_msg(f"{name:<20} │ {dv_v:>14} │ {mv_v:>14} │ {trend:>12}",
                  color=FontColor.PoseGraphOpt)

    print_msg("=" * 70, color=FontColor.PoseGraphOpt)
    sym_imp = dv_s['sym_mean'] - mv_s['sym_mean']
    conf_imp = mv_c['conf_mean'] - dv_c['conf_mean']
    if sym_imp > 0:
        print_msg(f"✅ 对称性误差降低 {sym_imp:.4f} "
                  f"({sym_imp/dv_s['sym_mean']*100:.1f}%)",
                  color=FontColor.PoseGraphOpt)
    if conf_imp > 0:
        print_msg(f"✅ 置信度提升 {conf_imp:.4f} "
                  f"({conf_imp/dv_c['conf_mean']*100:.1f}%)",
                  color=FontColor.PoseGraphOpt)
    extra_edges = mv_s['sym_cnt'] - dv_s['sym_cnt']
    if extra_edges > 0:
        print_msg(f"✅ 额外约束边 {extra_edges} 条 (MV提供更多冗余约束)",
                  color=FontColor.PoseGraphOpt)

    print_msg("", color=FontColor.INFO)


if __name__ == "__main__":
    main()
