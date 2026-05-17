# =========================================================
# DV vs MV —— 链式累积漂移对比实验
# 轻量级：只需前向推理，无需跑完整SLAM
# 核心思想：对比"链式累积"和"直接预测"的位姿差异
# =========================================================
import torch
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
    def cpu_compat_forward(self, x, xpos):
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
        blocks_mod.XFormer_Attention.forward = cpu_compat_forward

    frontend = STA()
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        frontend.load_state_dict(ckpt['model'], strict=True)
    frontend.to(device)
    frontend.eval()
    return frontend


def load_images(glob_pattern, max_frames, target_size=(224, 224)):
    paths = sorted(glob.glob(glob_pattern))
    if len(paths) == 0:
        raise FileNotFoundError(f"No images at {glob_pattern}")
    paths = paths[:max_frames]
    from PIL import Image
    from torchvision import transforms
    t = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
    frames = []
    for p in paths:
        img = t(Image.open(p).convert('RGB')).unsqueeze(0)
        frames.append(img)
    return frames, [Path(p).name for p in paths]


def predict_relative_pose(frontend, feat_i, pos_i, feat_j, pos_j, shape):
    """预测视图 i → j 的相对位姿"""
    with torch.no_grad():
        dec_ij, dec_ji = frontend._decode_stereo(feat_i, feat_j, pos_i, pos_j)
        pose_ij = frontend.head_pose_s(dec_ij[-1][:, 0, :])
        pose_ji = frontend.head_pose_s(dec_ji[-1][:, 0, :])
    T_ij = pose_ij['pose'][0].cpu()
    T_ji = pose_ji['pose'][0].cpu()
    conf = (pose_ij['conf'].item() + pose_ji['conf'].item()) / 2.0
    return T_ij, T_ji, conf


def compute_chain_drift(poses_chain, pose_direct):
    """
    计算链式累积漂移
    chain: 从0到N逐帧累积位姿
    direct: 直接从0到N预测的位姿
    理想情况下二者应该一致
    """
    T_chain = torch.eye(4)
    for T in poses_chain:
        T_chain = T_chain @ T

    # 漂移 = 链式预测与直接预测的差异
    drift = torch.norm(T_chain[:3, 3] - pose_direct[:3, 3]).item()
    # 旋转漂移
    R_diff = T_chain[:3, :3].T @ pose_direct[:3, :3]
    rot_drift = torch.acos(torch.clamp((torch.trace(R_diff) - 1) / 2, -1, 1)).item()
    return drift, rot_drift


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="pretrains/frontend_sta_weights.pth")
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    alt_ckpt = "/content/vista-slam/pretrains/frontend_sta_weights.pth"
    if not os.path.exists(args.ckpt) and os.path.exists(alt_ckpt):
        args.ckpt = alt_ckpt

    device = torch.device(args.device)
    print_msg(f"Loading {args.max_frames} frames...", color=FontColor.INFO)
    frames, names = load_images(args.images, args.max_frames)
    N = len(frames)
    print_msg(f"Loaded {N} frames, device={args.device}", color=FontColor.INFO)

    print_msg("Loading model...", color=FontColor.INFO)
    frontend = load_frontend(args.ckpt, device)

    # 编码所有帧
    print_msg("Encoding frames...", color=FontColor.INFO)
    feats, poss = [], []
    for i, img in enumerate(frames):
        shape = torch.tensor(img.shape[2:4]).unsqueeze(0)
        f, p = frontend._encode_image(img.to(device), shape, normalize=False)
        feats.append(f)
        poss.append(p)
        if i % 10 == 0:
            print_msg(f"  [{i}/{N}]", color=FontColor.INFO)

    # ==============================
    # 双视图：仅链式累积（只连相邻帧）
    # ==============================
    print_msg("=" * 60, color=FontColor.INFO)
    print_msg("[DV] Chain of adjacent edges...", color=FontColor.INFO)

    dv_chain_poses = []
    dv_confidences = []

    for i in range(N - 1):
        T_ij, _, conf = predict_relative_pose(frontend, feats[i], poss[i],
                                               feats[i+1], poss[i+1], None)
        dv_chain_poses.append(T_ij)
        dv_confidences.append(conf)

    # 对每个间隔 d=2,3,...,N-1 计算链式 vs 直接的漂移
    print_msg(f"\n{'间隔':>6} {'DV链式漂移(平移↓)':>18} {'DV旋转漂移(弧度↓)':>20} "
              f"{'MV链式漂移(平移↓)':>18} {'MV旋转漂移(弧度↓)':>20} "
              f"{'改善%':>10}", color=FontColor.INFO)
    print_msg("-" * 100, color=FontColor.INFO)

    total_dv_drift = 0
    total_mv_drift = 0
    sample_count = 0

    for gap in range(2, min(N, 20)):  # 间隔从2到19
        dv_drifts = []
        mv_drifts = []

        for start in range(N - gap):
            end = start + gap

            # 直接预测 0 → gap
            T_direct, _, _ = predict_relative_pose(
                frontend, feats[start], poss[start], feats[end], poss[end], None)

            # ---- DV链式：只走相邻边 ----
            chain = dv_chain_poses[start:end]
            dv_d, dv_r = compute_chain_drift(chain, T_direct)
            dv_drifts.append(dv_d)

            # ---- MV链式：每步包含最多3个前向边 ----
            mv_chain_poses = []
            for k in range(start, end):
                # MV 方式：使用更远的边，减少链长
                step_size = min(gap, 3)  # MV窗口=3
                if end - k <= step_size:
                    # 最后一跳直接预测
                    T_step, _, _ = predict_relative_pose(
                        frontend, feats[k], poss[k], feats[end], poss[end], None)
                    mv_chain_poses.append(T_step)
                    break
                else:
                    T_step, _, _ = predict_relative_pose(
                        frontend, feats[k], poss[k], feats[k+step_size], poss[k+step_size], None)
                    mv_chain_poses.append(T_step)

            mv_d, mv_r = compute_chain_drift(mv_chain_poses, T_direct)
            mv_drifts.append(mv_d)

        avg_dv = np.mean(dv_drifts) * 100  # 转 cm
        avg_mv = np.mean(mv_drifts) * 100
        improv = (avg_dv - avg_mv) / max(avg_dv, 1e-8) * 100

        total_dv_drift += avg_dv
        total_mv_drift += avg_mv
        sample_count += 1

        # 只打印部分间隔，避免过多输出
        if gap <= 5 or gap % 5 == 0:
            print_msg(f"{gap:>6}  {avg_dv:>14.3f}cm           {dv_r:>14.4f}          "
                      f"{avg_mv:>14.3f}cm           {mv_r:>14.4f}          "
                      f"{'✅' if improv > 10 else ''} {improv:>+6.1f}%",
                      color=FontColor.INFO)

    print_msg("=" * 100, color=FontColor.INFO)
    print_msg(f"\n{'':>6}  DV平均漂移: {total_dv_drift/max(sample_count,1):.3f}cm  |  "
              f"MV平均漂移: {total_mv_drift/max(sample_count,1):.3f}cm  |  "
              f"改善: {(total_dv_drift-total_mv_drift)/max(total_dv_drift,1e-8)*100:.1f}%",
              color=FontColor.PoseGraphOpt)
    print_msg("", color=FontColor.INFO)


if __name__ == "__main__":
    main()
