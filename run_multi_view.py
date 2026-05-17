import torch
import numpy as np
from vista_slam.utils.slam_utils import compute_local_pointclouds, FontColor, print_msg
from colorama import Fore
from vista_slam.multi_view_slam import MultiViewOnlineSLAM
import torch.backends.cudnn as cudnn
from vista_slam.datasets.slam_images_only import SLAM_image_only
import glob, time, argparse, yaml, munch, os, tqdm


def run_multi_view_demo():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--images", type=str, required=True)
    parser.add_argument("--output", type=str, default="output/multi_view_demo")
    parser.add_argument("--window-size", type=int, default=5, help="Multi-view window size")
    parser.add_argument("--topology", type=str, default="hierarchical",
                        choices=["baseline", "star", "hierarchical", "sparse_hierarchical"])
    parser.add_argument("--enable-mv-loss", action="store_true", help="Enable multi-view loss")
    parser.add_argument("--enable-ba", action="store_true", help="Enable bundle adjustment")
    parser.add_argument("--gpu-memory", type=float, default=8.0, help="Available GPU memory in GB")
    parser.add_argument("--target-fps", type=int, default=15, help="Target FPS")
    parser.add_argument("--no-mv", action="store_true", help="Force baseline mode")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    cfg['output_dir'] = args.output
    cfg['images_path'] = args.images
    cfg = munch.Munch(cfg)

    torch.manual_seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    cudnn.benchmark = True

    output_folder = cfg.output_dir
    os.makedirs(output_folder, exist_ok=True)
    dataset = SLAM_image_only(glob.glob(cfg.images_path), resolution=(224, 224))

    multi_view_config = {
        'enabled': not args.no_mv,
        'window_size': args.window_size,
        'topology': args.topology,
        'enable_consistency_loss': args.enable_mv_loss,
        'enable_bundle_adjustment': args.enable_ba,
        'gpu_memory_gb': args.gpu_memory,
        'target_fps': args.target_fps,
        'force_topology': None if args.no_mv else args.topology,
        'use_learned_weights': args.enable_mv_loss,
    }

    slam = MultiViewOnlineSLAM(
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
        pgo_every=cfg.pgo_every,
        multi_view_config=multi_view_config,
    )

    mode_str = "MULTI-VIEW" if not args.no_mv else "BASELINE (double-view)"
    print_msg(f"Running ViSTA-SLAM in {mode_str} mode", color=FontColor.INFO)
    if not args.no_mv:
        print_msg(f"  Window size: {args.window_size}", color=FontColor.INFO)
        print_msg(f"  Topology: {args.topology}", color=FontColor.INFO)

    stride = cfg.stride
    last = len(dataset)

    if cfg.keyframe_detection == "stride":
        stride_idxes = list(range(1, last, stride))
        if len(stride_idxes) > cfg.max_view_num:
            stride_idxes = list(torch.linspace(0, last - 1, steps=cfg.max_view_num).long())

    using_stride_for_kf = (cfg.keyframe_detection == "stride")
    t = 0
    is_optimized = False
    pbar = tqdm.tqdm(total=last, position=0,
                     bar_format=Fore.GREEN+"[Progress] "+Fore.RESET+"{percentage:3.0f}%|{bar}| [{n_fmt}/{total_fmt} frames]")

    while t < last:
        pbar.n = int(t+1)
        pbar.refresh()
        if using_stride_for_kf:
            is_keyframe = (t in stride_idxes)
        else:
            data = dataset[t]
            img_gray = (data.gray.squeeze(0).numpy() * 255).astype(np.uint8)
            is_keyframe = slam.flow_tracker.compute_disparity(img_gray)

        if not is_keyframe:
            if t == last - 1 and not is_optimized:
                slam.pose_graph_optimize()
                torch.cuda.empty_cache()
            pbar.update()
            t += 1
            continue

        data = dataset[t]
        img_gray = (data.gray.squeeze(0).numpy() * 255).astype(np.uint8)
        img = data.rgb.unsqueeze(0).to(slam.device)
        img_shape = torch.tensor(data.rgb.shape[1:3]).unsqueeze(0)

        input_value = {'rgb': img, 'shape': img_shape, 'gray': img_gray, 'view_name': data.img_name}
        is_optimized = slam.step(input_value, force_pgo=(t == last - 1))
        pbar.update()
        t += 1

    pbar.close()
    print_msg(f"Total keyframes: {slam.view_num}", color=FontColor.INFO)

    time_dict = slam.get_time_dict()
    print_msg(f"Total time: {time_dict['total']:.1f} s", color=FontColor.INFO)

    print_msg(f"Saving data to {output_folder} ...", color=FontColor.INFO, end=" ")
    slam.save_data_all(f"{output_folder}",
                        save_view_graph=True,
                        save_poses=True, save_images=True, save_scales=True,
                        save_depths=True, save_intrinsics=True,
                        save_confs=True, save_ply=True)
    print_msg("Done.", color=FontColor.INFO)


if __name__ == "__main__":
    run_multi_view_demo()
