import torch
import pypose as pp
import time

from .slam import OnlineSLAM
from .utils.slam_utils import FontColor, print_msg
from .multi_view.adaptive_window import WindowManager, AdaptiveSymmetricWindow
from .multi_view.topology import TopologySelector, SparseMultiViewAttention
from .multi_view.consistency_loss import MultiViewConsistencyLoss
from .multi_view.view_graph import ViewRelationGraph
from .multi_view.optimizer import MultiViewBundleAdjustment


class MultiViewOnlineSLAM(OnlineSLAM):
    def __init__(self, *args, multi_view_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = multi_view_config or {}

        self.use_multi_view = cfg.get('enabled', False)
        self.multi_view_window_size = cfg.get('window_size', 5)
        self.multi_view_topology = cfg.get('topology', 'hierarchical')
        self.multi_view_enable_loss = cfg.get('enable_consistency_loss', False)
        self.multi_view_enable_ba = cfg.get('enable_bundle_adjustment', False)

        if self.use_multi_view:
            self.window_manager = WindowManager({'window_size': self.multi_view_window_size})
            self.topology_selector = TopologySelector({
                'gpu_memory_gb': cfg.get('gpu_memory_gb', 8.0),
                'target_fps': cfg.get('target_fps', 15),
                'force_topology': cfg.get('force_topology', None),
            })
            self.view_graph = ViewRelationGraph(cfg.get('edge_threshold', 0.3))
            self.consistency_loss = MultiViewConsistencyLoss(
                use_learned_weights=cfg.get('use_learned_weights', True),
            )
            self.bundle_adjustment = MultiViewBundleAdjustment(
                max_iterations=cfg.get('ba_iterations', 20),
            )
            self._mv_cache_feats = []
            self._mv_cache_pos = []
            self._mv_cache_poses = []
            self._mv_cache_depths = []

    def reset(self):
        super().reset()
        if self.use_multi_view:
            self.window_manager.reset()
            self._mv_cache_feats.clear()
            self._mv_cache_pos.clear()
            self._mv_cache_poses.clear()
            self._mv_cache_depths.clear()

    def add_view(self, image, image_shape, view_name):
        super().add_view(image, image_shape, view_name)
        if self.use_multi_view:
            idx = self.view_num - 1
            self._mv_cache_feats.append(self.enc_features[-1])
            self._mv_cache_pos.append(self.enc_pos[-1])
            self.window_manager.add_frame(image)

    def _get_window_features(self):
        window_size = min(self.multi_view_window_size, len(self._mv_cache_feats))
        if window_size < 2:
            return [], []
        return (self._mv_cache_feats[-window_size:],
                self._mv_cache_pos[-window_size:])

    def _multi_view_regress(self):
        feats, poss = self._get_window_features()
        num_views = len(feats)
        if num_views < 2:
            return [], [], []

        topology = self.topology_selector.select(
            num_views,
            scene_complexity=self.window_manager.estimate_scene_complexity()
        )

        if topology == 'baseline' or topology == 'star':
            center_idx = num_views - 1
            center_feat = feats[center_idx]
            center_pos = poss[center_idx]
            all_poses = []
            all_depths = []
            all_confs = []
            for j in range(center_idx):
                se3, conf, confs, intri, depths = self.regress_two_views(
                    center_idx, j
                )
                all_poses.append(pp.mat2SE3(se3, atol=1e-3).data)
                all_depths.append(depths[0] if depths is not None else None)
                all_confs.append(confs[0] if confs is not None else None)

            main_view = self.view_num - 1
            if main_view > 0 and len(all_poses) > 0:
                self.connect_view_i_j(main_view, main_view - 1)
            return all_poses, all_depths, all_confs

        elif topology == 'hierarchical' or topology == 'sparse_hierarchical':
            center_idx = num_views - 1
            context_idxs = list(range(max(0, center_idx - self.neighbor_edge_num), center_idx))
            if len(context_idxs) == 0:
                return [], [], []

            all_poses = []
            all_depths = []
            all_confs = []
            for j in context_idxs:
                self.connect_view_i_j(center_idx + (self.view_num - 1 - center_idx), j)
                se3, conf, confs, intri, depths = self.regress_two_views(
                    center_idx, j
                )
                all_poses.append(pp.mat2SE3(se3, atol=1e-3).data)
                all_depths.append(depths[0] if depths is not None else None)
                all_confs.append(confs[0] if confs is not None else None)
            return all_poses, all_depths, all_confs

        return [], [], []

    def step(self, value, force_pgo=False, log_intermediate_results=False, output_folder=None):
        if not self.use_multi_view:
            return super().step(value, force_pgo, log_intermediate_results, output_folder)

        prepare_start = time.time()
        image = value['rgb']
        image_shape = value['shape']
        img_gray = value['gray']
        view_name = value['view_name']
        i = self.view_num

        if i == 0:
            sim3 = pp.identity_Sim3(1).to(self.device)
            self.pose_graph_nodes.poses[0] = sim3
        prepare_end = time.time()
        self.time_dict['prepare_data'] += (prepare_end - prepare_start)

        enc_start = time.time()
        self.add_view(image, image_shape, view_name)
        enc_end = time.time()
        self.time_dict['encoder'] += (enc_end - enc_start)

        graph_neighbor_start = time.time()
        farthest_neighbor = max(0, i - self.neighbor_edge_num)
        for j in range(farthest_neighbor, i):
            self.connect_view_i_j(i, j)
        graph_neighbor_end = time.time()

        loop_start = time.time()
        loop_candi_list = self.lc_detector.detect_loop(img_gray, farthest_neighbor)
        loop_end = time.time()
        self.time_dict['lc'] += (loop_end - loop_start)

        graph_loop_start = time.time()
        for j_sim in loop_candi_list[:self.loop_edge_num]:
            j = j_sim[0]
            self.connect_view_i_j(i, j)
        graph_loop_end = time.time()
        self.time_dict['graph_construction'] += (
            graph_neighbor_end - graph_neighbor_start +
            graph_loop_end - graph_loop_start
        )

        if self.view_num % self.pgo_every == 0 or force_pgo:
            if log_intermediate_results:
                self.save_data_all(f"{output_folder}",
                        save_view_graph=False, traj_name_postfix=f"{self.view_num-1}",
                        save_poses=True, save_images=False, save_scales=True,
                        save_depths=False, save_intrinsics=False,
                        save_confs=False, save_ply=False,
                        gt_poses=None, gt_depths=None, gt_intrinsics=None)

            opt_start = time.time()
            self.pose_graph_optimize()
            opt_end = time.time()
            self.time_dict['pgo'] += (opt_end - opt_start)

            torch.cuda.empty_cache()
            return True
        return False
