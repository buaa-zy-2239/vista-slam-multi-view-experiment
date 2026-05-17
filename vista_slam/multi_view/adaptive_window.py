import torch
import numpy as np

class AdaptiveSymmetricWindow:
    def __init__(self, max_window_size=5, min_window_size=2):
        self.max_window_size = max_window_size
        self.min_window_size = min_window_size
        self.window = []

    def add_view(self, view):
        self.window.append(view)
        if len(self.window) > self.max_window_size:
            self.window.pop(0)

    def get_window(self, complexity=None):
        if complexity is None:
            return self.window[-self.max_window_size:]
        window_size = self._compute_size(complexity)
        return self.window[-window_size:]

    def _compute_size(self, complexity):
        if complexity > 0.8:
            return self.min_window_size
        elif complexity > 0.5:
            return (self.min_window_size + self.max_window_size) // 2
        return self.max_window_size

    def reset(self):
        self.window.clear()


class WindowManager:
    def __init__(self, config=None):
        cfg = config or {}
        self.window_size = cfg.get('window_size', 5)
        self.sliding_interval = cfg.get('sliding_interval', 1)
        self.overlap_threshold = cfg.get('overlap_threshold', 0.3)
        self.frame_buffer = []
        self.view_buffer = []

    def add_frame(self, frame_data):
        self.frame_buffer.append(frame_data)
        if len(self.frame_buffer) > self.window_size * 3:
            self.frame_buffer.pop(0)

    def should_select_keyframe(self, new_frame, last_kf_idx):
        if last_kf_idx is None:
            return True
        gap = len(self.frame_buffer) - 1 - last_kf_idx
        return gap >= self.sliding_interval

    def get_current_window(self):
        if len(self.frame_buffer) < self.window_size:
            return list(self.frame_buffer)
        return list(self.frame_buffer[-self.window_size:])

    def estimate_overlap_ratio(self):
        window = self.get_current_window()
        if len(window) < 2:
            return 1.0
        return min(1.0, 0.5 * len(window))

    def estimate_scene_complexity(self, frame=None):
        if frame is None and len(self.frame_buffer) == 0:
            return 0.5
        img = frame if frame is not None else self.frame_buffer[-1]
        if isinstance(img, torch.Tensor):
            if img.ndim == 3:
                img = img.unsqueeze(0)
            grad_x = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).mean().item()
            grad_y = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).mean().item()
            texture = (grad_x + grad_y) / 2.0
            complexity = min(1.0, texture * 10.0)
            return complexity
        return 0.5

    def reset(self):
        self.frame_buffer.clear()
        self.view_buffer.clear()
