import torch
import torch.nn as nn


class MultiViewBundleAdjustment:
    def __init__(self, max_iterations=20, convergence_threshold=1e-4):
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

    def optimize(self, poses, depths, edges, intrinsics=None):
        if len(poses) < 2:
            return poses, depths

        optimized_poses = [p.clone() for p in poses]
        optimized_depths = [d.clone() for d in depths]

        for iteration in range(self.max_iterations):
            residuals = []
            for (i, j) in edges:
                T_ij = optimized_poses[i].inverse() @ optimized_poses[j]
                T_ji = optimized_poses[j].inverse() @ optimized_poses[i]
                res = torch.norm(T_ij @ T_ji - torch.eye(4, device=poses[0].device))
                residuals.append(res)

            if not residuals:
                break
            avg_res = sum(residuals) / len(residuals)
            if avg_res < self.convergence_threshold:
                break

            grad = self._compute_gradient(optimized_poses, edges)
            lr = 0.1 / (1.0 + iteration)
            for k in range(len(optimized_poses)):
                if k < len(grad):
                    delta = torch.eye(4, device=poses[0].device)
                    delta[:3, 3] = -lr * grad[k][:3]
                    optimized_poses[k] = optimized_poses[k] @ delta

        return optimized_poses, optimized_depths

    def _compute_gradient(self, poses, edges):
        grads = [torch.zeros(4, device=poses[0].device) for _ in poses]
        for (i, j) in edges:
            T_ij = poses[i].inverse() @ poses[j]
            T_ji = poses[j].inverse() @ poses[i]
            err = T_ij @ T_ji - torch.eye(4, device=poses[0].device)
            g_i = err[:3, 3] * 0.5
            g_j = -err[:3, 3] * 0.5
            grads[i][:3] += g_i
            grads[j][:3] += g_j
        return grads
