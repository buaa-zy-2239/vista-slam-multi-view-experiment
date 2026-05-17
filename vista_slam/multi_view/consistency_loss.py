import torch
import torch.nn as nn
import torch.nn.functional as F


class AleatoricLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_var = nn.Parameter(torch.zeros(1))

    def forward(self, loss_value):
        precision = torch.exp(-self.log_var)
        return precision * loss_value + 0.5 * self.log_var


class MultiViewConsistencyLoss(nn.Module):
    def __init__(self, use_learned_weights=True, w_symmetry=1.0, w_depth=1.0,
                 w_global=0.5, w_cycle=0.3):
        super().__init__()
        self.w_symmetry = w_symmetry
        self.w_depth = w_depth
        self.w_global = w_global
        self.w_cycle = w_cycle
        self.use_learned_weights = use_learned_weights

        if use_learned_weights:
            self.aleatoric_sym = AleatoricLoss()
            self.aleatoric_depth = AleatoricLoss()
            self.aleatoric_global = AleatoricLoss()
            self.aleatoric_cycle = AleatoricLoss()

    def forward(self, pred_poses, pred_depths, pred_confs=None):
        total_loss = 0.0
        details = {}
        N = len(pred_poses)
        if N < 2:
            return total_loss, details

        device = pred_poses[0].device
        I_4 = torch.eye(4, device=device)

        # 1) pairwise symmetry loss
        sym_loss = 0.0
        pair_count = 0
        for i in range(N):
            for j in range(i+1, N):
                T_ij = pred_poses[i].inverse() @ pred_poses[j]
                T_ji = pred_poses[j].inverse() @ pred_poses[i]
                err = torch.norm(T_ij @ T_ji - I_4)
                conf = 1.0
                if pred_confs is not None:
                    conf = (pred_confs[i].mean() + pred_confs[j].mean()).item()
                sym_loss += err * conf
                pair_count += 1

        sym_loss = sym_loss / max(pair_count, 1)
        if self.use_learned_weights:
            total_loss += self.aleatoric_sym(sym_loss)
        else:
            total_loss += self.w_symmetry * sym_loss
        details['symmetry_loss'] = sym_loss.item()

        # 2) depth consistency
        depth_loss = 0.0
        depth_count = 0
        for i in range(N):
            for j in range(i+1, N):
                d_i = pred_depths[i]
                d_j = pred_depths[j]
                s_i = d_i.mean()
                s_j = d_j.mean()
                if s_i > 0 and s_j > 0:
                    err = F.l1_loss(d_i / s_i, d_j / s_j)
                    depth_loss += err
                    depth_count += 1

        depth_loss = depth_loss / max(depth_count, 1)
        if self.use_learned_weights:
            total_loss += self.aleatoric_depth(depth_loss)
        else:
            total_loss += self.w_depth * depth_loss
        details['depth_consistency_loss'] = depth_loss.item()

        # 3) global consistency (cycle)
        if N >= 3:
            T_cycle = I_4
            for i in range(N):
                T_cycle = T_cycle @ (pred_poses[i].inverse() @ pred_poses[(i+1) % N])
            global_loss = torch.norm(T_cycle - I_4)

            triple_loss = 0.0
            triple_count = 0
            for i in range(N):
                for j in range(i+1, N):
                    for k in range(j+1, N):
                        T_ij = pred_poses[i].inverse() @ pred_poses[j]
                        T_jk = pred_poses[j].inverse() @ pred_poses[k]
                        T_ki = pred_poses[k].inverse() @ pred_poses[i]
                        triple_loss += torch.norm(T_ij @ T_jk @ T_ki - I_4)
                        triple_count += 1

            if self.use_learned_weights:
                total_loss += self.aleatoric_global(global_loss)
                if triple_count > 0:
                    total_loss += self.aleatoric_cycle(triple_loss / triple_count)
            else:
                total_loss += self.w_global * global_loss
                if triple_count > 0:
                    total_loss += self.w_cycle * (triple_loss / triple_count)

            details['global_cycle_loss'] = global_loss.item()
            details['triple_cycle_loss'] = (triple_loss / max(triple_count, 1)).item()

        return total_loss, details
