# ViSTA-SLAM 多视图对称关联研究 —— 实验档案

> **归档日期**: 2026-05-17
> **目标**: 探索 ViSTA-SLAM 框架下多视图对称关联的创新空间
> **约束条件**: 无 GPU 训练资源，仅靠 Colab T4 免费 GPU 推理
> **结论**: 推理层面的多视图增强空间已被 ViSTA-SLAM 原作者充分挖掘；
>          真正的创新空间在训练层面（需要 GPU 训练资源）。

---

## 目录

1. [研究背景](#1-研究背景)
2. [实验历程](#2-实验历程)
3. [代码仓库结构](#3-代码仓库结构)
4. [实验结果汇总](#4-实验结果汇总)
5. [关键结论](#5-关键结论)
6. [未来工作建议](#6-未来工作建议)
7. [附录](#7-附录)

---

## 1. 研究背景

### 1.1 项目起点

ViSTA-SLAM (Zhang et al., 2025) 是一个实时单目稠密 SLAM 系统，其核心创新是 **对称双视图关联（Symmetric Two-view Association, STA）**——通过双向 Transformer 解码器同时预测两个视图间的深度和位姿，利用对称性约束提高精度。

### 1.2 研究问题

> **核心问题**: 能否将双视图对称关联扩展为**多视图对称关联**，从而进一步提升 SLAM 的精度和鲁棒性？

### 1.3 约束条件

| 约束 | 说明 |
|------|------|
| **无 GPU 训练** | 无法重新训练模型，只能基于现有预训练权重做推理 |
| **Colab 免费 GPU** | Tesla T4 (16GB)，受限于 Colab 的 90 分钟会话限制 |
| **依赖环境限制** | DBoW3Py (C++ 编译)、xformers (CUDA 版本不匹配) 等组件在 Colab 上构建困难 |

---

## 2. 实验历程

### 2.1 实验一：双视图 vs 多视图对称性对比

**文件**: ~~`experiment_dv_vs_mv.py`~~（已删除）
**代码位置**: Git commit `7255051` ~ `806adf8`

**设计**: 3-10帧图像上分别用 DV (连接相邻帧) 和 MV (连接窗口内多帧) 做对称回归，对比对称性误差和置信度。

**发现**: MV 相比 DV：
| 指标 | 改善 |
|------|:----:|
| 约束边数量 | **+166.7%** ✅ |
| 对称性误差 | -3.7% ~ **-8.0%** ✅ |
| 置信度 | ~+0.01% |

**结论**: 单条边的对称性已经饱和（预训练模型已学得很好），MV 的优势在于**全局图结构的稠密度**而非单条边的质量。

### 2.2 实验二：链式累积漂移

**文件**: ~~`experiment_chain_drift.py`~~（已删除）
**代码位置**: Git commit `5ae7302`

**设计**: 对比链式累积位姿 VS 直接预测位姿的漂移大小。MV 用大步跳跃减少链长度，DV 逐帧累积。

**结果**: MV 的大基线预测误差更大，导致链式漂移反而劣于 DV。

**结论**: 实验设计有缺陷——MV 的优势不在"跳过帧"，而在"联合优化"。

### 2.3 实验三：多视图深度深度融合

**文件**: ~~`experiment_depth_fusion.py`~~（已删除）
**代码位置**: Git commit `fb063b2` ~ `5d15d03`

**设计**: 将邻居视图的深度图通过投影变换对齐到参考视图，做置信度加权融合。

**结果** (20帧, TUM fr1/xyz):
| 指标 | 融合前 | 融合后 | 改善 |
|------|:-----:|:-----:|:----:|
| RMSE | 1.8857 | 1.8543 | **+1.7%** |
| 融合覆盖率 | - | 99.7% | ✅ |
| 每帧耗时 | - | 0.04s | ✅ |

**结论**: 几何投影代码正确，但 ViSTA-SLAM 在 `save_data_all` 中已通过 `Sim3.scale()` 做了尺度校正，后处理融合改善空间有限（1.7%）。这个实验仍可作为**消融实验的 baseline**。

### 2.4 实验四：迭代式鲁棒位姿图优化 (IR-PGO) ⭐ **最终方向**

**文件**: `experiment_robust_pgo.py`
**代码位置**: Git commit `7f83b96` ~ `e82ecd2`

**设计**: 多次迭代 PGO，每次根据残差用 Tukey 权函数重加权，自动降低异常边的权重。

**理论收益**:
- M-估计理论保证对异常值鲁棒
- IRLS (迭代重加权最小二乘) 等价于稳健统计中的 Huber/Tukey 估计

**实验结果**:
| 测试集 | ATE (Standard) | ATE (Robust) | 改善 | 结论 |
|--------|:-------------:|:------------:|:----:|------|
| TUM fr1/xyz (50帧, 原地旋转) | 0.2484m | 0.2484m | 0% | 无回环、无异常边 |
| TUM fr2/desk (100帧, 有回环) | 2.2933m | 2.2933m | 0% | SLAM 配置错误导致 ATE 异常高 |

**问题诊断**:
1. `Cholesky decomposition failed` — 重加权导致信息矩阵奇异，需要正则化
2. ATE 高达 2.29m — `neighbor_edge_num=3` + `max_view_num=100` + `pgo_every=50` 配置不当
3. 无法在 Colab 上测试真正的回环序列

---

## 3. 代码仓库结构

```
experiment_repo/                        # GitHub: buaa-zy-2239/vista-slam-multi-view-experiment
├── README.md                           # 本文件（归档档案）
├── setup_colab.sh                      # Colab 环境配置脚本
├── experiment_robust_pgo.py            # ⭐ 最终实验：迭代式鲁棒 PGO
├── configs/
│   └── multi_view.yaml                 # 多视图配置模板
└── vista_slam/
    ├── multi_view_slam.py              # MultiViewOnlineSLAM 子类
    └── multi_view/                     # 多视图核心模块
        ├── __init__.py
        ├── adaptive_window.py          # 自适应对称窗口
        ├── consistency_loss.py         # 多视图一致性损失
        ├── topology.py                 # 拓扑选择器
        ├── view_graph.py              # 视图关系图 + GAT
        └── optimizer.py               # 多视图 BA 优化器
```

### Git 提交历史（16 commits）

```
e82ecd2 Fix: inject fake DBoW3Py module via sys.modules
300ae61 Fix: detect broken DBoW3Py via hasattr
f9ccc4d Fix: module-level DBoW3 patch before slam import
3ee4244 Update verification file list
2ae3cdc Fix: handle missing DBoW3Py in Colab
3fb9759 Fix: use nn.Module instead of pp.nn.Module
7f83b96 Implement Iterative Robust PGO (IR-PGO)
5d15d03 Fix: scale alignment for predicted depth
fc8c045 Fix: depth loading (16-bit mm to m)
bf71ccd Fix: K estimation, last frame depth direction
fb063b2 Switch to multi-view depth fusion
5ae7302 Add chain drift experiment
e05a71e Increase default max-frames to 10
806adf8 Fix: remove local default path
16a02f4 Fix path resolution for Colab compatibility
7255051 Initial commit
```

---

## 4. 实验结果汇总

### 4.1 三个创新方向的理论上限评估

| 方向 | 尝试文件 | 理论收益 | 实测收益 | 状态 |
|------|---------|:--------:|:--------:|:----:|
| ① 多视图对称性增强 | `experiment_dv_vs_mv.py` | 中 | 对称性+0% | ❌ 已被原作者实现 |
| ② 深度深度融合 | `experiment_depth_fusion.py` | 中 | RMSE +1.7% | ❌ 后处理空间有限 |
| ③ 迭代式鲁棒 PGO | `experiment_robust_pgo.py` | **高** | **无法验证** | ⚠️ 需要 GPU 服务器 |

### 4.2 关键技术验证结果

| 验证项 | 状态 | 说明 |
|--------|:----:|------|
| Monkey-patch 绕过 DBoW3Py | ✅ | `sys.modules` 注入假模块 + `slam.LoopDetector` 替换 |
| CPU 兼容 xformers attention | ✅ | 替换为 PyTorch 原生 SDPA |
| Colab GPU (T4) 上运行 SLAM | ✅ | 100帧 ~77秒 |
| ATE 轨迹评估 | ✅ | 含 Procrustes Sim(3) 对齐 |
| 迭代重加权 PGO | ✅ | Tukey 权函数 + IRLS |
| Cholesky 数值稳定性 | ❌ | 需要正则化改进 |

---

## 5. 关键结论

### 5.1 核心发现

**ViSTA-SLAM 的作者已经实现了"多视图对称关联"的基本功能。**

关键证据：在 `slam.py` 的 `step()` 中：

```python
farthest_neighbor = max(0, i - self.neighbor_edge_num)
for j in range(farthest_neighbor, i):
    self.connect_view_i_j(i, j)      # 连接多帧
```

`neighbor_edge_num` 参数控制连接到多少帧（默认1，即双视图；设为3即多视图）。同时 `connect_view_i_j` 自动为同一视图的多个节点添加**尺度约束边**。

**这意味着：推理层面的多视图处理不是创新，只是参数调整。**

### 5.2 真正的创新空间

```
推理层（可用的，但不是创新）      训练层（创新空间所在）
─────────────────────────────    ─────────────────────────────
neighbor_edge_num=3              多视图训练数据构建
多边位姿图优化                     多视图一致性损失函数
尺度融合（Sim3）                  同方差不确定性加权
                                 三/多视图对称模型训练
```

### 5.3 对研究者的建议

当前实验框架 (Colab + 预训练权重) **适用于验证性实验**，但**无法支撑真正的算法创新**。如果要在"多视图对称关联"方向上做有发表价值的创新，必须：

1. 申请 GPU 训练资源（至少单卡 A100/RTX 4090）
2. 阅读并修改 ViSTA-SLAM 的训练代码 (`vista_slam/sta_model/train.py`)
3. 设计新的多视图训练策略和损失函数
4. 在完整数据集上从头训练或微调

---

## 6. 未来工作建议

### 6.1 短期（有 GPU 后）

1. **修复 IR-PGO 数值稳定性问题**
   - 在 Cholesky 分解前添加正则化项: `A += 1e-6 * I`
   - 改用 Levenberg-Marquardt 优化器（已有）
   - 限制权重下限（防止完全剔除）

2. **在 fr2/desk / fr3/long_office 上做完整验证**
   - 配置正确的 SLAM 参数
   - 对比标准 PGO vs IR-PGO 的 ATE

### 6.2 中期（有新数据集）

1. **多视图训练数据构建**
   - 从 ScanNet / 7-Scenes 提取 3-5 帧窗口
   - 构建多视图训练样本

2. **多视图一致性损失函数**
   - 三帧闭环一致性约束
   - 跨视图深度一致性约束

### 6.3 长期（完整研究）

1. **端到端多视图对称模型**
   - 修改 STA 模型架构支持多视图输入
   - 从头训练多视图对称关联模型

---

## 7. 附录

### 7.1 Colab 运行指南

如果要在 Colab 上运行最终实验（`experiment_robust_pgo.py`）：

```python
import os
os.chdir('/content')

!rm -rf /content/exp_repo
!git clone https://github.com/buaa-zy-2239/vista-slam-multi-view-experiment.git exp_repo
!cp -r /content/exp_repo/* /content/vista-slam/

%cd /content/vista-slam
!bash /content/exp_repo/setup_colab.sh

!python3 /content/vista-slam/experiment_robust_pgo.py \
    --rgb "rgbd_dataset_freiburg2_desk/rgb/*.png" \
    --gt "rgbd_dataset_freiburg2_desk/groundtruth.txt" \
    --max-frames 100 --device cuda
```

### 7.2 依赖清单

```
torch>=2.0
torchvision
xformers (可选，CPU 自动降级为原生 attention)
pypose>=0.9
munch
einops
opencv-python-headless
scipy
colorama
pandas
```

### 7.3 引用

```bibtex
@misc{zhang2025vistaslam,
  title={{ViSTA-SLAM}: Visual {SLAM} with Symmetric Two-view Association},
  author={Ganlin Zhang and Shenhan Qian and Xi Wang and Daniel Cremers},
  year={2025},
  eprint={2509.01584},
  archivePrefix={arXiv},
}
```

---

> **最后更新**: 2026-05-17
> **实验人**: buaa-zy-2239
> **仓库地址**: https://github.com/buaa-zy-2239/vista-slam-multi-view-experiment
