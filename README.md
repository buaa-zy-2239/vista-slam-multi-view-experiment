# ViSTA-SLAM Multi-View Experiment

多视图对称关联对比实验工具包。对原始 ViSTA-SLAM 的增强，用于验证多视图对称关联相对于双视图基线的优势。

## 在 Colab 上运行

### 一键设置

```python
# 在 Colab Notebook 的第一个 cell 中运行
!git clone https://github.com/zhangganlin/vista-slam.git
!git clone <您的实验仓库URL> experiment
!cp -r experiment/* vista-slam/
%cd vista-slam
!bash experiment/setup_colab.sh
```

### 手动步骤

```bash
# 1. 克隆原始 ViSTA-SLAM
git clone https://github.com/zhangganlin/vista-slam.git
cd vista-slam

# 2. 克隆本实验仓库并覆盖文件
git clone <您的实验仓库URL> experiment_repo
cp -r experiment_repo/* ./

# 3. 安装依赖
pip install torch torchvision xformers pypose munch einops opencv-python-headless scipy
pip install -r requirements.txt

# 4. 下载预训练权重
mkdir -p pretrains
wget -O pretrains/frontend_sta_weights.pth \
  "https://huggingface.co/zhangganlin/vista_slam/resolve/main/frontend_sta_weights.pth?download=true"
wget -O pretrains/ORBvoc.txt \
  "https://huggingface.co/zhangganlin/vista_slam/resolve/main/ORBvoc.txt?download=true"

# 5. 运行实验（CPU上约需3-5分钟/帧）
python3 experiment_dv_vs_mv.py --images "PATH/TO/IMAGES/*.png" --max-frames 3

# 使用默认 TUM RGB-D 数据集
# 先下载 TUM 数据:
# wget -r -np -nH --cut-dirs=4 -R "index.html*" \
#   https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz
# tar -xzf rgbd_dataset_freiburg1_xyz.tgz
# python3 experiment_dv_vs_mv.py --images "rgbd_dataset_freiburg1_xyz/rgb/*.png"
```

## 文件清单

```
experiment_repo/
├── README.md                          # 本文件
├── setup_colab.sh                     # Colab 一键设置脚本
├── experiment_dv_vs_mv.py            # DV vs MV 对比实验主脚本
├── run_multi_view.py                  # 多视图 SLAM 运行入口
├── configs/
│   └── multi_view.yaml               # 多视图配置参数
└── vista_slam/
    ├── multi_view_slam.py             # MultiViewOnlineSLAM 子类
    └── multi_view/                    # 多视图核心模块
        ├── __init__.py
        ├── adaptive_window.py         # 自适应对称窗口
        ├── consistency_loss.py        # 多视图一致性损失
        ├── topology.py                # 拓扑选择器
        ├── view_graph.py             # 视图关系图 + GAT
        └── optimizer.py              # 多视图 BA 优化器
```

## 实验指标说明

| 指标 | 含义 | 更优方向 |
|------|------|----------|
| 对称性误差 | `||T_ij · T_ji - I||` | ↓ 越小越好 |
| 置信度 | 网络对位姿预测的自我评估 | ↑ 越大越好 |
| 深度尺度一致性 | 重叠视图深度比偏离1的程度 | ↓ 越小越好 |
| 边数量 | 视图间约束边数量 | ↑ 越多约束越强 |

## 引用

如使用本实验代码，请引用原始 ViSTA-SLAM：

```bibtex
@misc{zhang2025vistaslam,
  title={{ViSTA-SLAM}: Visual {SLAM} with Symmetric Two-view Association},
  author={Ganlin Zhang and Shenhan Qian and Xi Wang and Daniel Cremers},
  year={2025},
  eprint={2509.01584},
  archivePrefix={arXiv},
}
```
