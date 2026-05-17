#!/bin/bash
# ================================================
# ViSTA-SLAM Multi-View Experiment - Colab Setup
# ================================================
# 在 Colab 上克隆原始 ViSTA-SLAM 后运行此脚本
# 用法: bash setup_colab.sh
# ================================================

set -e

echo "========================================"
echo "  ViSTA-SLAM Multi-View Experiment Setup"
echo "========================================"

# 1. 安装依赖
echo ""
echo "[1/5] Installing Python dependencies..."
pip install -q torchvision xformers pypose munch einops opencv-python-headless scipy colorama pandas setuptools

# 构建 DBoW3Py（回环检测组件）
echo ""
echo "[1b/5] Building DBoW3Py (loop detection)..."
# 先初始化子模块（DBoW3Py 是原始仓库的 git submodule）
if [ -f .gitmodules ]; then
    echo "  Initializing git submodules..."
    git submodule update --init --recursive 2>/dev/null || true
fi

if [ -f DBoW3Py/setup.py ]; then
    echo "  Building DBoW3Py..."
    cd DBoW3Py && pip install -q --no-build-isolation . && cd ..
    echo "  Done."
else
    echo "  DBoW3Py/setup.py not found. Cloning from upstream..."
    rm -rf DBoW3Py 2>/dev/null
    git clone --depth 1 https://github.com/zhangganlin/vista-slam.git _tmp_clone 2>/dev/null
    if [ -d _tmp_clone/DBoW3Py ]; then
        cp -r _tmp_clone/DBoW3Py ./
        cd DBoW3Py && pip install -q --no-build-isolation . && cd ..
        rm -rf _tmp_clone
        echo "  Done."
    else
        echo "  Fallback: will use simulated loop detection."
        rm -rf _tmp_clone 2>/dev/null
    fi
fi
echo "  Done."

# 2. 预训练权重
echo ""
echo "[2/5] Downloading pretrained weights..."
mkdir -p pretrains

WEIGHT_URL="https://huggingface.co/zhangganlin/vista_slam/resolve/main/frontend_sta_weights.pth?download=true"
VOCAB_URL="https://huggingface.co/zhangganlin/vista_slam/resolve/main/ORBvoc.txt?download=true"

if [ ! -f pretrains/frontend_sta_weights.pth ]; then
    echo "  Downloading frontend_sta_weights.pth (1.6GB)..."
    wget -q --show-progress -O pretrains/frontend_sta_weights.pth "$WEIGHT_URL"
else
    echo "  frontend_sta_weights.pth already exists, skipping."
fi

if [ ! -f pretrains/ORBvoc.txt ]; then
    echo "  Downloading ORBvoc.txt (139MB)..."
    wget -q --show-progress -O pretrains/ORBvoc.txt "$VOCAB_URL"
else
    echo "  ORBvoc.txt already exists, skipping."
fi
echo "  Done."

# 3. 验证
echo ""
echo "[3/5] Verifying files..."
python3 -c "
import os
files = [
    'experiment_robust_pgo.py',
    'vista_slam/multi_view_slam.py',
    'vista_slam/multi_view/__init__.py',
    'configs/multi_view.yaml',
    'pretrains/frontend_sta_weights.pth',
    'pretrains/ORBvoc.txt',
]
for f in files:
    exists = os.path.exists(f)
    size = os.path.getsize(f) if exists else 0
    status = '✅' if exists else '❌'
    print(f'  {status} {f} ({size/1e6:.1f}MB)' if size > 1e6 else f'  {status} {f}')
"
echo "  Done."

# 4. 检查 CUDA
echo ""
echo "[4/5] Checking CUDA..."
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
"
echo "  Done."

echo ""
echo "========================================"
echo "  Setup complete!"
echo "========================================"
echo ""
echo "  Run experiment:"
echo "    python3 experiment_dv_vs_mv.py --max-frames 3"
echo "    python3 experiment_dv_vs_mv.py --max-frames 5"
echo ""
echo "  or with your own images:"
echo "    python3 experiment_dv_vs_mv.py --images 'path/to/*.png'"
echo ""
