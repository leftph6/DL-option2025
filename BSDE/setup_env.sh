#!/bin/bash
# Deep-BSDE PyTorch Environment Setup Script
# 使用方法: source setup_env.sh

echo "=== Deep-BSDE PyTorch 环境配置 ==="

# 检查Python版本
echo "检查Python版本..."
python --version

# 检查是否在正确的目录
if [ ! -f "mvp.py" ]; then
    echo "错误: 请在包含mvp.py的目录中运行此脚本"
    return 1
fi

# 检查并安装依赖
echo "检查PyTorch环境..."

# 检查torch是否已安装
if ! python -c "import torch" 2>/dev/null; then
    echo "PyTorch未安装，正在安装..."
    pip install torch
    if [ $? -ne 0 ]; then
        echo "错误: PyTorch安装失败"
        return 1
    fi
else
    echo "PyTorch已安装"
fi

# 检查numpy是否已安装
if ! python -c "import numpy" 2>/dev/null; then
    echo "NumPy未安装，正在安装..."
    pip install numpy
    if [ $? -ne 0 ]; then
        echo "错误: NumPy安装失败"
        return 1
    fi
else
    echo "NumPy已安装"
fi

# 验证环境
echo "验证环境配置..."
python -c "
import torch
import numpy as np
print(f'PyTorch版本: {torch.__version__}')
print(f'NumPy版本: {np.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
print(f'设备: {torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")}')
print('环境配置完成！')
"

if [ $? -eq 0 ]; then
    echo "=== 环境配置成功 ==="
    echo "现在可以运行: python mvp.py"
else
    echo "=== 环境配置失败 ==="
    return 1
fi
