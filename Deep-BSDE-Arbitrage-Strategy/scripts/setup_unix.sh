#!/bin/bash

echo "============================================================"
echo "Deep BSDE 套利策略回测系统 - Unix 环境配置"
echo "============================================================"
echo

echo "检查系统要求..."
echo

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3未安装"
    echo "请先安装Python 3.9+"
    echo "Ubuntu/Debian: sudo apt install python3 python3-pip"
    echo "CentOS/RHEL: sudo yum install python3 python3-pip"
    echo "macOS: brew install python3"
    exit 1
else
    echo "✓ Python3已安装"
fi

# 检查Conda
if ! command -v conda &> /dev/null; then
    echo "❌ Conda未安装"
    echo "请先安装Anaconda或Miniconda"
    echo "下载地址: https://www.anaconda.com/products/distribution"
    exit 1
else
    echo "✓ Conda已安装"
fi

echo
echo "选择安装方式:"
echo "  1. 完整安装 (推荐，包含GPU支持)"
echo "  2. 最小安装 (仅CPU版本)"
echo "  3. 跳过环境创建，仅安装依赖"
echo

read -p "请输入选择 (1-3): " choice

case $choice in
    1)
        echo
        echo "使用完整环境配置..."
        conda env create -f environment.yml
        if [ $? -ne 0 ]; then
            echo "❌ 环境创建失败！"
            exit 1
        fi
        ;;
    2)
        echo
        echo "使用最小环境配置..."
        conda env create -f environment_minimal.yml
        if [ $? -ne 0 ]; then
            echo "❌ 环境创建失败！"
            exit 1
        fi
        ;;
    3)
        echo
        echo "跳过环境创建，直接安装依赖..."
        pip install -r requirements.txt
        if [ $? -ne 0 ]; then
            echo "❌ 依赖安装失败！"
            exit 1
        fi
        ;;
    *)
        echo
        echo "默认使用完整环境配置..."
        conda env create -f environment.yml
        if [ $? -ne 0 ]; then
            echo "❌ 环境创建失败！"
            exit 1
        fi
        ;;
esac

echo
echo "✓ 环境配置完成！"
echo

echo "正在激活环境..."
source activate deep-bsde-pricing

echo
echo "验证安装..."
python -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA可用:', torch.cuda.is_available())"

echo
echo "============================================================"
echo "安装完成！现在可以运行程序了"
echo "============================================================"
echo
echo "运行命令:"
echo "  python scripts/run_arbitrage.py"
echo "  或"
echo "  ./scripts/run_arbitrage.sh"
echo
echo "项目结构:"
echo "  src/                    - 源代码"
echo "  scripts/                - 脚本文件"
echo "  docs/                   - 文档"
echo "  output/                 - 输出文件"
echo "  examples/               - 示例代码"
echo "============================================================"
echo
