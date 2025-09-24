#!/bin/bash
# Deep-BSDE 程序运行脚本
# 使用方法: ./run.sh

echo "=== Deep-BSDE 程序启动 ==="

# 设置环境
source setup_env.sh

if [ $? -eq 0 ]; then
    echo "开始运行Deep-BSDE程序..."
    echo "=========================================="
    python mvp.py
    echo "=========================================="
    echo "程序运行完成"
else
    echo "环境配置失败，程序退出"
    exit 1
fi
