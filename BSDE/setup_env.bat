@echo off
REM Deep-BSDE PyTorch Environment Setup Script for Windows
REM 使用方法: setup_env.bat

echo === Deep-BSDE PyTorch 环境配置 ===

REM 检查Python版本
echo 检查Python版本...
python --version
if %errorlevel% neq 0 (
    echo 错误: Python未安装或不在PATH中
    pause
    exit /b 1
)

REM 检查是否在正确的目录
if not exist "mvp.py" (
    echo 错误: 请在包含mvp.py的目录中运行此脚本
    pause
    exit /b 1
)

REM 检查并安装依赖
echo 检查PyTorch环境...

REM 检查torch是否已安装
python -c "import torch" >nul 2>&1
if %errorlevel% neq 0 (
    echo PyTorch未安装，正在安装...
    pip install torch
    if %errorlevel% neq 0 (
        echo 错误: PyTorch安装失败
        pause
        exit /b 1
    )
) else (
    echo PyTorch已安装
)

REM 检查numpy是否已安装
python -c "import numpy" >nul 2>&1
if %errorlevel% neq 0 (
    echo NumPy未安装，正在安装...
    pip install numpy
    if %errorlevel% neq 0 (
        echo 错误: NumPy安装失败
        pause
        exit /b 1
    )
) else (
    echo NumPy已安装
)

REM 验证环境
echo 验证环境配置...
python -c "import torch; import numpy as np; print(f'PyTorch版本: {torch.__version__}'); print(f'NumPy版本: {np.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'设备: {torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")}'); print('环境配置完成！')"

if %errorlevel% equ 0 (
    echo === 环境配置成功 ===
    echo 现在可以运行: python mvp.py
) else (
    echo === 环境配置失败 ===
    pause
    exit /b 1
)

pause
