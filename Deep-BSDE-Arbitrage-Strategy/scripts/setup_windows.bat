@echo off
chcp 65001 >nul
echo ============================================================
echo Deep BSDE 套利策略回测系统 - Windows 环境配置
echo ============================================================
echo.

echo 检查系统要求...
echo.

REM 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python未安装或未添加到PATH
    echo 请先安装Python 3.9+并添加到系统PATH
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
) else (
    echo ✓ Python已安装
)

REM 检查Conda
conda --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Conda未安装或未添加到PATH
    echo 请先安装Anaconda或Miniconda
    echo 下载地址: https://www.anaconda.com/products/distribution
    pause
    exit /b 1
) else (
    echo ✓ Conda已安装
)

echo.
echo 选择安装方式:
echo   1. 完整安装 (推荐，包含GPU支持)
echo   2. 最小安装 (仅CPU版本)
echo   3. 跳过环境创建，仅安装依赖
echo.

set /p choice="请输入选择 (1-3): "

if "%choice%"=="1" (
    echo.
    echo 使用完整环境配置...
    conda env create -f environment.yml
    if errorlevel 1 (
        echo ❌ 环境创建失败！
        pause
        exit /b 1
    )
) else if "%choice%"=="2" (
    echo.
    echo 使用最小环境配置...
    conda env create -f environment_minimal.yml
    if errorlevel 1 (
        echo ❌ 环境创建失败！
        pause
        exit /b 1
    )
) else if "%choice%"=="3" (
    echo.
    echo 跳过环境创建，直接安装依赖...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ 依赖安装失败！
        pause
        exit /b 1
    )
) else (
    echo.
    echo 默认使用完整环境配置...
    conda env create -f environment.yml
    if errorlevel 1 (
        echo ❌ 环境创建失败！
        pause
        exit /b 1
    )
)

echo.
echo ✓ 环境配置完成！
echo.

echo 正在激活环境...
call conda activate deep-bsde-pricing

echo.
echo 验证安装...
python -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA可用:', torch.cuda.is_available())"

echo.
echo ============================================================
echo 安装完成！现在可以运行程序了
echo ============================================================
echo.
echo 运行命令:
echo   python scripts/run_arbitrage.py
echo   或
echo   scripts\run_arbitrage.bat
echo.
echo 项目结构:
echo   src/                    - 源代码
echo   scripts/                - 脚本文件
echo   docs/                   - 文档
echo   output/                 - 输出文件
echo   examples/               - 示例代码
echo ============================================================
echo.

pause
