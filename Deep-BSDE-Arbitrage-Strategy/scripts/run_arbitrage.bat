@echo off
chcp 65001 >nul
echo ============================================================
echo Deep BSDE 套利策略回测系统
echo ============================================================
echo.

echo 正在激活conda环境...
call conda activate deep-bsde-pricing

if errorlevel 1 (
    echo.
    echo ❌ 环境激活失败！请先运行 setup_windows.bat 创建环境
    echo.
    pause
    exit /b 1
)

echo ✓ 环境激活成功！
echo.

echo 选择运行模式:
echo   1. 运行简单示例 (推荐)
echo   2. 运行高级示例
echo   3. 无图形界面模式
echo.

set /p choice="请输入选择 (1-3): "

if "%choice%"=="1" (
    echo.
    echo 启动简单示例...
    python scripts/run_arbitrage.py --mode simple
) else if "%choice%"=="2" (
    echo.
    echo 启动高级示例...
    python scripts/run_arbitrage.py --mode advanced
) else if "%choice%"=="3" (
    echo.
    echo 启动无图形界面模式...
    python scripts/run_arbitrage.py --mode simple --no-gui
) else (
    echo.
    echo ❌ 无效选择，启动简单示例...
    python scripts/run_arbitrage.py --mode simple
)

echo.
echo 程序执行完成！
echo.
echo 生成的文件位置:
echo   output/arbitrage_pnl_curve_*.png  - PNL曲线图
echo   output/summary_statistics_*.png   - 汇总统计图
echo   output/arbitrage_results_*.csv    - 汇总结果数据
echo   output/detailed_results_*.csv     - 详细结果数据
echo.
pause
