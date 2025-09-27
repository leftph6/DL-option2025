#!/bin/bash

echo "============================================================"
echo "Deep BSDE 套利策略回测系统"
echo "============================================================"
echo

echo "正在激活conda环境..."
source activate deep-bsde-pricing

if [ $? -ne 0 ]; then
    echo
    echo "❌ 环境激活失败！请先运行 setup_unix.sh 创建环境"
    echo
    exit 1
fi

echo "✓ 环境激活成功！"
echo

echo "选择运行模式:"
echo "  1. 运行简单示例 (推荐)"
echo "  2. 运行高级示例"
echo "  3. 无图形界面模式"
echo

read -p "请输入选择 (1-3): " choice

case $choice in
    1)
        echo
        echo "启动简单示例..."
        python scripts/run_arbitrage.py --mode simple
        ;;
    2)
        echo
        echo "启动高级示例..."
        python scripts/run_arbitrage.py --mode advanced
        ;;
    3)
        echo
        echo "启动无图形界面模式..."
        python scripts/run_arbitrage.py --mode simple --no-gui
        ;;
    *)
        echo
        echo "❌ 无效选择，启动简单示例..."
        python scripts/run_arbitrage.py --mode simple
        ;;
esac

echo
echo "程序执行完成！"
echo
echo "生成的文件位置:"
echo "  output/arbitrage_pnl_curve_*.png  - PNL曲线图"
echo "  output/summary_statistics_*.png   - 汇总统计图"
echo "  output/arbitrage_results_*.csv    - 汇总结果数据"
echo "  output/detailed_results_*.csv     - 详细结果数据"
echo
