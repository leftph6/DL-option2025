#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Deep BSDE 套利策略回测系统 - 主运行脚本
"""

import os
import sys
import argparse
from pathlib import Path

# 添加src目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

from deep_bsde_model import create_model
from arbitrage_strategy import create_strategy
from backtest_engine import create_backtest_engine

def print_banner():
    """打印横幅"""
    print("=" * 60)
    print("Deep BSDE 套利策略回测系统")
    print("=" * 60)
    print()

def check_environment():
    """检查环境"""
    print("检查环境...")
    
    try:
        import torch
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        print("✓ 所有依赖包已安装")
        print(f"  - PyTorch: {torch.__version__}")
        print(f"  - NumPy: {np.__version__}")
        print(f"  - Pandas: {pd.__version__}")
        print(f"  - Matplotlib: {plt.__version__}")
        print(f"  - CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  - CUDA版本: {torch.version.cuda}")
            print(f"  - GPU数量: {torch.cuda.device_count()}")
            print(f"  - GPU名称: {torch.cuda.get_device_name(0)}")
        return True
    except ImportError as e:
        print(f"❌ 缺少依赖包: {e}")
        print("请运行: pip install -r requirements.txt")
        return False

def run_simple_example():
    """运行简单示例"""
    print("运行简单示例...")
    
    # 参数设置
    T = 1.0
    N = 50
    d = 1
    r = 0.05
    sigma = 0.2
    S0 = 100.0
    K = 100.0
    batch_size = 5
    
    print(f"参数设置:")
    print(f"  到期时间: {T}")
    print(f"  时间步数: {N}")
    print(f"  无风险利率: {r}")
    print(f"  波动率: {sigma}")
    print(f"  初始价格: {S0}")
    print(f"  执行价格: {K}")
    print(f"  回测路径数: {batch_size}")
    print("=" * 60)
    
    # 1. 创建模型
    print("创建Deep BSDE模型...")
    model = create_model(d=d, N=N, T=T, r=r, hidden_size=64)
    
    # 2. 训练模型
    print("训练模型...")
    losses = model.train_model(num_epochs=100, batch_size=128, S0=S0, K=K, sigma=sigma)
    
    # 3. 创建策略
    print("创建套利策略...")
    strategy = create_strategy(model, risk_free_rate=r, transaction_cost=0.001)
    
    # 4. 创建回测引擎
    print("创建回测引擎...")
    engine = create_backtest_engine(strategy, output_dir="output")
    
    # 5. 生成价格路径
    print("生成价格路径...")
    price_paths = engine.simulate_gbm_paths(batch_size, T, N, d, r, sigma, S0)
    
    # 6. 运行回测
    print("运行回测...")
    backtest_results = engine.run_backtest(price_paths, K, initial_capital=10000.0)
    
    # 7. 显示结果
    summary = backtest_results['summary']
    print("\n" + "=" * 60)
    print("回测结果汇总:")
    print("=" * 60)
    print(f"平均最终PNL: {summary['mean_final_pnl']:.2f}")
    print(f"PNL标准差: {summary['std_final_pnl']:.2f}")
    print(f"平均最大回撤: {summary['mean_max_drawdown']:.2f}")
    print(f"平均夏普比率: {summary['mean_sharpe_ratio']:.4f}")
    print(f"胜率: {summary['win_rate']:.2%}")
    print(f"总路径数: {summary['total_paths']}")
    print("=" * 60)
    
    # 8. 生成图表
    print("生成PNL曲线图...")
    first_result = backtest_results['individual_results'][0]
    plot_filename = engine.plot_pnl_curve(first_result, save_plot=True)
    
    print("生成汇总统计图...")
    summary_filename = engine.plot_summary_statistics(backtest_results, save_plot=True)
    
    # 9. 保存结果
    print("保存结果...")
    csv_filename = engine.save_results_to_csv(backtest_results)
    detailed_filename = engine.save_detailed_results_to_csv(first_result)
    
    print("\n" + "=" * 60)
    print("回测完成！生成的文件:")
    print("=" * 60)
    print(f"  - PNL曲线图: {plot_filename}")
    print(f"  - 汇总统计图: {summary_filename}")
    print(f"  - 汇总结果: {csv_filename}")
    print(f"  - 详细结果: {detailed_filename}")
    print("=" * 60)
    
    return backtest_results

def run_advanced_example():
    """运行高级示例"""
    print("运行高级示例...")
    
    # 这里可以添加更复杂的配置和参数
    # 例如：多资产、不同波动率、更长的回测期间等
    pass

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Deep BSDE 套利策略回测系统')
    parser.add_argument('--mode', choices=['simple', 'advanced'], default='simple',
                       help='运行模式: simple (简单示例) 或 advanced (高级示例)')
    parser.add_argument('--no-gui', action='store_true',
                       help='不显示图形界面')
    
    args = parser.parse_args()
    
    print_banner()
    
    # 检查环境
    if not check_environment():
        return 1
    
    # 设置matplotlib后端
    if args.no_gui:
        import matplotlib
        matplotlib.use('Agg')
    
    try:
        if args.mode == 'simple':
            results = run_simple_example()
        else:
            results = run_advanced_example()
        
        print("\n程序执行完成！")
        return 0
        
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        return 1
    except Exception as e:
        print(f"\n程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
