#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
高级示例：多参数配置和批量回测
"""

import sys
import numpy as np
from pathlib import Path

# 添加src目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

from deep_bsde_model import create_model
from arbitrage_strategy import create_strategy
from backtest_engine import create_backtest_engine

def run_parameter_sweep():
    """参数扫描示例"""
    print("=" * 60)
    print("参数扫描示例")
    print("=" * 60)
    
    # 定义参数范围
    volatility_range = [0.1, 0.2, 0.3]
    strike_prices = [95, 100, 105]
    
    results = []
    
    for sigma in volatility_range:
        for K in strike_prices:
            print(f"\n测试参数: σ={sigma}, K={K}")
            
            # 创建模型
            model = create_model(d=1, N=50, T=1.0, r=0.05, hidden_size=64)
            
            # 快速训练
            losses = model.train_model(num_epochs=30, batch_size=64, S0=100.0, K=K, sigma=sigma)
            
            # 创建策略
            strategy = create_strategy(model, risk_free_rate=0.05, transaction_cost=0.001)
            
            # 创建回测引擎
            engine = create_backtest_engine(strategy, output_dir="output")
            
            # 生成价格路径
            price_paths = engine.simulate_gbm_paths(
                batch_size=10, T=1.0, N=50, d=1, 
                r=0.05, sigma=sigma, S0=100.0
            )
            
            # 运行回测
            backtest_results = engine.run_backtest(price_paths, K=K, initial_capital=10000.0)
            
            # 记录结果
            summary = backtest_results['summary']
            results.append({
                'sigma': sigma,
                'strike_price': K,
                'mean_pnl': summary['mean_final_pnl'],
                'std_pnl': summary['std_final_pnl'],
                'win_rate': summary['win_rate'],
                'sharpe_ratio': summary['mean_sharpe_ratio']
            })
            
            print(f"  平均PNL: {summary['mean_final_pnl']:.2f}")
            print(f"  胜率: {summary['win_rate']:.2%}")
    
    # 显示结果汇总
    print("\n" + "=" * 60)
    print("参数扫描结果汇总:")
    print("=" * 60)
    for result in results:
        print(f"σ={result['sigma']}, K={result['strike_price']}: "
              f"PNL={result['mean_pnl']:.2f}±{result['std_pnl']:.2f}, "
              f"胜率={result['win_rate']:.2%}, "
              f"夏普={result['sharpe_ratio']:.3f}")
    print("=" * 60)
    
    return results

def run_risk_analysis():
    """风险分析示例"""
    print("\n" + "=" * 60)
    print("风险分析示例")
    print("=" * 60)
    
    # 创建模型
    model = create_model(d=1, N=50, T=1.0, r=0.05, hidden_size=64)
    losses = model.train_model(num_epochs=50, batch_size=64, S0=100.0, K=100.0, sigma=0.2)
    
    # 创建策略
    strategy = create_strategy(model, risk_free_rate=0.05, transaction_cost=0.001)
    engine = create_backtest_engine(strategy, output_dir="output")
    
    # 生成大量路径进行风险分析
    print("生成大量路径进行风险分析...")
    price_paths = engine.simulate_gbm_paths(
        batch_size=100, T=1.0, N=50, d=1, 
        r=0.05, sigma=0.2, S0=100.0
    )
    
    # 运行回测
    backtest_results = engine.run_backtest(price_paths, K=100.0, initial_capital=10000.0)
    
    # 风险指标计算
    individual_results = backtest_results['individual_results']
    final_pnls = [r['final_pnl'] for r in individual_results]
    max_drawdowns = [r['max_drawdown'] for r in individual_results]
    
    # 计算VaR和CVaR
    pnl_array = np.array(final_pnls)
    var_95 = np.percentile(pnl_array, 5)  # 95% VaR
    cvar_95 = pnl_array[pnl_array <= var_95].mean()  # 95% CVaR
    
    print(f"风险指标:")
    print(f"  95% VaR: {var_95:.2f}")
    print(f"  95% CVaR: {cvar_95:.2f}")
    print(f"  最大回撤: {min(max_drawdowns):.2f}")
    print(f"  平均回撤: {np.mean(max_drawdowns):.2f}")
    
    # 生成风险分析图
    summary_filename = engine.plot_summary_statistics(backtest_results, save_plot=True)
    print(f"✓ 风险分析图已保存: {summary_filename}")

def main():
    """高级示例主函数"""
    print("=" * 60)
    print("Deep BSDE 套利策略回测系统 - 高级示例")
    print("=" * 60)
    
    # 1. 参数扫描
    parameter_results = run_parameter_sweep()
    
    # 2. 风险分析
    run_risk_analysis()
    
    print("\n" + "=" * 60)
    print("高级示例运行完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
