#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单示例：如何使用Deep BSDE套利策略回测系统
"""

import sys
from pathlib import Path

# 添加src目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

from deep_bsde_model import create_model
from arbitrage_strategy import create_strategy
from backtest_engine import create_backtest_engine

def main():
    """简单示例主函数"""
    print("=" * 60)
    print("Deep BSDE 套利策略回测系统 - 简单示例")
    print("=" * 60)
    
    # 1. 创建模型
    print("步骤1: 创建Deep BSDE模型")
    model = create_model(d=1, N=50, T=1.0, r=0.05, hidden_size=64)
    print(f"✓ 模型创建完成，设备: {model.device}")
    
    # 2. 训练模型
    print("\n步骤2: 训练模型")
    print("训练中...")
    losses = model.train_model(num_epochs=50, batch_size=64, S0=100.0, K=100.0, sigma=0.2)
    print(f"✓ 训练完成，最终损失: {losses[-1]:.6f}")
    
    # 3. 创建策略
    print("\n步骤3: 创建套利策略")
    strategy = create_strategy(model, risk_free_rate=0.05, transaction_cost=0.001)
    print("✓ 策略创建完成")
    
    # 4. 创建回测引擎
    print("\n步骤4: 创建回测引擎")
    engine = create_backtest_engine(strategy, output_dir="output")
    print("✓ 回测引擎创建完成")
    
    # 5. 生成价格路径
    print("\n步骤5: 生成价格路径")
    price_paths = engine.simulate_gbm_paths(
        batch_size=3, T=1.0, N=50, d=1, 
        r=0.05, sigma=0.2, S0=100.0
    )
    print(f"✓ 生成了 {price_paths.shape[0]} 条价格路径")
    
    # 6. 运行回测
    print("\n步骤6: 运行回测")
    backtest_results = engine.run_backtest(price_paths, K=100.0, initial_capital=10000.0)
    
    # 7. 显示结果
    summary = backtest_results['summary']
    print("\n" + "=" * 60)
    print("回测结果:")
    print("=" * 60)
    print(f"平均最终PNL: {summary['mean_final_pnl']:.2f}")
    print(f"PNL标准差: {summary['std_final_pnl']:.2f}")
    print(f"平均最大回撤: {summary['mean_max_drawdown']:.2f}")
    print(f"平均夏普比率: {summary['mean_sharpe_ratio']:.4f}")
    print(f"胜率: {summary['win_rate']:.2%}")
    print("=" * 60)
    
    # 8. 生成图表
    print("\n步骤7: 生成图表")
    first_result = backtest_results['individual_results'][0]
    plot_filename = engine.plot_pnl_curve(first_result, save_plot=True)
    print(f"✓ PNL曲线图已保存: {plot_filename}")
    
    # 9. 保存结果
    print("\n步骤8: 保存结果")
    csv_filename = engine.save_results_to_csv(backtest_results)
    detailed_filename = engine.save_detailed_results_to_csv(first_result)
    print(f"✓ 结果已保存: {csv_filename}, {detailed_filename}")
    
    print("\n" + "=" * 60)
    print("示例运行完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
