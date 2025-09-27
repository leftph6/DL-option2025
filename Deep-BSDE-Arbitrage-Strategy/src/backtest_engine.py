#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
回测引擎
用于执行套利策略的回测分析
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from datetime import datetime
import os
from .arbitrage_strategy import ArbitrageStrategy
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class BacktestEngine:
    """回测引擎"""
    
    def __init__(self, strategy: ArbitrageStrategy, output_dir: str = "output"):
        """
        初始化回测引擎
        
        Args:
            strategy: 套利策略实例
            output_dir: 输出目录
        """
        self.strategy = strategy
        self.output_dir = output_dir
        
        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def simulate_gbm_paths(self, batch_size: int, T: float, N: int, d: int, 
                          r: float, sigma: float, S0: float) -> torch.Tensor:
        """
        模拟GBM路径
        
        Args:
            batch_size: 批次大小
            T: 到期时间
            N: 时间步数
            d: 维度
            r: 无风险利率
            sigma: 波动率
            S0: 初始价格
            
        Returns:
            price_paths: 价格路径 (batch_size, N+1, d)
        """
        dt = T / N
        sqrt_dt = np.sqrt(dt)
        
        # 生成路径
        paths = np.zeros((batch_size, N + 1, d))
        paths[:, 0, :] = S0
        
        for i in range(1, N + 1):
            dW = np.random.randn(batch_size, d) * sqrt_dt
            paths[:, i, :] = paths[:, i-1, :] * np.exp((r - 0.5 * sigma**2) * dt + sigma * dW)
        
        return torch.tensor(paths, dtype=torch.float32)
    
    def run_backtest(self, price_paths: torch.Tensor, strike_price: float, 
                    initial_capital: float = 10000.0) -> Dict:
        """
        运行回测
        
        Args:
            price_paths: 价格路径 (batch_size, time_steps, d)
            strike_price: 执行价格
            initial_capital: 初始资本
            
        Returns:
            backtest_results: 回测结果
        """
        batch_size = price_paths.shape[0]
        results = []
        
        print(f"开始回测，共 {batch_size} 条路径...")
        
        for i in range(batch_size):
            price_path = price_paths[i, :, 0]  # 取第一个维度
            result = self.strategy.execute_arbitrage_strategy(price_path, strike_price, initial_capital)
            results.append(result)
        
        # 汇总结果
        final_pnls = [r['final_pnl'] for r in results]
        max_drawdowns = [r['max_drawdown'] for r in results]
        sharpe_ratios = [r['sharpe_ratio'] for r in results]
        
        return {
            'individual_results': results,
            'summary': {
                'mean_final_pnl': np.mean(final_pnls),
                'std_final_pnl': np.std(final_pnls),
                'mean_max_drawdown': np.mean(max_drawdowns),
                'mean_sharpe_ratio': np.mean(sharpe_ratios),
                'win_rate': np.mean([pnl > 0 for pnl in final_pnls]),
                'total_paths': batch_size
            }
        }
    
    def plot_pnl_curve(self, result: Dict, save_plot: bool = True, 
                      filename: Optional[str] = None) -> str:
        """
        绘制PNL曲线
        
        Args:
            result: 单条路径的回测结果
            save_plot: 是否保存图像
            filename: 自定义文件名
            
        Returns:
            filename: 保存的文件名
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        time_steps = result['time_steps']
        prices = result['prices']
        pnl = result['pnl']
        portfolio_values = result['portfolio_values']
        hedge_ratios = result['hedge_ratios']
        optimal_time = result['optimal_time']
        
        # 1. 价格路径和PNL曲线
        ax1_twin = ax1.twinx()
        ax1.plot(time_steps, prices, 'b-', alpha=0.7, label='价格路径')
        ax1.axvline(x=optimal_time, color='r', linestyle='--', alpha=0.8, label='最优停时')
        ax1_twin.plot(time_steps, pnl, 'g-', linewidth=2, label='PNL')
        ax1.set_xlabel('时间步')
        ax1.set_ylabel('价格', color='b')
        ax1_twin.set_ylabel('PNL', color='g')
        ax1.set_title('价格路径与PNL曲线')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 2. 组合价值变化
        ax2.plot(time_steps, portfolio_values, 'purple', linewidth=2, label='组合价值')
        ax2.axhline(y=10000, color='k', linestyle='--', alpha=0.7, label='初始资本')
        ax2.axvline(x=optimal_time, color='r', linestyle='--', alpha=0.8, label='最优停时')
        ax2.set_xlabel('时间步')
        ax2.set_ylabel('组合价值')
        ax2.set_title('组合价值变化')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 对冲比率
        ax3.plot(time_steps, hedge_ratios, 'orange', linewidth=2, label='对冲比率')
        ax3.axvline(x=optimal_time, color='r', linestyle='--', alpha=0.8, label='最优停时')
        ax3.set_xlabel('时间步')
        ax3.set_ylabel('对冲比率')
        ax3.set_title('动态对冲比率')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 仓位变化
        positions = result['positions']
        hedge_positions = result['hedge_positions']
        ax4.plot(time_steps, positions, 'b-', linewidth=2, label='期权仓位')
        ax4.plot(time_steps, hedge_positions, 'r-', linewidth=2, label='对冲仓位')
        ax4.axvline(x=optimal_time, color='r', linestyle='--', alpha=0.8, label='最优停时')
        ax4.set_xlabel('时间步')
        ax4.set_ylabel('仓位')
        ax4.set_title('仓位变化')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.output_dir}/arbitrage_pnl_curve_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✓ PNL曲线图已保存到: {filename}")
        
        plt.show()
        return filename if save_plot else None
    
    def plot_summary_statistics(self, backtest_results: Dict, save_plot: bool = True) -> str:
        """
        绘制汇总统计图
        
        Args:
            backtest_results: 回测结果
            save_plot: 是否保存图像
            
        Returns:
            filename: 保存的文件名
        """
        individual_results = backtest_results['individual_results']
        final_pnls = [r['final_pnl'] for r in individual_results]
        max_drawdowns = [r['max_drawdown'] for r in individual_results]
        sharpe_ratios = [r['sharpe_ratio'] for r in individual_results]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. PNL分布直方图
        ax1.hist(final_pnls, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(x=np.mean(final_pnls), color='red', linestyle='--', label=f'均值: {np.mean(final_pnls):.2f}')
        ax1.set_xlabel('最终PNL')
        ax1.set_ylabel('频次')
        ax1.set_title('PNL分布')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 最大回撤分布
        ax2.hist(max_drawdowns, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.axvline(x=np.mean(max_drawdowns), color='red', linestyle='--', label=f'均值: {np.mean(max_drawdowns):.2f}')
        ax2.set_xlabel('最大回撤')
        ax2.set_ylabel('频次')
        ax2.set_title('最大回撤分布')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 夏普比率分布
        ax3.hist(sharpe_ratios, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.axvline(x=np.mean(sharpe_ratios), color='red', linestyle='--', label=f'均值: {np.mean(sharpe_ratios):.2f}')
        ax3.set_xlabel('夏普比率')
        ax3.set_ylabel('频次')
        ax3.set_title('夏普比率分布')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. PNL vs 最大回撤散点图
        ax4.scatter(max_drawdowns, final_pnls, alpha=0.6, color='purple')
        ax4.set_xlabel('最大回撤')
        ax4.set_ylabel('最终PNL')
        ax4.set_title('PNL vs 最大回撤')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.output_dir}/summary_statistics_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✓ 汇总统计图已保存到: {filename}")
        
        plt.show()
        return filename if save_plot else None
    
    def save_results_to_csv(self, backtest_results: Dict, filename: Optional[str] = None) -> str:
        """
        保存结果到CSV文件
        
        Args:
            backtest_results: 回测结果
            filename: 自定义文件名
            
        Returns:
            filename: 保存的文件名
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.output_dir}/arbitrage_results_{timestamp}.csv"
        
        # 创建汇总结果
        summary_data = []
        for i, result in enumerate(backtest_results['individual_results']):
            summary_data.append({
                'Path_ID': i,
                'Final_PNL': result['final_pnl'],
                'Max_Drawdown': result['max_drawdown'],
                'Sharpe_Ratio': result['sharpe_ratio'],
                'Optimal_Time': result['optimal_time'],
                'Optimal_Value': result['optimal_value']
            })
        
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv(filename, index=False)
        
        print(f"✓ 汇总结果已保存到: {filename}")
        return filename
    
    def save_detailed_results_to_csv(self, result: Dict, filename: Optional[str] = None) -> str:
        """
        保存详细结果到CSV文件
        
        Args:
            result: 单条路径的详细结果
            filename: 自定义文件名
            
        Returns:
            filename: 保存的文件名
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.output_dir}/detailed_results_{timestamp}.csv"
        
        detailed_data = {
            'TimeStep': result['time_steps'],
            'Price': result['prices'],
            'Option_Position': result['positions'],
            'Hedge_Position': result['hedge_positions'],
            'Cash': result['cash'],
            'Portfolio_Value': result['portfolio_values'],
            'PNL': result['pnl'],
            'Hedge_Ratio': result['hedge_ratios']
        }
        
        df_detailed = pd.DataFrame(detailed_data)
        df_detailed.to_csv(filename, index=False)
        
        print(f"✓ 详细结果已保存到: {filename}")
        return filename

def create_backtest_engine(strategy: ArbitrageStrategy, output_dir: str = "output") -> BacktestEngine:
    """
    创建回测引擎
    
    Args:
        strategy: 套利策略实例
        output_dir: 输出目录
        
    Returns:
        engine: 回测引擎实例
    """
    return BacktestEngine(strategy, output_dir)

if __name__ == "__main__":
    # 示例使用
    from .deep_bsde_model import create_model
    from .arbitrage_strategy import create_strategy
    
    print("创建Deep BSDE模型...")
    model = create_model(d=1, N=50, T=1.0, r=0.05, hidden_size=64)
    
    print("训练模型...")
    losses = model.train_model(num_epochs=100, batch_size=128)
    
    print("创建套利策略...")
    strategy = create_strategy(model, risk_free_rate=0.05, transaction_cost=0.001)
    
    print("创建回测引擎...")
    engine = create_backtest_engine(strategy)
    
    print("回测引擎创建完成！")
