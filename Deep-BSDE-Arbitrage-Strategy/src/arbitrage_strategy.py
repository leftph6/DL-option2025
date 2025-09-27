#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
套利策略实现
基于最优停时理论的套利策略
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from .deep_bsde_model import DeepBSDE
import warnings
warnings.filterwarnings('ignore')

class ArbitrageStrategy:
    """基于Deep BSDE的套利策略类"""
    
    def __init__(self, model: DeepBSDE, risk_free_rate: float = 0.05, 
                 transaction_cost: float = 0.001, max_position: float = 1.0):
        """
        初始化套利策略
        
        Args:
            model: 训练好的Deep BSDE模型
            risk_free_rate: 无风险利率
            transaction_cost: 交易成本（比例）
            max_position: 最大仓位限制
        """
        self.model = model
        self.r = risk_free_rate
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        self.device = next(model.parameters()).device
        
    def calculate_black_scholes_price(self, S: float, K: float, T: float) -> float:
        """计算Black-Scholes期权价格"""
        if T <= 0:
            return max(S - K, 0)
        
        r = self.r
        sigma = 0.2  # 假设波动率
        
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        # 使用正态分布累积分布函数的近似
        def norm_cdf(x):
            return 0.5 * (1 + np.tanh(x * np.sqrt(2/np.pi)))
        
        price = S * norm_cdf(d1) - K * np.exp(-r*T) * norm_cdf(d2)
        return max(price, 0)
    
    def calculate_delta(self, S: float, K: float, T: float) -> float:
        """计算Delta（对冲比率）"""
        if T <= 0:
            return 1.0 if S > K else 0.0
        
        r = self.r
        sigma = 0.2  # 假设波动率
        
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        
        # 使用正态分布累积分布函数的近似
        def norm_cdf(x):
            return 0.5 * (1 + np.tanh(x * np.sqrt(2/np.pi)))
        
        return norm_cdf(d1)
    
    def calculate_optimal_stopping_time(self, price_path: torch.Tensor, 
                                      strike_price: float) -> Tuple[int, float]:
        """
        计算最优停时
        
        Args:
            price_path: 价格路径 (time_steps,)
            strike_price: 执行价格
            
        Returns:
            optimal_time: 最优停时索引
            optimal_value: 最优价值
        """
        time_steps = price_path.shape[0]
        dt = 1.0 / (time_steps - 1)  # 假设T=1
        
        # 计算每个时间点的期权价值
        option_values = []
        
        for t in range(time_steps):
            current_price = price_path[t].item()
            remaining_time = 1.0 - t * dt
            
            if remaining_time <= 0:
                # 到期时的价值
                option_value = max(current_price - strike_price, 0)
            else:
                # 使用Black-Scholes公式计算期权价值
                option_value = self.calculate_black_scholes_price(current_price, strike_price, remaining_time)
            
            option_values.append(option_value)
        
        # 找到最优停时（最大价值对应的时刻）
        optimal_time = np.argmax(option_values)
        optimal_value = option_values[optimal_time]
        
        return optimal_time, optimal_value
    
    def execute_arbitrage_strategy(self, price_path: torch.Tensor, 
                                 strike_price: float, initial_capital: float = 10000.0) -> Dict:
        """
        执行套利策略
        
        Args:
            price_path: 价格路径 (time_steps,)
            strike_price: 执行价格
            initial_capital: 初始资本
            
        Returns:
            strategy_results: 策略执行结果
        """
        time_steps = price_path.shape[0]
        dt = 1.0 / (time_steps - 1)
        
        # 初始化记录
        positions = []  # 期权仓位
        hedge_positions = []  # 对冲仓位
        cash = []  # 现金
        portfolio_values = []  # 组合价值
        pnl = []  # 损益
        hedge_ratios = []  # 对冲比率
        
        # 初始状态
        current_cash = initial_capital
        current_option_position = 0.0
        current_hedge_position = 0.0
        
        # 计算最优停时
        optimal_time, optimal_value = self.calculate_optimal_stopping_time(price_path, strike_price)
        
        print(f"最优停时: {optimal_time}/{time_steps-1}, 最优价值: {optimal_value:.4f}")
        
        for t in range(time_steps):
            current_price = price_path[t].item()
            current_time = t * dt
            remaining_time = 1.0 - current_time
            
            # 计算当前对冲比率
            hedge_ratio = self.calculate_delta(current_price, strike_price, remaining_time)
            hedge_ratios.append(hedge_ratio)
            
            # 策略逻辑
            if t == 0:
                # 初始建仓：买入期权
                option_cost = optimal_value * 0.8  # 假设以理论价值的80%买入
                current_option_position = 1.0
                current_cash -= option_cost
                
                # 初始对冲
                hedge_amount = hedge_ratio * current_price
                current_hedge_position = -hedge_ratio  # 做空标的进行对冲
                current_cash += hedge_amount
                
            elif t < optimal_time:
                # 在最优停时之前：动态调整对冲
                target_hedge = hedge_ratio
                hedge_adjustment = target_hedge - current_hedge_position
                
                if abs(hedge_adjustment) > 0.01:  # 避免频繁交易
                    # 调整对冲仓位
                    hedge_cost = hedge_adjustment * current_price * (1 + self.transaction_cost)
                    current_hedge_position = target_hedge
                    current_cash -= hedge_cost
                    
            elif t == optimal_time:
                # 最优停时：平仓
                # 卖出期权
                option_value = max(current_price - strike_price, 0)
                current_cash += option_value * current_option_position
                current_option_position = 0.0
                
                # 平掉对冲仓位
                hedge_value = current_hedge_position * current_price
                current_cash += hedge_value
                current_hedge_position = 0.0
                
            else:
                # 最优停时之后：保持现金
                pass
            
            # 记录当前状态
            positions.append(current_option_position)
            hedge_positions.append(current_hedge_position)
            cash.append(current_cash)
            
            # 计算组合价值
            portfolio_value = (current_cash + 
                             current_option_position * max(current_price - strike_price, 0) + 
                             current_hedge_position * current_price)
            portfolio_values.append(portfolio_value)
            
            # 计算损益
            current_pnl = portfolio_value - initial_capital
            pnl.append(current_pnl)
        
        return {
            'time_steps': np.arange(time_steps),
            'prices': price_path.numpy(),
            'positions': positions,
            'hedge_positions': hedge_positions,
            'cash': cash,
            'portfolio_values': portfolio_values,
            'pnl': pnl,
            'hedge_ratios': hedge_ratios,
            'optimal_time': optimal_time,
            'optimal_value': optimal_value,
            'final_pnl': pnl[-1],
            'max_drawdown': min(pnl) if pnl else 0,
            'sharpe_ratio': self.calculate_sharpe_ratio(pnl)
        }
    
    def calculate_sharpe_ratio(self, pnl: List[float]) -> float:
        """计算夏普比率"""
        if len(pnl) < 2:
            return 0.0
        
        pnl_returns = np.diff(pnl)
        if len(pnl_returns) == 0 or np.std(pnl_returns) == 0:
            return 0.0
        
        return np.mean(pnl_returns) / np.std(pnl_returns) * np.sqrt(252)  # 年化夏普比率

def create_strategy(model: DeepBSDE, risk_free_rate: float = 0.05, 
                   transaction_cost: float = 0.001) -> ArbitrageStrategy:
    """
    创建套利策略
    
    Args:
        model: Deep BSDE模型
        risk_free_rate: 无风险利率
        transaction_cost: 交易成本
        
    Returns:
        strategy: 套利策略实例
    """
    return ArbitrageStrategy(model, risk_free_rate, transaction_cost)

if __name__ == "__main__":
    # 示例使用
    from .deep_bsde_model import create_model
    
    print("创建Deep BSDE模型...")
    model = create_model(d=1, N=50, T=1.0, r=0.05, hidden_size=64)
    
    print("训练模型...")
    losses = model.train_model(num_epochs=100, batch_size=128)
    
    print("创建套利策略...")
    strategy = create_strategy(model, risk_free_rate=0.05, transaction_cost=0.001)
    
    print("策略创建完成！")
