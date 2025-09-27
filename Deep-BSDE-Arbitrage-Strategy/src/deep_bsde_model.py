#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Deep BSDE 期权定价模型
基于深度学习的BSDE求解器
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

def select_device() -> torch.device:
    """自动选择计算设备"""
    try:
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            return torch.device("cuda:0")
        return torch.device("cpu")
    except Exception:
        return torch.device("cpu")

class ControlNet(nn.Module):
    """控制网络，用于近似Z_t"""
    
    def __init__(self, d: int, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d + 1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, d)
        )
    
    def forward(self, S: torch.Tensor, t: float) -> torch.Tensor:
        """
        前向传播
        
        Args:
            S: 价格张量 (batch_size, d)
            t: 时间标量
            
        Returns:
            Z_t: 控制变量 (batch_size, d)
        """
        device = S.device
        t_input = torch.full((S.shape[0], 1), t, device=device)
        x = torch.cat([S, t_input], dim=1)
        return self.net(x)

class DeepBSDE(nn.Module):
    """Deep BSDE 模型"""
    
    def __init__(self, d: int, N: int, T: float, r: float, 
                 hidden_size: int = 64, device: Optional[torch.device] = None):
        super().__init__()
        self.d = d
        self.N = N
        self.T = T
        self.r = r
        self.dt = T / N
        self.device = device or select_device()
        
        # 模型参数
        self.Y0 = nn.Parameter(torch.tensor(0.0))  # 初始值
        self.Znets = nn.ModuleList([
            ControlNet(d, hidden_size) for _ in range(N)
        ])
        
        # 移动到指定设备
        self.to(self.device)
    
    def forward(self, S: torch.Tensor, dW: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            S: 价格路径 (batch_size, N+1, d)
            dW: 布朗运动增量 (batch_size, N, d)
            
        Returns:
            Y_T: 终端价值 (batch_size, 1)
        """
        batch_size = S.shape[0]
        Y = self.Y0.expand(batch_size, 1).to(self.device)
        
        for t in range(self.N):
            Z = self.Znets[t](S[:, t, :], t * self.dt)
            Y = Y - self.r * Y * self.dt + (Z * dW[:, t, :]).sum(dim=1, keepdim=True)
        
        return Y
    
    def train_model(self, num_epochs: int = 200, batch_size: int = 256, 
                   learning_rate: float = 1e-3, S0: float = 100.0, 
                   K: float = 100.0, sigma: float = 0.2) -> list:
        """
        训练模型
        
        Args:
            num_epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            S0: 初始价格
            K: 执行价格
            sigma: 波动率
            
        Returns:
            losses: 损失历史
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        losses = []
        
        # 混合精度训练
        use_cuda = self.device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)
        
        for epoch in range(num_epochs):
            # 生成训练数据
            S = self.simulate_gbm_paths(batch_size, S0, sigma)
            sqrt_dt = torch.sqrt(torch.tensor(self.dt, device=self.device))
            dW = torch.randn(batch_size, self.N, self.d, device=self.device) * sqrt_dt
            
            # 前向传播
            with torch.cuda.amp.autocast(enabled=use_cuda):
                Y_T = self.forward(S, dW)
                target = torch.clamp(S[:, -1, 0] - K, min=0).unsqueeze(1)
                loss = loss_fn(Y_T, target)
            
            # 反向传播
            optimizer.zero_grad()
            if use_cuda:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            losses.append(loss.item())
            
            if epoch % (num_epochs // 10) == 0:
                print(f"Epoch {epoch:4d}, Loss={loss.item():.6f}, Price(Y0)={self.Y0.item():.4f}")
        
        return losses
    
    def simulate_gbm_paths(self, batch_size: int, S0: float, sigma: float) -> torch.Tensor:
        """
        模拟GBM路径
        
        Args:
            batch_size: 批次大小
            S0: 初始价格
            sigma: 波动率
            
        Returns:
            S: 价格路径 (batch_size, N+1, d)
        """
        dt = self.dt
        sqrt_dt = torch.tensor(dt, device=self.device).sqrt()
        
        # 生成布朗运动增量
        dW = torch.randn(batch_size, self.N, self.d, device=self.device) * sqrt_dt
        W = torch.cumsum(dW, dim=1)
        W0 = torch.zeros(batch_size, 1, self.d, device=self.device)
        W_full = torch.cat([W0, W], dim=1)
        
        # 构建时间网格和漂移项
        t_grid = torch.linspace(0.0, self.T, self.N + 1, device=self.device).view(1, self.N + 1, 1)
        r_t = torch.tensor(self.r, device=self.device)
        sigma_t = torch.tensor(sigma, device=self.device)
        drift = (r_t - 0.5 * sigma_t * sigma_t) * t_grid
        
        # GBM 显式公式
        S = torch.tensor(S0, device=self.device) * torch.exp(drift + sigma_t * W_full)
        return S
    
    def predict_option_price(self, S0: float, K: float, sigma: float, 
                           num_paths: int = 1000) -> float:
        """
        预测期权价格
        
        Args:
            S0: 初始价格
            K: 执行价格
            sigma: 波动率
            num_paths: 路径数量
            
        Returns:
            option_price: 期权价格
        """
        self.eval()
        with torch.no_grad():
            S = self.simulate_gbm_paths(num_paths, S0, sigma)
            sqrt_dt = torch.sqrt(torch.tensor(self.dt, device=self.device))
            dW = torch.randn(num_paths, self.N, self.d, device=self.device) * sqrt_dt
            
            Y_T = self.forward(S, dW)
            option_price = Y_T.mean().item()
        
        return option_price

def create_model(d: int = 1, N: int = 50, T: float = 1.0, r: float = 0.05, 
                hidden_size: int = 64, device: Optional[torch.device] = None) -> DeepBSDE:
    """
    创建Deep BSDE模型
    
    Args:
        d: 维度
        N: 时间步数
        T: 到期时间
        r: 无风险利率
        hidden_size: 隐藏层大小
        device: 计算设备
        
    Returns:
        model: Deep BSDE模型
    """
    if device is None:
        device = select_device()
    
    model = DeepBSDE(d, N, T, r, hidden_size, device)
    return model

if __name__ == "__main__":
    # 示例使用
    print("创建Deep BSDE模型...")
    model = create_model(d=1, N=50, T=1.0, r=0.05, hidden_size=64)
    
    print(f"设备: {model.device}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 训练模型
    print("开始训练...")
    losses = model.train_model(num_epochs=100, batch_size=128)
    
    # 预测期权价格
    option_price = model.predict_option_price(S0=100.0, K=100.0, sigma=0.2)
    print(f"预测期权价格: {option_price:.4f}")
