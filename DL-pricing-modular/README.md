# Deep BSDE 套利策略回测系统 - 模块化版本

## 概述

本系统采用模块化架构，将Deep BSDE套利策略分为三个核心模块：

1. **数学建模模块** (`math_modeling/`) - 路径生成和数学建模
2. **机器学习模块** (`ml_models/`) - 神经网络模型训练
3. **套利回测模块** (`backtest/`) - 策略回测和金融分析

## 系统架构

```
DL-pricing-modular/
├── math_modeling/          # 数学建模模块
│   ├── __init__.py
│   ├── path_generators.py  # 路径生成器（GBM、FBM等）
│   ├── models.py           # 数学模型定义
│   └── utils.py            # 数学工具函数
├── ml_models/              # 机器学习模块
│   ├── __init__.py
│   ├── base_model.py       # 基础模型接口
│   ├── mlp_model.py        # MLP模型
│   ├── rnn_model.py        # RNN/LSTM模型
│   ├── transformer_model.py # Transformer模型（预留）
│   └── trainer.py          # 训练器
├── backtest/               # 套利回测模块
│   ├── __init__.py
│   ├── strategy.py         # 套利策略
│   ├── engine.py           # 回测引擎
│   ├── data_loader.py      # 数据加载器
│   ├── metrics.py           # 金融指标计算
│   └── visualizer.py        # 结果可视化
├── config/                 # 配置文件
│   ├── __init__.py
│   ├── default_config.py   # 默认配置
│   └── config_manager.py   # 配置管理器
├── utils/                  # 通用工具
│   ├── __init__.py
│   ├── device_manager.py   # 设备管理
│   └── logger.py           # 日志系统
├── main.py                # 主程序
├── requirements.txt        # 依赖包
└── README.md              # 说明文档
```

## 特性

### 数学建模模块
- **多种路径模型**：GBM、FBM、Vasicek等
- **可扩展接口**：易于添加新的数学模型
- **参数化配置**：支持灵活的参数设置

### 机器学习模块
- **统一接口**：所有模型继承自BaseModel
- **多种架构**：MLP、RNN、LSTM、Transformer（预留）
- **训练优化**：混合精度、梯度裁剪、学习率调度

### 套利回测模块
- **多种数据源**：模拟数据、CSV文件、实时API（预留）
- **完整指标**：夏普比率、最大回撤、资金周转率等
- **可视化分析**：PNL曲线、价格路径、风险分析

## 使用方法

### 1. 环境设置
```bash
# 创建conda环境
conda env create -f environment.yml
conda activate deep-bsde-pricing

# 或使用pip
pip install -r requirements.txt
```

### 2. 运行主程序
```bash
python main.py
```

### 3. 模块化使用
```python
from math_modeling import GBMGenerator
from ml_models import RNNModel
from backtest import ArbitrageEngine

# 生成训练数据
generator = GBMGenerator()
paths = generator.generate_paths(batch_size=1000)

# 训练模型
model = RNNModel()
model.train(paths)

# 运行回测
engine = ArbitrageEngine()
results = engine.run_backtest(model, paths)
```

## 扩展指南

### 添加新的路径模型
1. 在 `math_modeling/path_generators.py` 中继承 `BasePathGenerator`
2. 实现 `generate_paths()` 方法
3. 在配置文件中注册新模型

### 添加新的神经网络
1. 在 `ml_models/` 中创建新的模型文件
2. 继承 `BaseModel` 类
3. 实现必要的方法：`forward()`, `train()`, `predict()`

### 添加新的回测策略
1. 在 `backtest/strategy.py` 中继承 `BaseStrategy`
2. 实现 `execute()` 方法
3. 定义策略特定的参数

## 配置说明

系统使用配置文件管理参数，支持：
- 默认配置
- 用户自定义配置
- 运行时参数覆盖

## 性能优化

- **GPU加速**：自动检测和使用CUDA
- **混合精度**：支持fp16/bf16训练
- **批处理**：向量化计算优化
- **内存管理**：高效的数据加载和缓存

## 输出文件

系统会在 `output/` 目录下生成：
- 训练数据CSV文件
- 模型权重文件
- 回测结果CSV
- 可视化图表PNG
- 分析报告TXT

## 注意事项

1. 确保有足够的GPU内存用于训练
2. 大数据集建议使用批处理模式
3. 回测结果仅供参考，不构成投资建议
4. 定期备份重要的模型和结果文件
