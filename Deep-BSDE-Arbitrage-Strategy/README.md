# Deep BSDE 套利策略回测系统

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

基于深度BSDE（Backward Stochastic Differential Equations）的期权定价模型，实现套利策略在最优停时处的平仓回测系统。

## 🎯 项目概述

本项目结合了深度学习技术和金融工程理论，通过训练神经网络来近似求解BSDE，从而构建期权定价模型。在此基础上，实现了基于最优停时理论的套利策略，能够在GBM（几何布朗运动）路径上进行回测，并生成详细的PNL曲线分析。

### 核心功能

- **Deep BSDE期权定价**：使用神经网络求解BSDE进行期权定价
- **最优停时算法**：基于Black-Scholes理论计算最优平仓时机
- **套利策略回测**：在GBM路径上执行动态对冲策略
- **PNL曲线分析**：可视化策略执行过程和风险收益特征
- **多路径回测**：支持批量路径的统计分析

## 🚀 快速开始

### 环境要求

- Python 3.9+
- CUDA 11.0+ (可选，用于GPU加速)
- 8GB+ RAM
- 5GB+ 可用存储空间

### 一键安装

```bash
# 克隆项目
git clone https://github.com/your-username/Deep-BSDE-Arbitrage-Strategy.git
cd Deep-BSDE-Arbitrage-Strategy

# Windows用户
scripts\setup_windows.bat

# Linux/Mac用户
chmod +x scripts/setup_unix.sh
./scripts/setup_unix.sh
```

### 快速运行

```bash
# 运行套利策略回测
python scripts/run_arbitrage.py

# 或使用批处理脚本（Windows）
scripts\run_arbitrage.bat
```

## 📁 项目结构

```
Deep-BSDE-Arbitrage-Strategy/
├── src/                          # 源代码
│   ├── deep_bsde_model.py       # Deep BSDE模型
│   ├── arbitrage_strategy.py    # 套利策略实现
│   ├── backtest_engine.py       # 回测引擎
│   └── utils.py                 # 工具函数
├── scripts/                      # 脚本文件
│   ├── setup_windows.bat        # Windows环境配置
│   ├── setup_unix.sh            # Unix环境配置
│   ├── run_arbitrage.py         # 主运行脚本
│   └── run_arbitrage.bat        # Windows批处理
├── docs/                         # 文档
│   ├── user_guide.md            # 用户指南
│   ├── api_reference.md         # API参考
│   └── examples/                # 示例代码
├── output/                       # 输出文件
│   ├── arbitrage_pnl_*.png      # PNL曲线图
│   ├── arbitrage_results_*.csv  # 详细结果数据
│   └── bond_price_paths_*.png   # 价格路径图
├── examples/                     # 示例和测试
│   ├── simple_example.py        # 简单示例
│   └── advanced_example.py      # 高级示例
├── requirements.txt              # Python依赖
├── environment.yml               # Conda环境配置
└── README.md                    # 项目说明
```

## 🔧 详细安装

### 方法1：Conda环境（推荐）

```bash
# 创建conda环境
conda env create -f environment.yml

# 激活环境
conda activate deep-bsde-pricing

# 验证安装
python -c "import torch; print('PyTorch版本:', torch.__version__)"
```

### 方法2：Pip安装

```bash
# 创建虚拟环境
python -m venv venv

# 激活环境
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

## 📊 使用示例

### 基础使用

```python
from src.arbitrage_strategy import ArbitrageStrategy
from src.backtest_engine import BacktestEngine
from src.deep_bsde_model import DeepBSDE

# 1. 训练Deep BSDE模型
model = DeepBSDE(d=1, N=50, T=1.0, r=0.05, hidden_size=64)
# ... 训练过程 ...

# 2. 创建套利策略
strategy = ArbitrageStrategy(model, risk_free_rate=0.05)

# 3. 运行回测
engine = BacktestEngine(strategy)
results = engine.run_backtest(price_paths, strike_price=100.0)

# 4. 生成PNL曲线
engine.plot_pnl_curve(results[0], save_plot=True)
```

### 高级配置

```python
# 自定义参数
strategy = ArbitrageStrategy(
    model=model,
    risk_free_rate=0.05,
    transaction_cost=0.001,  # 交易成本
    max_position=1.0         # 最大仓位
)

# 批量回测
results = engine.run_backtest(
    price_paths=price_paths,
    strike_price=100.0,
    initial_capital=10000.0
)
```

## 📈 输出说明

### 生成文件

1. **PNL曲线图** (`arbitrage_pnl_curve_*.png`)：
   - 价格路径与PNL曲线对比
   - 组合价值变化趋势
   - 动态对冲比率变化
   - 仓位变化情况

2. **详细结果数据** (`arbitrage_results_*.csv`)：
   - 每个时间步的完整数据
   - 包含价格、仓位、现金、组合价值、PNL等

3. **价格路径图** (`bond_price_paths_*.png`)：
   - 模拟的GBM价格路径
   - 平均路径和初始价格线

### 结果解读

- **最优停时**：策略选择平仓的最佳时机
- **最终PNL**：策略执行完毕后的净损益
- **最大回撤**：策略执行过程中的最大亏损
- **夏普比率**：风险调整后的收益指标

## ⚙️ 配置参数

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| T | 1.0 | 到期时间（年） |
| N | 50 | 时间步数 |
| r | 0.05 | 无风险利率 |
| sigma | 0.2 | 波动率 |
| S0 | 100.0 | 初始价格 |
| K | 100.0 | 执行价格 |

### 策略参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| transaction_cost | 0.001 | 交易成本比例 |
| max_position | 1.0 | 最大仓位限制 |
| initial_capital | 10000.0 | 初始资本 |

## 🐛 故障排除

### 常见问题

1. **CUDA相关错误**
   ```bash
   # 检查CUDA安装
   nvidia-smi
   
   # 安装CPU版本
   conda install pytorch cpuonly -c pytorch
   ```

2. **依赖包冲突**
   ```bash
   # 重新创建环境
   conda env remove -n deep-bsde-pricing
   conda env create -f environment.yml
   ```

3. **内存不足**
   - 减少batch_size参数
   - 使用CPU版本
   - 减少时间步数N

### 性能优化

- 使用GPU加速训练
- 调整批次大小
- 使用混合精度训练
- 启用PyTorch编译优化

## 📚 理论背景

### Deep BSDE方法

本项目基于以下论文的理论基础：
- "Deep Learning-Based Numerical Methods for High-Dimensional Parabolic PDEs and BSDEs"
- "Solving high-dimensional partial differential equations using deep learning"

### 最优停时理论

套利策略基于最优停时理论，在Black-Scholes框架下寻找最优平仓时机。

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系方式

- 项目维护者：[您的姓名]
- 邮箱：[your.email@example.com]
- 项目链接：[https://github.com/your-username/Deep-BSDE-Arbitrage-Strategy](https://github.com/your-username/Deep-BSDE-Arbitrage-Strategy)

## 🙏 致谢

- PyTorch团队提供的深度学习框架
- 金融工程社区的理论支持
- 开源社区的贡献

---

**注意**：本项目仅供学习和研究使用，不构成投资建议。实际交易请谨慎评估风险。
