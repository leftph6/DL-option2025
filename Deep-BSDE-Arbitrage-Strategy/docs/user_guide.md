# 用户指南

## 快速开始

### 1. 环境配置

#### Windows用户
```bash
# 克隆项目
git clone https://github.com/leftph6/DL-option2025.git

# 运行环境配置脚本
scripts\setup_windows.bat
```

#### Linux/Mac用户
```bash
# 克隆项目
git clone https://github.com/leftph6/DL-option2025.git
cd Deep-BSDE-Arbitrage-Strategy

# 运行环境配置脚本
chmod +x scripts/setup_unix.sh
./scripts/setup_unix.sh
```

### 2. 运行程序

#### 方法1：使用脚本（推荐）
```bash
# Windows
scripts\run_arbitrage.bat

# Linux/Mac
./scripts/run_arbitrage.sh
```

#### 方法2：直接运行Python
```bash
python scripts/run_arbitrage.py
```

#### 方法3：运行示例
```bash
# 简单示例
python examples/simple_example.py

# 高级示例
python examples/advanced_example.py
```

## 详细使用说明

### 基本工作流程

1. **模型训练**：使用Deep BSDE方法训练期权定价模型
2. **策略创建**：基于训练好的模型创建套利策略
3. **回测执行**：在模拟的价格路径上执行策略
4. **结果分析**：生成PNL曲线和统计报告

### 参数配置

#### 模型参数
- `d`: 资产维度（默认：1）
- `N`: 时间步数（默认：50）
- `T`: 到期时间（默认：1.0年）
- `r`: 无风险利率（默认：0.05）
- `hidden_size`: 神经网络隐藏层大小（默认：64）

#### 策略参数
- `risk_free_rate`: 无风险利率（默认：0.05）
- `transaction_cost`: 交易成本比例（默认：0.001）
- `max_position`: 最大仓位限制（默认：1.0）

#### 回测参数
- `batch_size`: 回测路径数量（默认：5）
- `initial_capital`: 初始资本（默认：10000.0）
- `S0`: 初始价格（默认：100.0）
- `K`: 执行价格（默认：100.0）
- `sigma`: 波动率（默认：0.2）

### 输出文件说明

#### 图像文件
- `arbitrage_pnl_curve_*.png`: PNL曲线图
  - 价格路径与PNL曲线对比
  - 组合价值变化
  - 动态对冲比率
  - 仓位变化

- `summary_statistics_*.png`: 汇总统计图
  - PNL分布直方图
  - 最大回撤分布
  - 夏普比率分布
  - PNL vs 最大回撤散点图

#### 数据文件
- `arbitrage_results_*.csv`: 汇总结果
  - 每条路径的最终PNL
  - 最大回撤
  - 夏普比率
  - 最优停时

- `detailed_results_*.csv`: 详细结果
  - 每个时间步的完整数据
  - 价格、仓位、现金、组合价值等

### 结果解读

#### 关键指标
- **最终PNL**: 策略执行完毕后的净损益
- **最大回撤**: 策略执行过程中的最大亏损
- **夏普比率**: 风险调整后的收益指标
- **胜率**: 盈利路径占总路径的比例
- **最优停时**: 策略选择平仓的最佳时机

#### 图表分析
1. **PNL曲线**: 显示策略的实时表现
2. **组合价值**: 反映总资产的变化
3. **对冲比率**: 显示风险管理的动态调整
4. **仓位变化**: 展示交易决策的执行

### 常见问题

#### Q1: 程序运行很慢怎么办？
A: 可以尝试以下优化：
- 减少batch_size参数
- 减少时间步数N
- 使用GPU加速（如果可用）
- 减少训练轮数

#### Q2: 内存不足怎么办？
A: 可以尝试：
- 减少batch_size
- 使用CPU版本
- 减少路径数量
- 增加系统内存

#### Q3: 如何调整策略参数？
A: 可以通过修改以下文件：
- `src/arbitrage_strategy.py`: 策略逻辑
- `scripts/run_arbitrage.py`: 运行参数
- `examples/`: 示例配置

#### Q4: 如何添加新的指标？
A: 可以修改：
- `src/backtest_engine.py`: 添加新的计算函数
- `src/arbitrage_strategy.py`: 添加新的策略逻辑

### 高级功能

#### 自定义策略
```python
from src.arbitrage_strategy import ArbitrageStrategy

class CustomStrategy(ArbitrageStrategy):
    def calculate_optimal_stopping_time(self, price_path, strike_price):
        # 自定义最优停时计算
        pass
    
    def calculate_hedge_ratio(self, price, time, strike_price):
        # 自定义对冲比率计算
        pass
```

#### 批量回测
```python
from src.backtest_engine import create_backtest_engine

# 创建回测引擎
engine = create_backtest_engine(strategy)

# 运行批量回测
results = engine.run_backtest(price_paths, strike_price=100.0)

# 生成汇总统计
engine.plot_summary_statistics(results)
```

#### 参数扫描
```python
# 扫描不同参数组合
for sigma in [0.1, 0.2, 0.3]:
    for K in [95, 100, 105]:
        # 运行回测
        results = run_backtest(sigma=sigma, K=K)
        # 记录结果
        save_results(results, sigma, K)
```

### 性能优化

#### GPU加速
```python
# 检查CUDA可用性
import torch
print(f"CUDA可用: {torch.cuda.is_available()}")

# 使用GPU训练
model = create_model(device=torch.device("cuda:0"))
```

#### 混合精度训练
```python
# 在模型训练中启用混合精度
model.train_model(use_amp=True)
```

#### 并行处理
```python
# 使用多进程进行批量回测
from multiprocessing import Pool

def run_single_backtest(params):
    # 单次回测逻辑
    pass

# 并行执行
with Pool(processes=4) as pool:
    results = pool.map(run_single_backtest, parameter_list)
```

### 故障排除

#### 环境问题
1. 检查Python版本（需要3.9+）
2. 检查依赖包安装
3. 检查CUDA版本兼容性

#### 程序错误
1. 查看错误日志
2. 检查输入参数
3. 验证数据格式

#### 性能问题
1. 监控内存使用
2. 检查GPU利用率
3. 优化批处理大小

### 技术支持

如果遇到问题，请：
1. 查看本文档的故障排除部分
2. 检查GitHub Issues
3. 联系项目维护者

### 更新日志

#### v1.0.0
- 初始版本发布
- 支持基本的套利策略回测
- 包含PNL曲线可视化
- 支持Windows和Unix系统
