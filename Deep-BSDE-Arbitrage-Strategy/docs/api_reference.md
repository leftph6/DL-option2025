# API 参考

## 核心模块

### deep_bsde_model

#### DeepBSDE类

```python
class DeepBSDE(nn.Module):
    def __init__(self, d, N, T, r, hidden_size=64, device=None):
        """
        初始化Deep BSDE模型
        
        Args:
            d (int): 资产维度
            N (int): 时间步数
            T (float): 到期时间
            r (float): 无风险利率
            hidden_size (int): 隐藏层大小
            device (torch.device): 计算设备
        """
```

#### 主要方法

##### train_model
```python
def train_model(self, num_epochs=200, batch_size=256, learning_rate=1e-3, 
                S0=100.0, K=100.0, sigma=0.2):
    """
    训练模型
    
    Args:
        num_epochs (int): 训练轮数
        batch_size (int): 批次大小
        learning_rate (float): 学习率
        S0 (float): 初始价格
        K (float): 执行价格
        sigma (float): 波动率
        
    Returns:
        list: 损失历史
    """
```

##### predict_option_price
```python
def predict_option_price(self, S0, K, sigma, num_paths=1000):
    """
    预测期权价格
    
    Args:
        S0 (float): 初始价格
        K (float): 执行价格
        sigma (float): 波动率
        num_paths (int): 路径数量
        
    Returns:
        float: 期权价格
    """
```

#### 工具函数

##### create_model
```python
def create_model(d=1, N=50, T=1.0, r=0.05, hidden_size=64, device=None):
    """
    创建Deep BSDE模型
    
    Args:
        d (int): 维度
        N (int): 时间步数
        T (float): 到期时间
        r (float): 无风险利率
        hidden_size (int): 隐藏层大小
        device (torch.device): 计算设备
        
    Returns:
        DeepBSDE: 模型实例
    """
```

##### select_device
```python
def select_device():
    """
    自动选择计算设备
    
    Returns:
        torch.device: 计算设备
    """
```

### arbitrage_strategy

#### ArbitrageStrategy类

```python
class ArbitrageStrategy:
    def __init__(self, model, risk_free_rate=0.05, transaction_cost=0.001, max_position=1.0):
        """
        初始化套利策略
        
        Args:
            model (DeepBSDE): 训练好的模型
            risk_free_rate (float): 无风险利率
            transaction_cost (float): 交易成本
            max_position (float): 最大仓位
        """
```

#### 主要方法

##### execute_arbitrage_strategy
```python
def execute_arbitrage_strategy(self, price_path, strike_price, initial_capital=10000.0):
    """
    执行套利策略
    
    Args:
        price_path (torch.Tensor): 价格路径
        strike_price (float): 执行价格
        initial_capital (float): 初始资本
        
    Returns:
        dict: 策略执行结果
    """
```

##### calculate_optimal_stopping_time
```python
def calculate_optimal_stopping_time(self, price_path, strike_price):
    """
    计算最优停时
    
    Args:
        price_path (torch.Tensor): 价格路径
        strike_price (float): 执行价格
        
    Returns:
        tuple: (最优停时索引, 最优价值)
    """
```

##### calculate_delta
```python
def calculate_delta(self, S, K, T):
    """
    计算Delta（对冲比率）
    
    Args:
        S (float): 当前价格
        K (float): 执行价格
        T (float): 剩余时间
        
    Returns:
        float: 对冲比率
    """
```

#### 工具函数

##### create_strategy
```python
def create_strategy(model, risk_free_rate=0.05, transaction_cost=0.001):
    """
    创建套利策略
    
    Args:
        model (DeepBSDE): 模型实例
        risk_free_rate (float): 无风险利率
        transaction_cost (float): 交易成本
        
    Returns:
        ArbitrageStrategy: 策略实例
    """
```

### backtest_engine

#### BacktestEngine类

```python
class BacktestEngine:
    def __init__(self, strategy, output_dir="output"):
        """
        初始化回测引擎
        
        Args:
            strategy (ArbitrageStrategy): 套利策略
            output_dir (str): 输出目录
        """
```

#### 主要方法

##### run_backtest
```python
def run_backtest(self, price_paths, strike_price, initial_capital=10000.0):
    """
    运行回测
    
    Args:
        price_paths (torch.Tensor): 价格路径
        strike_price (float): 执行价格
        initial_capital (float): 初始资本
        
    Returns:
        dict: 回测结果
    """
```

##### plot_pnl_curve
```python
def plot_pnl_curve(self, result, save_plot=True, filename=None):
    """
    绘制PNL曲线
    
    Args:
        result (dict): 单条路径结果
        save_plot (bool): 是否保存图像
        filename (str): 自定义文件名
        
    Returns:
        str: 保存的文件名
    """
```

##### plot_summary_statistics
```python
def plot_summary_statistics(self, backtest_results, save_plot=True):
    """
    绘制汇总统计图
    
    Args:
        backtest_results (dict): 回测结果
        save_plot (bool): 是否保存图像
        
    Returns:
        str: 保存的文件名
    """
```

##### save_results_to_csv
```python
def save_results_to_csv(self, backtest_results, filename=None):
    """
    保存结果到CSV文件
    
    Args:
        backtest_results (dict): 回测结果
        filename (str): 自定义文件名
        
    Returns:
        str: 保存的文件名
    """
```

##### simulate_gbm_paths
```python
def simulate_gbm_paths(self, batch_size, T, N, d, r, sigma, S0):
    """
    模拟GBM路径
    
    Args:
        batch_size (int): 批次大小
        T (float): 到期时间
        N (int): 时间步数
        d (int): 维度
        r (float): 无风险利率
        sigma (float): 波动率
        S0 (float): 初始价格
        
    Returns:
        torch.Tensor: 价格路径
    """
```

#### 工具函数

##### create_backtest_engine
```python
def create_backtest_engine(strategy, output_dir="output"):
    """
    创建回测引擎
    
    Args:
        strategy (ArbitrageStrategy): 策略实例
        output_dir (str): 输出目录
        
    Returns:
        BacktestEngine: 回测引擎实例
    """
```

## 数据结构

### 策略执行结果

```python
{
    'time_steps': np.ndarray,        # 时间步数组
    'prices': np.ndarray,            # 价格路径
    'positions': List[float],        # 期权仓位
    'hedge_positions': List[float],  # 对冲仓位
    'cash': List[float],             # 现金
    'portfolio_values': List[float], # 组合价值
    'pnl': List[float],              # 损益
    'hedge_ratios': List[float],     # 对冲比率
    'optimal_time': int,             # 最优停时
    'optimal_value': float,          # 最优价值
    'final_pnl': float,              # 最终PNL
    'max_drawdown': float,           # 最大回撤
    'sharpe_ratio': float            # 夏普比率
}
```

### 回测汇总结果

```python
{
    'individual_results': List[dict],  # 各路径结果
    'summary': {
        'mean_final_pnl': float,       # 平均最终PNL
        'std_final_pnl': float,        # PNL标准差
        'mean_max_drawdown': float,    # 平均最大回撤
        'mean_sharpe_ratio': float,    # 平均夏普比率
        'win_rate': float,             # 胜率
        'total_paths': int             # 总路径数
    }
}
```

## 配置参数

### 模型参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| d | int | 1 | 资产维度 |
| N | int | 50 | 时间步数 |
| T | float | 1.0 | 到期时间（年） |
| r | float | 0.05 | 无风险利率 |
| hidden_size | int | 64 | 隐藏层大小 |

### 策略参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| risk_free_rate | float | 0.05 | 无风险利率 |
| transaction_cost | float | 0.001 | 交易成本比例 |
| max_position | float | 1.0 | 最大仓位限制 |

### 回测参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| batch_size | int | 5 | 回测路径数 |
| initial_capital | float | 10000.0 | 初始资本 |
| S0 | float | 100.0 | 初始价格 |
| K | float | 100.0 | 执行价格 |
| sigma | float | 0.2 | 波动率 |

## 错误处理

### 常见异常

#### ImportError
```python
try:
    import torch
except ImportError:
    print("PyTorch未安装，请运行: pip install torch")
```

#### CUDAError
```python
try:
    model = create_model(device=torch.device("cuda:0"))
except torch.cuda.CUDAError:
    print("CUDA不可用，使用CPU")
    model = create_model(device=torch.device("cpu"))
```

#### FileNotFoundError
```python
try:
    results = engine.load_results("results.csv")
except FileNotFoundError:
    print("结果文件不存在，请先运行回测")
```

### 调试技巧

1. **启用详细日志**：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. **检查中间结果**：
```python
# 在关键步骤添加打印
print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
print(f"当前损失: {loss.item()}")
```

3. **验证数据格式**：
```python
# 检查张量形状
print(f"价格路径形状: {price_paths.shape}")
print(f"数据类型: {price_paths.dtype}")
```

## 性能优化

### GPU加速
```python
# 检查CUDA可用性
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"使用GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("使用CPU")
```

### 混合精度训练
```python
# 启用自动混合精度
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 批处理优化
```python
# 调整批处理大小
optimal_batch_size = find_optimal_batch_size(model, device)
print(f"最优批处理大小: {optimal_batch_size}")
```

## 扩展开发

### 自定义策略
```python
class CustomArbitrageStrategy(ArbitrageStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 自定义初始化
    
    def calculate_optimal_stopping_time(self, price_path, strike_price):
        # 自定义最优停时算法
        return super().calculate_optimal_stopping_time(price_path, strike_price)
```

### 自定义指标
```python
def calculate_custom_metric(results):
    """计算自定义指标"""
    pnls = [r['final_pnl'] for r in results]
    return {
        'custom_metric': np.mean(pnls) / np.std(pnls),
        'other_metric': len([p for p in pnls if p > 0])
    }
```

### 自定义可视化
```python
def plot_custom_chart(results, save_path=None):
    """绘制自定义图表"""
    fig, ax = plt.subplots(figsize=(10, 6))
    # 自定义绘图逻辑
    if save_path:
        plt.savefig(save_path)
    plt.show()
```
