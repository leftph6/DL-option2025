# 团队使用指南

## 项目概述

本项目是一个基于深度BSDE（Backward Stochastic Differential Equations）的期权定价和套利策略回测系统，由2025年吉林大学数学学院期权定价大创小组制作

## 快速开始

### 1. 环境准备

#### 系统要求
- **操作系统**: Windows 10/11, Linux (Ubuntu 18.04+), macOS 10.15+
- **Python**: 3.9+
- **内存**: 8GB+ (推荐16GB)
- **存储**: 5GB+ 可用空间
- **GPU**: 可选，NVIDIA GPU (CUDA 11.0+)

#### 开发环境设置
```bash
# 1. 克隆项目
git clone https://github.com/your-org/Deep-BSDE-Arbitrage-Strategy.git
cd Deep-BSDE-Arbitrage-Strategy

# 2. 创建开发分支
git checkout -b feature/your-feature-name

# 3. 配置环境
# Windows
scripts\setup_windows.bat

# Linux/Mac
chmod +x scripts/setup_unix.sh
./scripts/setup_unix.sh
```

### 2. 项目结构说明

```
Deep-BSDE-Arbitrage-Strategy/
├── src/                          # 核心源代码
│   ├── __init__.py              # 包初始化
│   ├── deep_bsde_model.py       # Deep BSDE模型
│   ├── arbitrage_strategy.py    # 套利策略
│   └── backtest_engine.py       # 回测引擎
├── scripts/                      # 可执行脚本
│   ├── setup_windows.bat        # Windows环境配置
│   ├── setup_unix.sh            # Unix环境配置
│   ├── run_arbitrage.py         # 主运行脚本
│   ├── run_arbitrage.bat        # Windows批处理
│   └── run_arbitrage.sh         # Unix脚本
├── docs/                         # 文档
│   ├── user_guide.md            # 用户指南
│   ├── api_reference.md         # API参考
│   └── team_guide.md            # 团队指南
├── examples/                     # 示例代码
│   ├── simple_example.py        # 简单示例
│   └── advanced_example.py      # 高级示例
├── output/                       # 输出文件
│   ├── arbitrage_pnl_*.png      # PNL曲线图
│   ├── arbitrage_results_*.csv  # 结果数据
│   └── summary_statistics_*.png # 统计图表
├── tests/                        # 测试文件
├── requirements.txt              # Python依赖
├── environment.yml               # Conda环境
├── .gitignore                   # Git忽略文件
├── LICENSE                      # 许可证
└── README.md                    # 项目说明
```

## 开发工作流

### 1. 代码开发

#### 分支管理
```bash
# 主分支
main                    # 生产环境代码
develop                 # 开发环境代码

# 功能分支
feature/strategy-enhancement    # 策略增强
feature/risk-metrics           # 风险指标
feature/visualization          # 可视化改进

# 修复分支
hotfix/critical-bug-fix        # 紧急修复
bugfix/minor-issue-fix         # 小问题修复
```

#### 提交规范
```bash
# 提交信息格式
<type>(<scope>): <description>

# 类型
feat:     新功能
fix:      修复
docs:     文档更新
style:    代码格式
refactor: 重构
test:     测试
chore:    构建/工具

# 示例
feat(strategy): 添加动态对冲策略
fix(backtest): 修复PNL计算错误
docs(api): 更新API文档
```

### 2. 测试流程

#### 单元测试
```bash
# 运行所有测试
python -m pytest tests/

# 运行特定测试
python -m pytest tests/test_arbitrage_strategy.py

# 生成覆盖率报告
python -m pytest --cov=src tests/
```

#### 集成测试
```bash
# 运行完整回测流程
python scripts/run_arbitrage.py --mode simple

# 运行高级示例
python examples/advanced_example.py
```

### 3. 代码审查

#### 审查清单
- [ ] 代码符合项目规范
- [ ] 测试覆盖率达标
- [ ] 文档更新完整
- [ ] 性能影响评估
- [ ] 安全性检查

#### 审查流程
1. 创建Pull Request
2. 自动测试检查
3. 团队成员审查
4. 修改建议反馈
5. 代码合并

## 功能模块

### 1. Deep BSDE模型 (`src/deep_bsde_model.py`)

#### 核心功能
- 神经网络求解BSDE
- 期权定价
- 路径模拟

#### 扩展点
```python
# 自定义控制网络
class CustomControlNet(ControlNet):
    def __init__(self, d, hidden_size):
        super().__init__(d, hidden_size)
        # 添加自定义层
    
    def forward(self, S, t):
        # 自定义前向传播
        return super().forward(S, t)

# 自定义损失函数
def custom_loss_fn(Y_T, target):
    # 自定义损失计算
    return loss
```

### 2. 套利策略 (`src/arbitrage_strategy.py`)

#### 核心功能
- 最优停时计算
- 动态对冲
- 风险管理

#### 扩展点
```python
# 自定义策略
class CustomArbitrageStrategy(ArbitrageStrategy):
    def calculate_optimal_stopping_time(self, price_path, strike_price):
        # 自定义最优停时算法
        pass
    
    def calculate_hedge_ratio(self, price, time, strike_price):
        # 自定义对冲比率计算
        pass
```

### 3. 回测引擎 (`src/backtest_engine.py`)

#### 核心功能
- 批量回测
- 结果可视化
- 统计分析

#### 扩展点
```python
# 自定义指标
def calculate_custom_metrics(results):
    """计算自定义风险指标"""
    pass

# 自定义可视化
def plot_custom_chart(results, save_path=None):
    """绘制自定义图表"""
    pass
```

## 性能优化

### 1. GPU加速

#### CUDA配置
```python
# 检查CUDA可用性
import torch
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"使用GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("使用CPU")
```

#### 混合精度训练
```python
# 启用自动混合精度
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    output = model(input)
    loss = criterion(output, target)
```

### 2. 内存优化

#### 批处理优化
```python
# 动态调整批处理大小
def find_optimal_batch_size(model, device, max_batch_size=1024):
    """寻找最优批处理大小"""
    for batch_size in [64, 128, 256, 512, 1024]:
        try:
            test_input = torch.randn(batch_size, model.N+1, model.d).to(device)
            with torch.no_grad():
                _ = model(test_input, torch.randn(batch_size, model.N, model.d).to(device))
            return batch_size
        except RuntimeError:
            continue
    return 64
```

#### 梯度累积
```python
# 使用梯度累积处理大批次
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    with torch.cuda.amp.autocast():
        output = model(batch)
        loss = criterion(output, target) / accumulation_steps
    
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 3. 并行处理

#### 多进程回测
```python
from multiprocessing import Pool
import functools

def run_single_backtest(params):
    """单次回测"""
    model, strategy, price_path, strike_price = params
    return strategy.execute_arbitrage_strategy(price_path, strike_price)

# 并行执行
with Pool(processes=4) as pool:
    results = pool.map(run_single_backtest, parameter_list)
```

## 部署指南

### 1. 开发环境

#### Docker部署
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "scripts/run_arbitrage.py"]
```

#### 本地部署
```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt

# 运行程序
python scripts/run_arbitrage.py
```

### 2. 生产环境

#### 服务器配置
- **CPU**: 8核+ Intel/AMD处理器
- **内存**: 32GB+ RAM
- **GPU**: NVIDIA RTX 3080+ 或 Tesla V100+
- **存储**: 100GB+ SSD

#### 监控设置
```python
# 性能监控
import psutil
import time

def monitor_performance():
    """监控系统性能"""
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent
    gpu_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    return {
        'cpu_percent': cpu_percent,
        'memory_percent': memory_percent,
        'gpu_memory': gpu_memory
    }
```

## 故障排除

### 1. 常见问题

#### 环境问题
```bash
# 检查Python版本
python --version

# 检查依赖包
pip list

# 检查CUDA
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

#### 内存问题
```python
# 监控内存使用
import psutil
print(f"内存使用: {psutil.virtual_memory().percent}%")

# 清理GPU内存
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

#### 性能问题
```python
# 性能分析
import cProfile
cProfile.run('your_function()')

# 内存分析
import memory_profiler
@memory_profiler.profile
def your_function():
    pass
```

### 2. 调试技巧

#### 日志配置
```python
import logging

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
```

#### 断点调试
```python
# 使用pdb调试
import pdb; pdb.set_trace()

# 或使用IDE断点
# VS Code: 在代码行左侧点击设置断点
# PyCharm: 在代码行左侧点击设置断点
```

## 团队协作

### 1. 代码规范

#### Python代码风格
- 遵循PEP 8规范
- 使用Black进行代码格式化
- 使用flake8进行代码检查

```bash
# 代码格式化
black src/ scripts/ examples/

# 代码检查
flake8 src/ scripts/ examples/
```

#### 文档规范
- 使用Google风格的docstring
- 为所有公共函数添加文档
- 保持README和API文档同步

### 2. 版本控制

#### Git工作流
```bash
# 1. 创建功能分支
git checkout -b feature/new-strategy

# 2. 开发功能
git add .
git commit -m "feat(strategy): 添加新策略算法"

# 3. 推送分支
git push origin feature/new-strategy

# 4. 创建Pull Request
# 在GitHub上创建PR

# 5. 代码审查和合并
# 团队成员审查后合并
```

#### 标签管理
```bash
# 创建版本标签
git tag -a v1.0.0 -m "版本1.0.0发布"
git push origin v1.0.0

# 查看标签
git tag -l
```

### 3. 持续集成

#### GitHub Actions配置
```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    - name: Run tests
      run: |
        python -m pytest tests/
```

## 贡献指南

### 1. 如何贡献

#### 报告问题
1. 在GitHub Issues中创建问题
2. 提供详细的错误信息和复现步骤
3. 包含系统环境信息

#### 提交代码
1. Fork项目仓库
2. 创建功能分支
3. 编写代码和测试
4. 提交Pull Request

#### 文档贡献
1. 更新相关文档
2. 添加代码注释
3. 更新API参考

### 2. 代码审查

#### 审查标准
- 代码质量和可读性
- 测试覆盖率
- 性能影响
- 安全性考虑

#### 审查流程
1. 自动测试检查
2. 代码风格检查
3. 功能测试
4. 性能测试
5. 安全扫描

## 联系信息

- **项目维护者**: [团队负责人姓名]
- **邮箱**: team@your-org.com
- **Slack频道**: #deep-bsde-project
- **项目地址**: https://github.com/your-org/Deep-BSDE-Arbitrage-Strategy

---

**注意**: 本指南会随着项目发展持续更新，请定期查看最新版本。
