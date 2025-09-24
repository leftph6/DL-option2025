# Deep-BSDE PyTorch 环境配置

这是一个用于Deep-BSDE（深度反向随机微分方程）的PyTorch环境配置包。

## 文件说明

- `mvp.py` - 主要的Deep-BSDE程序
- `setup_env.sh` - Linux/macOS环境配置脚本
- `setup_env.bat` - Windows环境配置脚本
- `run.sh` - 一键运行脚本（Linux/macOS）
- `requirements.txt` - Python依赖包列表
- `README.md` - 本说明文件

## 快速开始

### 方法1: 使用环境脚本（推荐）

#### Linux/macOS:
```bash
# 给脚本执行权限
chmod +x setup_env.sh run.sh

# 配置环境并运行
source setup_env.sh
python mvp.py

# 或者一键运行
./run.sh
```

#### Windows:
```cmd
# 配置环境
setup_env.bat

# 运行程序
python mvp.py
```

### 方法2: 手动安装

```bash
# 安装依赖
pip install -r requirements.txt

# 运行程序
python mvp.py
```

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- NumPy 1.20+

## 程序说明

这是一个用于求解10维篮子看涨期权的Deep-BSDE求解器：

- **问题**: 10维篮子看涨期权定价
- **方法**: 深度反向随机微分方程（Deep-BSDE）
- **参数**: 
  - 资产数量: 10
  - 到期时间: 1年
  - 无风险利率: 3%
  - 波动率: 20%
  - 初始价格: 100
  - 执行价格: 100

## 输出说明

程序会输出：
1. 训练过程中的损失值和Y0参数
2. Deep-BSDE估算的期权价格
3. Monte Carlo方法估算的价格（用于对比）

## 故障排除

如果遇到问题：

1. 确保Python版本正确（3.8+）
2. 检查网络连接（安装PyTorch需要下载）
3. 在包含`mvp.py`的目录中运行脚本
4. 查看错误信息并相应调整

## 联系

如有问题，请检查环境配置或查看程序输出信息。
