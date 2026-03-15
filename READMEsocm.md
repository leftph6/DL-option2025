# SOCM-BSDE 复现程序

本目录是根据 `guideline` 文档实现的复现代码，包含：
- SOCM 训练目标的 Deep BSDE 控制学习
- 经典 Deep BSDE 终端损失基线
- 对比实验：定价误差、收敛速度、梯度方差

所有需要的程序统一放在本目录内，便于管理和一键运行。

## 目录结构与路径清单

```
socm_bsde_repro/
├── environment.yml
├── requirements.txt
├── README.md
├── scripts/
│   ├── run_socm.py
│   ├── run_bsde.py
│   ├── run_experiment.py
│   └── run_optuna.py
└── src/
    └── socm_bsde/
        ├── __init__.py
        ├── black_scholes.py
        ├── experiment.py
        ├── nets.py
        ├── path_generators.py
        ├── optuna_hpo.py
        ├── train_bsde.py
        ├── train_socm.py
        └── utils.py
```

## 环境配置

使用 Conda（推荐）：

```bash
cd /Users/mashengyu/Desktop/dlproject/code/scombsde/socm_bsde_repro
conda env create -f environment.yml
conda activate deep-bsde-pricing
```

或使用 pip：

```bash
cd /Users/mashengyu/Desktop/dlproject/code/scombsde/socm_bsde_repro
pip install -r requirements.txt
```

## 快速运行

SOCM 训练：

```bash
cd /Users/mashengyu/Desktop/dlproject/code/scombsde/socm_bsde_repro
PYTHONPATH=src python scripts/run_socm.py --epochs 200 --batch_size 1024
```

经典 Deep BSDE 训练：

```bash
PYTHONPATH=src python scripts/run_bsde.py --epochs 200 --batch_size 1024
```

对比实验（SOCM vs BSDE）：

```bash
PYTHONPATH=src python scripts/run_experiment.py --epochs 200 --batch_size 1024
```

超参数优化（Optuna）：

```bash
PYTHONPATH=src python scripts/run_optuna.py --n_trials 20 --epochs_per_trial 50 --batch_size 512
```

如果当前 Python 环境未安装 `optuna`，项目会尝试从 `vendor/optuna` 加载离线拷贝。

## 输出说明

- `outputs_socm/socm_history.json`
- `outputs_bsde/bsde_history.json`
- `outputs_experiment/summary.json`
- `outputs_experiment/history.json`
- `outputs_experiment/plots.json`
- `outputs_experiment/loss_curves.png`
- `outputs_experiment/grad_norm_curves.png`
- `outputs_experiment/price_error_curves.png`
- `outputs_experiment/socm_y0_curve.png`
- `outputs_experiment/bsde_y0_curve.png`
- `outputs_experiment/metrics_bar.png`
- `outputs_optuna/optuna_summary.json`
- `outputs_optuna/socm_trial_losses.png`
- `outputs_optuna/bsde_trial_losses.png`
- `outputs_optuna/socm_optimization_history.png`
- `outputs_optuna/bsde_optimization_history.png`
- `outputs_optuna/socm_param_importances.png`
- `outputs_optuna/bsde_param_importances.png`

`summary.json` 包含：
- Black-Scholes 解析定价
- SOCM/BSDE 的价格估计、定价误差
- 收敛速度（loss 降至初始 10% 所需 epoch）
- 梯度方差（多批次梯度范数方差）
- 额外统计：loss/grad_norm 统计量、相对误差、SOCM 的 Z-matching MSE

## 关键实现说明

- `train_socm.py`: 对应 guideline 中的 SOCM 目标
- `train_bsde.py`: 经典终端损失 Deep BSDE
- `experiment.py`: 指标对比与汇总
- `optuna_hpo.py`: Optuna 超参数优化与对比
- `black_scholes.py`: 解析 Delta 与定价，用于构造 SOCM 目标和误差评估
- `path_generators.py`: GBM 路径生成（风险中性）
- `nets.py`: 控制网络（policy）与 Deep BSDE 模型

如需改动参数（T、N、r、sigma、strike 等），可直接通过命令行参数覆盖。
