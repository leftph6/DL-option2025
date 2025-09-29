#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Deep BSDE 套利策略回测系统 - 主程序
模块化架构的主入口
"""

import os
import sys
import argparse
from typing import Dict, Any, Optional
import torch
import numpy as np
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入各个模块
from math_modeling import get_path_generator, get_model
from ml_models import ModelFactory
from backtest import BacktestEngine, DataLoaderFactory, get_strategy
from config import ConfigManager, get_config
from utils import DeviceManager, setup_logger, log_system_info


class DeepBSDESystem:
    """Deep BSDE系统主类"""
    
    def __init__(self, config_file: Optional[str] = None):
        """初始化系统"""
        # 设置日志
        self.logger = setup_logger('deep_bsde_system')
        
        # 加载配置
        self.config_manager = ConfigManager(config_file)
        self.config = self.config_manager.get_config()
        
        # 设置设备
        device_config = self.config.get('device', {})
        self.device_manager = DeviceManager(device_config)
        self.device = self.device_manager.get_device()
        
        # 初始化组件
        self.path_generator = None
        self.model = None
        self.strategy = None
        self.data_loader = None
        self.backtest_engine = None
        
        self.logger.info("Deep BSDE系统初始化完成")
        self.device_manager.print_device_info()
    
    def setup_data_generation(self, generator_type: str = 'gbm', **kwargs):
        """设置数据生成"""
        self.logger.info(f"设置数据生成器: {generator_type}")
        
        # 获取默认配置
        default_config = self.config['math_modeling']['path_generators'].get(generator_type, {})
        default_config.update(kwargs)
        
        # 创建路径生成器
        self.path_generator = get_path_generator(generator_type, **default_config)
        
        self.logger.info(f"✓ 数据生成器设置完成: {generator_type}")
        return self.path_generator
    
    def generate_training_data(self, batch_size: int = 1000, **kwargs) -> tuple:
        """生成训练数据"""
        if self.path_generator is None:
            raise ValueError("请先设置数据生成器")
        
        self.logger.info(f"生成训练数据: {batch_size}条路径")
        
        # 生成路径
        paths, times = self.path_generator.generate_paths(batch_size, **kwargs)
        
        self.logger.info(f"✓ 训练数据生成完成: {paths.shape}")
        return paths, times
    
    def setup_model(self, model_type: str = 'deep_bsde_rnn', **kwargs):
        """设置模型"""
        self.logger.info(f"设置模型: {model_type}")
        
        # 获取默认配置
        default_config = self.config['ml_models'].get(model_type, {})
        default_config.update(kwargs)
        
        # 创建模型
        self.model = ModelFactory.create_model(model_type, device=self.device, **default_config)
        
        self.logger.info(f"✓ 模型设置完成: {model_type}")
        return self.model
    
    def train_model(self, train_data: tuple, **kwargs):
        """训练模型"""
        if self.model is None:
            raise ValueError("请先设置模型")
        
        self.logger.info("开始训练模型")
        
        # 获取训练配置
        training_config = self.config['ml_models']['training'].copy()
        training_config.update(kwargs)
        
        # 创建训练器
        trainer = ModelFactory.create_trainer(
            self.model.__class__.__name__.replace('Model', 'Trainer'),
            self.model,
            device=self.device,
            **training_config
        )
        
        # 准备数据
        paths, times = train_data
        if isinstance(paths, torch.Tensor):
            paths = paths.cpu().numpy()
        
        # 创建数据加载器（简化版本）
        from torch.utils.data import DataLoader, TensorDataset
        
        # 根据模型类型准备数据
        if hasattr(self.model, 'forward') and len(self.model.forward.__code__.co_varnames) > 2:
            # Deep BSDE模型需要特殊处理
            if hasattr(self.model, 'Y0'):  # Deep BSDE模型
                # 计算dS和log returns
                dS = paths[:, 1:] - paths[:, :-1]
                eps = 1e-9
                log_returns = np.log(paths[:, 1:] + eps) - np.log(paths[:, :-1] + eps)
                log_returns_full = np.concatenate([np.zeros((paths.shape[0], 1)), log_returns], axis=1)
                
                # 计算目标（期权收益）
                strike = kwargs.get('strike', 100.0)
                targets = np.maximum(paths[:, -1] - strike, 0.0)
                
                # 创建数据集
                dataset = TensorDataset(
                    torch.tensor(log_returns_full, dtype=torch.float32),
                    torch.tensor(paths, dtype=torch.float32),
                    torch.tensor(dS, dtype=torch.float32),
                    torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
                )
            else:
                # 普通模型
                dataset = TensorDataset(
                    torch.tensor(paths, dtype=torch.float32),
                    torch.tensor(paths, dtype=torch.float32)  # 简化：输入输出相同
                )
        else:
            # 简单模型
            dataset = TensorDataset(
                torch.tensor(paths, dtype=torch.float32),
                torch.tensor(paths, dtype=torch.float32)
            )
        
        dataloader = DataLoader(dataset, batch_size=training_config['batch_size'], shuffle=True)
        
        # 训练模型
        history = trainer.train(
            dataloader,
            epochs=training_config['epochs'],
            verbose=True
        )
        
        self.logger.info("✓ 模型训练完成")
        return history
    
    def setup_strategy(self, strategy_type: str = 'arbitrage', **kwargs):
        """设置策略"""
        self.logger.info(f"设置策略: {strategy_type}")
        
        # 获取默认配置
        default_config = self.config['backtest']['strategies'].get(strategy_type, {})
        default_config.update(kwargs)
        
        # 创建策略
        self.strategy = get_strategy(strategy_type, device=self.device, **default_config)
        
        self.logger.info(f"✓ 策略设置完成: {strategy_type}")
        return self.strategy
    
    def run_backtest(self, test_data: tuple, **kwargs):
        """运行回测"""
        if self.strategy is None:
            raise ValueError("请先设置策略")
        
        self.logger.info("开始回测")
        
        # 准备数据
        paths, times = test_data
        if isinstance(paths, torch.Tensor):
            paths = paths.cpu().numpy()
        
        # 如果有模型，计算Y序列
        Y_seqs = None
        if self.model is not None:
            self.logger.info("计算模型预测的Y序列")
            self.model.eval()
            
            with torch.no_grad():
                # 根据模型类型计算Y序列
                if hasattr(self.model, 'Y0'):  # Deep BSDE模型
                    Y_seqs = []
                    for i in range(paths.shape[0]):
                        path = paths[i:i+1]
                        dS = path[:, 1:] - path[:, :-1]
                        eps = 1e-9
                        log_returns = np.log(path[:, 1:] + eps) - np.log(path[:, :-1] + eps)
                        log_returns_full = np.concatenate([np.zeros((1, 1)), log_returns], axis=1)
                        
                        # 使用模型预测
                        if hasattr(self.model, 'forward'):
                            try:
                                Y_T, Y_seq = self.model(
                                    torch.tensor(log_returns_full, dtype=torch.float32, device=self.device),
                                    torch.tensor(path, dtype=torch.float32, device=self.device),
                                    torch.tensor(dS, dtype=torch.float32, device=self.device),
                                    torch.tensor(np.diff(times), dtype=torch.float32, device=self.device)
                                )
                                Y_seqs.append(Y_seq.cpu().numpy().flatten())
                            except Exception as e:
                                self.logger.warning(f"模型预测失败: {e}")
                                Y_seqs.append(np.zeros(len(times)))
                        else:
                            Y_seqs.append(np.zeros(len(times)))
                    
                    Y_seqs = np.array(Y_seqs)
                else:
                    # 简单模型，使用价格作为Y序列
                    Y_seqs = paths
        
        # 创建回测引擎
        self.backtest_engine = BacktestEngine(self.strategy, device=self.device)
        
        # 运行回测
        results = self.backtest_engine.run_backtest(
            torch.tensor(paths, dtype=torch.float32),
            torch.tensor(Y_seqs, dtype=torch.float32) if Y_seqs is not None else None,
            **kwargs
        )
        
        self.logger.info("✓ 回测完成")
        return results
    
    def save_results(self, results: Dict[str, Any], output_dir: str = "output"):
        """保存结果"""
        if self.backtest_engine is None:
            raise ValueError("请先运行回测")
        
        self.logger.info("保存回测结果")
        
        # 保存CSV结果
        saved_files = self.backtest_engine.save_results(output_dir)
        
        # 生成可视化图表
        from backtest import ResultVisualizer
        visualizer = ResultVisualizer(output_dir)
        
        plot_files = {}
        try:
            plot_files['pnl_analysis'] = visualizer.plot_pnl_curves(results, save_plot=True)
            plot_files['price_paths'] = visualizer.plot_price_paths(results, save_plot=True)
            plot_files['risk_metrics'] = visualizer.plot_risk_metrics(results, save_plot=True)
            plot_files['interactive'] = visualizer.create_interactive_plot(results, save_html=True)
        except Exception as e:
            self.logger.warning(f"生成图表失败: {e}")
        
        # 生成汇总报告
        try:
            from backtest import PlotGenerator
            plot_generator = PlotGenerator(output_dir)
            report_file = plot_generator.generate_summary_report(results)
            plot_files['summary_report'] = report_file
        except Exception as e:
            self.logger.warning(f"生成报告失败: {e}")
        
        self.logger.info("✓ 结果保存完成")
        return {**saved_files, **plot_files}
    
    def run_complete_pipeline(self, 
                            generator_type: str = 'gbm',
                            model_type: str = 'deep_bsde_rnn',
                            strategy_type: str = 'arbitrage',
                            train_size: int = 1000,
                            test_size: int = 100,
                            **kwargs) -> Dict[str, Any]:
        """运行完整流程"""
        self.logger.info("开始运行完整流程")
        
        try:
            # 1. 设置数据生成
            self.setup_data_generation(generator_type, **kwargs)
            
            # 2. 生成训练数据
            train_data = self.generate_training_data(train_size, **kwargs)
            
            # 3. 设置模型
            self.setup_model(model_type, **kwargs)
            
            # 4. 训练模型
            training_history = self.train_model(train_data, **kwargs)
            
            # 5. 生成测试数据
            test_data = self.generate_training_data(test_size, **kwargs)
            
            # 6. 设置策略
            self.setup_strategy(strategy_type, **kwargs)
            
            # 7. 运行回测
            results = self.run_backtest(test_data, **kwargs)
            
            # 8. 保存结果
            saved_files = self.save_results(results)
            
            self.logger.info("✓ 完整流程运行成功")
            
            return {
                'results': results,
                'training_history': training_history,
                'saved_files': saved_files,
                'config': self.config
            }
            
        except Exception as e:
            self.logger.error(f"流程运行失败: {e}")
            raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Deep BSDE 套利策略回测系统')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--generator', type=str, default='gbm', 
                       choices=['gbm', 'fbm', 'vasicek', 'jump_diffusion'],
                       help='路径生成器类型')
    parser.add_argument('--model', type=str, default='deep_bsde_rnn',
                       choices=['mlp', 'deep_bsde_mlp', 'rnn', 'deep_bsde_rnn', 'transformer'],
                       help='模型类型')
    parser.add_argument('--strategy', type=str, default='arbitrage',
                       choices=['arbitrage', 'callable_bond', 'mean_reversion'],
                       help='策略类型')
    parser.add_argument('--train_size', type=int, default=1000, help='训练数据大小')
    parser.add_argument('--test_size', type=int, default=100, help='测试数据大小')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=256, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='学习率')
    parser.add_argument('--output_dir', type=str, default='output', help='输出目录')
    parser.add_argument('--interactive', action='store_true', help='交互式模式')
    
    args = parser.parse_args()
    
    # 记录系统信息
    log_system_info()
    
    try:
        # 创建系统实例
        system = DeepBSDESystem(args.config)
        
        if args.interactive:
            # 交互式模式
            run_interactive_mode(system)
        else:
            # 命令行模式
            results = system.run_complete_pipeline(
                generator_type=args.generator,
                model_type=args.model,
                strategy_type=args.strategy,
                train_size=args.train_size,
                test_size=args.test_size,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate
            )
            
            print("\n" + "="*60)
            print("运行完成！")
            print("="*60)
            print(f"生成的文件保存在: {args.output_dir}")
            for file_type, file_path in results['saved_files'].items():
                print(f"  {file_type}: {file_path}")
            print("="*60)
    
    except Exception as e:
        print(f"❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def run_interactive_mode(system: DeepBSDESystem):
    """交互式模式"""
    print("\n" + "="*60)
    print("Deep BSDE 套利策略回测系统 - 交互式模式")
    print("="*60)
    
    while True:
        print("\n请选择操作:")
        print("1. 设置数据生成器")
        print("2. 生成训练数据")
        print("3. 设置模型")
        print("4. 训练模型")
        print("5. 设置策略")
        print("6. 运行回测")
        print("7. 保存结果")
        print("8. 运行完整流程")
        print("9. 查看配置")
        print("0. 退出")
        
        choice = input("\n请输入选择 (0-9): ").strip()
        
        try:
            if choice == '0':
                print("退出程序")
                break
            elif choice == '1':
                generator_type = input("请输入生成器类型 (gbm/fbm/vasicek/jump_diffusion): ").strip()
                system.setup_data_generation(generator_type)
            elif choice == '2':
                batch_size = int(input("请输入数据大小 (默认1000): ") or "1000")
                train_data = system.generate_training_data(batch_size)
                print(f"✓ 生成了 {train_data[0].shape[0]} 条路径")
            elif choice == '3':
                model_type = input("请输入模型类型 (mlp/rnn/deep_bsde_rnn): ").strip()
                system.setup_model(model_type)
            elif choice == '4':
                epochs = int(input("请输入训练轮数 (默认200): ") or "200")
                system.train_model(system.generate_training_data(1000), epochs=epochs)
            elif choice == '5':
                strategy_type = input("请输入策略类型 (arbitrage/callable_bond): ").strip()
                system.setup_strategy(strategy_type)
            elif choice == '6':
                test_size = int(input("请输入测试数据大小 (默认100): ") or "100")
                test_data = system.generate_training_data(test_size)
                results = system.run_backtest(test_data)
                print(f"✓ 回测完成，平均PNL: {results['summary']['mean_final_pnl']:.2f}")
            elif choice == '7':
                output_dir = input("请输入输出目录 (默认output): ").strip() or "output"
                saved_files = system.save_results(system.run_backtest(system.generate_training_data(100)), output_dir)
                print(f"✓ 结果已保存到: {output_dir}")
            elif choice == '8':
                results = system.run_complete_pipeline()
                print("✓ 完整流程运行成功")
            elif choice == '9':
                config_summary = system.config_manager.get_config_summary()
                print(f"配置摘要: {config_summary}")
            else:
                print("❌ 无效选择")
        
        except Exception as e:
            print(f"❌ 操作失败: {e}")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
