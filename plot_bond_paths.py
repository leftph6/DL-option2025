import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib import rcParams

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_bond_price_paths(csv_file_path, num_paths_to_plot=50, figsize=(12, 8)):
    """
    绘制债券价格路径的折线图
    
    参数:
    csv_file_path (str): CSV文件路径
    num_paths_to_plot (int): 要绘制的路径数量（默认50条）
    figsize (tuple): 图形大小
    """
    print(f"正在读取数据文件: {csv_file_path}")
    
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_file_path)
        print(f"成功读取数据，共 {df.shape[0]} 条路径，{df.shape[1]} 个时间点")
    except Exception as e:
        print(f"读取文件出错: {e}")
        return
    
    # 创建图形
    plt.figure(figsize=figsize)
    
    # 设置颜色映射
    colors = plt.cm.viridis(np.linspace(0, 1, min(num_paths_to_plot, df.shape[0])))
    
    # 绘制指定数量的路径
    paths_to_plot = min(num_paths_to_plot, df.shape[0])
    
    for i in range(paths_to_plot):
        # 获取第i条路径的所有时间点数据
        path_data = df.iloc[i].values
        
        # 绘制折线图，使用半透明效果
        plt.plot(path_data, color=colors[i], alpha=0.7, linewidth=0.8)
    
    # 计算并绘制平均路径
    mean_path = df.mean(axis=0)
    plt.plot(mean_path, color='red', linewidth=2, label=f'平均路径 (基于{df.shape[0]}条路径)')
    
    # 计算并绘制置信区间
    std_path = df.std(axis=0)
    upper_bound = mean_path + 1.96 * std_path  # 95%置信区间上界
    lower_bound = mean_path - 1.96 * std_path  # 95%置信区间下界
    
    plt.fill_between(range(len(mean_path)), lower_bound, upper_bound, 
                     color='red', alpha=0.2, label='95% 置信区间')
    
    # 设置图形属性
    plt.title(f'Vasicek模型驱动的债券价格路径\n(分数布朗运动, H=0.7, 显示前{paths_to_plot}条路径)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('时间步', fontsize=12)
    plt.ylabel('债券价格', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 添加统计信息
    initial_price = df.iloc[0, 0]
    final_prices = df.iloc[:, -1]
    final_mean = final_prices.mean()
    final_std = final_prices.std()
    
    stats_text = f'初始价格: {initial_price:.2f}\n'
    stats_text += f'最终价格均值: {final_mean:.2f}\n'
    stats_text += f'最终价格标准差: {final_std:.2f}\n'
    stats_text += f'总路径数: {df.shape[0]}\n'
    stats_text += f'时间步数: {df.shape[1]-1}'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图形
    output_file = 'bond_price_paths_plot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"图形已保存为: {output_file}")
    
    # 显示图形
    plt.show()

def plot_statistics(csv_file_path, figsize=(15, 10)):
    """
    绘制详细的统计分析图
    """
    print(f"正在生成统计分析图...")
    
    # 读取数据
    df = pd.read_csv(csv_file_path)
    
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('债券价格路径统计分析', fontsize=16, fontweight='bold')
    
    # 1. 所有路径的折线图（前100条）
    ax1 = axes[0, 0]
    num_paths = min(100, df.shape[0])
    for i in range(num_paths):
        ax1.plot(df.iloc[i].values, alpha=0.3, linewidth=0.5)
    mean_path = df.mean(axis=0)
    ax1.plot(mean_path, color='red', linewidth=2, label='平均路径')
    ax1.set_title(f'价格路径 (前{num_paths}条)')
    ax1.set_xlabel('时间步')
    ax1.set_ylabel('价格')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 最终价格分布直方图
    ax2 = axes[0, 1]
    final_prices = df.iloc[:, -1]
    ax2.hist(final_prices, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(final_prices.mean(), color='red', linestyle='--', linewidth=2, label=f'均值: {final_prices.mean():.2f}')
    ax2.axvline(final_prices.median(), color='green', linestyle='--', linewidth=2, label=f'中位数: {final_prices.median():.2f}')
    ax2.set_title('最终价格分布')
    ax2.set_xlabel('最终价格')
    ax2.set_ylabel('频数')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 价格波动性（标准差随时间变化）
    ax3 = axes[0, 2]
    price_std = df.std(axis=0)
    ax3.plot(price_std, color='purple', linewidth=2)
    ax3.set_title('价格波动性随时间变化')
    ax3.set_xlabel('时间步')
    ax3.set_ylabel('标准差')
    ax3.grid(True, alpha=0.3)
    
    # 4. 价格范围（最大值和最小值）
    ax4 = axes[1, 0]
    max_prices = df.max(axis=0)
    min_prices = df.min(axis=0)
    ax4.plot(max_prices, color='red', linewidth=2, label='最大值')
    ax4.plot(min_prices, color='blue', linewidth=2, label='最小值')
    ax4.fill_between(range(len(mean_path)), min_prices, max_prices, alpha=0.3, color='gray')
    ax4.set_title('价格范围')
    ax4.set_xlabel('时间步')
    ax4.set_ylabel('价格')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 收益率分布（最终收益率）
    ax5 = axes[1, 1]
    initial_price = df.iloc[0, 0]
    returns = (final_prices - initial_price) / initial_price * 100
    ax5.hist(returns, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    ax5.axvline(returns.mean(), color='red', linestyle='--', linewidth=2, label=f'均值: {returns.mean():.2f}%')
    ax5.set_title('最终收益率分布')
    ax5.set_xlabel('收益率 (%)')
    ax5.set_ylabel('频数')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 分位数分析
    ax6 = axes[1, 2]
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    quantile_data = np.percentile(df.values, [q*100 for q in quantiles], axis=0)
    
    for i, q in enumerate(quantiles):
        ax6.plot(quantile_data[i], linewidth=2, label=f'{q*100}% 分位数')
    
    ax6.set_title('价格分位数分析')
    ax6.set_xlabel('时间步')
    ax6.set_ylabel('价格')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图形
    output_file = 'bond_price_statistics.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"统计分析图已保存为: {output_file}")
    
    plt.show()

if __name__ == '__main__':
    # CSV文件路径
    csv_file = 'fbm_bond_price_paths.csv'
    
    print("=== 债券价格路径可视化程序 ===")
    print("1. 绘制价格路径折线图")
    print("2. 生成详细统计分析图")
    print("3. 同时生成两种图形")
    
    choice = input("请选择要生成的图形类型 (1/2/3): ").strip()
    
    if choice == '1':
        plot_bond_price_paths(csv_file, num_paths_to_plot=50)
    elif choice == '2':
        plot_statistics(csv_file)
    elif choice == '3':
        plot_bond_price_paths(csv_file, num_paths_to_plot=50)
        plot_statistics(csv_file)
    else:
        print("无效选择，默认生成折线图")
        plot_bond_price_paths(csv_file, num_paths_to_plot=50)
    
    print("程序执行完成！")

