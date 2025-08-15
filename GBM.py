import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适合命令行环境
import matplotlib.pyplot as plt

def simulate_gbm_paths(S0, mu, sigma, T, N, M, seed=None):
    """
    模拟 GBM 路径
    :param S0: 初始价格
    :param mu: 漂移率（年化收益率）
    :param sigma: 波动率（年化标准差）
    :param T: 总时间（单位：年）
    :param N: 时间步数
    :param M: 模拟路径条数
    :param seed: 随机种子
    :return: 模拟出的价格路径数组 shape=(M, N+1)
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / N
    t = np.linspace(0, T, N + 1)
    paths = np.zeros((M, N + 1))
    paths[:, 0] = S0

    # 生成路径
    for i in range(1, N + 1):
        Z = np.random.standard_normal(M)
        paths[:, i] = paths[:, i - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

    return t, paths

# 参数设置
S0 = 100      # 初始价格
mu = 0.1      # 漂移率
sigma = 0.2   # 波动率
T = 1         # 模拟1年
N = 252       # 每年252个交易日
M = 10        # 模拟10条路径

print("开始模拟GBM路径...")
# 模拟路径
t, paths = simulate_gbm_paths(S0, mu, sigma, T, N, M, seed=42)
print(f"模拟完成！生成了 {M} 条路径，每条路径有 {N+1} 个时间点")

# 显示一些统计信息
print(f"初始价格: {S0}")
print(f"最终价格范围: {paths[:, -1].min():.2f} - {paths[:, -1].max():.2f}")
print(f"平均最终价格: {paths[:, -1].mean():.2f}")

# 绘图
plt.figure(figsize=(12, 6))
for i in range(M):
    plt.plot(t, paths[i], lw=1)
plt.title('Geometric Brownian Motion (GBM) Simulated Paths')
plt.xlabel('Time (Years)')
plt.ylabel('Asset Price')
plt.grid(True)

# 保存图片到文件而不是显示
plt.savefig('gbm_simulation.png', dpi=300, bbox_inches='tight')
print("图片已保存为 'gbm_simulation.png'")

# 如果环境支持，也尝试显示
try:
    plt.show()
    print("图形窗口已显示")
except:
    print("无法显示图形窗口，但图片已保存到文件")