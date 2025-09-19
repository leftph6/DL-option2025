import numpy as np
import pandas as pd
from fbm import FBM

def simulate_vasicek_fbm_price_paths(
    n_paths: int,
    n_steps: int,
    T: float,
    initial_price: float,
    kappa: float,
    theta: float,
    sigma: float,
    hurst: float,
    seed: int,
    output_csv_path: str
) -> pd.DataFrame:
    """
    使用由分数布朗运动(fBm)驱动的Vasicek模型模拟债券价格路径。

    参数:
    n_paths (int): 要模拟的路径数量。
    n_steps (int): 每个路径的时间步数。
    T (float): 总模拟时长（年）。
    initial_price (float): 债券的初始价格 P_0。
    kappa (float): 均值回归速度。
    theta (float): 长期均值价格（例如，债券面值）。
    sigma (float): 波动率。
    hurst (float): 分数布朗运动的Hurst指数 (0 < H < 1)。
    seed (int): 用于复现结果的随机种子。
    output_csv_path (str): 保存输出路径的CSV文件名。

    返回:
    pandas.DataFrame: 包含所有模拟路径的DataFrame。
    """
    print(f"--- 开始模拟 {n_paths} 条价格路径 ---")
    
    # 1. 设置参数
    dt = T / n_steps
    
    # 设置随机种子以保证结果可复现
    np.random.seed(seed)
    
    # 2. 初始化分数布朗运动生成器
    # 使用 Davies-Harte 方法，它是一种精确的模拟方法
    fbm_generator = FBM(n=n_steps, hurst=hurst, length=T, method='daviesharte')
    
    # 3. 准备存储路径的容器
    # 每一行代表一条模拟路径
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = initial_price
    
    # 4. 模拟所有路径
    for i in range(n_paths):
        # 生成一条分数布朗运动样本路径
        fbm_sample = fbm_generator.fbm()
        
        # 计算fBm的增量，对应于 dW_t
        fbm_increments = np.diff(fbm_sample)
        
        # 使用欧拉-丸山法进行离散化迭代
        for t in range(n_steps):
            # Vasicek SDE: dP_t = kappa * (theta - P_t) * dt + sigma * dB_H(t)
            price_drift = kappa * (theta - paths[i, t]) * dt
            price_shock = sigma * fbm_increments[t]
            paths[i, t+1] = paths[i, t] + price_drift + price_shock

        if (i + 1) % 1000 == 0:
            print(f"已生成 {i + 1}/{n_paths} 条路径...")

    # 5. 转换为DataFrame并保存
    print("\n--- 模拟完成，正在保存到CSV文件 ---")
    
    # 创建列名，如 Time_0, Time_1, ...
    columns = [f"Time_{j}" for j in range(n_steps + 1)]
    df_paths = pd.DataFrame(paths, columns=columns)
    
    # 保存到CSV文件，不包含索引
    df_paths.to_csv(output_csv_path, index=False)
    
    print(f"成功将 {n_paths} 条路径数据保存到: {output_csv_path}")
    
    return df_paths

# --- 主程序入口 ---
if __name__ == '__main__':
    # --- 可调参数 ---
    
    # 模拟设置 - 减少路径数量以便快速测试
    N_PATHS = 1000               # 减少到1000条路径进行测试
    N_STEPS = 252 * 2            # 减少到2年，每年252个交易日
    T_HORIZON = 2.0              # 总时长（年）
    RANDOM_SEED = 42             # 随机种子，改变它可以得到不同的随机路径

    # 债券和Vasicek模型参数
    INITIAL_PRICE = 980.0        # 债券初始价格
    KAPPA = 1.0                  # 均值回归速度 (越高回归越快)
    THETA = 1000.0               # 长期目标价 (设为债券面值)
    SIGMA = 15.0                 # 价格年化波动率
    HURST_EXPONENT = 0.7         # Hurst指数 ( > 0.5 表示趋势持续)
    
    # 输出文件
    OUTPUT_FILE = 'fbm_bond_price_paths.csv'

    print("程序开始运行...")
    print(f"参数设置: {N_PATHS} 条路径, {N_STEPS} 步, {T_HORIZON} 年")

    # --- 执行模拟 ---
    try:
        simulated_data = simulate_vasicek_fbm_price_paths(
            n_paths=N_PATHS,
            n_steps=N_STEPS,
            T=T_HORIZON,
            initial_price=INITIAL_PRICE,
            kappa=KAPPA,
            theta=THETA,
            sigma=SIGMA,
            hurst=HURST_EXPONENT,
            seed=RANDOM_SEED,
            output_csv_path=OUTPUT_FILE
        )

        # 显示生成数据的前5条路径的前5个时间点，以供预览
        print("\n--- 数据预览 (前5条路径, 前5个时间点) ---")
        print(simulated_data.head(5).iloc[:, :5])
        
        print("\n程序执行完成！")
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()

