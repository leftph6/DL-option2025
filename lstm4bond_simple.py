import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 使用更简单的机器学习方法替代LSTM
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def create_simple_model():
    """
    创建一个简单的随机森林模型来替代LSTM
    """
    return RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

def price_callable_bond_simple(
    paths_df: pd.DataFrame,
    call_price: float,
    call_schedule: list,
    maturity: float,
    risk_free_rate: float,
    n_estimators=100
) -> tuple:
    """
    使用预生成的价格路径和简单机器学习模型为可赎回债券定价。
    """
    # 1. 数据准备
    paths = paths_df.to_numpy()
    n_paths, n_steps_plus_1 = paths.shape
    n_steps = n_steps_plus_1 - 1
    dt = maturity / n_steps
    time_points = np.linspace(0, maturity, n_steps + 1)
    call_indices = {np.abs(time_points - t).argmin() for t in call_schedule}
    
    print("--- 阶段1: 收集训练数据 ---")
    # --------------------------------------------------------------------------
    # 第一遍向后迭代: 收集训练数据
    # --------------------------------------------------------------------------
    all_features = []
    all_labels = []
    
    cash_flow_values_for_training = paths[:, -1].copy()
    
    for t_idx in range(n_steps - 1, 0, -1):
        discount_factor = np.exp(-risk_free_rate * dt)
        discounted_future_value = cash_flow_values_for_training * discount_factor
        
        if t_idx in call_indices:
            print(f"在时间点 t={time_points[t_idx]:.2f} 收集数据...")
            current_prices = paths[:, t_idx]
            itm_indices = np.where(current_prices > call_price)[0]
            
            if len(itm_indices) > 0:
                # 提取特征: 当前价格、时间、价格历史统计
                for idx in itm_indices:
                    price_history = paths[idx, :t_idx+1]
                    features = [
                        current_prices[idx],  # 当前价格
                        time_points[t_idx] / maturity,  # 归一化时间
                        np.mean(price_history),  # 历史平均价格
                        np.std(price_history),  # 历史价格标准差
                        np.max(price_history),  # 历史最高价格
                        np.min(price_history),  # 历史最低价格
                        price_history[-1] - price_history[0],  # 价格变化
                        len(price_history)  # 序列长度
                    ]
                    all_features.append(features)
                    all_labels.append(discounted_future_value[idx])

        cash_flow_values_for_training = discounted_future_value

    # 准备训练数据
    X_train = np.array(all_features)
    y_train = np.array(all_labels)
    
    print(f"收集到 {len(y_train)} 个训练样本")

    # 数据缩放
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    
    print("\n--- 阶段2: 训练随机森林模型 ---")
    # --------------------------------------------------------------------------
    # 训练随机森林模型
    # --------------------------------------------------------------------------
    model = create_simple_model()
    
    # 划分训练和验证集
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_scaled, y_train_scaled, test_size=0.2, random_state=42
    )
    
    model.fit(X_train_split, y_train_split)
    
    # 验证模型
    y_pred_val = model.predict(X_val_split)
    val_mse = mean_squared_error(y_val_split, y_pred_val)
    print(f"验证集MSE: {val_mse:.6f}")
    
    print("\n--- 阶段3: 使用训练好的模型执行定价 ---")
    # --------------------------------------------------------------------------
    # 第二遍向后迭代: 使用训练好的模型进行定价
    # --------------------------------------------------------------------------
    cash_flow_values = paths[:, -1].copy()

    for t_idx in range(n_steps - 1, 0, -1):
        discount_factor = np.exp(-risk_free_rate * dt)
        discounted_future_value = cash_flow_values * discount_factor
        
        if t_idx in call_indices:
            print(f"在时间点 t={time_points[t_idx]:.2f} 进行决策...")
            current_prices = paths[:, t_idx]
            
            # 准备所有路径的特征
            all_features = []
            for idx in range(n_paths):
                price_history = paths[idx, :t_idx+1]
                features = [
                    current_prices[idx],  # 当前价格
                    time_points[t_idx] / maturity,  # 归一化时间
                    np.mean(price_history),  # 历史平均价格
                    np.std(price_history),  # 历史价格标准差
                    np.max(price_history),  # 历史最高价格
                    np.min(price_history),  # 历史最低价格
                    price_history[-1] - price_history[0],  # 价格变化
                    len(price_history)  # 序列长度
                ]
                all_features.append(features)
            
            all_features_scaled = scaler_X.transform(all_features)
            
            # 预测"继续持有价值"
            predicted_continuation_value_scaled = model.predict(all_features_scaled)
            predicted_continuation_value = scaler_y.inverse_transform(
                predicted_continuation_value_scaled.reshape(-1, 1)
            ).flatten()
            
            # 决策
            exercise_mask = call_price >= predicted_continuation_value
            cash_flow_values = np.where(exercise_mask, call_price, discounted_future_value)
            
            print(f"  执行决策的路径数: {np.sum(exercise_mask)}/{n_paths}")
        else:
            cash_flow_values = discounted_future_value

    # 最终定价
    final_discount = np.exp(-risk_free_rate * dt)
    callable_bond_price = np.mean(cash_flow_values * final_discount)
    straight_bond_price = np.mean(paths[:, 0])
    embedded_option_value = straight_bond_price - callable_bond_price
    
    return callable_bond_price, embedded_option_value, straight_bond_price

# --- 主程序入口 ---
if __name__ == '__main__':
    print("=== 简化版可赎回债券定价程序启动 ===")
    CSV_FILE_PATH = 'fbm_bond_price_paths.csv'
    try:
        print(f"正在读取数据文件: {CSV_FILE_PATH}")
        price_paths_df = pd.read_csv(CSV_FILE_PATH)
        print(f"成功读取数据，共 {price_paths_df.shape[0]} 条路径，{price_paths_df.shape[1]} 个时间点")
    except FileNotFoundError:
        print(f"错误: 文件 '{CSV_FILE_PATH}' 未找到。请先生成它。")
        exit()
    except Exception as e:
        print(f"读取文件时出错: {e}")
        exit()

    BOND_PARAMS = {
        'call_price': 1020.0,
        'call_schedule': [2.0, 3.0, 4.0, 5.0],
        'maturity': 5.0
    }
    
    PRICING_PARAMS = {
        'risk_free_rate': 0.03,
        'n_estimators': 100
    }
    
    print("\n开始定价计算...")
    callable_price, option_value, straight_price = price_callable_bond_simple(
        paths_df=price_paths_df,
        **BOND_PARAMS,
        **PRICING_PARAMS
    )
    
    print("\n--- 定价结果 ---")
    print(f"普通债券的价值 (基于模拟): ${straight_price:.4f}")
    print(f"可赎回债券的价值 (机器学习模型): ${callable_price:.4f}")
    print(f"发行方持有的内含看涨期权价值: ${option_value:.4f}")
    
    print("\n程序执行完成！")
