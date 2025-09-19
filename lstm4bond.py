import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import MinMaxScaler
    print("TensorFlow和Keras导入成功")
except ImportError as e:
    print(f"TensorFlow导入失败: {e}")
    print("尝试使用替代方案...")
    try:
        import keras
        from keras.models import Model
        from keras.layers import Input, LSTM, Dense, Concatenate, Dropout
        from keras.callbacks import EarlyStopping
        from sklearn.preprocessing import MinMaxScaler
        print("使用独立Keras库")
    except ImportError:
        print("无法导入深度学习库，程序将无法运行")
        exit(1)

def create_lstm_model(max_seq_len: int, n_static_features: int):
    """
    创建并编译一个用于LSMC的LSTM模型。
    模型有两个输入: 序列输入(价格历史)和静态输入(当前时间)。
    """
    # 序列输入
    sequence_input = Input(shape=(max_seq_len, 1), name='sequence_input')
    
    # 静态特征输入
    static_input = Input(shape=(n_static_features,), name='static_input')
    
    # LSTM层处理序列数据
    lstm_layer = LSTM(units=50, activation='relu')(sequence_input)
    lstm_layer = Dropout(0.2)(lstm_layer)
    
    # 将LSTM的输出和静态特征拼接起来
    concatenated = Concatenate()([lstm_layer, static_input])
    
    # 全连接层
    dense_layer = Dense(units=32, activation='relu')(concatenated)
    dense_layer = Dropout(0.2)(dense_layer)
    output = Dense(units=1, activation='linear')(dense_layer)
    
    # 创建模型
    model = Model(inputs=[sequence_input, static_input], outputs=output)
    
    # 编译模型
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

def price_callable_bond_lstm(
    paths_df: pd.DataFrame,
    call_price: float,
    call_schedule: list,
    maturity: float,
    risk_free_rate: float,
    epochs=20,
    batch_size=256
) -> tuple:
    """
    使用预生成的价格路径和LSTM模型为可赎回债券定价。
    """
    # 1. 数据准备
    paths = paths_df.to_numpy()
    n_paths, n_steps_plus_1 = paths.shape
    n_steps = n_steps_plus_1 - 1
    dt = maturity / n_steps
    time_points = np.linspace(0, maturity, n_steps + 1)
    call_indices = {np.abs(time_points - t).argmin() for t in call_schedule}
    
    print("--- 阶段1: 收集LSTM训练数据 ---")
    # --------------------------------------------------------------------------
    # 第一遍向后迭代: 收集训练数据
    # --------------------------------------------------------------------------
    all_sequences = []
    all_static_features = []
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
                # 提取序列数据: 从0到t_idx(包含)
                sequences = paths[itm_indices, :t_idx+1]
                # 当前时间作为静态特征
                static_features = np.full((len(itm_indices), 1), time_points[t_idx] / maturity) # 时间归一化
                # 标签
                labels = discounted_future_value[itm_indices]
                
                all_sequences.extend(sequences)
                all_static_features.extend(static_features)
                all_labels.extend(labels)

        cash_flow_values_for_training = discounted_future_value

    # 准备训练数据
    X_seq = np.array(all_sequences)[:, :, np.newaxis] # 增加特征维度
    X_static = np.array(all_static_features)
    y_train = np.array(all_labels)

    # 数据缩放
    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    
    # 对每个路径的序列数据进行缩放
    for i in range(X_seq.shape[0]):
        scaler_seq = MinMaxScaler()
        X_seq[i, :, 0] = scaler_seq.fit_transform(X_seq[i, :, 0].reshape(-1, 1)).flatten()

    print(f"\n--- 阶段2: 训练统一的LSTM决策模型 (共 {len(y_train)} 个样本) ---")
    # --------------------------------------------------------------------------
    # 训练统一的LSTM模型
    # --------------------------------------------------------------------------
    max_seq_len = n_steps + 1
    # 使用tf.keras.utils.pad_sequences来填充序列
    X_seq_padded = tf.keras.utils.pad_sequences(X_seq, maxlen=max_seq_len, padding='pre', dtype='float32')

    lstm_model = create_lstm_model(max_seq_len=max_seq_len, n_static_features=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    lstm_model.fit(
        [X_seq_padded, X_static],
        y_train_scaled,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    print("\n--- 阶段3: 使用训练好的LSTM模型执行定价 ---")
    # --------------------------------------------------------------------------
    # 第二遍向后迭代: 使用训练好的模型进行定价
    # --------------------------------------------------------------------------
    cash_flow_values = paths[:, -1].copy()

    for t_idx in range(n_steps - 1, 0, -1):
        discount_factor = np.exp(-risk_free_rate * dt)
        discounted_future_value = cash_flow_values * discount_factor
        
        if t_idx in call_indices:
            # 准备所有路径的数据以供预测
            all_current_sequences = paths[:, :t_idx+1][:, :, np.newaxis]
            # 缩放
            for i in range(all_current_sequences.shape[0]):
                scaler_seq = MinMaxScaler()
                all_current_sequences[i, :, 0] = scaler_seq.fit_transform(all_current_sequences[i, :, 0].reshape(-1, 1)).flatten()
            
            all_current_sequences_padded = tf.keras.utils.pad_sequences(all_current_sequences, maxlen=max_seq_len, padding='pre', dtype='float32')
            all_current_static = np.full((n_paths, 1), time_points[t_idx] / maturity)

            # 预测“继续持有价值”
            predicted_continuation_value_scaled = lstm_model.predict([all_current_sequences_padded, all_current_static])
            predicted_continuation_value = scaler_y.inverse_transform(predicted_continuation_value_scaled).flatten()
            
            # 决策
            exercise_mask = call_price >= predicted_continuation_value
            cash_flow_values = np.where(exercise_mask, call_price, discounted_future_value)
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
    print("=== LSTM可赎回债券定价程序启动 ===")
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
        'epochs': 25, # 增加训练轮数
        'batch_size': 512
    }
    
    callable_price, option_value, straight_price = price_callable_bond_lstm(
        paths_df=price_paths_df,
        **BOND_PARAMS,
        **PRICING_PARAMS
    )
    
    print("\n--- LSTM 定价结果 ---")
    print(f"普通债券的价值 (基于模拟): ${straight_price:.4f}")
    print(f"可赎回债券的价值 (LSTM模型): ${callable_price:.4f}")
    print(f"发行方持有的内含看涨期权价值: ${option_value:.4f}")