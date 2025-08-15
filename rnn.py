import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

# --- 1. 数据准备 ---
# 假设我们有以下特征来预测美式期权价格：
# S: 标的资产价格
# K: 行权价格
# T: 剩余到期时间 (年)
# r: 无风险利率 (年化)
# sigma: 标的资产波动率 (年化)
# 以下是示例数据，你需要用你真实的历史数据替换它
def generate_sample_data(num_samples=1000):
    np.random.seed(42)
    S = np.random.uniform(90, 110, num_samples) # 股票价格
    K = np.random.uniform(95, 105, num_samples) # 行权价格
    T = np.random.uniform(0.1, 1.0, num_samples) # 到期时间
    r = 0.05 # 无风险利率
    sigma = np.random.uniform(0.15, 0.35, num_samples) # 波动率

    # 简单模拟美式看涨期权价格 (这里只是一个非常简化的模拟，实际应使用期权定价模型或历史数据)
    # 实际的美式期权定价通常需要数值方法 (如二叉树、有限差分)
    # 这里的C是基于Black-Scholes公式的简化，没有考虑早期行权，
    # 你需要更复杂的逻辑来生成美式期权价格或使用真实数据
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    # 这是一个简化的看涨期权价格，不完全是美式期权的行为
    C = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    # 引入一些随机噪声以模拟真实数据的不确定性
    C += np.random.normal(0, 0.5, num_samples)
    C = np.maximum(C, np.maximum(S - K, 0)) # 美式看涨期权的内在价值

    features = np.vstack([S, K, T, r * np.ones(num_samples), sigma]).T
    labels = C.reshape(-1, 1)
    return features, labels

from scipy.stats import norm
X, y = generate_sample_data(num_samples=5000)

# 数据标准化
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# RNN 输入需要是 (样本数, 时间步, 特征数)
# 对于期权定价，通常可以将单个时间点的特征视为一个时间步
# 如果你有序列数据（例如，一天内的股票价格走势），那么时间步就更有意义
# 这里我们假设每个样本是一个独立的观测，所以时间步为1
X_train_rnn = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test_rnn = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# --- 2. 构建RNN模型 ---
def build_rnn_model(input_shape):
    model = Sequential([
        # SimpleRNN层，return_sequences=False 表示我们只关注最后一个时间步的输出
        # 如果你有多个时间步的序列数据，并且希望每个时间步都有输出，则设置为True
        SimpleRNN(units=64, activation='relu', input_shape=input_shape),
        Dropout(0.2), # 防止过拟合
        Dense(32, activation='relu'),
        Dense(1) # 输出层，预测期权价格
    ])
    model.compile(optimizer='adam', loss='mse') # 使用均方误差作为损失函数
    return model

input_shape = (X_train_rnn.shape[1], X_train_rnn.shape[2]) # (时间步, 特征数)
model = build_rnn_model(input_shape)
model.summary()

# --- 3. 训练模型 ---
history = model.fit(X_train_rnn, y_train,
                    epochs=50, # 训练轮数
                    batch_size=32, # 批次大小
                    validation_split=0.1, # 10%训练数据用于验证
                    verbose=1)

# --- 4. 模型评估 ---
loss = model.evaluate(X_test_rnn, y_test, verbose=0)
print(f"测试集均方误差 (MSE): {loss:.4f}")

# 绘制训练损失和验证损失
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('模型损失')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.savefig('model_loss.png')
plt.close()

# --- 5. 预测与反标准化 ---
y_pred_scaled = model.predict(X_test_rnn)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_original = scaler_y.inverse_transform(y_test)

# 打印一些预测结果
print("\n部分预测结果：")
for i in range(10):
    print(f"真实值: {y_test_original[i][0]:.4f}, 预测值: {y_pred[i][0]:.4f}")

# 绘制真实值 vs 预测值
plt.figure(figsize=(10, 6))
plt.scatter(y_test_original, y_pred, alpha=0.3)
plt.plot([min(y_test_original), max(y_test_original)],
         [min(y_test_original), max(y_test_original)],
         'r--', label='理想预测')
plt.title('真实值 vs. 预测值')
plt.xlabel('真实期权价格')
plt.ylabel('预测期权价格')
plt.legend()
plt.grid(True)
plt.savefig('true_vs_predicted.png')
plt.close()