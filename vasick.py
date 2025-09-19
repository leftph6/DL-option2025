import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================
# Vasicek Model Parameters
# =====================
r0 = 0.04        # 初始短利率
kappa = 0.5      # 均值回复速度
theta = 0.05     # 长期均值 (真实测度下)
sigma = 0.02     # 波动率
dt = 1.0         # 步长 = 1 年
N = 2            # 期限 = 2 年
face_value = 100
coupon = 5
call_price = 101

market_price_of_risk = 0.0  # lambda

# Risk-neutral drift adjustment
theta_Q = theta - market_price_of_risk * sigma / kappa
print("模型参数 (风险中性 theta_Q = theta - lambda*sigma/kappa):\n",
      f"r0={r0:.4f}, kappa={kappa:.4f}, theta={theta:.4f}, sigma={sigma:.4f}, market_lambda={market_price_of_risk:.4f}\n",
      f"theta_Q={theta_Q:.6f}, dt={dt}, N={N}, h={(sigma*np.sqrt(dt)):.6e}, var_step={(sigma**2*dt):.6e}\n")

# =====================
# 构建利率二叉树 (Ho-Lee/Vasicek离散化近似)
# =====================
h = sigma * np.sqrt(dt)

def q_prob(r):
    return 0.5 + 0.5 * (kappa * (theta_Q - r) * dt) / h

# 存储利率树
rates = {}
rates[(0,0)] = r0
for n in range(1, N+1):
    for j in range(n+1):
        if j == n:  # 最上节点
            rates[(n,j)] = rates[(n-1,j-1)] + h
        else:
            rates[(n,j)] = rates[(n-1,j)] - h

# =====================
# 定价树：自下而上
# =====================
def price_bond(callable_flag=True):
    V = {}
    # 到期时价值
    for j in range(N+1):
        V[(N,j)] = face_value + coupon

    # 向后归纳
    for n in range(N-1, -1, -1):
        for j in range(n+1):
            r = rates[(n,j)]
            q = q_prob(r)
            disc = np.exp(-r * dt)
            value_hold = coupon + disc * (q * V[(n+1,j+1)] + (1-q) * V[(n+1,j)])
            if callable_flag and n == 1:  # 可赎回时刻
                V[(n,j)] = min(value_hold, call_price)
            else:
                V[(n,j)] = value_hold
    return V

# 直债 & 可赎回债定价
V_straight = price_bond(callable_flag=False)
V_callable = price_bond(callable_flag=True)

print("Straight bond price (t=0):", V_straight[(0,0)])
print("Callable bond price (t=0):", V_callable[(0,0)])
print("Embedded call option value:", V_straight[(0,0)] - V_callable[(0,0)])

# =====================
# 构建可视化表格
# =====================
table = []
for n in range(N+1):
    row = {'time': n}
    for j in range(n+1):
        row[f'V_straight_n{j}'] = V_straight.get((n,j), np.nan)
        row[f'V_callable_n{j}'] = V_callable.get((n,j), np.nan)
    table.append(row)
df = pd.DataFrame(table)
print(df)

# =====================
# 绘制利率二叉树
# =====================
plt.figure(figsize=(6,4))
for n in range(N+1):
    for j in range(n+1):
        plt.scatter(n, rates[(n,j)], color='blue')
        if n < N:
            plt.plot([n, n+1],[rates[(n,j)], rates[(n+1,j)]],'k--',alpha=0.5)
            plt.plot([n, n+1],[rates[(n,j)], rates[(n+1,j+1)]],'k--',alpha=0.5)
plt.title("Vasicek Risk-Neutral Short Rate Tree")
plt.xlabel("Time step")
plt.ylabel("Short rate")
plt.grid(True)
plt.show()