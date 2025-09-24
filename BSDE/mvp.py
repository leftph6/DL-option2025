# deep_bsde_pytorch_min.py
# Minimal Deep-BSDE solver (PyTorch) for a d-dim Basket Call under Black-Scholes dynamics
import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------ problem parameters -------------
d = 10               # number of assets
T = 1.0              # maturity
N = 50               # time steps
dt = T / N
r = 0.03             # risk-free rate
sigma = 0.2          # volatility (for simplicity same for all assets)
S0 = 100.0           # initial price (same for all assets)
K = 100.0            # strike
batch_size = 512
num_epochs = 2000
lr = 1e-3
seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)

# ------------- network for Z(t, x) -------------
class ZNet(nn.Module):
    def __init__(self, input_dim, out_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, t_and_x):
        return self.net(t_and_x)

# initial Y0 as learnable parameter
y0 = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32, device=device))

# network: input = [t, x_1, ..., x_d] -> output Z (d)
z_net = ZNet(input_dim=1 + d, out_dim=d, hidden=256).to(device)

optimizer = optim.Adam(list(z_net.parameters()) + [y0], lr=lr)
mse_loss = nn.MSELoss()

# ------------- payoff (basket call) --------------
def payoff(S_T):
    # S_T: (batch, d)
    basket = S_T.mean(dim=1)  # average price
    pay = torch.relu(basket - K).unsqueeze(1)  # shape (batch,1)
    return pay

# ------------- GBM simulator (vectorized) -------------
def simulate_paths(batch_size):
    # returns S_paths shape (batch, N+1, d), dW shape (batch, N, d)
    # use log-Euler exact step for GBM increments
    S_paths = torch.zeros(batch_size, N+1, d, device=device)
    S_paths[:,0,:] = S0
    dW = torch.randn(batch_size, N, d, device=device) * math.sqrt(dt)  # normal increments
    for i in range(N):
        # Euler-Maruyama on log price: S_{t+dt} = S_t * exp( (r - 0.5 sigma^2) dt + sigma * dW )
        exp_term = (r - 0.5 * sigma**2) * dt + sigma * dW[:,i,:]
        S_paths[:, i+1, :] = S_paths[:, i, :] * torch.exp(exp_term)
    return S_paths, dW

# ------------- training loop -------------
print("Starting training...")
print(f"Device: {device}")
print(f"Problem: {d}-dim basket call, T={T}, N={N} steps")
print(f"Batch size: {batch_size}, Learning rate: {lr}")
print("="*50)

for epoch in range(1, num_epochs+1):
    optimizer.zero_grad()
    # sample batch
    S_paths, dW = simulate_paths(batch_size)
    Y = y0.expand(batch_size, 1)   # shape (batch,1)
    # forward BSDE march
    for i in range(N):
        t_i = torch.full((batch_size,1), float(i)*dt, device=device)
        X_i = S_paths[:, i, :]   # (batch, d)
        net_input = torch.cat([t_i, X_i], dim=1)  # (batch, 1+d)
        Z_i = z_net(net_input)   # (batch, d)
        # driver for Black-Scholes pricing: f = -r * Y  (since PDE: u_t + L u - r u = 0 => f = -r u)
        f_val = - r * Y          # (batch,1)
        # dW increment
        dW_i = dW[:, i, :]       # (batch,d)
        # inner product Z_i * dW_i  -> (batch,1)
        ZdW = (Z_i * dW_i).sum(dim=1, keepdim=True)
        # Euler stepping for Y: Y_{i+1} = Y_i - f dt + Z dW
        Y = Y - f_val * dt + ZdW
    # terminal: compare with payoff at S_T
    S_T = S_paths[:, -1, :]
    G = payoff(S_T)   # (batch,1)
    loss = mse_loss(Y, G)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0 or epoch==1:
        print(f"Epoch {epoch:4d}  loss={loss.item():.6e}  y0={y0.item():.6f}")

# after training y0.item() is the approximated price at t=0
approx_price = y0.item()
print("Approximated price (Y0):", approx_price)

# ------------- Monte Carlo for verification (risk-neutral analytical style) -------------
def mc_price_multiprocess(nmc=200000):
    # pure Monte Carlo price of basket call (using lognormal exact sampling)
    Z = np.random.randn(nmc, d) * np.sqrt(T)
    S_T_np = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * Z)
    basket = S_T_np.mean(axis=1)
    pay = np.maximum(basket - K, 0.0)
    price = np.exp(-r*T) * pay.mean()
    return price

# compute quick MC (smaller sample) for comparison
mc_est = mc_price_multiprocess(nmc=20000)
print("Monte Carlo (20k) estimate (discounted):", mc_est)