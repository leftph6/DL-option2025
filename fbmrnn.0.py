"""
deepbsde_rnn_callable.py

Experimental implementation:
 - Train RNN (LSTM) based Deep-BSDE on supplied price paths (FBM or GBM).
 - After training, compute pathwise Y_t and Z_t by forward propagation.
 - Use model's Y_t as continuation estimator for issuer optimal stopping decisions on callable bond.
 - Output callable bond price (MC averaged), diagnostics and plots.

CSV format expected:
 - Each row = 1 simulated path
 - Columns = time steps from t0..tN (no header) OR optional header of times
 - If your CSV has 'datetime' or index, adapt load_paths() accordingly

Usage:
  python deepbsde_rnn_callable.py --paths your_paths.csv
"""

import argparse
import math
import os
import time
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------
# Configurable defaults
# ---------------------------
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------
# Utilities: load paths
# ---------------------------
def load_paths(csv_path: str, expect_header: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load price paths from CSV.
    - If file has numeric header row (time grid), will parse times
    - Else times are assumed uniform in [0,1] (you can rescale)
    Returns:
      S_paths: ndarray shape (n_paths, n_steps)
      times: ndarray shape (n_steps,)
    """
    df = pd.read_csv(csv_path, header=0 if expect_header else None)
    arr = df.values.astype(float)
    n_paths, n_steps = arr.shape
    # try to detect header times if expect_header True
    if expect_header:
        try:
            times = np.array([float(c) for c in df.columns])
        except:
            times = np.linspace(0.0, 1.0, n_steps)
    else:
        times = np.linspace(0.0, 1.0, n_steps)
    return arr, times

# ---------------------------
# RNN model: outputs Z_t sequence
# ---------------------------
class RNN_Z_net(nn.Module):
    """
    LSTM-based network that takes full price sequence (or returns) and outputs Z for each time step.
    Input: (batch, T, input_dim=1)  -> outputs (batch, T, 1)
    """
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=False)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, T, 1)
        out, _ = self.lstm(x)       # out: (batch, T, hidden_dim)
        z = self.head(out)          # (batch, T, 1)
        return z

# ---------------------------
# Training / forward-discretization utilities
# ---------------------------
def compute_dS_and_dt(paths: np.ndarray, times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    paths: (n_paths, n_steps)
    returns:
      dS: (n_paths, n_steps-1)
      dt: scalar (assume uniform grid) or array (n_steps-1)
    """
    dS = paths[:, 1:] - paths[:, :-1]
    dt_arr = np.diff(times)
    return dS, dt_arr

def forward_bsde_from_Z(y0_param: torch.nn.Parameter, Z_seq: torch.Tensor,
                        S_batch: torch.Tensor, dS_batch: torch.Tensor,
                        r: float, dt_arr: np.ndarray, device: torch.device):
    """
    Discrete forward BSDE using given Z sequence (empirical scheme).
    - y0_param: nn.Parameter scalar
    - Z_seq: (batch, T-1, 1) predicted by RNN (we use first T-1 outputs since we have dS of length T-1)
    - S_batch: (batch, T)  torch tensor
    - dS_batch: (batch, T-1) torch tensor
    - returns: Y_T_pred (batch, 1) and Y_seq (batch, T) list of Y at each time (including Y0)
    Discretization:
       Y_{i+1} = Y_i - f(Y_i) * dt + Z_i * dS_i
    Here f = - r * Y (Black-Scholes style). Note sign choices consistent with earlier code.
    """
    batch = S_batch.shape[0]
    T_full = S_batch.shape[1]  # number of time nodes
    T_steps = T_full - 1
    Y = y0_param.expand(batch, 1).to(device)  # (batch,1)
    Y_seq = [Y.clone()]
    # dt_arr can be array length T_steps
    if isinstance(dt_arr, np.ndarray):
        dt_torch = torch.tensor(dt_arr, dtype=Y.dtype, device=device)
    else:
        dt_torch = torch.tensor([dt_arr]*T_steps, dtype=Y.dtype, device=device)
    for i in range(T_steps):
        Z_i = Z_seq[:, i, :]         # (batch,1)
        # driver f = - r * Y -> so Y update: Y_{i+1} = Y_i - f*dt + Z dS = Y_i + r*Y_i*dt + Z dS
        f_val = - r * Y
        dS_i = dS_batch[:, i].unsqueeze(1)  # (batch,1)
        Y = Y - f_val * dt_torch[i] + Z_i * dS_i
        Y_seq.append(Y.clone())
    Y_T = Y
    Y_seq_tensor = torch.cat(Y_seq, dim=1)  # (batch, T_full)
    return Y_T, Y_seq_tensor

# ---------------------------
# Payoff for callable bond (we train on final payoff first)
# ---------------------------
def european_call_payoff(S_T: np.ndarray, strike: float) -> np.ndarray:
    return np.maximum(S_T - strike, 0.0)

# ---------------------------
# Main training loop
# ---------------------------
def train_rnn_bsde(paths: np.ndarray, times: np.ndarray,
                   device: str = DEFAULT_DEVICE,
                   hidden_dim: int = 64, num_layers: int = 2,
                   lr: float = 1e-3, batch_size: int = 256,
                   epochs: int = 200, r: float = 0.03, strike: float = 100.0,
                   save_path: str = "rnn_bsde_model.pt", verbose: bool = True):
    """
    Train RNN-based Deep-BSDE on provided paths (paths shape: n_paths x n_steps).
    Returns trained model and y0 param.
    """
    device = torch.device(device)
    n_paths, n_steps = paths.shape
    S0 = float(paths[:,0].mean())
    dS, dt_arr = compute_dS_and_dt(paths, times)   # dS shape (n_paths, n_steps-1)
    # normalize inputs: use returns or scaled prices to stabilize training
    # Use log returns as input features to RNN
    eps = 1e-9
    log_returns = np.log(paths[:,1:] + eps) - np.log(paths[:,:-1] + eps)  # shape (n_paths, n_steps-1)
    # For RNN input, we provide full sequence length = n_steps (we prepend 0 at t0)
    log_returns_full = np.concatenate([np.zeros((n_paths,1)), log_returns], axis=1)  # (n_paths, n_steps)
    # convert to torch
    dataset_S = torch.tensor(paths, dtype=torch.float32, device=device)
    dataset_dS = torch.tensor(dS, dtype=torch.float32, device=device)
    dataset_lr = torch.tensor(log_returns_full, dtype=torch.float32, device=device).unsqueeze(2)  # (n_paths, T, 1)

    model = RNN_Z_net(input_dim=1, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    y0 = nn.Parameter(torch.tensor([0.0], dtype=torch.float32, device=device))
    optimizer = optim.Adam(list(model.parameters()) + [y0], lr=lr)
    loss_fn = nn.MSELoss()

    n_batches = max(1, n_paths // batch_size)
    indices = np.arange(n_paths)

    print("Training RNN-BSDE on device:", device)
    for epoch in range(1, epochs+1):
        np.random.shuffle(indices)
        epoch_loss = 0.0
        for b in range(n_batches):
            idx = indices[b*batch_size:(b+1)*batch_size]
            S_batch = dataset_S[idx]            # (batch, T)
            dS_batch = dataset_dS[idx]          # (batch, T-1)
            lr_batch = dataset_lr[idx]          # (batch, T, 1)
            optimizer.zero_grad()
            Z_pred = model(lr_batch)            # (batch, T, 1)
            # We only have dS for first T-1 steps, so use Z_pred[:, :T-1, :]
            Z_use = Z_pred[:, : (n_steps-1), :] # (batch, T-1, 1)
            Y_T_pred, _ = forward_bsde_from_Z(y0, Z_use, S_batch, dS_batch, r, dt_arr, device)
            # compute target payoff (discounting optional - we use undiscounted payoff at T because BSDE forward accounted for drift)
            payoff = torch.tensor(european_call_payoff(S_batch[:,-1].cpu().numpy(), strike), dtype=torch.float32, device=device).unsqueeze(1)
            loss = loss_fn(Y_T_pred, payoff)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + [y0], max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item() * S_batch.size(0)
        epoch_loss /= n_paths
        if verbose and (epoch % 10 == 0 or epoch == 1 or epoch == epochs):
            print(f"Epoch {epoch}/{epochs}  loss={epoch_loss:.6e}  y0={y0.item():.6f}")
    # save model
    torch.save({
        "model_state_dict": model.state_dict(),
        "y0": y0.detach().cpu().numpy()
    }, save_path)
    print("Saved trained model to", save_path)
    return model, y0.detach().cpu().numpy(), (dataset_S, dataset_dS, dataset_lr)

# ---------------------------
# Use trained model to compute pathwise Y_t and Z_t for all paths
# ---------------------------
def compute_pathwise_YZ(model: nn.Module, y0_val: float,
                        paths: np.ndarray, times: np.ndarray,
                        device: str = DEFAULT_DEVICE):
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    n_paths, n_steps = paths.shape
    eps = 1e-9
    # compute log returns
    log_returns = np.log(paths[:,1:] + eps) - np.log(paths[:,:-1] + eps)
    lr_full = np.concatenate([np.zeros((n_paths,1)), log_returns], axis=1)
    with torch.no_grad():
        lr_th = torch.tensor(lr_full, dtype=torch.float32, device=device).unsqueeze(2)  # (n_paths, T, 1)
        Z_all = model(lr_th)  # (n_paths, T, 1)
        dS = torch.tensor(paths[:,1:] - paths[:,:-1], dtype=torch.float32, device=device)
        # use only first T-1 Z
        Z_use = Z_all[:, : (n_steps-1), :]   # (n_paths, T-1, 1)
        y0_param = torch.tensor([float(y0_val)], dtype=torch.float32, device=device)
        Y_T_pred, Y_seq = forward_bsde_from_Z(y0_param, Z_use, torch.tensor(paths, dtype=torch.float32, device=device),
                                             dS, r=0.03, dt_arr=np.diff(times), device=device)
    Z_all_np = Z_all.cpu().numpy().squeeze(2)   # (n_paths, T)
    Y_seq_np = Y_seq.cpu().numpy()             # (n_paths, T)
    return Y_seq_np, Z_all_np

# ---------------------------
# Callable bond pricing using model's pathwise Y as continuation value
# ---------------------------
def price_callable_from_pathwise(Y_seq: np.ndarray, paths: np.ndarray, times: np.ndarray,
                                coupon_times: Optional[List[float]] = None,
                                coupon_amt: float = 0.0,
                                call_times: Optional[List[float]] = None,
                                call_prices: Optional[List[float]] = None,
                                r: float = 0.03,
                                call_after_coupon: bool = True):
    """
    Use model-predicted Y_seq (n_paths, T) as continuation estimator at each time.
    For each call date, issuer calls if continuation_value > call_price.
    Returns PV estimate (mean over paths) and details.
    """
    n_paths, T = Y_seq.shape
    # map requested call/coupon times to indices in times
    idx_map = lambda tval: int(np.argmin(np.abs(times - tval)))
    if coupon_times is None:
        coupon_times = []
    coupon_indices = [idx_map(t) for t in coupon_times]
    if call_times is None:
        call_times = []
        call_prices = []
    call_indices = [idx_map(t) for t in call_times]
    # construct cashflow matrix
    cashflows = np.zeros((n_paths, T))
    for ci in coupon_indices:
        cashflows[:, ci] += coupon_amt
    cashflows[:, -1] += 100.0   # face value add (you can parameterize)
    alive = np.ones(n_paths, dtype=bool)

    details = {"calls": []}
    # iterate call dates in chronological order (issuer decides at that time)
    for k_idx, call_idx in enumerate(call_indices):
        call_price = call_prices[k_idx]
        # continuation value: model's Y at that time (we assume Y corresponds to undiscounted value aligned to path)
        cont_vals = Y_seq[:, call_idx]   # shape (n_paths,)
        # If call_after_coupon True, coupon at call_idx is already in cashflows; continuation should reflect *excluding* current coupon (we used Y_seq which includes effect of future CFs and past).
        # Heuristic: we compare cont_vals > call_price
        call_mask = (cont_vals > call_price) & alive
        if call_mask.sum() > 0:
            # issuing calls: pay call_price at call_idx (in addition to coupon if call_after_coupon True)
            cashflows[call_mask, call_idx] += call_price
            # zero future cashflows beyond call_idx
            if call_idx + 1 < T:
                cashflows[call_mask, call_idx+1:] = 0.0
            alive[call_mask] = False
        details["calls"].append({
            "time": times[call_idx],
            "call_idx": call_idx,
            "n_called": int(call_mask.sum())
        })
    # discount cashflows to t=0
    # assume times are in years or normalized to [0,1]
    disc = np.exp(-r * times)   # shape (T,)
    pv_paths = (cashflows * disc[None, :]).sum(axis=1)
    price = pv_paths.mean()
    stderr = pv_paths.std(ddof=1) / math.sqrt(n_paths)
    return price, stderr, {"pv_paths": pv_paths, "cashflows": cashflows, "details": details}

# ---------------------------
# Plot helpers
# ---------------------------
def plot_example_paths(paths: np.ndarray, times: np.ndarray, n_show: int = 50, outname: Optional[str] = None):
    plt.figure(figsize=(10,5))
    n_paths = paths.shape[0]
    idx = np.random.choice(n_paths, min(n_show, n_paths), replace=False)
    for i in idx:
        plt.plot(times, paths[i], linewidth=0.8, alpha=0.6)
    plt.plot(times, paths.mean(axis=0), color="red", linewidth=2.0, label="mean")
    plt.axhline(paths[:,0].mean(), color="green", linestyle="--", label=f"S0={paths[:,0].mean():.1f}")
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("price")
    plt.title(f"Sample price paths (n={len(idx)})")
    if outname:
        plt.savefig(outname, dpi=150, bbox_inches="tight")
    plt.show()

def plot_YZ_sample(Y_seq: np.ndarray, Z_seq: np.ndarray, times: np.ndarray, n_show: int = 6):
    # plot some sample Y and Z
    n_paths = Y_seq.shape[0]
    idx = np.random.choice(n_paths, min(n_show, n_paths), replace=False)
    plt.figure(figsize=(12,5))
    for i in idx:
        plt.plot(times, Y_seq[i], alpha=0.7)
    plt.title("Sample Y_t (model predicted) for few paths")
    plt.xlabel("time")
    plt.ylabel("Y_t")
    plt.show()

    plt.figure(figsize=(12,5))
    for i in idx:
        plt.plot(times, Z_seq[i], alpha=0.7)
    plt.title("Sample Z_t (model predicted) for few paths")
    plt.xlabel("time")
    plt.ylabel("Z_t")
    plt.show()

# ---------------------------
# CLI / main
# ---------------------------
def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=str, required=True, help="CSV file of simulated paths (rows=paths, cols=time steps)")
    parser.add_argument("--expect_header", action="store_true", help="CSV has header row with time grid")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--strike", type=float, default=100.0)
    parser.add_argument("--r", type=float, default=0.03)
    parser.add_argument("--save", type=str, default="rnn_bsde_model.pt")
    parser.add_argument("--plot_examples", action="store_true")
    parser.add_argument("--call_times", type=float, nargs="*", default=[0.5, 1.0], help="call times in same units as times grid")
    parser.add_argument("--call_prices", type=float, nargs="*", default=[102.0, 100.0])
    parser.add_argument("--coupon_times", type=float, nargs="*", default=[], help="coupon times")
    parser.add_argument("--coupon_amt", type=float, default=0.0)
    args = parser.parse_args()

    print("Loading paths from", args.paths)
    paths, times = load_paths(args.paths, expect_header=args.expect_header)
    print("paths shape:", paths.shape, "time grid len:", len(times))
    if args.plot_examples:
        plot_example_paths(paths, times, n_show=50, outname="example_paths.png")

    # train
    model, y0_val, _ = train_rnn_bsde(paths, times,
                                     device=args.device,
                                     hidden_dim=args.hidden_dim,
                                     num_layers=args.num_layers,
                                     lr=args.lr,
                                     batch_size=args.batch_size,
                                     epochs=args.epochs,
                                     r=args.r,
                                     strike=args.strike,
                                     save_path=args.save,
                                     verbose=True)
    # compute pathwise Y and Z
    Y_seq, Z_seq = compute_pathwise_YZ(model, float(y0_val), paths, times, device=args.device)
    # diagnostics
    plot_YZ_sample(Y_seq, Z_seq, times, n_show=6)

    # price callable via model's continuation Y_seq
    price, stderr, info = price_callable_from_pathwise(Y_seq, paths, times,
                                                       coupon_times=args.coupon_times,
                                                       coupon_amt=args.coupon_amt,
                                                       call_times=args.call_times,
                                                       call_prices=args.call_prices,
                                                       r=args.r,
                                                       call_after_coupon=True)
    print("Callable bond price estimate:", price, "stderr:", stderr)
    # save final PV histogram
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,4))
    plt.hist(info["pv_paths"], bins=50)
    plt.title("Distribution of pathwise PV under model decisions")
    plt.xlabel("PV")
    plt.ylabel("frequency")
    plt.savefig("pv_hist.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main_cli()
