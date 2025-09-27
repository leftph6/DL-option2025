# backtest_deepbsde.py
# Backtest Deep-BSDE replication strategy on historical price series and plot PnL curve.
# Assumes model.Znets[t] (or a time-parametrized net) outputs hedge quantity (units of underlying).
# Save/Load model sample usage included.

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict

# ---------------------------
# Config / hyperparams
# ---------------------------
MODEL_WEIGHTS_PATH = "deepbsde_model.pt"   # 你的训练后模型权重 (示例)
PRICE_CSV = "history_prices.csv"           # 输入 price CSV 路径
PRICE_COL = "price"                        # CSV 中价格列名（若无，代码会 guess）
DTYPE = torch.float32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

R = 0.00              # 无风险利率，用于现金账户计息（用年化小数表示，例如0.03）
TRANSACTION_COST_RATE = 0.0  # 交易费率（按成交额比例），eg 0.001 = 0.1%
TAKE_SHORT = True     # 我们模拟做空期权（True）或做多（False）
INSTRUMENT_MULTIPLIER = 1.0  # 每张期权对应的标的份数（通常1）
PAYOFF_TYPE = "call"  # 期权类型示例，"call" or "put" or "custom"
STRIKE = 100.0
MATURITY_TIME = None  # 若历史数据包括到期时间可设置为具体 datetime/string，否则若 None 将以 CSV 最后一时刻为到期

# ---------------------------
# 1) 需要和你的模型结构匹配的 wrapper
#    把你的模型封装成下面的 API：
#      - model.load_state_dict(torch.load(...))
#      - model.eval()
#      - model.get_Y0() -> float
#      - model.compute_Z(t_index, price) -> hedge_qty (float per path)
#
# 这里包含一个示例 Minimal Model wrapper（你应替换为你训练时的实际类）
# ---------------------------
class ExampleDeepBSDEWrapper:
    """
    示例 wrapper：这个类**必须**被替换为你实际训练模型的类或加载方式。
    要求：
      - __init__(self, device)
      - load_state(path)
      - get_Y0() -> float (initial option premium from model)
      - compute_Z(t_idx, price) -> hedge quantity (float)  # t_idx is index in times array
    """

    def __init__(self, device=DEVICE):
        # 这里用一个 toy net 作为占位；真实使用时请换成你的模型和网络结构
        self.device = device
        # example: single MLP that takes [t, price_norm] and outputs hedge quantity
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(device)
        # example initial premium (若模型里已包含 y0 参数则 load_state 会覆盖)
        self.y0 = 5.0

    def load_state(self, path: str):
        # 载入示例权重（如果路径存在）
        try:
            ck = torch.load(path, map_location=self.device)
            if "state_dict" in ck:
                self.net.load_state_dict(ck["state_dict"])
            else:
                self.net.load_state_dict(ck)
            if "y0" in ck:
                self.y0 = float(ck["y0"])
            print("Model weights loaded from", path)
        except Exception as e:
            print("Warning: could not load weights:", e)
            print("Using random-initialized example network (replace wrapper with your model).")

    def get_Y0(self) -> float:
        return float(self.y0)

    def compute_Z(self, t_idx: int, t_val: float, price: float) -> float:
        # price normalization (simple)
        inp = torch.tensor([[t_val, price / max(price, 1.0)]], dtype=DTYPE, device=self.device)
        with torch.no_grad():
            z = self.net(inp).cpu().numpy().ravel()[0]
        # interpret z as number-of-shares to hold
        return float(z)

# ---------------------------
# 2) 读取历史价格 CSV
# ---------------------------
def load_price_series(csv_path: str, price_col_hint: str = "price") -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    # find price column
    if price_col_hint in df.columns:
        prices = df[price_col_hint].values.astype(float)
    else:
        # try common names
        for c in ["close", "Close", "price", "Price", "adj_close", "Adj Close"]:
            if c in df.columns:
                prices = df[c].values.astype(float)
                break
        else:
            # fallback: use first numeric column
            num_cols = df.select_dtypes(include=[np.number]).columns
            if len(num_cols) == 0:
                raise ValueError("No numeric column found in CSV for price.")
            prices = df[num_cols[0]].values.astype(float)
    # times
    if "datetime" in df.columns:
        times = pd.to_datetime(df["datetime"]).values
    elif "time" in df.columns:
        times = pd.to_datetime(df["time"]).values
    else:
        # create index-based times (0..N-1), we interpret them as equally spaced units (e.g., days)
        times = np.arange(len(prices))
    return prices, times

# ---------------------------
# 3) 回测引擎（离散再平衡）
# ---------------------------
def backtest_replication(
    prices: np.ndarray,
    times: np.ndarray,
    model_wrapper: ExampleDeepBSDEWrapper,
    r: float = R,
    tx_cost_rate: float = TRANSACTION_COST_RATE,
    take_short: bool = TAKE_SHORT,
    instrument_multiplier: float = INSTRUMENT_MULTIPLIER,
    strike: float = STRIKE,
    payoff_type: str = PAYOFF_TYPE,
    maturity_time = MATURITY_TIME
) -> Dict:
    """
    prices: ndarray shape (T,)
    times: ndarray shape (T,) (can be np.datetime64 or numeric)
    model_wrapper: instance that provides get_Y0() and compute_Z(...)
    """
    n = len(prices)
    # time grid deltas for interest accrual (in fraction of year) -> here we assume times either numeric or datetime
    # if datetime, compute year fractions:
    if np.issubdtype(times.dtype, np.datetime64):
        # convert to pandas datetime index for convenience
        pd_times = pd.to_datetime(times)
        # compute year fraction between steps using days / 365
        deltas = pd_times.to_series().diff().dt.total_seconds().fillna(0).values / (365.0*24*3600)
    else:
        # numeric grid: assume unit corresponds to year fraction 1 / freq? default assume uniform step of 1 unit -> treat as days?
        # For safety assume times numeric and normalized to years (user can pass times in years)
        deltas = np.diff(times, prepend=times[0])
        # if first delta is zero (prepend), set it same as second
        if deltas[0] == 0 and n > 1:
            deltas[0] = deltas[1]

    # bookkeeping arrays
    cash = np.zeros(n)       # cash account after rebalancing at time i (immediately after trades)
    hedge_pos = np.zeros(n)  # number of underlying shares held after rebalancing at time i
    portfolio_val = np.zeros(n)  # cash + hedge_pos * price_i
    option_liability = np.zeros(n)  # mark-to-market of option liability if desired (we use final payoff)
    pnl = np.zeros(n)

    # initial sell of 1 option contract
    Y0 = model_wrapper.get_Y0()
    # if we are short the option we receive premium
    if take_short:
        cash0 = Y0 * instrument_multiplier
        short_sign = -1.0
    else:
        cash0 = -Y0 * instrument_multiplier
        short_sign = 1.0

    # initial hedge at time 0
    t0 = 0
    price0 = float(prices[0])
    z0 = model_wrapper.compute_Z(t_idx=0, t_val=0.0, price=price0)  # desired hedge (number of shares)
    # pay for initial purchase of hedge: cost = z0 * price0 * multiplier
    trade0 = z0 * instrument_multiplier
    trade_cost0 = abs(trade0) * price0 * tx_cost_rate
    cash0 = cash0 - trade0 * price0 - trade_cost0
    cash[0] = cash0
    hedge_pos[0] = trade0
    portfolio_val[0] = cash[0] + hedge_pos[0] * price0
    # no PnL yet; we can set pnl[0] = portfolio_val[0] - (-short_sign * 0?)  but define final PnL = portfolio - option_settlement
    pnl[0] = portfolio_val[0]  # initial mark (premium minus initial hedge cost)

    # iterate through times 1..n-1, at each time:
    #  - cash accrues interest over delta
    #  - portfolio mark-to-market before trade
    #  - compute desired new hedge z(t)
    #  - trade delta_h = z_new - z_old, pay trade cost and update cash and hedge_pos
    for i in range(1, n):
        dt = float(deltas[i]) if deltas[i] > 0 else 1e-6
        # accrue interest on cash
        cash_prev = cash[i-1]
        cash_accrued = cash_prev * (1.0 + r * dt)  # simple interest for dt fraction of year
        # mark before trade
        price_i = float(prices[i])
        # get previous hedge pos
        prev_pos = hedge_pos[i-1]
        preportfolio = cash_accrued + prev_pos * price_i
        # compute new desired hedge from model (model expects t index and price; we approximate t_val as fraction of total time)
        # t_val normalization: use i/(n-1) as [0,1] scaled time
        t_val = float(i) / max(1, n-1)
        z_new = model_wrapper.compute_Z(t_idx=i, t_val=t_val, price=price_i)
        desired_pos = z_new * instrument_multiplier
        # trade
        trade = desired_pos - prev_pos
        trade_cost = abs(trade) * price_i * tx_cost_rate
        cash_after_trade = cash_accrued - trade * price_i - trade_cost
        # record
        cash[i] = cash_after_trade
        hedge_pos[i] = desired_pos
        portfolio_val[i] = cash_after_trade + hedge_pos[i] * price_i
        # Option liability: at maturity we will settle payoff; for interim we can mark to model price (optional)
        # Here we don't mark liability; final PnL computed after maturity settlement
        pnl[i] = portfolio_val[i]  # interim mark

    # finally, settle option at maturity (assumed at last timestamp unless maturity_time provided)
    # compute option payoff at last price
    ST = float(prices[-1])
    if payoff_type == "call":
        payoff = max(ST - strike, 0.0)
    elif payoff_type == "put":
        payoff = max(strike - ST, 0.0)
    else:
        # custom: user can compute externally
        payoff = 0.0

    # as the replicating portfolio holder (we hold hedge_pos[-1] shares and cash[-1]), and we are short the option (we must pay payoff)
    # final PnL (for the strategy that started by selling 1 option and delta-hedging):
    final_portfolio = cash[-1] + hedge_pos[-1] * ST
    if take_short:
        final_pnl = final_portfolio - payoff * instrument_multiplier  # we must pay payoff to option buyer
    else:
        final_pnl = final_portfolio + payoff * instrument_multiplier  # if we were long option
    # build timeline of realized PnL: mark-to-market minus outstanding liability if you want running PnL -> here we use
    # running_pnl = portfolio_val - expected liability (we didn't compute expected liability each time). So we'll produce:
    #  - mark_curve: portfolio_val
    #  - final_pnl scalar
    # compute cumulative PnL series by subtracting initial mark (so starts at 0)
    mark_curve = portfolio_val
    mark_curve0 = mark_curve[0]
    cum_pnl = mark_curve - mark_curve0
    # store results
    results = {
        "times": times,
        "prices": prices,
        "cash": cash,
        "hedge_pos": hedge_pos,
        "portfolio_val": portfolio_val,
        "cum_pnl": cum_pnl,
        "final_pnl": final_pnl,
        "final_portfolio": final_portfolio,
        "option_payoff": payoff,
        "Y0": Y0
    }
    return results

# ---------------------------
# 4) Plot utility
# ---------------------------
def plot_pnl(results: Dict, out_png: Optional[str] = None):
    times = results["times"]
    if np.issubdtype(times.dtype, np.datetime64):
        x = pd.to_datetime(times)
    else:
        x = np.arange(len(times))
    plt.figure(figsize=(10,5))
    plt.plot(x, results["cum_pnl"], label="Cumulative PnL (mark-to-market - initial)")
    plt.plot(x, results["portfolio_val"] - results["portfolio_val"][0], alpha=0.4, label="Portfolio (delta-hedge) change")
    plt.xlabel("Time")
    plt.ylabel("PnL / Portfolio change")
    plt.title(f"Replication cumulative PnL, final PnL={results['final_pnl']:.4f}")
    plt.legend()
    plt.grid(True)
    if out_png:
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------
# 5) Main: load model, load csv, run backtest, plot
# ---------------------------
def main():
    # load price data
    prices, times = load_price_series(PRICE_CSV, PRICE_COL)
    print("Loaded price series:", len(prices), "points")

    # load model wrapper and weights
    model = ExampleDeepBSDEWrapper(device=DEVICE)
    model.load_state(MODEL_WEIGHTS_PATH)

    results = backtest_replication(
        prices=prices,
        times=times,
        model_wrapper=model,
        r=R,
        tx_cost_rate=TRANSACTION_COST_RATE,
        take_short=TAKE_SHORT,
        instrument_multiplier=INSTRUMENT_MULTIPLIER,
        strike=STRIKE,
        payoff_type=PAYOFF_TYPE
    )

    print("Final PnL:", results["final_pnl"])
    plot_pnl(results, out_png="pnl_curve.png")

if __name__ == "__main__":
    main()