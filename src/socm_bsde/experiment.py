import argparse
import json
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.optim as optim

from .black_scholes import call_delta, call_price
from .nets import ControlNet, DeepBSDEModel, grad_norm
from .path_generators import GBMGenerator
from .utils import ensure_dir, get_device, set_seed


@dataclass
class ExperimentConfig:
    epochs: int = 200
    batch_size: int = 1024
    lr: float = 1e-3
    hidden_size: int = 64
    depth: int = 2
    seed: int = 42
    device: str = ""
    output_dir: str = "outputs_experiment"

    T: float = 1.0
    N: int = 50
    d: int = 1
    r: float = 0.05
    sigma: float = 0.2
    S0: float = 100.0
    strike: float = 100.0

    grad_batches: int = 20
    eval_interval: int = 10
    eval_batch_size: int = 512
    plot: bool = True


def socm_loss(policy: ControlNet, S: torch.Tensor, times: torch.Tensor, sigma: float, r: float, strike: float) -> torch.Tensor:
    batch_size, steps, d = S.shape
    S_flat = S[:, :-1, :].reshape(-1, d)
    t_flat = times[:-1].view(1, -1).repeat(batch_size, 1).reshape(-1, 1)

    Z_pred = policy(S_flat, t_flat)
    delta = call_delta(S_flat, t_flat, strike, times[-1].item(), r, sigma)
    Z_target = sigma * S_flat * delta
    return torch.mean((Z_pred - Z_target) ** 2)


def bsde_loss(model: DeepBSDEModel, S: torch.Tensor, dW: torch.Tensor, times: torch.Tensor, r: float, strike: float) -> torch.Tensor:
    Y_T = model(S, dW, times, r)
    payoff = torch.clamp(S[:, -1, :] - strike, min=0.0)
    return torch.mean((Y_T - payoff) ** 2)


def train_socm_model(config: ExperimentConfig) -> Tuple[ControlNet, Dict[str, List[float]]]:
    device = get_device(config.device or None)
    generator = GBMGenerator(
        T=config.T,
        N=config.N,
        d=config.d,
        r=config.r,
        sigma=config.sigma,
        S0=config.S0,
        device=device,
    )
    policy = ControlNet(d=config.d, hidden_size=config.hidden_size, depth=config.depth).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=config.lr)

    history = {"loss": [], "grad_norm": [], "y0_est": [], "price_error": [], "eval_epoch": []}
    for epoch in range(1, config.epochs + 1):
        with torch.no_grad():
            S, _, times = generator.generate_paths(config.batch_size)
        loss = socm_loss(policy, S, times, config.sigma, config.r, config.strike)
        optimizer.zero_grad()
        loss.backward()
        gnorm = grad_norm(policy.parameters()).item()
        optimizer.step()

        history["loss"].append(float(loss.item()))
        history["grad_norm"].append(float(gnorm))

        if config.eval_interval > 0 and epoch % config.eval_interval == 0:
            with torch.no_grad():
                S_eval, dW_eval, times_eval = generator.generate_paths(config.eval_batch_size)
                y0_est = estimate_y0_from_policy(policy, S_eval, dW_eval, times_eval, config.r, config.strike)
                bs_price = float(call_price(torch.tensor(config.S0, device=device), config.strike, config.T, config.r, config.sigma).item())
                history["y0_est"].append(y0_est)
                history["price_error"].append(abs(y0_est - bs_price))
                history["eval_epoch"].append(epoch)
    return policy, history


def train_bsde_model(config: ExperimentConfig) -> Tuple[DeepBSDEModel, Dict[str, List[float]]]:
    device = get_device(config.device or None)
    generator = GBMGenerator(
        T=config.T,
        N=config.N,
        d=config.d,
        r=config.r,
        sigma=config.sigma,
        S0=config.S0,
        device=device,
    )
    model = DeepBSDEModel(d=config.d, hidden_size=config.hidden_size, depth=config.depth).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    history = {"loss": [], "grad_norm": [], "Y0": [], "price_error": [], "eval_epoch": []}
    for epoch in range(1, config.epochs + 1):
        with torch.no_grad():
            S, dW, times = generator.generate_paths(config.batch_size)
        loss = bsde_loss(model, S, dW, times, config.r, config.strike)
        optimizer.zero_grad()
        loss.backward()
        gnorm = grad_norm(model.parameters()).item()
        optimizer.step()

        history["loss"].append(float(loss.item()))
        history["grad_norm"].append(float(gnorm))
        history["Y0"].append(float(model.Y0.item()))
        if config.eval_interval > 0 and epoch % config.eval_interval == 0:
            bs_price = float(call_price(torch.tensor(config.S0, device=device), config.strike, config.T, config.r, config.sigma).item())
            history["price_error"].append(abs(float(model.Y0.item()) - bs_price))
            history["eval_epoch"].append(epoch)
    return model, history


def estimate_y0_from_policy(
    policy: ControlNet,
    S: torch.Tensor,
    dW: torch.Tensor,
    times: torch.Tensor,
    r: float,
    strike: float,
) -> float:
    dt = times[1] - times[0]
    decay = (1 - r * dt)
    N = dW.shape[1]
    weights = torch.pow(decay, torch.arange(N - 1, -1, -1, device=S.device))

    Z_list = []
    for i in range(N):
        t = times[i]
        Z_list.append(policy(S[:, i, :], t))
    Z = torch.stack(Z_list, dim=1)

    incr = (Z * dW).sum(dim=2)  # (batch, N)
    weighted = incr * weights
    B = weighted.sum(dim=1, keepdim=True)

    payoff = torch.clamp(S[:, -1, :] - strike, min=0.0)
    y0 = (payoff - B).mean() / torch.pow(decay, N)
    return float(y0.item())


def grad_variance_socm(policy: ControlNet, generator: GBMGenerator, config: ExperimentConfig) -> float:
    norms = []
    for _ in range(config.grad_batches):
        with torch.no_grad():
            S, _, times = generator.generate_paths(config.batch_size)
        loss = socm_loss(policy, S, times, config.sigma, config.r, config.strike)
        policy.zero_grad()
        loss.backward()
        norms.append(grad_norm(policy.parameters()).item())
    return float(np.var(norms))


def grad_variance_bsde(model: DeepBSDEModel, generator: GBMGenerator, config: ExperimentConfig) -> float:
    norms = []
    for _ in range(config.grad_batches):
        with torch.no_grad():
            S, dW, times = generator.generate_paths(config.batch_size)
        loss = bsde_loss(model, S, dW, times, config.r, config.strike)
        model.zero_grad()
        loss.backward()
        norms.append(grad_norm(model.parameters()).item())
    return float(np.var(norms))


def find_epoch_to_threshold(losses: List[float], ratio: float = 0.1) -> int:
    if not losses:
        return -1
    threshold = losses[0] * ratio
    for i, v in enumerate(losses, start=1):
        if v <= threshold:
            return i
    return -1


def summarize_series(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"start": 0.0, "end": 0.0, "min": 0.0, "argmin_epoch": -1, "mean": 0.0, "std": 0.0}
    arr = np.array(values, dtype=float)
    return {
        "start": float(arr[0]),
        "end": float(arr[-1]),
        "min": float(arr.min()),
        "argmin_epoch": int(arr.argmin() + 1),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def compute_eval_metrics(
    policy: ControlNet,
    model: DeepBSDEModel,
    generator: GBMGenerator,
    config: ExperimentConfig,
) -> Dict[str, Dict[str, float]]:
    with torch.no_grad():
        S_eval, dW_eval, times_eval = generator.generate_paths(config.eval_batch_size)
    bs_price = float(call_price(torch.tensor(config.S0, device=S_eval.device), config.strike, config.T, config.r, config.sigma).item())
    y0_socm = estimate_y0_from_policy(policy, S_eval, dW_eval, times_eval, config.r, config.strike)
    y0_bsde = float(model.Y0.item())

    socm_price_error = abs(y0_socm - bs_price)
    bsde_price_error = abs(y0_bsde - bs_price)

    # Z-field matching error for SOCM
    batch_size, steps, d = S_eval.shape
    S_flat = S_eval[:, :-1, :].reshape(-1, d)
    t_flat = times_eval[:-1].view(1, -1).repeat(batch_size, 1).reshape(-1, 1)
    Z_pred = policy(S_flat, t_flat)
    delta = call_delta(S_flat, t_flat, config.strike, times_eval[-1].item(), config.r, config.sigma)
    Z_target = config.sigma * S_flat * delta
    z_mse = float(torch.mean((Z_pred - Z_target) ** 2).item())

    return {
        "black_scholes_price": bs_price,
        "socm": {
            "price_estimate": y0_socm,
            "price_error": socm_price_error,
            "price_rel_error": socm_price_error / (abs(bs_price) + 1e-9),
            "z_mse": z_mse,
        },
        "bsde": {
            "price_estimate": y0_bsde,
            "price_error": bsde_price_error,
            "price_rel_error": bsde_price_error / (abs(bs_price) + 1e-9),
        },
    }


def generate_plots(
    output_dir: str,
    socm_hist: Dict[str, List[float]],
    bsde_hist: Dict[str, List[float]],
    summary: Optional[Dict[str, Dict[str, object]]] = None,
) -> Dict[str, str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: Dict[str, str] = {}

    # Loss curves
    plt.figure(figsize=(7, 4))
    plt.plot(socm_hist["loss"], label="SOCM loss")
    plt.plot(bsde_hist["loss"], label="BSDE loss")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.title("Training Loss")
    plt.legend()
    plt.tight_layout()
    loss_path = f"{output_dir}/loss_curves.png"
    plt.savefig(loss_path, dpi=150)
    plt.close()
    paths["loss_curves"] = loss_path

    # Grad norm curves
    plt.figure(figsize=(7, 4))
    plt.plot(socm_hist["grad_norm"], label="SOCM grad_norm")
    plt.plot(bsde_hist["grad_norm"], label="BSDE grad_norm")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Grad Norm (log scale)")
    plt.title("Gradient Norm")
    plt.legend()
    plt.tight_layout()
    grad_path = f"{output_dir}/grad_norm_curves.png"
    plt.savefig(grad_path, dpi=150)
    plt.close()
    paths["grad_norm_curves"] = grad_path

    # Price error curves (evaluated every eval_interval)
    if socm_hist.get("price_error") and bsde_hist.get("price_error"):
        plt.figure(figsize=(7, 4))
        plt.plot(socm_hist["eval_epoch"], socm_hist["price_error"], label="SOCM price error")
        plt.plot(bsde_hist["eval_epoch"], bsde_hist["price_error"], label="BSDE price error")
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Abs Price Error (log scale)")
        plt.title("Price Error Over Training")
        plt.legend()
        plt.tight_layout()
        pe_path = f"{output_dir}/price_error_curves.png"
        plt.savefig(pe_path, dpi=150)
        plt.close()
        paths["price_error_curves"] = pe_path

    # SOCM y0 estimates
    if socm_hist.get("y0_est"):
        plt.figure(figsize=(7, 4))
        plt.plot(socm_hist["eval_epoch"], socm_hist["y0_est"], label="SOCM y0 estimate")
        plt.xlabel("Epoch")
        plt.ylabel("y0 estimate")
        plt.title("SOCM Price Estimate Over Training")
        plt.legend()
        plt.tight_layout()
        y0_path = f"{output_dir}/socm_y0_curve.png"
        plt.savefig(y0_path, dpi=150)
        plt.close()
        paths["socm_y0_curve"] = y0_path

    # BSDE Y0 curve
    if bsde_hist.get("Y0"):
        plt.figure(figsize=(7, 4))
        plt.plot(bsde_hist["Y0"], label="BSDE Y0")
        plt.xlabel("Epoch")
        plt.ylabel("Y0")
        plt.title("BSDE Y0 Over Training")
        plt.legend()
        plt.tight_layout()
        y0b_path = f"{output_dir}/bsde_y0_curve.png"
        plt.savefig(y0b_path, dpi=150)
        plt.close()
        paths["bsde_y0_curve"] = y0b_path

    # Summary bar chart
    if summary:
        plt.figure(figsize=(7, 4))
        labels = ["Price Error", "Grad Variance"]
        socm_vals = [summary["socm"]["price_error"], summary["socm"]["grad_variance"]]
        bsde_vals = [summary["bsde"]["price_error"], summary["bsde"]["grad_variance"]]
        x = np.arange(len(labels))
        width = 0.35
        plt.bar(x - width / 2, socm_vals, width, label="SOCM")
        plt.bar(x + width / 2, bsde_vals, width, label="BSDE")
        plt.xticks(x, labels)
        plt.title("Key Metrics Comparison")
        plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        bar_path = f"{output_dir}/metrics_bar.png"
        plt.savefig(bar_path, dpi=150)
        plt.close()
        paths["metrics_bar"] = bar_path

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="SOCM vs Classical Deep BSDE experiment")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="outputs_experiment")
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--N", type=int, default=50)
    parser.add_argument("--d", type=int, default=1)
    parser.add_argument("--r", type=float, default=0.05)
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--S0", type=float, default=100.0)
    parser.add_argument("--strike", type=float, default=100.0)
    parser.add_argument("--grad_batches", type=int, default=20)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--no_plot", action="store_true")

    args = parser.parse_args()
    args_dict = vars(args)
    if args_dict.get("no_plot"):
        args_dict["plot"] = False
    args_dict.pop("no_plot", None)
    config = ExperimentConfig(**args_dict)

    set_seed(config.seed)
    device = get_device(config.device or None)

    generator = GBMGenerator(
        T=config.T,
        N=config.N,
        d=config.d,
        r=config.r,
        sigma=config.sigma,
        S0=config.S0,
        device=device,
    )

    policy, socm_hist = train_socm_model(config)
    model, bsde_hist = train_bsde_model(config)

    eval_metrics = compute_eval_metrics(policy, model, generator, config)

    socm_conv = find_epoch_to_threshold(socm_hist["loss"], ratio=0.1)
    bsde_conv = find_epoch_to_threshold(bsde_hist["loss"], ratio=0.1)
    socm_conv_1pct = find_epoch_to_threshold(socm_hist["loss"], ratio=0.01)
    bsde_conv_1pct = find_epoch_to_threshold(bsde_hist["loss"], ratio=0.01)

    socm_grad_var = grad_variance_socm(policy, generator, config)
    bsde_grad_var = grad_variance_bsde(model, generator, config)

    summary = {
        "black_scholes_price": eval_metrics["black_scholes_price"],
        "socm": {
            **eval_metrics["socm"],
            "epochs_to_10pct_loss": socm_conv,
            "epochs_to_1pct_loss": socm_conv_1pct,
            "grad_variance": socm_grad_var,
            "loss_stats": summarize_series(socm_hist["loss"]),
            "grad_norm_stats": summarize_series(socm_hist["grad_norm"]),
        },
        "bsde": {
            **eval_metrics["bsde"],
            "epochs_to_10pct_loss": bsde_conv,
            "epochs_to_1pct_loss": bsde_conv_1pct,
            "grad_variance": bsde_grad_var,
            "loss_stats": summarize_series(bsde_hist["loss"]),
            "grad_norm_stats": summarize_series(bsde_hist["grad_norm"]),
            "y0_stats": summarize_series(bsde_hist["Y0"]),
        },
    }

    ensure_dir(config.output_dir)
    summary_path = f"{config.output_dir}/summary.json"
    history_path = f"{config.output_dir}/history.json"
    with open(summary_path, "w") as f:
        json.dump({"config": asdict(config), "summary": summary}, f, indent=2)
    with open(history_path, "w") as f:
        json.dump({"socm": socm_hist, "bsde": bsde_hist}, f, indent=2)

    plots = {}
    if config.plot:
        plots = generate_plots(config.output_dir, socm_hist, bsde_hist, summary)
        with open(f"{config.output_dir}/plots.json", "w") as f:
            json.dump(plots, f, indent=2)

    print("Experiment complete.")
    print(f"Summary saved to {summary_path}")
    if plots:
        print(f"Plots saved to {config.output_dir}")


if __name__ == "__main__":
    main()
