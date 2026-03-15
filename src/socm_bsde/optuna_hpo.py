import argparse
import json
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.optim as optim
try:
    import optuna
except Exception:
    import sys
    from pathlib import Path
    vendor = Path(__file__).resolve().parents[2] / "vendor"
    sys.path.insert(0, str(vendor))
    import optuna

from .black_scholes import call_price
from .nets import ControlNet, DeepBSDEModel
from .path_generators import GBMGenerator
from .experiment import socm_loss, bsde_loss, estimate_y0_from_policy
from .utils import ensure_dir, get_device, set_seed


@dataclass
class HPOConfig:
    n_trials: int = 10
    epochs_per_trial: int = 50
    batch_size: int = 512
    eval_batch_size: int = 512
    seed: int = 42
    device: str = ""
    output_dir: str = "outputs_optuna"

    T: float = 1.0
    N: int = 50
    d: int = 1
    r: float = 0.05
    sigma: float = 0.2
    S0: float = 100.0
    strike: float = 100.0


def _train_socm_trial(config: HPOConfig, params: Dict[str, float]) -> Tuple[float, List[float]]:
    device = get_device(config.device or None)
    generator = GBMGenerator(T=config.T, N=config.N, d=config.d, r=config.r, sigma=config.sigma, S0=config.S0, device=device)

    policy = ControlNet(d=config.d, hidden_size=params["hidden_size"], depth=params["depth"]).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=params["lr"])

    losses: List[float] = []
    for _ in range(config.epochs_per_trial):
        with torch.no_grad():
            S, _, times = generator.generate_paths(config.batch_size)
        loss = socm_loss(policy, S, times, config.sigma, config.r, config.strike)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    with torch.no_grad():
        S_eval, dW_eval, times_eval = generator.generate_paths(config.eval_batch_size)
        y0 = estimate_y0_from_policy(policy, S_eval, dW_eval, times_eval, config.r, config.strike)
        bs_price = float(call_price(torch.tensor(config.S0, device=device), config.strike, config.T, config.r, config.sigma).item())
    price_error = abs(y0 - bs_price)
    return price_error, losses


def _train_bsde_trial(config: HPOConfig, params: Dict[str, float]) -> Tuple[float, List[float]]:
    device = get_device(config.device or None)
    generator = GBMGenerator(T=config.T, N=config.N, d=config.d, r=config.r, sigma=config.sigma, S0=config.S0, device=device)

    model = DeepBSDEModel(d=config.d, hidden_size=params["hidden_size"], depth=params["depth"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params["lr"])

    losses: List[float] = []
    for _ in range(config.epochs_per_trial):
        with torch.no_grad():
            S, dW, times = generator.generate_paths(config.batch_size)
        loss = bsde_loss(model, S, dW, times, config.r, config.strike)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    with torch.no_grad():
        bs_price = float(call_price(torch.tensor(config.S0, device=device), config.strike, config.T, config.r, config.sigma).item())
    price_error = abs(float(model.Y0.item()) - bs_price)
    return price_error, losses


def run_study(model_type: str, config: HPOConfig) -> Dict[str, object]:
    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            "lr": trial.suggest_float("lr", 1e-4, 3e-2, log=True),
            "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
            "depth": trial.suggest_int("depth", 1, 3),
        }
        if model_type == "socm":
            error, losses = _train_socm_trial(config, params)
        else:
            error, losses = _train_bsde_trial(config, params)
        trial.set_user_attr("losses", losses)
        return error

    sampler = optuna.samplers.TPESampler(seed=config.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=config.n_trials)

    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "trials": study.trials,
        "study": study,
    }


def plot_optuna(study: optuna.Study, output_dir: str, prefix: str) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    try:
        from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances

        fig = plot_optimization_history(study)
        history_path = f"{output_dir}/{prefix}_optimization_history.png"
        fig.figure.savefig(history_path, dpi=150, bbox_inches="tight")
        paths["optimization_history"] = history_path

        fig = plot_param_importances(study)
        importances_path = f"{output_dir}/{prefix}_param_importances.png"
        fig.figure.savefig(importances_path, dpi=150, bbox_inches="tight")
        paths["param_importances"] = importances_path
    except Exception:
        pass

    # Trial loss curves
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    for trial in study.trials:
        losses = trial.user_attrs.get("losses")
        if losses:
            plt.plot(losses, alpha=0.6)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{prefix.upper()} Trial Loss Curves")
    plt.tight_layout()
    loss_path = f"{output_dir}/{prefix}_trial_losses.png"
    plt.savefig(loss_path, dpi=150)
    plt.close()
    paths["trial_losses"] = loss_path

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna HPO for SOCM vs BSDE")
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--epochs_per_trial", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="outputs_optuna")
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--N", type=int, default=50)
    parser.add_argument("--d", type=int, default=1)
    parser.add_argument("--r", type=float, default=0.05)
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--S0", type=float, default=100.0)
    parser.add_argument("--strike", type=float, default=100.0)

    args = parser.parse_args()
    config = HPOConfig(**vars(args))
    set_seed(config.seed)
    ensure_dir(config.output_dir)

    socm_result = run_study("socm", config)
    bsde_result = run_study("bsde", config)

    socm_plots = plot_optuna(socm_result["study"], config.output_dir, "socm")
    bsde_plots = plot_optuna(bsde_result["study"], config.output_dir, "bsde")

    summary = {
        "config": asdict(config),
        "socm": {
            "best_params": socm_result["best_params"],
            "best_value": socm_result["best_value"],
            "plots": socm_plots,
        },
        "bsde": {
            "best_params": bsde_result["best_params"],
            "best_value": bsde_result["best_value"],
            "plots": bsde_plots,
        },
    }

    with open(f"{config.output_dir}/optuna_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Optuna HPO complete.")
    print(f"Summary saved to {config.output_dir}/optuna_summary.json")


if __name__ == "__main__":
    main()
