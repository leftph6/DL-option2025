import argparse
import json
from dataclasses import asdict, dataclass
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim

from .black_scholes import call_delta
from .nets import ControlNet, grad_norm
from .path_generators import GBMGenerator
from .utils import ensure_dir, get_device, set_seed


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 1024
    lr: float = 1e-3
    hidden_size: int = 64
    depth: int = 2
    seed: int = 42
    device: str = ""
    output_dir: str = "outputs_socm"

    T: float = 1.0
    N: int = 50
    d: int = 1
    r: float = 0.05
    sigma: float = 0.2
    S0: float = 100.0
    strike: float = 100.0


def _socm_loss(
    policy: ControlNet,
    S: torch.Tensor,
    times: torch.Tensor,
    sigma: float,
    r: float,
    strike: float,
) -> torch.Tensor:
    batch_size, steps, d = S.shape
    S_flat = S[:, :-1, :].reshape(-1, d)
    t_flat = times[:-1].view(1, -1).repeat(batch_size, 1).reshape(-1, 1)

    Z_pred = policy(S_flat, t_flat)
    delta = call_delta(S_flat, t_flat, strike, times[-1].item(), r, sigma)
    Z_target = sigma * S_flat * delta
    return torch.mean((Z_pred - Z_target) ** 2)


def train(config: TrainConfig) -> Dict[str, List[float]]:
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

    policy = ControlNet(d=config.d, hidden_size=config.hidden_size, depth=config.depth).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=config.lr)

    history = {"loss": [], "grad_norm": []}

    for epoch in range(1, config.epochs + 1):
        with torch.no_grad():
            S, _, times = generator.generate_paths(config.batch_size)

        loss = _socm_loss(policy, S, times, config.sigma, config.r, config.strike)

        optimizer.zero_grad()
        loss.backward()
        gnorm = grad_norm(policy.parameters()).item()
        optimizer.step()

        history["loss"].append(float(loss.item()))
        history["grad_norm"].append(float(gnorm))

        if epoch % 20 == 0 or epoch == 1:
            print(f"[SOCM] Epoch {epoch:04d} | loss={loss.item():.6f} | grad_norm={gnorm:.6f}")

    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="SOCM-BSDE training")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="outputs_socm")
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--N", type=int, default=50)
    parser.add_argument("--d", type=int, default=1)
    parser.add_argument("--r", type=float, default=0.05)
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--S0", type=float, default=100.0)
    parser.add_argument("--strike", type=float, default=100.0)

    args = parser.parse_args()
    config = TrainConfig(**vars(args))

    history = train(config)

    ensure_dir(config.output_dir)
    history_path = f"{config.output_dir}/socm_history.json"
    with open(history_path, "w") as f:
        json.dump({"config": asdict(config), "history": history}, f, indent=2)

    print(f"Saved history to {history_path}")


if __name__ == "__main__":
    main()
