import argparse
import json
from dataclasses import asdict, dataclass
from typing import Dict, List

import torch
import torch.optim as optim

from .black_scholes import call_price
from .nets import DeepBSDEModel, grad_norm
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
    output_dir: str = "outputs_bsde"

    T: float = 1.0
    N: int = 50
    d: int = 1
    r: float = 0.05
    sigma: float = 0.2
    S0: float = 100.0
    strike: float = 100.0


def _bsde_loss(model: DeepBSDEModel, S: torch.Tensor, dW: torch.Tensor, times: torch.Tensor, r: float, strike: float) -> torch.Tensor:
    Y_T = model(S, dW, times, r)
    payoff = torch.clamp(S[:, -1, :] - strike, min=0.0)
    return torch.mean((Y_T - payoff) ** 2)


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

    model = DeepBSDEModel(d=config.d, hidden_size=config.hidden_size, depth=config.depth).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    history = {"loss": [], "grad_norm": [], "Y0": []}

    for epoch in range(1, config.epochs + 1):
        with torch.no_grad():
            S, dW, times = generator.generate_paths(config.batch_size)

        loss = _bsde_loss(model, S, dW, times, config.r, config.strike)

        optimizer.zero_grad()
        loss.backward()
        gnorm = grad_norm(model.parameters()).item()
        optimizer.step()

        history["loss"].append(float(loss.item()))
        history["grad_norm"].append(float(gnorm))
        history["Y0"].append(float(model.Y0.item()))

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"[BSDE] Epoch {epoch:04d} | loss={loss.item():.6f} | grad_norm={gnorm:.6f} | Y0={model.Y0.item():.4f}"
            )

    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Classical Deep BSDE training")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="outputs_bsde")
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
    history_path = f"{config.output_dir}/bsde_history.json"
    with open(history_path, "w") as f:
        json.dump({"config": asdict(config), "history": history}, f, indent=2)

    print(f"Saved history to {history_path}")


if __name__ == "__main__":
    main()
