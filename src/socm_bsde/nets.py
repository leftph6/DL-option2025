from typing import Iterable

import torch
import torch.nn as nn


class ControlNet(nn.Module):
    def __init__(self, d: int, hidden_size: int = 64, depth: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(d + 1, hidden_size), nn.ReLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        layers.append(nn.Linear(hidden_size, d))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, S: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t_in = torch.full((S.shape[0], 1), t.item(), device=S.device)
        elif t.ndim == 1:
            t_in = t.view(-1, 1)
        else:
            t_in = t
        x = torch.cat([S, t_in], dim=1)
        return self.net(x)


class DeepBSDEModel(nn.Module):
    def __init__(self, d: int, hidden_size: int = 64, depth: int = 2) -> None:
        super().__init__()
        self.Y0 = nn.Parameter(torch.tensor(0.0))
        self.policy = ControlNet(d=d, hidden_size=hidden_size, depth=depth)

    def forward(self, S: torch.Tensor, dW: torch.Tensor, times: torch.Tensor, r: float) -> torch.Tensor:
        batch_size = S.shape[0]
        Y = self.Y0.expand(batch_size, 1).to(S.device)
        dt = times[1] - times[0]
        for i in range(dW.shape[1]):
            t = times[i]
            Z = self.policy(S[:, i, :], t)
            Y = Y - r * Y * dt + (Z * dW[:, i, :]).sum(dim=1, keepdim=True)
        return Y


def grad_norm(parameters: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    norms = []
    for p in parameters:
        if p.grad is None:
            continue
        norms.append(p.grad.detach().reshape(-1))
    if not norms:
        return torch.tensor(0.0)
    return torch.cat(norms).norm(p=2)
