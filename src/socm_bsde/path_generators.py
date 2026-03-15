from typing import Tuple

import torch


class GBMGenerator:
    def __init__(
        self,
        T: float = 1.0,
        N: int = 50,
        d: int = 1,
        r: float = 0.05,
        sigma: float = 0.2,
        S0: float = 100.0,
        device: torch.device = None,
    ) -> None:
        self.T = T
        self.N = N
        self.d = d
        self.r = r
        self.sigma = sigma
        self.S0 = S0
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate_paths(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dt = self.T / self.N
        sqrt_dt = torch.sqrt(torch.tensor(dt, device=self.device))

        dW = torch.randn(batch_size, self.N, self.d, device=self.device) * sqrt_dt
        W = torch.cumsum(dW, dim=1)
        W0 = torch.zeros(batch_size, 1, self.d, device=self.device)
        W_full = torch.cat([W0, W], dim=1)

        t_grid = torch.linspace(0.0, self.T, self.N + 1, device=self.device).view(1, self.N + 1, 1)
        drift = (self.r - 0.5 * self.sigma**2) * t_grid

        S = torch.tensor(self.S0, device=self.device) * torch.exp(drift + self.sigma * W_full)
        times = torch.linspace(0.0, self.T, self.N + 1, device=self.device)
        return S, dW, times
