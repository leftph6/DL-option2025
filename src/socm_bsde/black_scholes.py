import torch


def _norm_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0, device=x.device))))


def call_price(S: torch.Tensor, K: float, T: float, r: float, sigma: float) -> torch.Tensor:
    T_t = torch.tensor(T, device=S.device)
    d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * T_t) / (sigma * torch.sqrt(T_t))
    d2 = d1 - sigma * torch.sqrt(T_t)
    disc = torch.exp(torch.tensor(-r * T, device=S.device))
    return S * _norm_cdf(d1) - K * disc * _norm_cdf(d2)


def call_delta(S: torch.Tensor, t: torch.Tensor, K: float, T: float, r: float, sigma: float) -> torch.Tensor:
    if not torch.is_tensor(t):
        t = torch.tensor(t, device=S.device)
    T_t = torch.tensor(T, device=S.device)
    tau = torch.clamp(T_t - t, min=1e-6)
    d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * torch.sqrt(tau))
    return _norm_cdf(d1)
