import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def cosine_alpha_bar(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cumulative signal rate alpha_bar_t of the cosine schedule (Nichol & Dhariwal, 2021)."""
    f = torch.cos((torch.arange(timesteps + 1) / timesteps + s) / (1 + s) * math.pi / 2) ** 2
    betas = (1 - f[1:] / f[:-1]).clamp(max=0.999)
    return torch.cumprod(1 - betas, dim=0)


class DDPM(nn.Module):
    """Denoising Diffusion Probabilistic Model (Ho et al., 2020) with an epsilon-predicting network."""

    def __init__(self, model: nn.Module, sample_shape: Tuple[int, ...], timesteps: int = 1000):
        super().__init__()
        self.model = model
        self.sample_shape = sample_shape
        self.timesteps = timesteps
        self.register_buffer('alpha_bar', cosine_alpha_bar(timesteps))

    def loss(self, x0: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """MSE between the true and predicted noise at a random timestep."""
        t = torch.randint(0, self.timesteps, (x0.size(0),), device=x0.device)
        noise = torch.randn_like(x0)
        ab = self.alpha_bar[t].view(-1, 1, 1, 1)
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
        return F.mse_loss(self.model(xt, t / self.timesteps, y), noise)

    @torch.no_grad()
    def sample(self, y: torch.Tensor, steps: int = 100, guidance: float = 2.0, eta: float = 1.0) -> torch.Tensor:
        """Generate samples in [-1, 1]. eta=1 gives DDPM ancestral sampling, eta=0 deterministic DDIM."""
        x = torch.randn(len(y), *self.sample_shape, device=y.device)
        ts = torch.linspace(self.timesteps - 1, 0, steps, device=y.device).round().long()
        for i, t in enumerate(ts):
            ab = self.alpha_bar[t]
            ab_prev = self.alpha_bar[ts[i + 1]] if i + 1 < steps else torch.ones_like(ab)
            eps = self.model.guided(x, t.repeat(len(y)) / self.timesteps, y, guidance)
            x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
            eps = (x - ab.sqrt() * x0) / (1 - ab).sqrt()
            sigma = eta * ((1 - ab_prev) / (1 - ab) * (1 - ab / ab_prev)).sqrt()
            x = ab_prev.sqrt() * x0 + (1 - ab_prev - sigma ** 2).clamp(min=0).sqrt() * eps
            if i + 1 < steps:
                x = x + sigma * torch.randn_like(x)
        return x
