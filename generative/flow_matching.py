import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class FlowMatching(nn.Module):
    """Conditional Flow Matching (Lipman et al., 2023) on the straight noise-to-data path (rectified flow)."""

    def __init__(self, model: nn.Module, sample_shape: Tuple[int, ...]):
        super().__init__()
        self.model = model
        self.sample_shape = sample_shape

    def loss(self, x1: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """MSE between the predicted and true velocity x1 - x0 at a random point on the path."""
        x0 = torch.randn_like(x1)
        t = torch.sigmoid(torch.randn(x1.size(0), device=x1.device))  # logit-normal: focus on mid-path
        tt = t.view(-1, 1, 1, 1)
        xt = (1 - tt) * x0 + tt * x1
        return F.mse_loss(self.model(xt, t, y), x1 - x0)

    @torch.no_grad()
    def sample(self, y: torch.Tensor, steps: int = 50, guidance: float = 2.0) -> torch.Tensor:
        """Integrate the learned ODE from noise (t=0) to data (t=1) with Heun's method."""
        x = torch.randn(len(y), *self.sample_shape, device=y.device)
        ts = torch.linspace(0, 1, steps + 1, device=y.device)
        for t0, t1 in zip(ts[:-1], ts[1:]):
            dt = t1 - t0
            v0 = self.model.guided(x, t0.repeat(len(y)), y, guidance)
            x_euler = x + dt * v0
            v1 = self.model.guided(x_euler, t1.repeat(len(y)), y, guidance)
            x = x + dt * (v0 + v1) / 2
        return x.clamp(-1, 1)
