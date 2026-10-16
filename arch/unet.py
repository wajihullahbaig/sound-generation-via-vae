import math
import torch
import torch.nn as nn
from typing import Sequence


class TimeEmbedding(nn.Module):
    """Sinusoidal embedding of a continuous time t in [0, 1]."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = 1000 * t[:, None] * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=1)


class ResBlock(nn.Module):
    """Residual conv block modulated by the time/class embedding."""

    def __init__(self, c_in: int, c_out: int, emb_dim: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Sequential(nn.GroupNorm(8, c_in), nn.SiLU(), nn.Conv2d(c_in, c_out, 3, padding=1))
        self.emb = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, c_out))
        self.conv2 = nn.Sequential(nn.GroupNorm(8, c_out), nn.SiLU(), nn.Dropout(dropout),
                                   nn.Conv2d(c_out, c_out, 3, padding=1))
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x) + self.emb(emb)[:, :, None, None]
        return self.conv2(h) + self.skip(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention over all spatial positions."""

    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = self.norm(x).flatten(2).transpose(1, 2)
        out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        return x + out.transpose(1, 2).reshape(b, c, h, w)


class UNet(nn.Module):
    """Class-conditional U-Net predicting noise (DDPM) or velocity (Flow Matching)."""

    def __init__(self,
                 in_channels: int = 1,
                 base_channels: int = 64,
                 channel_mults: Sequence[int] = (1, 2, 2, 4),
                 num_classes: int = 10,
                 dropout: float = 0.1,
                 p_uncond: float = 0.1,
                 patch_size: int = 1):
        super().__init__()
        emb_dim = 4 * base_channels
        channels = [base_channels * m for m in channel_mults]
        self.null_class = num_classes          # extra label used for unconditional predictions
        self.p_uncond = p_uncond               # label dropout rate for classifier-free guidance

        self.time_emb = nn.Sequential(TimeEmbedding(base_channels), nn.Linear(base_channels, emb_dim),
                                      nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.class_emb = nn.Embedding(num_classes + 1, emb_dim)
        # patch_size > 1 folds p x p pixel patches into channels: same output resolution, ~p^2 less compute
        patch_channels = in_channels * patch_size ** 2
        self.stem = nn.Sequential(nn.PixelUnshuffle(patch_size),
                                  nn.Conv2d(patch_channels, base_channels, 3, padding=1))

        self.down_blocks, self.downsamples = nn.ModuleList(), nn.ModuleList()
        c = base_channels
        for i, ch in enumerate(channels):
            self.down_blocks.append(ResBlock(c, ch, emb_dim, dropout))
            last = i == len(channels) - 1
            self.downsamples.append(nn.Identity() if last else nn.Conv2d(ch, ch, 3, stride=2, padding=1))
            c = ch

        self.mid1 = ResBlock(c, c, emb_dim, dropout)
        self.mid_attn = SelfAttention(c)
        self.mid2 = ResBlock(c, c, emb_dim, dropout)

        self.up_blocks, self.upsamples = nn.ModuleList(), nn.ModuleList()
        for i in reversed(range(len(channels))):
            c_out = channels[i - 1] if i > 0 else base_channels
            self.up_blocks.append(ResBlock(c + channels[i], c_out, emb_dim, dropout))
            self.upsamples.append(nn.Identity() if i == 0 else nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(c_out, c_out, 3, padding=1)))
            c = c_out

        self.head = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(),
                                  nn.Conv2d(c, patch_channels, 3, padding=1), nn.PixelShuffle(patch_size))
        nn.init.zeros_(self.head[-2].weight)
        nn.init.zeros_(self.head[-2].bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Predict the target for noisy input x at time t (in [0, 1]) with labels y."""
        if self.training and self.p_uncond > 0:
            drop = torch.rand(y.shape, device=y.device) < self.p_uncond
            y = torch.where(drop, torch.full_like(y, self.null_class), y)
        emb = self.time_emb(t) + self.class_emb(y)

        h = self.stem(x)
        skips = []
        for block, down in zip(self.down_blocks, self.downsamples):
            h = block(h, emb)
            skips.append(h)
            h = down(h)

        h = self.mid2(self.mid_attn(self.mid1(h, emb)), emb)

        for block, up in zip(self.up_blocks, self.upsamples):
            h = up(block(torch.cat([h, skips.pop()], dim=1), emb))
        return self.head(h)

    def guided(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, scale: float) -> torch.Tensor:
        """Classifier-free guided prediction: uncond + scale * (cond - uncond)."""
        if scale == 1.0:
            return self(x, t, y)
        null = torch.full_like(y, self.null_class)
        cond, uncond = self(torch.cat([x, x]), torch.cat([t, t]), torch.cat([y, null])).chunk(2)
        return uncond + scale * (cond - uncond)
