from typing import List, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from common.audio import stft_magnitude


class ConvNeXtBlock(nn.Module):
    """1-D ConvNeXt block: depthwise conv -> LayerNorm -> pointwise MLP, with layer scale."""

    def __init__(self, dim: int, hidden: int, layer_scale: float):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.gamma = nn.Parameter(torch.full((dim,), layer_scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.mlp(self.norm(self.dwconv(x).transpose(1, 2)))
        return x + (self.gamma * h).transpose(1, 2)


class Vocoder(nn.Module):
    """Vocos-style neural vocoder (Siuzdak, 2023).

    A ConvNeXt backbone runs at frame rate and predicts STFT magnitude and phase for every frame;
    one inverse STFT turns them into a waveform. No sample-rate upsampling, so it is very fast.
    """

    def __init__(self, n_mels: int = 64, dim: int = 256, hidden: int = 768, layers: int = 8,
                 n_fft: int = 128, hop_length: int = 24):
        super().__init__()
        self.n_fft, self.hop_length = n_fft, hop_length
        self.embed = nn.Conv1d(n_mels, dim, 7, padding=3)
        self.norm = nn.LayerNorm(dim)
        self.blocks = nn.Sequential(*[ConvNeXtBlock(dim, hidden, 1 / layers) for _ in range(layers)])
        self.final_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_fft + 2)          # magnitude + phase for n_fft // 2 + 1 bins
        self.register_buffer('window', torch.hann_window(n_fft), persistent=False)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """(B, n_mels, frames) normalised mel in [0, 1] -> (B, (frames - 1) * hop_length) waveform."""
        x = self.norm(self.embed(mel).transpose(1, 2)).transpose(1, 2)
        x = self.final_norm(self.blocks(x).transpose(1, 2))
        log_mag, phase = self.head(x).float().transpose(1, 2).chunk(2, dim=1)
        mag = log_mag.exp().clamp(max=100.0)
        spec = torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))
        return torch.istft(spec, self.n_fft, self.hop_length, window=self.window, center=True)


class SpectrogramDiscriminator(nn.Module):
    """2-D conv discriminator on the log-magnitude spectrogram at one STFT resolution."""

    def __init__(self, n_fft: int, hop_length: int, channels: int = 32):
        super().__init__()
        self.n_fft, self.hop_length = n_fft, hop_length
        conv = lambda c_in, kernel, stride: weight_norm(
            nn.Conv2d(c_in, channels, kernel, stride=stride, padding=(kernel[0] // 2, kernel[1] // 2)))
        self.convs = nn.ModuleList([conv(1, (3, 9), 1), conv(channels, (3, 9), (1, 2)),
                                    conv(channels, (3, 9), (1, 2)), conv(channels, (3, 9), (1, 2)),
                                    conv(channels, (3, 3), 1)])
        self.out = weight_norm(nn.Conv2d(channels, 1, (3, 3), padding=1))

    def forward(self, wav: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = torch.log(stft_magnitude(wav, self.n_fft, self.hop_length)).unsqueeze(1)
        features = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            features.append(x)
        return self.out(x), features


class MultiResolutionDiscriminator(nn.Module):
    """Spectrogram discriminators at several STFT resolutions (as in UnivNet / Vocos)."""

    def __init__(self, resolutions: Sequence[Tuple[int, int]] = ((64, 16), (128, 32), (256, 64))):
        super().__init__()
        self.discriminators = nn.ModuleList([SpectrogramDiscriminator(n, h) for n, h in resolutions])

    def forward(self, wav: torch.Tensor) -> List[Tuple[torch.Tensor, List[torch.Tensor]]]:
        return [d(wav) for d in self.discriminators]
