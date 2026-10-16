import torch
import torch.nn as nn
import torchaudio


def stft_magnitude(wav: torch.Tensor, n_fft: int, hop_length: int, eps: float = 1e-7) -> torch.Tensor:
    """|STFT| of (B, samples) audio -> (B, freq, frames).

    Uses view_as_real instead of complex abs(), which needs NVRTC on CUDA and is unavailable on some Windows setups.
    """
    window = torch.hann_window(n_fft, device=wav.device)
    spec = torch.view_as_real(torch.stft(wav.float(), n_fft, hop_length, window=window, return_complex=True))
    return spec.pow(2).sum(-1).clamp(min=eps).sqrt()


class MelSpectrogram(nn.Module):
    """Power mel spectrogram matching ProcessingConfig, as a single filterbank matmul (GPU friendly)."""

    def __init__(self, sample_rate: int, n_fft: int, hop_length: int, n_mels: int, f_min: float, f_max: float):
        super().__init__()
        self.n_fft, self.hop_length = n_fft, hop_length
        fbanks = torchaudio.functional.melscale_fbanks(n_fft // 2 + 1, f_min, f_max, n_mels, sample_rate)
        self.register_buffer('fbanks', fbanks, persistent=False)   # (freq, mels)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """(B, samples) -> (B, mels, frames) power."""
        power = stft_magnitude(wav, self.n_fft, self.hop_length).pow(2)
        return (power.transpose(1, 2) @ self.fbanks).transpose(1, 2)


def normalize_mel(mel_power: torch.Tensor, top_db: float) -> torch.Tensor:
    """Per-clip peak-referenced dB in [0, 1], identical to the dataset spectrograms."""
    db = 10 * torch.log10(mel_power.clamp(min=1e-10))
    db = db - db.amax(dim=(-2, -1), keepdim=True)
    return (db.clamp(min=-top_db) + top_db) / top_db
