import torch
import torchaudio
from torch import device as torch_device
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pickle


def load_db_range(spectrograms_dir: str) -> Tuple[float, float]:
    """Return the dataset-wide (min, max) dB range used to normalise the spectrograms."""
    with open(Path(spectrograms_dir) / "min_max_values.pkl", "rb") as f:
        values = next(iter(pickle.load(f).values()))
    return values["min"], values["max"]


@dataclass
class ProcessingConfig:
    """Configuration for audio preprocessing with exact 256 time bins."""
    # Common parameters
    sample_rate: int = 22050
    n_fft: int = 512
    hop_length: int = 128
    n_mels: int = 64   
    duration: float = 0.74
    f_min: float = 0.0
    f_max: Optional[float] = None   # None -> Nyquist (sample_rate / 2)
    top_db: float = 80.0            # dynamic range kept below each clip's peak
    griffin_lim_iters: int = 100
        
    
    def __post_init__(self):
        """Validate configuration and verify exact bin count."""
        if self.f_max is None:
            self.f_max = self.sample_rate / 2
        self.total_samples = int(self.sample_rate * self.duration)
        self.time_bins = int((self.total_samples + self.hop_length) // self.hop_length)        
        self.time_resolution = (self.hop_length / self.sample_rate * 1000)
        self.frequency_resolution= ((self.f_max - self.f_min) / self.n_mels)                               
        
        if self.n_fft < 2 * self.hop_length:
            raise ValueError(f"n_fft ({self.n_fft}) should be >= 2 * hop_length ({2 * self.hop_length})")
        
    def print_config(self):
        print("\n=== Spectrogram Configuration ===")
        print(f"Duration: {self.duration:.6f} seconds")
        print(f"Sample rate: {self.sample_rate} Hz")
        print(f"Total samples: {self.total_samples}")
        print(f"Hop length: {self.hop_length}")
        print(f"Time bins: {self.time_bins}")
        print(f"Mel bins: {self.n_mels} ({self.f_min:.0f}-{self.f_max:.0f} Hz)")
        print(f"Time resolution: {self.time_resolution:.2f} ms per bin")
        print(f"Frequency resolution: {self.frequency_resolution:.2f} Hz per bin")
        print(f"Output shape - mels x time_bin - > HxW: {self.n_mels} x {self.time_bins}")
        
class SpectrogramProcessor:
    """Creates mel spectrograms"""
    
    def __init__(self, config: ProcessingConfig,device:torch_device):
        self.device = device
        self.config = config
        self.num_samples = int(config.sample_rate * config.duration)
        
        # Initialize transforms we can use from Pytorch        
        self.mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=config.sample_rate,
        n_fft= config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        f_min=config.f_min,
        f_max=config.f_max,
        power=2.0
        ).to(self.device)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='power').to(self.device)
        # Storage for min max values per audio file. We will use this in reconstruction of audio
        self.min_max_values = {}
        
        self.inverse_mel_scale_transform = torchaudio.transforms.InverseMelScale(
            n_stft=self.config.n_fft // 2 + 1,
            n_mels=self.config.n_mels,
            sample_rate=self.config.sample_rate,
            f_min=self.config.f_min,
            f_max=self.config.f_max
        ).to(self.device)

        self.griffin_lim_transform = torchaudio.transforms.GriffinLim(
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_iter=self.config.griffin_lim_iters
        ).to(self.device)

           
    
    def create_audio_from_spectrogram(self, S_db: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:                
        # First we have De-normalize the signal. The order in which
        # we reconstruction the audio is important. 
        if S_db.dim() == 3:
            if S_db.size(2) == 1:
                S_db = S_db.permute(2,0,1)
        spectrogram_denormalized = S_db * (max_val - min_val) + min_val
        
        
        # Second, we need to convert back from dB to power
        S_reverse = torch.exp(spectrogram_denormalized * 0.11512925464970229)  # 10^(x/10) = e^(x * ln(10)/10)

        # Now we can invert the mel spectrogram
        y_reverse = self.griffin_lim_transform(self.inverse_mel_scale_transform(S_reverse))
        
        # Peak-normalise instead of a fixed gain + hard clip (which distorts loud outputs)
        peak = y_reverse.abs().max().clamp(min=1e-8)
        return y_reverse * (0.9 / peak)
    
    def create_audio_from_spectrograms(self, S_db: list, min_max_values:list) -> torch.Tensor:                
        signals = []
        for spectrogram, min_max_value in zip(S_db, min_max_values):
            spectrogram = torch.from_numpy(spectrogram).to(self.device)
            recreated_signal = self.create_audio_from_spectrogram(spectrogram, min_max_value['min'], min_max_value['max'])
            # append signal to "signals"
            signals.append(recreated_signal)
        return signals
            
    def create_spectrogram_from_audio(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """Return the (1, n_mels, time_bins) mel spectrogram in dB."""
        # Move waveform to the correct device first
        waveform = waveform.to(self.device)
        
        # Preprocessing steps 
        # Resample if necessary
        if sr != self.config.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.config.sample_rate).to(self.device)
            waveform = resampler(waveform)
        
        # Convert to mono if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Exact padding/trimming for precise number of samples
        if waveform.shape[1] < self.num_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.num_samples - waveform.shape[1])
            )
        elif waveform.shape[1] > self.num_samples:
            waveform = waveform[:, :self.num_samples]
        
        # Create the Mel Spectrograms
        S = self.mel_spectrogram_transform(waveform)  # (channel, n_mels, time)
        
        S_db = self.amplitude_to_db(S)
        
        if S_db.shape[1] != self.config.n_mels:
            raise ValueError(f"Expected {self.config.n_mels} mel bins, but got {S_db.shape[1]}")
        if S_db.shape[2] != self.config.time_bins:
            raise ValueError(f"Expected {self.config.time_bins} time bins, but got {S_db.shape[2]}")
        
        return S_db

    def normalize(self, S_db: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
        """Map dB values to [0, 1], clipping everything below min_val."""
        return ((S_db.clamp(min=min_val, max=max_val) - min_val) / (max_val - min_val))

    def create_spectrograms_and_save(self, 
                            audio_dir: str, 
                            save_dir: str,
                            audio_ext: str = ".wav") -> None:
            """Process all audio files and save spectrograms."""
            # Create save directory if it doesn't exist
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Normalise each clip relative to its own peak: dB - max(dB), clipped at -top_db.
            # - recording gain (which varies ~30 dB between speakers) is removed; the output audio
            #   is peak-normalised anyway, so absolute level carries no information
            # - every clip uses the full [0, 1] range and silence/padding sits at 0
            # - the range is the same for all clips, so any decoded spectrogram can be inverted
            #   without per-file statistics
            db_min, db_max = -self.config.top_db, 0.0
            audio_files = sorted(Path(audio_dir).rglob(f"*{audio_ext}"))
            for audio_path in audio_files:
                waveform, sr = torchaudio.load(str(audio_path))
                S_db = self.create_spectrogram_from_audio(waveform, sr)
                save_path = save_dir / f"{audio_path.stem}.npy"
                np.save(save_path, self.normalize(S_db - S_db.max(), db_min, db_max).cpu().numpy())
                # Kept per file for compatibility with create_audios.py
                self.min_max_values[str(save_path)] = {"min": db_min, "max": db_max}
            print(f"Processed and saved {len(audio_files)} spectrograms to {save_dir} "
                  f"(peak-referenced, {db_min:.0f} .. {db_max:.0f} dB)")
            
            # Save min/max values
            with open(save_dir / "min_max_values.pkl", "wb") as f:
                pickle.dump(self.min_max_values, f)
