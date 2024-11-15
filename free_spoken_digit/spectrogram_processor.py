import torch
import torchaudio
from pathlib import Path
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

@dataclass
class ProcessingConfig:
    """Configuration for audio preprocessing with exact 256 time bins."""
    # Common parameters
    sample_rate: int = 22050
    n_fft: int = 512
    hop_length: int = 128
    n_mels: int = 64   
    duration: float = 0.74
    
    def __post_init__(self):
        """Validate configuration and verify exact bin count."""
        total_samples = int(self.sample_rate * self.duration)
        time_bins = (total_samples + self.hop_length) // self.hop_length
        
        print("\n=== Spectrogram Configuration ===")
        print(f"Duration: {self.duration:.6f} seconds")
        print(f"Sample rate: {self.sample_rate} Hz")
        print(f"Total samples: {total_samples}")
        print(f"Hop length: {self.hop_length}")
        print(f"Time bins: {time_bins}")
        print(f"Mel bins: {self.n_mels}")
        print(f"Time resolution: {(self.hop_length / self.sample_rate * 1000):.2f} ms per bin")
        print(f"Frequency resolution: {(self.sample_rate / 2 / self.n_mels):.2f} Hz per bin")
        print(f"Output shape: {time_bins}x{self.n_mels}")
                       
        if self.n_fft < 2 * self.hop_length:
            raise ValueError(f"n_fft ({self.n_fft}) should be >= 2 * hop_length ({2 * self.hop_length})")

class SpectrogramProcessor:
    """Creates mel spectrograms with exact 256 time bins."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.num_samples = int(config.sample_rate * config.duration)
        
        # Initialize transforms we can use from Pytorch        
        self.mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=config.sample_rate,
        n_fft= config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        power=2.0
        )
        # Storage for min max values per audio file. We will use this in reconstruction of audio
        self.min_max_values = {}
        
    def _show_spectrogram(self,S_db):        
        plt.figure(figsize=(12, 8))
        plt.imshow(S_db[0].numpy(), aspect='auto', origin='lower', interpolation='nearest')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram (Normalized)')
        plt.xlabel('Time')
        plt.ylabel('Mel Frequency')
        plt.tight_layout()
        plt.show()
    
    def create_audio_from_spectrogram(self, spectrogram: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:                
        # First, we need to convert back from dB to power
        S_reverse = torch.exp(S_db * 0.11512925464970229)  # 10^(x/10) = e^(x * ln(10)/10)

        # Now we can invert the mel spectrogram
        inverse_mel_scale = torchaudio.transforms.InverseMelScale(
            n_stft=n_fft // 2 + 1,
            n_mels=n_mels,
            sample_rate=sr
        )

        griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            hop_length=hop_length,
            n_iter=100
        )

        y_reverse = griffin_lim(inverse_mel_scale(S_reverse))

        # 6. Denormalize the reconstructed audio
        y_reverse_denormalized = y_reverse * (waveform_max - waveform_min) + waveform_min

        # 7. Enhance volume
        volume_enhance_factor = 2.0  # Adjust this value to increase or decrease volume
        y_reverse_enhanced = y_reverse_denormalized * volume_enhance_factor

        # Clip the enhanced audio to avoid distortion
        y_reverse_enhanced = torch.clamp(y_reverse_enhanced, -1.0, 1.0)

        return y_reverse_enhanced
            
    def create_spectrogram_from_audio(self, waveform: torch.Tensor, sr: int) -> Tuple[torch.Tensor, float,float]:
        """Process a single audio file to mel spectrogram
            1- Find min/max for each file and store individually - we will need them at reconstruction
            2- Do any necessary preprocessing
            2- Create the mel spectrograms
            3- Convert to decibels
            4- Save everything  that is needed for reconstruction and model training. 
            
        """
       
        
        # Apply Min/Max Normalization
        waveform_min = torch.min(waveform)
        waveform_max = torch.max(waveform)        
        waveform_normalized = (waveform - waveform_min) / (waveform_max - waveform_min)

        # Preprocessing steps 
        # Resample if necessary
        if sr != self.config.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.config.sample_rate)
            waveform_normalized = resampler(waveform_normalized)
        
        # Convert to mono if necessary
        if waveform_normalized.shape[0] > 1:
            waveform_normalized = torch.mean(waveform_normalized, dim=0, keepdim=True)
        
        # Exact padding/trimming for precise number of samples
        if waveform_normalized.shape[1] < self.num_samples:
            waveform_normalized = torch.nn.functional.pad(
                waveform_normalized, (0, self.num_samples - waveform_normalized.shape[1])
            )
        elif waveform_normalized.shape[1] > self.num_samples:
            waveform_normalized = waveform_normalized[:, :self.num_samples]
        
        # Create the Mel Spectrograms
        S = self.mel_spectrogram_transform(waveform_normalized)
        # Convert to dB scale
        S_db = torchaudio.transforms.AmplitudeToDB()(S)
        if S_db.shape[1] != self.config.n_mels:
            raise ValueError(f"Expected {self.config.n_mels} mel bins, but got {S_db.shape[1]}")
        if S_db.shape[2] != self.config.n_mels:
            raise ValueError(f"Expected {self.config.n_mels} mel bins, but got {S_db.shape[2]}")
        #self._show_spectrogram(S_db)
        
        return S_db, waveform_min, waveform_max

    def create_spectrograms_and_save(self, 
                            audio_dir: str, 
                            save_dir: str,
                            audio_ext: str = ".wav") -> None:
            """Process all audio files and save spectrograms."""
            # Create save directory if it doesn't exist
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Process all audio files
            audio_files = list(Path(audio_dir).rglob(f"*{audio_ext}"))
            for audio_path in audio_files:
                # Process audio to mel spectrogram
                # Load audio
                waveform, sr = torchaudio.load(str(audio_path))
                mel_spec, waveform_min, waveform_max = self.create_spectrogram_from_audio(waveform,sr)
                                
                # Generate save path
                save_path = save_dir / f"{audio_path.stem}.npy"
                
                # Save spectrogram
                np.save(save_path, mel_spec.numpy())
                
                # Store min/max values
                self.min_max_values[str(save_path)] = {
                   "min": waveform_min.item(),
                    "max": waveform_max.item()
                }
                
                print(f"Processed and saved: {save_path}")
            
            # Save min/max values
            with open(save_dir / "min_max_values.pkl", "wb") as f:
                pickle.dump(self.min_max_values, f)