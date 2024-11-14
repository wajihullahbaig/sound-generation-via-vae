import torch
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
import numpy as np
import pickle
import os

@dataclass
class PreprocessingConfig:
    """Configuration for audio preprocessing with exact 256 time bins."""
    # Common parameters
    sample_rate: int = 22050
    n_fft: int = 512
    n_mels: int = 64
    normalize: bool = True
    
    # === OPTION 1: Short duration, small hop ===
    duration: float = 0.740181  # Exact duration for 256 bins
    hop_length: int = 64
    
    # === OPTION 2: Longer duration, standard hop ===
    # duration: float = 2.960590  # Exact duration for 256 bins
    # hop_length: int = 256
    
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

class SpectrogramPreprocessor:
    """Creates mel spectrograms with exact 256 time bins."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.num_samples = int(config.sample_rate * config.duration)
        
        # Initialize transforms
        self.mel_spec = T.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            center=True,
            pad_mode="reflect",
            power=2.0,
        )
        
        self.amplitude_to_db = T.AmplitudeToDB(top_db=80)
        
    
    def process_audio_file(self, audio_path: str) -> torch.Tensor:
        """Process a single audio file to mel spectrogram with exact 256 bins."""
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sr != self.config.sample_rate:
            resampler = T.Resample(sr, self.config.sample_rate)
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
        
        # Convert to mel spectrogram
        mel_spec = self.mel_spec(waveform)
        
        # Convert to dB
        mel_spec_db = self.amplitude_to_db(mel_spec)
               
        return mel_spec_db

    def process_and_save(self, 
                            audio_dir: str, 
                            save_dir: str,
                            audio_ext: str = ".wav",
                            resize:bool=True) -> None:
            """Process all audio files and save spectrograms."""
            # Create save directory if it doesn't exist
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Process all audio files
            audio_files = list(Path(audio_dir).rglob(f"*{audio_ext}"))
            for audio_path in audio_files:
                # Process audio to mel spectrogram
                mel_spec = self.process_audio_file(str(audio_path))
                
                # Save min/max values before normalization
                spec_min = mel_spec.min().item()
                spec_max = mel_spec.max().item()
                
                # Normalize if configured
                if self.config.normalize:
                    mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()
                
                # Generate save path
                save_path = save_dir / f"{audio_path.stem}.npy"
                
                # Save spectrogram
                np.save(save_path, mel_spec.numpy())
                
                # Store min/max values
                #self.min_max_values[str(save_path)] = {
                #    "min": spec_min,
                #    "max": spec_max
                #}
                
                print(f"Processed and saved: {save_path}")
            
            # Save min/max values
           # with open(save_dir / "min_max_values.pkl", "wb") as f:
           #     pickle.dump(self.min_max_values, f)


