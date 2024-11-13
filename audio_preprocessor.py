# preprocessor.py

import torch
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import numpy as np
import pickle
import os

@dataclass
class PreprocessingConfig:
    sample_rate: int = 22050
    duration: float = 0.74  # in seconds
    n_fft: int = 512
    hop_length: int = 256
    n_mels: int = 64
    normalize: bool = True
    
class SpectrogramPreprocessor:
    """Creates and saves mel spectrograms from audio files."""
    
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
        self.min_max_values = {}
        
    def process_audio_file(self, audio_path: str) -> torch.Tensor:
        """Process a single audio file to mel spectrogram."""
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sr != self.config.sample_rate:
            resampler = T.Resample(sr, self.config.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Pad or trim
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
        
         # Resize to 28x28 using interpolation
        mel_spec_db = torch.nn.functional.interpolate(
            mel_spec_db.unsqueeze(0),  # Add batch dimension for interpolate
            size=(28, 28),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)  # Remove batch dimension
        
        return mel_spec_db
    
    def process_and_save(self, 
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
            self.min_max_values[str(save_path)] = {
                "min": spec_min,
                "max": spec_max
            }
            
            print(f"Processed and saved: {save_path}")
        
        # Save min/max values
        with open(save_dir / "min_max_values.pkl", "wb") as f:
            pickle.dump(self.min_max_values, f)

