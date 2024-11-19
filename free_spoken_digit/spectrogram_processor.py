import torch
import torchaudio
from torch import device as torch_device
from pathlib import Path
from typing import Optional, Tuple, Dict, Union
from dataclasses import dataclass
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from datetime import datetime
from visualizations import show_spectrogram


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
        self.total_samples = int(self.sample_rate * self.duration)
        self.time_bins = int((self.total_samples + self.hop_length) // self.hop_length)        
        self.time_resolution = (self.hop_length / self.sample_rate * 1000)
        self.frequency_resolution= (self.sample_rate / 2 / self.n_mels)                               
        
        if self.n_fft < 2 * self.hop_length:
            raise ValueError(f"n_fft ({self.n_fft}) should be >= 2 * hop_length ({2 * self.hop_length})")
        
    def print_config(self):
        print("\n=== Spectrogram Configuration ===")
        print(f"Duration: {self.duration:.6f} seconds")
        print(f"Sample rate: {self.sample_rate} Hz")
        print(f"Total samples: {self.total_samples}")
        print(f"Hop length: {self.hop_length}")
        print(f"Time bins: {self.time_bins}")
        print(f"Mel bins: {self.n_mels}")
        print(f"Time resolution: {self.time_resolution:.2f} ms per bin")
        print(f"Frequency resolution: {self.frequency_resolution:.2f} Hz per bin")
        print(f"Output shape: {self.n_mels}x{self.time_bins}")
        
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
        power=2.0
        )
        # Storage for min max values per audio file. We will use this in reconstruction of audio
        self.min_max_values = {}
        
        self.inverse_mel_scale_transform = torchaudio.transforms.InverseMelScale(
            n_stft=self.config.n_fft // 2 + 1,
            n_mels=self.config.n_mels,
            sample_rate=self.config.sample_rate
        )

        self.griffin_lim_transform = torchaudio.transforms.GriffinLim(
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_iter=100
        )

           
    
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
        
        # Now we can enhance volume
        volume_enhance_factor = 2.0  # Adjust this value to increase or decrease volume
        y_reverse_enhanced = y_reverse * volume_enhance_factor

        # Clip the enhanced audio to avoid distortion
        y_reverse_enhanced = torch.clamp(y_reverse_enhanced, -1.0, 1.0)

        return y_reverse_enhanced
    
    def create_audio_from_spectrograms(self, S_db: list, min_max_values:list) -> torch.Tensor:                
        signals = []
        for spectrogram, min_max_value in zip(S_db, min_max_values):
            spectrogram = torch.from_numpy(spectrogram).to(self.device)
            recreated_signal = self.create_audio_from_spectrogram(spectrogram, min_max_value['min'], min_max_value['max'])
            # append signal to "signals"
            signals.append(recreated_signal)
        return signals
            
    def create_spectrogram_from_audio(self, waveform: torch.Tensor, sr: int) -> Tuple[torch.Tensor, float,float]:
        """Process a single audio file to mel spectrogram
            1- Do any necessary preprocessing
            2- Create the mel spectrograms
            3- Convert to decibels
            4- Get min/max values for reconstruction - then normalize
            4- Save everything  that is needed for reconstruction and model training. 
            
            
        """
       
        # Preprocessing steps 
        # Resample if necessary
        if sr != self.config.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.config.sample_rate)
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
        S = self.mel_spectrogram_transform(waveform)# (channel, n_mels, time)
        # Convert to dB scale
        S_db = torchaudio.transforms.AmplitudeToDB()(S)
        if S_db.shape[1] != self.config.n_mels:
            raise ValueError(f"Expected {self.config.n_mels} mel bins, but got {S_db.shape[1]}")
        if S_db.shape[2] != self.config.time_bins:
            raise ValueError(f"Expected {self.config.time_bins} time bins, but got {S_db.shape[2]}")
        
        # Get min/max, then normalize the spectrogram
        S_min = torch.min(S_db)
        S_max = torch.max(S_db)
        S_normalized = (S_db - S_min) / (S_max - S_min)
        
        #show_spectrogram(S_normalized,display=True)
        
        return S_normalized, S_min, S_max

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