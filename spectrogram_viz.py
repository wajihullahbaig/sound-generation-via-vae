import matplotlib.pyplot as plt
import numpy as np
import torch
import librosa
import librosa.display
from pathlib import Path
from typing import Optional, Union, List
import seaborn as sns

def plot_spectrogram(
    spec_path: Union[str, Path],
    figsize: tuple = (10, 4),
    save_path: Optional[str] = None
) -> None:
    """Plot a single spectrogram from file."""
    # Load spectrogram
    spec = np.load(spec_path)
    if isinstance(spec, torch.Tensor):
        spec = spec.numpy()
        
    plt.figure(figsize=figsize)
    librosa.display.specshow(
        spec[0] if spec.ndim == 3 else spec,
        y_axis='mel',
        x_axis='time',
        sr=22050,  # Default sample rate
        hop_length=256  # Default hop length
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel Spectrogram: {Path(spec_path).stem}')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_batch_spectrograms(
    specs: Union[torch.Tensor, np.ndarray],
    num_examples: int = 4,
    figsize: tuple = (15, 3),
    save_path: Optional[str] = None,
    titles: Optional[List[str]] = None
) -> None:
    """Plot multiple spectrograms from a batch."""
    if isinstance(specs, torch.Tensor):
        specs = specs.detach().cpu().numpy()
        
    num_examples = min(num_examples, specs.shape[0])
    fig, axes = plt.subplots(1, num_examples, figsize=figsize)
    
    if num_examples == 1:
        axes = [axes]
    
    for i in range(num_examples):
        spec = specs[i]
        img = librosa.display.specshow(
            spec,
            y_axis='mel',
            x_axis='time',
            sr=22050,  # Default sample rate
            hop_length=256,  # Default hop length
            ax=axes[i]
        )
        title = f'Sample {i+1}' if titles is None else titles[i]
        axes[i].set_title(title)
        plt.colorbar(img, ax=axes[i], format='%+2.0f dB')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def compare_spectrograms(
    spec_paths: List[str],
    titles: Optional[List[str]] = None,
    figsize: tuple = (15, 3),
    save_path: Optional[str] = None
) -> None:
    """Compare multiple spectrograms side by side."""
    specs = [np.load(path) for path in spec_paths]
    if titles is None:
        titles = [Path(path).stem for path in spec_paths]
    
    plot_batch_spectrograms(
        np.stack(specs),
        num_examples=len(specs),
        figsize=figsize,
        save_path=save_path,
        titles=titles
    )
