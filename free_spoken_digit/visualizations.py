import matplotlib.pyplot as plt
import numpy as np
import torch
import librosa
import librosa.display
from pathlib import Path
from typing import Optional, Union, List
import seaborn as sns

def show_spectrogram(S_db, display=True, save_dir=None):
        plt.figure(figsize=(12, 8))
        if torch.is_tensor(S_db[0]):
            plt.imshow(S_db[0].numpy(), aspect='auto', origin='lower', interpolation='nearest')
        else:
            plt.imshow(S_db[0], aspect='auto', origin='lower', interpolation='nearest')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram (Normalized)')
        plt.xlabel('Time')
        plt.ylabel('Mel Frequency')
        plt.tight_layout()

        if display:
            plt.draw()
            plt.pause(0.001)
            plt.show()
            return None
        else:
            if save_dir is None:
                raise ValueError("save_dir must be provided when display is False")
            
            # Create the filename with the current datetime
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"spectrogram_{current_time}.png"
            
            # Ensure the directory exists
            os.makedirs(save_dir, exist_ok=True)
            
            # Combine the directory and filename
            save_path = os.path.join(save_dir, filename)
            
            # Save the plot
            plt.savefig(save_path)
            plt.close()  # Close the figure to free up memory

            return save_path
        
def plot_spectrogram_tensor_or_array(
    spec: Union[torch.Tensor, np.ndarray],
    sr : int = 22050,
    hop_length : int = 128,
    figsize: tuple = (10, 4),
    save_path: Optional[str] = None
) -> None:
    """Plot a single spectrogram from Tensor or numpy array."""
    if isinstance(spec, torch.Tensor):
        spec = spec.numpy()
        
    plt.figure(figsize=figsize)
    librosa.display.specshow(
        spec[0] if spec.ndim == 3 else spec,
        y_axis='mel',
        x_axis='time',
        sr=sr,  
        hop_length=hop_length 
    )
    plt.colorbar(format='%+2.0f dB')
    if save_path:
        plt.title(f'Mel Spectrogram: {Path(spec_path).stem}')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.draw()
        plt.pause(0.01)  # Short pause to update the plot        
        plt.show()

def plot_spectrogram(
    spec_path: Union[str, Path],
    sr : int = 22050,
    hop_length : int = 128,
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
        sr=sr,  
        hop_length=hop_length 
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel Spectrogram: {Path(spec_path).stem}')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.draw()
        plt.pause(0.01)  # Short pause to update the plot        
        plt.show()

def plot_batch_spectrograms(
    specs: Union[torch.Tensor, np.ndarray],
    num_examples: int = 4,
    sr = 22050,
    hop_length = 128,
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
        # Ensure spec is 2D by squeezing or selecting the appropriate slice
        if spec.ndim == 3:
            spec = np.squeeze(spec)  # Remove singleton dimensions
        if spec.ndim != 2:
            raise ValueError(f"Unexpected spectrogram shape: {spec.shape}")
        
        img = librosa.display.specshow(
            spec,
            y_axis='mel',
            x_axis='time',
            sr=sr, 
            hop_length=hop_length,
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
        plt.draw()
        plt.pause(0.01)  # Short pause to update the plot
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
