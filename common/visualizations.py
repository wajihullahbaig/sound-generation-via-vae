import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import torch
from sklearn.decomposition import PCA
import numpy as np
from typing import Optional

import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import numpy as np

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
        
def visualize_reconstructions(
    model, 
    data_loader, 
    device, 
    num_images=8,
    display=True, 
    save_path=None,
    color_mode='auto',
    spectrogram_mode=False,
    cmap='viridis',
    fig_size_multiplier=(2, 4)
):
    """
    Visualize or save original and reconstructed images with enhanced color support.
    
    Args:
        model: VAE model
        data_loader: DataLoader instance
        device: torch device
        num_images: Number of images to visualize
        display: If True, displays the plot. If False, saves to disk
        save_path: Directory to save the visualization. If None, uses '/viz_outputs'
        color_mode: 'auto', 'color', or 'grayscale'. 'auto' determines from input shape
        spectrogram_mode: If True, applies spectrogram-specific visualization settings
        cmap: Colormap for visualization (e.g., 'viridis', 'magma', 'gray')
        fig_size_multiplier: Tuple of (width, height) multipliers for figure size
    """
    model.eval()
    save_path = Path(save_path) if save_path else Path('/viz_outputs')
    save_path.mkdir(parents=True, exist_ok=True)
    if data_loader.batch_size < num_images:
        num_images = data_loader.batch_size
        
    with torch.no_grad():
        images, _ = next(iter(data_loader))
        images = images[:num_images].to(device)
        reconstructions, _, _, _ = model(images)
        
        # Determine if images are color based on shape or color_mode
        is_color = (color_mode == 'color' or 
                   (color_mode == 'auto' and images.shape[1] == 3))
        
        # Create figure with appropriate size
        fig_width = fig_size_multiplier[0] * num_images
        fig_height = fig_size_multiplier[1]
        fig, axes = plt.subplots(2, num_images, figsize=(fig_width, fig_height))
        
        # Visualization loop
        for i in range(num_images):
            # Original images
            img_orig = images[i].cpu()
            if is_color:
                # Handle color images (C,H,W) -> (H,W,C)
                img_orig = img_orig.permute(1, 2, 0)
                img_recon = reconstructions[i].cpu().permute(1, 2, 0)
                
                # Ensure values are in valid range
                img_orig = torch.clamp(img_orig, 0, 1)
                img_recon = torch.clamp(img_recon, 0, 1)
                
                axes[0,i].imshow(img_orig)
                axes[1,i].imshow(img_recon)
            else:
                # Handle grayscale images (spectrograms)
                img_orig = img_orig.squeeze()
                img_recon = reconstructions[i].cpu().squeeze()
                
                # Use consistent visualization parameters with show_spectrogram
                axes[0,i].imshow(img_orig, cmap=cmap, origin='lower', 
                               aspect='auto', interpolation='nearest')
                axes[1,i].imshow(img_recon, cmap=cmap, origin='lower',
                               aspect='auto', interpolation='nearest')
            
            # Turn off axes for cleaner look
            axes[0,i].axis('off')
            axes[1,i].axis('off')
            
            # Add colorbar if in spectrogram mode
            if spectrogram_mode:
                plt.colorbar(axes[0,i].images[0], ax=axes[0,i], fraction=0.046, pad=0.04)
                plt.colorbar(axes[1,i].images[0], ax=axes[1,i], fraction=0.046, pad=0.04)
        
        # Add row labels
        axes[0,0].set_ylabel('Original', size='large')
        axes[1,0].set_ylabel('Reconstructed', size='large')
        
        plt.tight_layout()
        
        if display:
            plt.show()
        else:
            # Generate filename with timestamp and relevant info
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            mode = 'spec' if spectrogram_mode else 'img'
            color_info = 'color' if is_color else 'gray'
            filename = f'reconstructions_{mode}_{color_info}_{timestamp}.png'
            
            fig.savefig(save_path / filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved reconstructions to {save_path / filename}")


def visualize_latent_space(model, 
                          data_loader, 
                          device, 
                          display: bool = True, 
                          save_path: Optional[str] = None):
    """
    Visualize or save latent space distribution with PCA projection.
    
    Args:
        model: VAE model
        data_loader: DataLoader instance
        device: torch device
        display: If True, displays the plot. If False, saves to disk
        save_path: Directory to save the visualization. If None, uses '/viz_outputs'
    """
    model.eval()
    save_path = Path(save_path) if save_path else Path('/viz_outputs')
    save_path.mkdir(parents=True, exist_ok=True)
    
    z_points = []
    labels = []
    
    # Collect latent vectors
    with torch.no_grad():
        for x, y in data_loader:
            mu, _ = model.encoder(x.to(device))
            z_points.append(mu.cpu())
            labels.extend(y.numpy())
            
    # Combine all latent vectors and convert to numpy
    z_points = torch.cat(z_points, dim=0).numpy()
    labels = np.array(labels)
    
    # Apply PCA
    pca = PCA(n_components=2)
    z_points_2d = pca.fit_transform(z_points)
    
    # Calculate explained variance
    explained_var = pca.explained_variance_ratio_
    total_var = sum(explained_var)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot PCA projection
    scatter = ax1.scatter(z_points_2d[:, 0], z_points_2d[:, 1], 
                         c=labels, cmap='tab10', alpha=0.6)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.set_title(f'Latent Space PCA Projection\nTotal Explained Variance: {total_var:.3f}')
    ax1.set_xlabel(f'First PC (Var: {explained_var[0]:.3f})')
    ax1.set_ylabel(f'Second PC (Var: {explained_var[1]:.3f})')
    plt.colorbar(scatter, ax=ax1)
    
    # Plot explained variance
    n_components = len(pca.explained_variance_ratio_)
    variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(variance_ratio)
    
    ax2.plot(range(1, n_components + 1), 
             cumulative_variance_ratio, 
             'bo-', label='Cumulative Explained Variance')
    ax2.plot(range(1, n_components + 1), 
             variance_ratio, 
             'ro-', label='Individual Explained Variance')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Explained Variance Ratio')
    ax2.set_title('Explained Variance Analysis')
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    if display:
        plt.show()
    else:
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'latent_space_pca_{timestamp}.png'
        fig.savefig(save_path / filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved latent space plot to {save_path / filename}")
        
        # Save PCA components and explained variance
        np.save(save_path / f'pca_projection_{timestamp}.npy', z_points_2d)
        np.save(save_path / f'pca_components_{timestamp}.npy', pca.components_)
        
        # Save metadata
        metadata = {
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'singular_values': pca.singular_values_,
            'n_components': pca.n_components_,
            'total_variance': total_var
        }
        np.save(save_path / f'pca_metadata_{timestamp}.npy', metadata)
        
        return 