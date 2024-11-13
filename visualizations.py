import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import torch
from sklearn.decomposition import PCA
import numpy as np
from typing import Optional

def visualize_reconstructions(model, data_loader, device, num_images=8, 
                            display=True, save_path=None):
    """
    Visualize or save original and reconstructed images.
    
    Args:
        model: VAE model
        data_loader: DataLoader instance
        device: torch device
        num_images: Number of images to visualize
        display: If True, displays the plot. If False, saves to disk
        save_path: Directory to save the visualization. If None, uses './viz_outputs'
    """
    model.eval()
    save_path = Path(save_path) if save_path else Path('./viz_outputs')
    save_path.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        images, _ = next(iter(data_loader))
        images = images[:num_images].to(device)
        reconstructions, _, _,_ = model(images)
        
        # Create figure
        fig, axes = plt.subplots(2, num_images, figsize=(2*num_images, 4))
        for i in range(num_images):
            axes[0,i].imshow(images[i].cpu().squeeze(), cmap='gray')
            axes[0,i].axis('off')
            axes[1,i].imshow(reconstructions[i].cpu().squeeze(), cmap='gray')
            axes[1,i].axis('off')
        
        plt.tight_layout()
        
        if display:
            plt.show()
        else:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'reconstructions_{timestamp}.png'
            fig.savefig(save_path / filename)
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
        save_path: Directory to save the visualization. If None, uses './viz_outputs'
    """
    model.eval()
    save_path = Path(save_path) if save_path else Path('./viz_outputs')
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
