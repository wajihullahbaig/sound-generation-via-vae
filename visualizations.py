import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import torch

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
        reconstructions, _, _ = model(images)
        
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

def visualize_latent_space(model, data_loader, device, 
                          display=True, save_path=None):
    """
    Visualize or save latent space distribution.
    
    Args:
        model: VAE model
        data_loader: DataLoader instance
        device: torch device
        display: If True, displays the plot. If False, saves to disk
        save_path: Directory to save the visualization. If None, uses './visualizations'
    """
    model.eval()
    save_path = Path(save_path) if save_path else Path('./visualizations')
    save_path.mkdir(parents=True, exist_ok=True)
    
    z_points = []
    labels = []
    
    with torch.no_grad():
        for x, y in data_loader:
            mu, _ = model.encoder(x.to(device))
            z_points.append(mu.cpu())
            labels.extend(y.numpy())
            
    z_points = torch.cat(z_points, dim=0)
    
    # Create figure
    fig = plt.figure(figsize=(10, 10))
    scatter = plt.scatter(z_points[:, 0], z_points[:, 1], 
                         c=labels, cmap='tab10')
    plt.colorbar(scatter)
    plt.title('Latent Space Distribution')
    
    if display:
        plt.show()
    else:
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'latent_space_{timestamp}.png'
        fig.savefig(save_path / filename)
        plt.close(fig)
        print(f"Saved latent space plot to {save_path / filename}")
