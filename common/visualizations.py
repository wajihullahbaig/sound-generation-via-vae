import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import torch
from sklearn.decomposition import PCA
import numpy as np
from typing import Optional

       
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
    fig_size_multiplier=(2, 4),
    dataset_type='general',  # Added parameter to specify dataset type
    filename=None
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
        dataset_type: String indicating dataset type ('mnist', 'spectrogram', or 'general')
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
                # Handle grayscale images
                img_orig = img_orig.squeeze()
                img_recon = reconstructions[i].cpu().squeeze()
                
                if spectrogram_mode or dataset_type == 'spectrogram':
                    # Spectrogram-specific visualization
                    axes[0,i].imshow(img_orig, cmap=cmap, origin='lower', 
                                   aspect='auto', interpolation='nearest')
                    axes[1,i].imshow(img_recon, cmap=cmap, origin='lower',
                                   aspect='auto', interpolation='nearest')
                else:
                    # MNIST or other image visualization (no flipping)
                    axes[0,i].imshow(img_orig, cmap=cmap, interpolation='nearest')
                    axes[1,i].imshow(img_recon, cmap=cmap, interpolation='nearest')
            
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
            filename = filename or f'reconstructions_{mode}_{color_info}_{timestamp}.png'
            
            fig.savefig(save_path / filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved reconstructions to {save_path / filename}")


def visualize_latent_space(model, 
                          data_loader, 
                          device, 
                          display: bool = True, 
                          save_path: Optional[str] = None,
                          filename: Optional[str] = None):
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
        fig.savefig(save_path / (filename or f'latent_space_pca_{timestamp}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved latent space plot to {save_path / (filename or f'latent_space_pca_{timestamp}.png')}")
        if filename:
            return
        
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

# Shared figure style for README plots (light surface, recessive axes, validated series colours)
STYLE = {'surface': '#fcfcfb', 'text': '#0b0b0b', 'muted': '#52514e', 'grid': '#e4e3df',
         'series': ['#2a78d6', '#eb6834']}


def plot_sample_grid(rows: dict, save_path: str, col_labels: Optional[list] = None, cmap: str = 'magma') -> None:
    """Grid of spectrograms: one row per model ({name: tensor (N, 1, H, W) in [0, 1]})."""
    n_cols = max(len(v) for v in rows.values())
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(1.6 * n_cols, 1.25 * len(rows) + 0.4),
                             facecolor=STYLE['surface'], squeeze=False)
    for r, (name, images) in enumerate(rows.items()):
        for c in range(n_cols):
            ax = axes[r, c]
            ax.set_xticks([]), ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if c < len(images):
                ax.imshow(images[c].squeeze().cpu(), origin='lower', aspect='auto', cmap=cmap, vmin=0, vmax=1)
            if c == 0:
                ax.set_ylabel(name, rotation=0, ha='right', va='center', fontsize=10, color=STYLE['text'])
            if r == 0 and col_labels:
                ax.set_title(col_labels[c], fontsize=10, color=STYLE['muted'])
    fig.tight_layout(w_pad=0.3, h_pad=0.4)
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=STYLE['surface'])
    plt.close(fig)


def plot_training_curves(histories: dict, save_path: str) -> None:
    """One panel per model with train and validation loss ({name: {'step', 'train_loss', 'val_loss'}})."""
    fig, axes = plt.subplots(1, len(histories), figsize=(5.2 * len(histories), 3.4), facecolor=STYLE['surface'])
    for ax, (name, h) in zip(np.atleast_1d(axes), histories.items()):
        ax.set_facecolor(STYLE['surface'])
        for key, label, color in [('train_loss', 'Train', STYLE['series'][0]),
                                  ('val_loss', 'Validation (EMA)', STYLE['series'][1])]:
            ax.plot(h['step'], h[key], color=color, linewidth=2, label=label)
            ax.annotate(label, (h['step'][-1], h[key][-1]), xytext=(6, 0), textcoords='offset points',
                        va='center', fontsize=9, color=STYLE['muted'])
        ax.set_title(name, loc='left', fontsize=12, color=STYLE['text'])
        ax.set_xlabel('Training step', color=STYLE['muted'])
        ax.set_ylabel('Loss (MSE)', color=STYLE['muted'])
        ax.grid(True, axis='y', color=STYLE['grid'], linewidth=0.8)
        ax.tick_params(colors=STYLE['muted'], length=0)
        for side in ['top', 'right', 'left']:
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color(STYLE['grid'])
        ax.legend(frameon=False, fontsize=9, labelcolor=STYLE['muted'], loc='upper right')
        ax.margins(x=0.12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=STYLE['surface'])
    plt.close(fig)


def _style_axis(ax, title: str) -> None:
    """Apply the shared recessive axis style."""
    ax.set_facecolor(STYLE['surface'])
    ax.set_title(title, loc='left', fontsize=11, color=STYLE['text'])
    ax.tick_params(colors=STYLE['muted'], length=0)
    for side in ['top', 'right', 'left']:
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(STYLE['grid'])


def plot_metric_bars(results: list, save_path: str) -> None:
    """One small panel per metric (Frechet distance, digit accuracy, speed) with a bar per model."""
    panels = [('fd', 'Fréchet distance (lower is better)', '{:.1f}'),
              ('accuracy', 'Digit accuracy (higher is better)', '{:.0%}'),
              ('ms_per_sample', 'Generation time, ms/sample (lower is better)', '{:.1f}')]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), facecolor=STYLE['surface'])
    for ax, (key, title, fmt) in zip(axes, panels):
        rows = [r for r in results if r.get(key) is not None]
        names = [r['model'].split(' (')[0] for r in rows]
        values = [r[key] for r in rows]
        colors = [STYLE['muted'] if n.startswith('Real') else STYLE['series'][0] for n in names]
        bars = ax.barh(names, values, color=colors, height=0.6)
        ax.bar_label(bars, labels=[fmt.format(v) for v in values], padding=4, fontsize=9, color=STYLE['text'])
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.margins(x=0.25)
        _style_axis(ax, title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=STYLE['surface'])
    plt.close(fig)


def plot_confusion_matrices(matrices: dict, save_path: str) -> None:
    """Requested digit vs digit heard by the classifier, one row-normalised matrix per model."""
    fig, axes = plt.subplots(1, len(matrices), figsize=(5.2 * len(matrices), 4.6), facecolor=STYLE['surface'])
    for ax, (name, matrix) in zip(np.atleast_1d(axes), matrices.items()):
        m = np.array(matrix)
        image = ax.imshow(m, cmap='Blues', vmin=0, vmax=1)
        for i in range(10):
            for j in range(10):
                if m[i, j] >= 0.05:
                    ax.text(j, i, f"{m[i, j]:.0%}".replace('%', ''), ha='center', va='center', fontsize=7,
                            color='white' if m[i, j] > 0.6 else STYLE['text'])
        ax.set_xticks(range(10)), ax.set_yticks(range(10))
        ax.set_xlabel('Digit heard by classifier', color=STYLE['muted'])
        ax.set_ylabel('Digit requested', color=STYLE['muted'])
        _style_axis(ax, f"{name}: accuracy {np.trace(m) / 10:.0%}")
        ax.spines['bottom'].set_visible(False)
    fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02, label='Share of samples')
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=STYLE['surface'])
    plt.close(fig)
