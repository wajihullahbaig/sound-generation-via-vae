import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict, Optional
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import logging
from collections import defaultdict
from tqdm import tqdm
import json
from common.visualizations import visualize_reconstructions


    
class VAETrainer:
    """Handles training of the VAE model."""
    
    def __init__(self, 
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 save_dir: str = './checkpoints',
                 reconstruction_save_dir=None,
                 model_name: str = 'vae_mnist',
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
                 ):
        """
        Args:
            model: VAE model to train
            optimizer: Optimizer to use for training
            device: Device to train on
            save_dir: Directory to save checkpoints
            model_name: Name of the model for saving
            scheduler: Learning rate scheduler (optional)
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.save_dir = Path(save_dir)
        self.reconstruction_save_dir = reconstruction_save_dir
        self.model_name = model_name
        self.scheduler = scheduler
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize best validation loss
        self.best_val_loss = float('inf')
    
    def compute_loss(self, 
                    recon_x: torch.Tensor, 
                    x: torch.Tensor, 
                    mu: torch.Tensor, 
                    log_var: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute VAE loss (reconstruction + KL divergence).
        
        Args:
            recon_x: Reconstructed input
            x: Original input
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
            
        Returns:
            total_loss: Combined VAE loss
            metrics: Dictionary containing individual loss components
        """
        # Reconstruction loss (binary cross entropy)
        recon_loss = nn.BCELoss(reduction='sum')(
            recon_x.view(-1), 
            x.view(-1)
        )
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(
            1 + log_var - mu.pow(2) - log_var.exp()
        )
        
        # Total loss
        total_loss = recon_loss + kl_loss
        
        # Metrics for logging
        metrics = {
            'loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item()
        }
        
        return total_loss, metrics
    
    def train_epoch(self, 
                   train_loader: DataLoader, 
                   epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            metrics: Dictionary containing average losses for the epoch
        """
        self.model.train()
        total_metrics = defaultdict(float)     

        with tqdm(train_loader, desc=f'Epoch {epoch}') as pbar:
            for batch_idx, (data, _) in enumerate(pbar):
                # Move data to device
                data = data.to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass - noisy_x will be none if VAE was constructed with noise_factor = 0.0
                recon_x, mu, log_var, noisey_x = self.model(data)
                
                
                # Compute loss
                loss, metrics = self.compute_loss(recon_x, data, mu, log_var)
                
                # Backward pass
                loss.backward()
                
                # Update weights
                self.optimizer.step()
                
                # Update metrics
                for k, v in metrics.items():
                    total_metrics[k] += v
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{metrics["loss"]:.4f}',
                    'recon': f'{metrics["recon_loss"]:.4f}',
                    'kl': f'{metrics["kl_loss"]:.4f}'
                })
        
        # Calculate average metrics
        avg_metrics = {
            k: v / len(train_loader) for k, v in total_metrics.items()
        }
        
        return avg_metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            metrics: Dictionary containing average validation losses
        """
        self.model.eval()
        total_metrics = defaultdict(float)
        
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(self.device)
                # noisey_x will be None if VAE was constructed with noise_factor == 0.0
                recon_x, mu, log_var, noisey_x = self.model(data)
                loss, metrics = self.compute_loss(recon_x, data, mu, log_var)
                
                for k, v in metrics.items():
                    total_metrics[k] += v
        
        # Calculate average metrics
        avg_metrics = {
            k: v / len(val_loader) for k, v in total_metrics.items()
        }
        
        return avg_metrics
    
    def save_checkpoint(self, 
                       epoch: int, 
                       metrics: Dict[str, float], 
                       is_best: bool = False) -> None:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary containing metrics to save
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        
        # Save latest checkpoint
        checkpoint_path = self.save_dir / f'{self.model_name}_latest.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if this is the best so far
        if is_best:
            best_path = self.save_dir / f'{self.model_name}_best.pt'
            torch.save(checkpoint, best_path)
        
        # Save metrics
        metrics_path = self.save_dir / f'{self.model_name}_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int=20) -> Dict[str, list]:
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train for
            
        Returns:
            history: Dictionary containing training history
        """
        history = defaultdict(list)
        
        for epoch in range(num_epochs):
            # Train for one epoch
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update history
            for k, v in train_metrics.items():
                history[f'train_{k}'].append(v)
            for k, v in val_metrics.items():
                history[f'val_{k}'].append(v)
            
            # Check if this is the best model
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            
            # Save checkpoint
            self.save_checkpoint(epoch, 
                               {**train_metrics, **val_metrics}, 
                               is_best)
            
            # Log metrics
            self.logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}"
            )
            
            # Optional: Visualize reconstructions every N epochs
            if epoch % 5 == 0 and self.reconstruction_save_dir:
                visualize_reconstructions(self.model, val_loader, self.device, display=False,save_path=self.reconstruction_save_dir)
        
        
        return dict(history)