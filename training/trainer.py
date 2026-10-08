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
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 kl_weight: float = 1.0,
                 kl_warmup_epochs: int = 0,
                 grad_clip: Optional[float] = None
                 ):
        """
        Args:
            model: VAE model to train
            optimizer: Optimizer to use for training
            device: Device to train on
            save_dir: Directory to save checkpoints
            model_name: Name of the model for saving
            scheduler: Learning rate scheduler (optional)
            kl_weight: Final weight (beta) of the KL term
            kl_warmup_epochs: Linearly ramp the KL weight from 0 to kl_weight over this many epochs
            grad_clip: Max gradient norm (optional)
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.save_dir = Path(save_dir)
        self.reconstruction_save_dir = reconstruction_save_dir
        self.model_name = model_name
        self.scheduler = scheduler
        self.kl_weight = kl_weight
        self.kl_warmup_epochs = kl_warmup_epochs
        self.grad_clip = grad_clip
        self.current_kl_weight = kl_weight
        
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
        total_loss = recon_loss + self.current_kl_weight * kl_loss
        
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
        if self.kl_warmup_epochs > 0:
            self.current_kl_weight = self.kl_weight * min(1.0, (epoch + 1) / self.kl_warmup_epochs)
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
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
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
    ''
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
              num_epochs: int=20,dataset_type="general") -> Dict[str, list]:
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
            val_metrics = {f'val_{k}': v for k, v in val_metrics.items()}
            
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()
            
            # Update history
            for k, v in train_metrics.items():
                history[f'train_{k}'].append(v)
            for k, v in val_metrics.items():
                history[k].append(v)
            
            # Check if this is the best model. During KL warm-up the loss is not
            # comparable across epochs, so only track the best once beta is final.
            warmed_up = epoch + 1 >= self.kl_warmup_epochs
            is_best = warmed_up and val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
            
            # Save checkpoint
            self.save_checkpoint(epoch, 
                               {**train_metrics, **val_metrics}, 
                               is_best)
            
            # Log metrics
            self.logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f} "
                f"(recon {val_metrics['val_recon_loss']:.1f}, kl {val_metrics['val_kl_loss']:.1f}, "
                f"beta {self.current_kl_weight:.3f}, lr {self.optimizer.param_groups[0]['lr']:.2e})"
            )
            
            # Optional: Visualize reconstructions every N epochs
            if epoch % 5 == 0 and self.reconstruction_save_dir:
                visualize_reconstructions(self.model, val_loader, self.device, display=False,save_path=self.reconstruction_save_dir,dataset_type=dataset_type)                        
        
        
        return dict(history)