import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict, Optional
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from collections import defaultdict
from tqdm import tqdm
import json
from common.visualizations import visualize_reconstructions
from training.early_stopping import EarlyStopping
from common.seeding import get_rng_state, set_rng_state

    
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
                 grad_clip: Optional[float] = None,
                 patience: Optional[int] = None
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
            patience: Early-stopping patience in epochs, counted after KL warm-up (optional)
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
        self.patience = patience
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
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool,
                        history: Dict[str, list], stopper: Optional[EarlyStopping]) -> None:
        """Save the full training state to <name>_latest.pt (used by resume) and to <name>_best.pt if best."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'early_stopping': stopper.state_dict() if stopper else None,
            'best_val_loss': self.best_val_loss,
            'history': dict(history),
            'rng_state': get_rng_state(),
            'metrics': metrics
        }
        torch.save(checkpoint, self.save_dir / f'{self.model_name}_latest.pt')
        if is_best:
            torch.save(checkpoint, self.save_dir / f'{self.model_name}_best.pt')
        with open(self.save_dir / f'{self.model_name}_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)

    def load_checkpoint(self, history: Dict[str, list], stopper: Optional[EarlyStopping]) -> int:
        """Restore the full training state from <name>_latest.pt; returns the next epoch."""
        path = self.save_dir / f'{self.model_name}_latest.pt'
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if stopper and checkpoint['early_stopping']:
            stopper.load_state_dict(checkpoint['early_stopping'])
        self.best_val_loss = checkpoint['best_val_loss']
        history.update(checkpoint['history'])
        set_rng_state(checkpoint['rng_state'])
        self.logger.info(f"Resumed from {path} (epoch {checkpoint['epoch']}, best val {self.best_val_loss:.4f})")
        return checkpoint['epoch'] + 1

    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int = 20,
              dataset_type: str = "general",
              resume: bool = False) -> Dict[str, list]:
        """Train for up to num_epochs (optionally resuming); Ctrl+C stops after the last saved epoch."""
        history = defaultdict(list)
        stopper = EarlyStopping(self.patience) if self.patience else None
        start_epoch = self.load_checkpoint(history, stopper) if resume else 0
        try:
            for epoch in range(start_epoch, num_epochs):
                if self.run_epoch(epoch, train_loader, val_loader, history, stopper, dataset_type):
                    break
        except KeyboardInterrupt:
            self.logger.info(f"Interrupted. State up to the last finished epoch is in "
                             f"{self.save_dir / (self.model_name + '_latest.pt')}; continue with --resume")
            raise SystemExit(130)
        return dict(history)

    def run_epoch(self, epoch: int, train_loader: DataLoader, val_loader: DataLoader,
                  history: Dict[str, list], stopper: Optional[EarlyStopping], dataset_type: str) -> bool:
        """Train, validate, checkpoint and log one epoch; returns True when early stopping triggers."""
        train_metrics = self.train_epoch(train_loader, epoch)
        val_metrics = {f'val_{k}': v for k, v in self.validate(val_loader).items()}

        if self.scheduler is not None:
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['val_loss'])
            else:
                self.scheduler.step()

        for k, v in train_metrics.items():
            history[f'train_{k}'].append(v)
        for k, v in val_metrics.items():
            history[k].append(v)

        # During KL warm-up the loss is not comparable across epochs, so best model and
        # early stopping only start once beta is final.
        warmed_up = epoch + 1 >= self.kl_warmup_epochs
        is_best = warmed_up and val_metrics['val_loss'] < self.best_val_loss
        if is_best:
            self.best_val_loss = val_metrics['val_loss']
        if stopper and warmed_up:
            stopper.step(val_metrics['val_loss'])
        self.save_checkpoint(epoch, {**train_metrics, **val_metrics}, is_best, history, stopper)

        patience = f", patience {stopper.bad_checks}/{stopper.patience}" if stopper and warmed_up else ""
        self.logger.info(
            f"Epoch {epoch}: Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['val_loss']:.4f} "
            f"(recon {val_metrics['val_recon_loss']:.1f}, kl {val_metrics['val_kl_loss']:.1f}, "
            f"beta {self.current_kl_weight:.3f}, lr {self.optimizer.param_groups[0]['lr']:.2e}"
            f"{', best' if is_best else ''}{patience})")

        if epoch % 5 == 0 and self.reconstruction_save_dir:
            visualize_reconstructions(self.model, val_loader, self.device, display=False,
                                      save_path=self.reconstruction_save_dir, dataset_type=dataset_type)

        if stopper and warmed_up and stopper.should_stop:
            self.logger.info(f"Early stopping at epoch {epoch}: no improvement for {self.patience} epochs")
            return True
        return False
