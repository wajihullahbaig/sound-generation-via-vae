
from datetime import datetime
import os
from mnist_data_module import MNISTDataModule
from training.trainer import VAETrainer
import torch
from arch.auto_encoder import VAE
import torch.optim as optim
from typing import List, Tuple, Optional
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import numpy as np
from common.seeding import set_seed
from common.visualizations import visualize_reconstructions,visualize_latent_space
import pandas as pd

        
def test(test_loader):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    set_seed(111)   
    
    # Create model - set noise_factor > 0.0 to make VAE a DVAE
    model = VAE(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_dim=32,
        dropout_rate=0.1,
        noise_factor=0.2,
        seed=42
    ).to(device)
    
    # If you have a saved model, load it
    checkpoint = torch.load('mnist/checkpoints/vae_mnist_20241119_102442_best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set model to evaluation mode
    model.eval()

    # Visualize reconstructions
    print("Validation Set Reconstructions:")
    visualize_reconstructions(model, test_loader, device, num_images=8,display=False,save_path="mnist/viz_outputs",spectrogram_mode=False,color_mode='grayscale')

    # Visualize latent space
    print("\nValidation Set Latent Space:")
    visualize_latent_space(model, test_loader, device,display=False,save_path="mnist/viz_outputs")
    
def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 16
    set_seed(111)
    # Setup data
    data_module = MNISTDataModule(
        batch_size=batch_size,
        train_val_split=0.9
    )
    data_module.prepare_data()
    data_module.setup()
    
    # Get dataloaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
   
    # Create model - set noise_factor > 0.0 to make VAE a DVAE
    model = VAE(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_dim=32,
        dropout_rate=0.1,
        noise_factor=0.2,
        seed=42
    ).to(device)
    
    print(model.summary(batch_size=batch_size ))
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # Setup learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    # Create trainer
    trainer = VAETrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        save_dir='mnist/checkpoints',
        reconstruction_save_dir='mnist/viz_outputs',
        model_name=f'vae_mnist_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        scheduler=scheduler
    )
    
    # Train model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=100,
        dataset_type="mnist"
    )
    
   
    return history,test_loader


if __name__ == '__main__':
    # Load history
    history,test_loader = main()
    print(pd.DataFrame.from_dict(history).to_markdown(index=False))
    test(test_loader)

    
