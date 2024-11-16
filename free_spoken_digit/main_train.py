
from datetime import datetime
import os
from training.trainer import VAETrainer
import torch
from arch.auto_encoder import VAE
import torch.optim as optim
from typing import List, Tuple, Optional
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import torchvision.transforms as transforms
from spectrogram_processor import SpectrogramProcessor, ProcessingConfig
from spectrogram_data_module import SpectrogramDataModule
from common.visualizations import visualize_reconstructions,visualize_latent_space
from common.seeding import set_seed


        
def test(test_loader):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    set_seed(111)   
    
    model = VAE().to(device)
    
    # If you have a saved model, load it
    checkpoint = torch.load('checkpoints/vae_free_spoken_digit_20241114_152708_best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set model to evaluation mode
    model.eval()

    # Visualize reconstructions
    print("Validation Set Reconstructions:")
    visualize_reconstructions(model, test_loader, device, num_images=8,display=True,spectrogram_mode=True,cmap='magma')

    # Visualize latent space
    print("\nValidation Set Latent Space:")
    visualize_latent_space(model, test_loader, device,display=True)
    
def main():    
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 16
    set_seed(111)
    data_module = SpectrogramDataModule(
    data_dir="C:/Users/Acer/work/data/free-audio-spectrogram",
    batch_size=batch_size,
    train_val_split=0.9,
    num_workers=4
)
    
    # Setup the datasets
    data_module.setup()
    
    # Get dataloaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
             
    # Create model - set noise_factor > 0.0 to make VAE a DVAE
    model = VAE(
        input_shape=(128, 128, 1),
        conv_filters=(128, 96,64),
        conv_kernels=(3, 3, 3),
        conv_strides=(1, 2, 2),
        latent_dim=32,
        dropout_rate=0.2,
        noise_factor=0.1,
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
        save_dir='free_spoken_digit/checkpoints',
        reconstruction_save_dir='free_spoken_digit/viz_outputs',
        model_name=f'vae_free_spoken_digit_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        scheduler=scheduler
    )
    
    # Train model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=100
    )
    
   
    return history,test_loader

def generate_spectrograms():
    """
    Generate spectrograms from audio files.
    """
    # Configure preprocessing
    processing_config_128X128_mels = ProcessingConfig(
        sample_rate=22050,
        duration=0.74,
        n_fft=2048,
        hop_length=128,
        n_mels=128,
    )
    
    processor = SpectrogramProcessor(processing_config_128X128_mels)
    processor.create_spectrograms_and_save(
        audio_dir="C:/Users/Acer/work/git/free-spoken-digit-dataset/recordings",
        save_dir="C:/Users/Acer/work/data/free-audio-spectrogram",
    )


if __name__ == '__main__':
    # First create the spectrorgrams, you probably need to do this once.
    # generate_spectrograms()    
    # Load history    
    history, test_loader = main()
    # test(test_loader)

    
