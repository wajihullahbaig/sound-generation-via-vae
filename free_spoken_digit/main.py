
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
from torch import device as torch_device
from torch.utils.data import DataLoader


        
def test(test_loader,model:VAE,device:torch_device=None):   
         
    # If you have a saved model, load it
    checkpoint = torch.load('free_spoken_digit/checkpoints/vae_free_spoken_digit_20241118_171519_best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set model to evaluation mode
    model.eval()

    # Visualize reconstructions
    print("Test Set Reconstructions:")
    visualize_reconstructions(model, test_loader, device, num_images=8,display=True)

    # Visualize latent space
    print("\nTest Set Latent Space:")
    visualize_latent_space(model, test_loader, device,display=True)
    
def main( train_loader: DataLoader,validation_loader:DataLoader,model:VAE,batch_size = 8, device:torch_device = None):            
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
        num_epochs=200,
        dataset_type="spectrogram"
    )
    
   
    return history

def generate_spectrograms(processing_config:ProcessingConfig,device:torch_device = None):
    """
    Generate spectrograms from audio files.
    """
   
    processor = SpectrogramProcessor(processing_config,device=device)
    processor.create_spectrograms_and_save(
        audio_dir="C:/Users/Acer/work/git/free-spoken-digit-dataset/recordings",
        save_dir="C:/Users/Acer/work/data/free-audio-spectrogram",
    )


if __name__ == '__main__':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(111)    
    # First create the spectrorgrams, you probably need to do this once.
     # Configure preprocessing - Ensure you have correct configuratiosn    
    processing_configurations = {
    "64x256": ProcessingConfig(
        sample_rate=22050,
        duration=0.7425,      
        n_fft=2048,        
        hop_length=64,     
        n_mels=64,         
        ),
    "128x128": ProcessingConfig(
        sample_rate=22050,
        duration=0.7425,      
        n_fft=2048,        
        hop_length=128,     
        n_mels=128,         
        ),    
    "32x32": ProcessingConfig(
        sample_rate=22050,
        duration=0.7425,      
        n_fft=2048,        
        hop_length=512,     
        n_mels=32,         
        )
    }
    
    selection = "128x128"
    processing_configurations[selection].print_config()
    generate_spectrograms(processing_configurations[selection],device=device)    
    
    # Let setup data for training
    batch_size = 16 
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
    # Note - The VAE input shape depends on the shape of spectrograms
    H = processing_configurations[selection].n_mels
    W = processing_configurations[selection].time_bins
    model = VAE(
        input_shape=(H, W,1),
        conv_filters=(64,32,16),
        conv_kernels=(3, 3, 3),
        conv_strides=(1, 2, 2),
        latent_dim=16,
        dropout_rate=0.2,
        noise_factor=0.1,
        seed=42
    ).to(device)
    print(model.summary(batch_size=batch_size ))
    
    #Load history    
    history = main(train_loader,val_loader,model,batch_size=batch_size,device=device)
    #print(history)
    
    test(test_loader,model,device)

    
