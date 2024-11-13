
from datetime import datetime
import os
from trainer import VAETrainer
import torch
from auto_encoder import VAE
import torch.optim as optim
from typing import List, Tuple, Optional
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import numpy as np
import torchvision.transforms as transforms
from audio_preprocessor import SpectrogramPreprocessor, PreprocessingConfig
from spectrogram_data_module import SpectrogramDataModule
from visualizations import visualize_reconstructions,visualize_latent_space

def set_seed(seed: Optional[int] = 42) -> None:
    """
    Set all random seeds for reproducibility.
    
    Args:
        seed: Integer seed for reproducibility. If None, seeds are not set.
    """
    if seed is not None:
        # Python random
        random.seed(seed)
        
        # Numpy
        np.random.seed(seed)
        
        # PyTorch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Set CUDA backend to deterministic mode
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Set Python hash seed
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        print(f"Random seed set to {seed} for reproducibility")
        
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
    checkpoint = torch.load('./checkpoints/vae_mnist_20241113_091742_best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set model to evaluation mode
    model.eval()

    # Visualize reconstructions
    print("Validation Set Reconstructions:")
    visualize_reconstructions(model, test_loader, device, num_images=8,display=True)

    # Visualize latent space
    print("\nValidation Set Latent Space:")
    visualize_latent_space(model, test_loader, device,display=True)
    
def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    set_seed(111)
    data_module = SpectrogramDataModule(
    data_dir="C:/Users/Acer/work/data/free-audio-spectrogram",
    batch_size=256,
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
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_dim=32,
        dropout_rate=0.1,
        noise_factor=0.2,
        seed=42
    ).to(device)
    
    print(model.summary(batch_size=256 ))
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
        save_dir='./checkpoints',
        model_name=f'vae_audio_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
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
    preprocess_config = PreprocessingConfig(
        sample_rate=22050,
        duration=0.74,
        n_fft=512,
        hop_length=256,
        n_mels=64,
        normalize=True
    )

    # Create preprocessor and run - note that the preprocessor resizes the mel spectrogram from 64x64 to 28x28. This is to address less then a second of data
    preprocessor = SpectrogramPreprocessor(preprocess_config)
    preprocessor.process_and_save(
        audio_dir="C:/Users/Acer/work/git/free-spoken-digit-dataset/recordings",
        save_dir="C:/Users/Acer/work/data/free-audio-spectrogram"
    )


if __name__ == '__main__':
    # First create the spectrorgrams, you probably need to do this once.
    #generate_spectrograms()    
    # Load history    
    history, test_loader = main()
    test(test_loader)

    
