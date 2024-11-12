
import json
from datetime import datetime
from data_module import MNISTDataModule
from trainer import VAETrainer
import torch
from auto_encoder import VAE
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from visualizations import visualize_reconstructions,visualize_latent_space


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup data
    data_module = MNISTDataModule(
        batch_size=128,
        train_val_split=0.9
    )
    data_module.prepare_data()
    data_module.setup()
    
        # Get a batch
    train_loader = data_module.train_dataloader()
    images, _ = next(iter(train_loader))

    print(f"Data range: [{images.min():.3f}, {images.max():.3f}]")  # Should be [0, 1]

    # Create model
    model = VAE(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_dim=2
    ).to(device)
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
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
        model_name=f'vae_mnist_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    
    # Train model
    history = trainer.train(
        train_loader=data_module.train_dataloader(),
        val_loader=data_module.val_dataloader(),
        num_epochs=50
    )
    
    return history


if __name__ == '__main__':
    # Load history
    history = main()

    
    data_module = MNISTDataModule(batch_size=128)
    data_module.prepare_data()
    data_module.setup()

    model = VAE(...).to(device)
    optimizer = optim.Adam(model.parameters())

    trainer = VAETrainer(model, optimizer, device)
    history = trainer.train(
        data_module.train_dataloader(),
        data_module.val_dataloader(),
        num_epochs=50
    )
    
    # Get the validation loader
    test_loader = data_module.test_dataloader()

    # If you have a saved model, load it
    checkpoint = torch.load('vae_mnist_20241112_120030_best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set model to evaluation mode
    model.eval()

    # Visualize reconstructions
    print("Validation Set Reconstructions:")
    visualize_reconstructions(model, test_loader, device, num_images=8,display=True)

    # Visualize latent space
    print("\nValidation Set Latent Space:")
    visualize_latent_space(model, test_loader, device,display=True)