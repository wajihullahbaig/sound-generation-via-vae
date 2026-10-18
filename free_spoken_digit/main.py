from training.trainer import VAETrainer
import torch
from arch.auto_encoder import VAE
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from spectrogram_processor import SpectrogramProcessor, ProcessingConfig, load_db_range
from spectrogram_data_module import SpectrogramDataModule
from common.visualizations import visualize_reconstructions,visualize_latent_space
from common.seeding import set_seed
from common.logging_utils import setup_logging
from torch import device as torch_device
from torch.utils.data import DataLoader
import argparse
import json
import logging
import configs


@torch.no_grad()
def reconstruction_error_db(model: VAE, loader: DataLoader, device: torch_device) -> dict:
    """Mean absolute reconstruction error in dB, over all bins and over speech bins (within 60 dB of the clip peak)."""
    db_min, db_max = load_db_range(configs.SPECTROGRAMS_DIR)
    scale = db_max - db_min
    errors, speech = [], []
    for x, _ in loader:
        x = x.to(device)
        err = (model(x)[0] - x).abs() * scale
        errors.append(err.flatten())
        speech.append(err[x > 1 - 60 / scale])
    return {'mae_db_all': torch.cat(errors).mean().item(), 'mae_db_speech': torch.cat(speech).mean().item()}


def test(test_loader,model:VAE,device:torch_device=None,checkpoint_path:str=configs.CHECKPOINT_PATH):
    """Score test-set reconstructions and save the VAE figures and metrics to assets/."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    metrics = reconstruction_error_db(model, test_loader, device)
    logging.info(f"Test reconstruction error: {metrics['mae_db_speech']:.2f} dB (speech), "
                 f"{metrics['mae_db_all']:.2f} dB (all bins)")
    with open('assets/vae_results.json', 'w') as f:
        json.dump({'epoch': checkpoint.get('epoch'), **metrics}, f, indent=2)

    visualize_reconstructions(model, test_loader, device, num_images=10, display=False, save_path='assets',
                              dataset_type="spectrogram", filename='reconstructions.png')
    visualize_latent_space(model, test_loader, device, display=False, save_path='assets', filename='latent_space.png')


def main( train_loader: DataLoader,validation_loader:DataLoader,model:VAE, device:torch_device = None, resume: bool = False):            
    train_config = configs.TRAIN_CONFIG
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=train_config['lr'])
    # Setup learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=8
    )
    # Create trainer
    model_name = 'vae_free_spoken_digit'   # fixed name so --resume finds <name>_latest.pt
    trainer = VAETrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        save_dir='free_spoken_digit/checkpoints',
        reconstruction_save_dir=None,
        model_name=model_name,
        scheduler=scheduler,
        kl_weight=train_config['kl_weight'],
        kl_warmup_epochs=train_config['kl_warmup_epochs'],
        grad_clip=train_config['grad_clip'],
        patience=train_config['patience']
    )
    
    # Train model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=validation_loader,
        num_epochs=train_config['num_epochs'],
        dataset_type="spectrogram",
        resume=resume
    )
    
    # Export the best weights (without optimizer state) to the path test()/create_audios.py load from
    best = torch.load(f'free_spoken_digit/checkpoints/{model_name}_best.pt', map_location='cpu')
    torch.save({'epoch': best['epoch'], 'metrics': best['metrics'],
                'model_state_dict': best['model_state_dict']}, configs.CHECKPOINT_PATH)
    logging.info(f"Best weights exported to {configs.CHECKPOINT_PATH}")
   
    return history

def generate_spectrograms(processing_config:ProcessingConfig,device:torch_device = None):
    """
    Generate spectrograms from audio files.
    """
   
    processor = SpectrogramProcessor(processing_config,device=device)
    processor.create_spectrograms_and_save(
        audio_dir=configs.AUDIO_DIR,
        save_dir=configs.SPECTROGRAMS_DIR,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-spectrograms', action='store_true',
                        help='Create spectrograms from audio (needed once per processing config)')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--resume', action='store_true', help='Continue training from the latest checkpoint')
    parser.add_argument('--test', action='store_true', help='Visualise reconstructions and latent space')
    args = parser.parse_args()
    if not (args.generate_spectrograms or args.train or args.test):
        args.test = True
    
    setup_logging('vae_train' if args.train else 'vae')
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(111)    
    
    # Configure preprocessing in configs.py - training and create_audios.py share it
    processing_config = configs.PROCESSING_CONFIGS[configs.SELECTION]
    processing_config.print_config()
    if args.generate_spectrograms:
        # Audio transforms are cheap; CPU also avoids CUDA STFT/NVRTC issues on some setups
        generate_spectrograms(processing_config,device=torch.device('cpu'))    
    
    # Let setup data for training
    data_module = SpectrogramDataModule(
    data_dir=configs.SPECTROGRAMS_DIR,
    batch_size=configs.TRAIN_CONFIG['batch_size'],
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
    model = VAE(**configs.build_model_kwargs()).to(device)
    
    if args.train:
        history = main(train_loader,val_loader,model,device=device,resume=args.resume)
    
    if args.test:
        test(test_loader,model,device)
