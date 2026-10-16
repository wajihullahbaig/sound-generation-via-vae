"""Train DDPM / Flow Matching on mel spectrograms and render one audio sample per digit."""
import argparse
import logging
import os
import torch
import torchaudio
import configs
from common.logging_utils import setup_logging
from common.seeding import set_seed
from generative import build_method
from spectrogram_data_module import SpectrogramDataModule
from audio_renderer import build_processor
from spectrogram_processor import load_db_range
from training.generative_trainer import GenerativeTrainer


def sample_shape() -> tuple:
    """Spectrogram shape (channels, mels, frames) of the selected processing config."""
    config = configs.PROCESSING_CONFIGS[configs.SELECTION]
    return (1, config.n_mels, config.time_bins)


def load_method(name: str, device: torch.device) -> torch.nn.Module:
    """Build a method and load its trained EMA weights."""
    method = build_method(name, sample_shape(), **configs.UNET_CONFIG).to(device)
    checkpoint = torch.load(configs.GENERATIVE_CHECKPOINTS[name], map_location=device, weights_only=True)
    method.model.load_state_dict(checkpoint['model_state_dict'])
    return method.eval()


def generate(method: torch.nn.Module, name: str, labels: torch.Tensor, batch_size: int = 100) -> torch.Tensor:
    """Sample spectrograms in [0, 1] for the given digit labels."""
    batches = [method.sample(labels[i:i + batch_size], **configs.SAMPLER_CONFIG[name])
               for i in range(0, len(labels), batch_size)]
    return (torch.cat(batches) + 1) / 2


def train(name: str, device: torch.device, resume: bool = False) -> None:
    """Train the selected method on the train split, validating on the val split."""
    data = SpectrogramDataModule(configs.SPECTROGRAMS_DIR, num_workers=0)
    data.setup()
    # The models work in [-1, 1]; spectrograms are stored in [0, 1]
    train_set, val_set = [(x * 2 - 1, y) for x, y in (data.load_tensors(data.train_dataset, device),
                                                       data.load_tensors(data.val_dataset, device))]

    cfg = configs.GENERATIVE_TRAIN_CONFIG
    method = build_method(name, sample_shape(), **configs.UNET_CONFIG)
    logging.info(f"{name}: {sum(p.numel() for p in method.parameters()) / 1e6:.1f}M parameters")
    trainer = GenerativeTrainer(method, device, lr=cfg['lr'], warmup_steps=cfg['warmup_steps'],
                                ema_decay=cfg['ema_decay'], grad_clip=cfg['grad_clip'])
    trainer.fit(train_set, val_set, steps=cfg['steps'], batch_size=cfg['batch_size'], eval_every=cfg['eval_every'],
                save_path=configs.GENERATIVE_CHECKPOINTS[name], patience=cfg['patience'], log_every=cfg['log_every'], resume=resume)


def render_audio(name: str, device: torch.device, out_dir: str = "free_spoken_digit/samples") -> None:
    """Generate one sample per digit and write it as a .wav file."""
    config = configs.PROCESSING_CONFIGS[configs.SELECTION]
    processor = build_processor()
    db_min, db_max = load_db_range(configs.SPECTROGRAMS_DIR)
    spectrograms = generate(load_method(name, device), name, torch.arange(10, device=device)).cpu()

    out_dir = os.path.join(out_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    for digit, spectrogram in enumerate(spectrograms):
        audio = processor.create_audio_from_spectrogram(spectrogram, db_min, db_max)
        torchaudio.save(os.path.join(out_dir, f"digit_{digit}.wav"), audio, config.sample_rate)
    logging.info(f"Wrote 10 samples to {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['ddpm', 'flow'], required=True)
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--resume', action='store_true', help='Continue training from the resume checkpoint')
    parser.add_argument('--sample', action='store_true', help='Write one .wav per digit')
    args = parser.parse_args()

    setup_logging(f"{args.method}_{'train' if args.train else 'sample'}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(111)
    if args.train:
        train(args.method, device, resume=args.resume)
    if args.sample:
        render_audio(args.method, device)
