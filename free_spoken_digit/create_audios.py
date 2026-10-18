"""Render standardized VAE audio samples: real test clips, VAE reconstructions (digits 0-9), and 10 prior samples."""
from pathlib import Path
import torch
import torchaudio
import configs
from arch.auto_encoder import VAE
from audio_renderer import build_processor
from common.logging_utils import setup_logging
from common.seeding import set_seed
from spectrogram_data_module import SpectrogramDataModule
from spectrogram_processor import load_db_range

SAMPLES_DIR = Path("free_spoken_digit/samples")


def load_real_waveform(stem: str, sample_rate: int, total_samples: int) -> torch.Tensor:
    """Load, trim/pad and peak-normalise a raw test recording to match rendered clip formatting."""
    wav, sr = torchaudio.load(str(Path(configs.AUDIO_DIR) / f"{stem}.wav"))
    if sr != sample_rate:
        wav = torchaudio.transforms.Resample(sr, sample_rate)(wav)
    wav = wav.mean(0, keepdim=True)[:, :total_samples]
    if wav.shape[1] < total_samples:
        wav = torch.nn.functional.pad(wav, (0, total_samples - wav.shape[1]))
    return wav * (0.9 / wav.abs().max().clamp(min=1e-8))


def main() -> None:
    """Write 10 test clips per folder: samples/real/, samples/vae_recon/, and samples/vae_prior/."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(111)
    config = configs.PROCESSING_CONFIGS[configs.SELECTION]
    sp = build_processor()
    db_min, db_max = load_db_range(configs.SPECTROGRAMS_DIR)

    data = SpectrogramDataModule(configs.SPECTROGRAMS_DIR, num_workers=0)
    data.setup()
    test_subset = data.test_dataset

    model = VAE(**configs.build_model_kwargs()).to(device)
    checkpoint = torch.load(configs.CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    for sub in ('real', 'vae_recon', 'vae_prior'):
        (SAMPLES_DIR / sub).mkdir(parents=True, exist_ok=True)

    # 1. One test clip per digit (0..9): actual recording and its VAE reconstruction
    for digit in range(10):
        idx_in_subset = next(i for i in range(len(test_subset)) if test_subset[i][1] == digit)
        spec, _ = test_subset[idx_in_subset]
        stem = test_subset.dataset.file_paths[test_subset.indices[idx_in_subset]].stem

        real_wav = load_real_waveform(stem, config.sample_rate, config.total_samples)
        torchaudio.save(str(SAMPLES_DIR / 'real' / f'digit_{digit}.wav'), real_wav, config.sample_rate)

        with torch.no_grad():
            recon_spec, _, _, _ = model(spec.unsqueeze(0).to(device))
        recon_wav = sp.create_audio_from_spectrogram(recon_spec.squeeze(0).cpu(), db_min, db_max)
        torchaudio.save(str(SAMPLES_DIR / 'vae_recon' / f'digit_{digit}.wav'), recon_wav.cpu(), config.sample_rate)

    # 2. Ten unconditional prior samples z ~ N(0, I)
    with torch.no_grad():
        z = torch.randn(10, model.latent_dim, device=device)
        prior_specs = model.decoder(z).cpu()
    for i, prior_spec in enumerate(prior_specs):
        prior_wav = sp.create_audio_from_spectrogram(prior_spec, db_min, db_max)
        torchaudio.save(str(SAMPLES_DIR / 'vae_prior' / f'sample_{i}.wav'), prior_wav.cpu(), config.sample_rate)

    print(f"Saved standardized audio samples to {SAMPLES_DIR}/{{real,vae_recon,vae_prior}}")


if __name__ == "__main__":
    setup_logging('vae_audio')
    main()
