"""Spectrogram-to-audio rendering shared by all scripts: neural vocoder when trained, Griffin-Lim otherwise."""
import logging
import os
import torch
import configs
from arch.vocoder import Vocoder
from spectrogram_processor import SpectrogramProcessor


def load_vocoder(device: torch.device = torch.device('cpu')) -> Vocoder:
    """Load the trained vocoder, or return None if it has not been trained yet."""
    if not os.path.exists(configs.VOCODER_CHECKPOINT):
        logging.info("No vocoder checkpoint found: rendering audio with Griffin-Lim")
        return None
    vocoder = Vocoder(**configs.VOCODER_CONFIG).to(device)
    vocoder.load_state_dict(torch.load(configs.VOCODER_CHECKPOINT, map_location=device, weights_only=True)['model_state_dict'])
    logging.info(f"Rendering audio with the neural vocoder ({configs.VOCODER_CHECKPOINT})")
    return vocoder.eval()


def build_processor(use_vocoder: bool = True) -> SpectrogramProcessor:
    """Spectrogram processor on CPU (cheap, avoids CUDA STFT issues), with the vocoder when available."""
    config = configs.PROCESSING_CONFIGS[configs.SELECTION]
    return SpectrogramProcessor(config, torch.device('cpu'), vocoder=load_vocoder() if use_vocoder else None)
