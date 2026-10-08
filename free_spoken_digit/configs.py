"""
Shared configuration for training (main.py) and audio generation (create_audios.py).
Keeping these in one place ensures both scripts build identical spectrograms and models.
"""
from spectrogram_processor import ProcessingConfig

# Paths
AUDIO_DIR = "C:/Users/Precision/Onus/GitHub/free-spoken-digit-dataset/recordings"
SPECTROGRAMS_DIR = "C:/Users/Precision/Onus/Data/free-audio-spectrogram-8k"

# Spectrogram configurations. Output shape is n_mels x time_bins.
PROCESSING_CONFIGS = {
    # The FSDD recordings are 8 kHz, so work at the native rate.
    # Upsampling to 22.05 kHz leaves everything above 4 kHz empty, wasting ~1/3
    # of the mel bins on resampling artefacts.
    "8k_64x256": ProcessingConfig(
        sample_rate=8000,
        duration=0.765,       # covers ~97% of recordings
        n_fft=512,            # 64 ms window
        hop_length=24,        # 3 ms hop -> 256 frames
        n_mels=64,
        top_db=80.0,
        griffin_lim_iters=256,
        ),
}
SELECTION = "8k_64x256"

# Model configuration (input_shape is derived from the processing config)
MODEL_CONFIG = dict(
    conv_filters=(128, 64, 32),
    conv_kernels=(7, 5, 3),
    conv_strides=(1, 2, 2),
    latent_dim=128,
    dropout_rate=0.1,
    decoder_dropout_rate=0.0,
    noise_factor=0.05,
    seed=42,
)

# Training configuration
TRAIN_CONFIG = dict(
    batch_size=32,
    lr=3e-4,
    num_epochs=150,
    kl_weight=1.0,
    kl_warmup_epochs=20,
    grad_clip=5.0,
)

# Checkpoint used by test() and create_audios.py
CHECKPOINT_PATH = "free_spoken_digit/checkpoints/vae_free_spoken_digit_8k.pt"


def build_model_kwargs(selection: str = SELECTION) -> dict:
    config = PROCESSING_CONFIGS[selection]
    return dict(input_shape=(config.n_mels, config.time_bins, 1), **MODEL_CONFIG)
