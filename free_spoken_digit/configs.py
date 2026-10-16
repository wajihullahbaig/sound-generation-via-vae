"""
Shared configuration for training (main.py) and audio generation (create_audios.py).
Keeping these in one place ensures both scripts build identical spectrograms and models.
"""
from spectrogram_processor import ProcessingConfig

# Paths
AUDIO_DIR = "C:/Users/Precision/Onus/GitHub/free-spoken-digit-dataset/recordings"
SPECTROGRAMS_DIR = "C:/Users/Precision/Onus/Data/free-audio-spectrogram-8k-peaknorm"

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
    patience=15,           # epochs without val improvement (after KL warm-up) before stopping
)

# Checkpoint used by test() and create_audios.py
CHECKPOINT_PATH = "free_spoken_digit/checkpoints/vae_free_spoken_digit_8k.pt"


# Diffusion (DDPM) and Flow Matching share one class-conditional U-Net
UNET_CONFIG = dict(
    base_channels=64,
    channel_mults=(1, 2, 3, 4),
    num_classes=10,
    dropout=0.1,
    p_uncond=0.1,          # label dropout for classifier-free guidance
    patch_size=2,          # U-Net runs at 32x128 (pixel-unshuffle), ~3.7x faster than full resolution
)
GENERATIVE_TRAIN_CONFIG = dict(
    steps=60_000,          # upper bound; early stopping usually ends sooner
    batch_size=64,
    lr=2e-4,
    warmup_steps=1000,
    ema_decay=0.9995,
    grad_clip=1.0,
    eval_every=2000,
    patience=5,            # validations without improvement before stopping
    log_every=500,
)
# Both samplers use 100 network evaluations per sample (Heun = 2 per step)
SAMPLER_CONFIG = {
    'ddpm': dict(steps=100, guidance=2.0, eta=1.0),
    'flow': dict(steps=50, guidance=2.0),
}
GENERATIVE_CHECKPOINTS = {
    'ddpm': "free_spoken_digit/checkpoints/ddpm_8k.pt",
    'flow': "free_spoken_digit/checkpoints/flow_matching_8k.pt",
}
CLASSIFIER_CHECKPOINT = "free_spoken_digit/checkpoints/digit_classifier_8k.pt"

# Neural vocoder (Vocos-style): mel spectrogram -> waveform, replacing Griffin-Lim
VOCODER_CONFIG = dict(
    n_mels=64,
    dim=256,
    hidden=768,
    layers=8,
    n_fft=128,             # synthesis STFT; hop must equal the spectrogram hop
    hop_length=24,
)
VOCODER_TRAIN_CONFIG = dict(
    steps=40_000,          # upper bound; early stopping usually ends sooner
    batch_size=32,
    crop_frames=64,        # random 0.19 s crops: fast steps, many distinct examples
    lr=5e-4,
    gan_start=2000,        # reconstruction-only warm-up before adversarial training
    lambda_mel=45.0,
    lambda_fm=2.0,
    eval_every=1000,
    patience=6,
    log_every=200,
)
VOCODER_CHECKPOINT = "free_spoken_digit/checkpoints/vocoder_8k.pt"


def build_model_kwargs(selection: str = SELECTION) -> dict:
    config = PROCESSING_CONFIGS[selection]
    return dict(input_shape=(config.n_mels, config.time_bins, 1), **MODEL_CONFIG)
