# sound-generation-via-vae

Generating spoken digits from mel spectrograms with three generative models in PyTorch: a **convolutional VAE**, a **DDPM** (denoising diffusion) and **Flow Matching**. All three share one spectrogram pipeline.

Raw audio has too many samples per second to model directly. Each recording becomes a small mel spectrogram (64 × 256), the models learn that image, and a neural vocoder turns generated spectrograms back into audio.

```mermaid
flowchart LR
    A["WAV<br/>8 kHz"] --> B["Mel spectrogram<br/>64 mels × 256 frames"]
    B --> C["dB relative to clip peak<br/>clip at −80 dB → [0, 1]"]
    C --> D["VAE · DDPM · Flow Matching"]
    D -->|reconstruct / sample| E["Generated spectrogram"]
    E --> F["Neural vocoder (Vocos-style)"]
    F --> G["WAV"]
```

## Results

**[Read the full comparison report →](REPORT.md)**: findings, per-digit analysis, training dynamics, vocoder evaluation, training cost and limitations.

All models are trained on the [Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset): 3,000 recordings of the digits 0–9 from 6 speakers. The splits are 2,160 clips for training, 240 for validation and 600 for testing.

### Generation: VAE vs DDPM vs Flow Matching

<!-- results:generation -->
| Model | Fréchet distance ↓ | Digit accuracy ↑ | ms / sample ↓ |
|---|---|---|---|
| Real recordings (reference) | 0.17 | 98.7% | — |
| VAE (prior samples) | 16.26 | — | 1.2 |
| DDPM | 6.39 | 95.1% | 218.6 |
| Flow Matching | **4.13** | 99.9% | 229.8 |
<!-- /results:generation -->

- **Fréchet distance:** distance between generated and real test spectrograms in the feature space of a digit classifier trained on real data (as in FID and FAD). Lower is better.
- **Digit accuracy:** how often that classifier hears the digit that was requested. The VAE is unconditional, so it has no score here.
- **Perceived audio quality:** a blind listening test (30 clips per model, rated 1–5) found **no significant difference**: DDPM 4.00 vs Flow Matching 4.13 (p = 0.59). See the [report](REPORT.md#6-blind-listening-test).
- **Reference row:** compares real training clips with real test clips. It shows the best achievable Fréchet distance and the classifier's own accuracy on real test recordings.
- **Speed:** measured on an RTX 5000 in batches of 100. DDPM and Flow Matching both use 100 network evaluations per sample, plus classifier-free guidance (scale 2).

One sample per digit (uncurated; the VAE row is random digits):

![Generated samples](assets/generated_samples.png)

![Metric comparison](assets/generative_metrics.png)

The confusion matrices show which digit each model actually produced for every requested digit:

![Digit confusion](assets/digit_confusion.png)

![Training curves](assets/generative_training.png)

### VAE reconstruction

<!-- results:vae -->
| VAE version | Spectrogram error on speech (dB) ↓ |
|---|---|
| v1: 22.05 kHz, per-file min/max normalisation | 4.15 |
| v2: 8 kHz, dataset-wide dB range | 2.61 |
| **v3 (current): 8 kHz, per-clip peak normalisation** | **3.69** |

_The rows are not directly comparable. v1 and v2 clipped quiet speakers' low-level detail to the floor, which is trivially easy to reconstruct. v3 keeps the full 80 dB below each clip's own peak, so the same 60 dB "speech" region now includes much more quiet detail. The v3 number is the honest baseline for future changes._
<!-- /results:vae -->

![Reconstructions](assets/reconstructions.png)

![Latent space](assets/latent_space.png)

### Spectrogram → audio: Griffin-Lim vs neural vocoder

A spectrogram stores only *how much* energy each frequency has, not its phase. Griffin-Lim **guesses** the phase iteratively, which produces the metallic, robotic sound. The vocoder **learns** the phase (and the detail lost by the 64-bin mel) from the real recordings. The table scores both on test-set spectrograms against the true recordings:

<!-- results:vocoder -->
| Spectrogram → audio | PESQ (narrowband, 1–4.5) ↑ | STOI (0–1) ↑ | ms / clip (CPU) ↓ |
|---|---|---|---|
| Griffin-Lim (256 iterations) | 1.97 | 0.416 | 336.0 |
| Neural vocoder | 3.42 | 0.484 | 10.2 |
<!-- /results:vocoder -->

- **PESQ:** ITU-T P.862 perceived speech quality, narrowband mode for 8 kHz.
- **STOI:** intelligibility.

The figure shows one test clip at a finer STFT than the models use. Griffin-Lim's phase errors appear as smeared, wavy harmonics and noise in the silent tail.

![Vocoder comparison](assets/vocoder_comparison.png)

### Audio samples

Listen in [`free_spoken_digit/samples/`](free_spoken_digit/samples):

| Folder | Contents |
|---|---|
| `ddpm/`, `flow/` | One generated clip per digit (`digit_0.wav` … `digit_9.wav`) |
| `sampled_from_prior/` | VAE samples decoded from random latents z ~ N(0, I) |
| `generated/` | VAE reconstructions of real clips |
| `original/` | The original spectrograms converted straight back to audio, the upper bound on quality |
| `vocoder_comparison/` | Test clips as recording / Griffin-Lim / vocoder |

## Architectures

### VAE

```mermaid
flowchart TB
    subgraph ENC ["Encoder"]
        direction LR
        X["Spectrogram<br/>1×64×256"] --> E1["Conv 7×7, s1<br/>128×64×256"]
        E1 --> E2["Conv 5×5, s2<br/>64×32×128"]
        E2 --> E3["Conv 3×3, s2<br/>32×16×64"]
        E3 --> MU["μ, log σ²<br/>2 × 128"]
        MU --> Z(("z = μ + σ·ε"))
    end
    subgraph DEC ["Decoder"]
        direction LR
        D0["Linear<br/>32×16×64"] --> D1["ConvT 3×3, s2<br/>64×32×128"]
        D1 --> D2["ConvT 5×5, s2<br/>128×64×256"]
        D2 --> D3["ConvT 7×7, s1 + sigmoid<br/>1×64×256"]
    end
    ENC --> DEC
```

Each encoder block is Conv → ReLU → BatchNorm → Dropout. Training adds Gaussian noise to the input (σ = 0.05, i.e. a denoising VAE) and uses a BCE reconstruction loss plus a KL term warmed up over 20 epochs. At inference the model decodes μ directly.

### U-Net (shared by DDPM and Flow Matching)

```mermaid
flowchart TB
    subgraph cond ["Conditioning"]
        T["time t ∈ [0, 1]"] --> TE["Sinusoidal → MLP (256)"]
        Y["digit 0–9 or ∅"] --> YE["Embedding (256)"]
        TE --> S(("+"))
        YE --> S
    end
    IN["Noisy spectrogram<br/>1×64×256"] --> PU["PixelUnshuffle 2×2<br/>4×32×128"]
    PU --> ST["Conv 3×3 → 64×32×128"]
    ST --> R0["ResBlock 64 · 32×128"]
    R0 --> R1["↓ ResBlock 128 · 16×64"]
    R1 --> R2["↓ ResBlock 192 · 8×32"]
    R2 --> R3["↓ ResBlock 256 · 4×16"]
    R3 --> M["ResBlock → Self-attention → ResBlock<br/>256 · 4×16"]
    M --> U3["ResBlock 192 · 4×16 ↑"]
    U3 --> U2["ResBlock 128 · 8×32 ↑"]
    U2 --> U1["ResBlock 64 · 16×64 ↑"]
    U1 --> U0["ResBlock 64 · 32×128"]
    U0 --> PS["Conv → 4×32×128<br/>PixelShuffle → 1×64×256"]
    R3 -. skip .-> U3
    R2 -. skip .-> U2
    R1 -. skip .-> U1
    R0 -. skip .-> U0
    S -. "added in every ResBlock" .-> M
```

- **ResBlock:** GroupNorm → SiLU → Conv, add the embedding, then GroupNorm → SiLU → Dropout → Conv, plus a residual connection. Total 8.4M parameters.
- **Pixel-unshuffle:** folds 2×2 pixel patches into channels, so the network runs at 32×128 while still predicting every pixel. This is about 3.7× faster than full resolution.
- **Classifier-free guidance:** labels are dropped to ∅ 10% of the time during training. Sampling uses `uncond + 2·(cond − uncond)`.

### DDPM vs Flow Matching

Both wrap the same U-Net and differ only in what it learns to predict, and how samples are drawn.

```mermaid
flowchart LR
    subgraph DDPM ["DDPM: predicts the noise"]
        direction TB
        A1["Add noise at step t ∈ {0…999}<br/>x_t = √ᾱ_t·x + √(1−ᾱ_t)·ε<br/>(cosine schedule)"] --> A2["U-Net(x_t, t, digit) → ε_pred"]
        A2 --> A3["Loss = ‖ε − ε_pred‖²"]
        A3 -.->|after training| A4["Sample: 100 ancestral steps<br/>N(0, I) → spectrogram"]
    end
    subgraph FM ["Flow Matching: predicts the velocity"]
        direction TB
        B1["Interpolate at t ∈ (0, 1), logit-normal<br/>x_t = (1−t)·noise + t·x<br/>(straight path)"] --> B2["U-Net(x_t, t, digit) → v_pred"]
        B2 --> B3["Loss = ‖(x − noise) − v_pred‖²"]
        B3 -.->|after training| B4["Sample: Heun ODE, 50 steps<br/>N(0, I) → spectrogram"]
    end
    DDPM ~~~ FM
```

### Neural vocoder

```mermaid
flowchart LR
    M["Mel spectrogram<br/>64 × 256 frames"] --> E["Conv1d 7 → 256"]
    E --> B["8 × ConvNeXt block<br/>depthwise conv 7 · LayerNorm · MLP 768"]
    B --> H["Linear → magnitude + phase<br/>65 bins per frame"]
    H --> I["Inverse STFT<br/>n_fft 128 · hop 24"]
    I --> W["Waveform<br/>6,120 samples"]
    W -. "training only" .-> D["Multi-resolution spectrogram<br/>discriminators"]
```

The vocoder runs at frame rate (256 steps per clip) and produces the waveform with a single inverse FFT, so it needs no sample-by-sample upsampling. It has 3.3M parameters and is trained on real (spectrogram, waveform) pairs from the training split with:
- mel L1 loss;
- multi-resolution STFT loss;
- adversarial and feature-matching losses after a 2k-step warm-up.

### Evaluation classifier

```mermaid
flowchart LR
    X["Spectrogram 1×64×256"] --> C["4 × (Conv 3×3 → BN → ReLU → MaxPool)<br/>32 → 64 → 128 → 128"]
    C --> P["Global avg pool<br/>128-d features"]
    P --> H["Linear → 10 digits"]
    P -. "Fréchet distance" .-> FD["vs real test features"]
    H -. "digit accuracy" .-> ACC["vs requested digit"]
```

## Usage

```bash
pip install -r requirements.txt
export PYTHONPATH=.        # run everything from the repo root
```

Set the dataset paths and all hyperparameters in [`free_spoken_digit/configs.py`](free_spoken_digit/configs.py). Every VS Code launch entry is in [`.vscode/launch.json`](.vscode/launch.json).

Run everything end to end (VAE → DDPM → Flow Matching → comparison report). In VS Code, this is the **Pipeline: all methods + report** launch entry:

```bash
python free_spoken_digit/run_pipeline.py                     # all stages
python free_spoken_digit/run_pipeline.py --from ddpm --resume  # continue an interrupted stage, then the rest
python free_spoken_digit/run_pipeline.py --only vocoder vae_audio samples report   # selected stages
```

Or run the stages one by one:

```bash
python free_spoken_digit/main.py --generate-spectrograms                  # once: audio -> spectrograms
python free_spoken_digit/main.py --train                                  # VAE (early stopping)
python free_spoken_digit/main.py --test                                   # VAE figures -> assets/
python free_spoken_digit/train_vocoder.py --train --evaluate              # vocoder + Griffin-Lim comparison
python free_spoken_digit/create_audios.py                                 # VAE .wav samples
python free_spoken_digit/train_generative.py --method ddpm --train --sample
python free_spoken_digit/train_generative.py --method flow --train --sample
python free_spoken_digit/evaluate_generative.py                           # comparison table + figures
```

**Stopping and resuming.** Press Ctrl+C at any time. The VAE keeps its state from the last finished epoch, and DDPM and Flow Matching save the current step. Add `--resume` to continue exactly where training stopped: model, EMA, optimizer, learning-rate schedule, early-stopping counter, history and random-number state are all restored. Resume state lives in `free_spoken_digit/checkpoints/*_resume.pt` and `vae_free_spoken_digit_latest.pt`.

Every run logs to the console and to `logs/<run>_<timestamp>.log`. The logs of the runs behind these results are in [`logs/`](logs). Pretrained weights for every model are in `free_spoken_digit/checkpoints/`.

## Design notes

- **Native 8 kHz audio.** The recordings are 8 kHz. Resampling to 22.05 kHz would leave every mel bin above 4 kHz empty.
- **Per-clip peak normalization.** dB values are taken relative to each clip's own peak and clipped at −80 dB. Recording gain varies by about 30 dB between speakers, and the output audio is peak-normalized anyway, so absolute level carries no information. Because every clip shares the same [−80, 0] dB range, any generated spectrogram can be inverted without per-file statistics.
- **Early stopping everywhere.**
  - VAE: patience of 15 epochs after KL warm-up.
  - DDPM and Flow Matching: 5 validations (10k steps) on the EMA model, capped at 60k steps.
  
  Only the best checkpoint is kept.
- **Training efficiency.** The whole dataset sits on the GPU, with mixed precision, EMA weights (decay 0.9995) and gradient clipping.
- **Reproducibility.** Seeds are fixed, cuDNN is deterministic, and all settings live in `configs.py`.
- **Neural vocoder instead of Griffin-Lim.** Mel spectrograms discard phase. Griffin-Lim's estimate sounds metallic; the vocoder learns phase from real speech. All audio is rendered with it, and Griffin-Lim is the fallback when no vocoder checkpoint exists. Rendering runs on the CPU in about 16 ms per clip, versus about 550 ms for Griffin-Lim.

## Project layout

```
arch/               VAE, U-Net, vocoder and evaluation-classifier networks
generative/         DDPM and Flow Matching (loss + sampler)
training/           VAE, generative and vocoder trainers, early stopping
common/             Seeding, logging, audio ops, metrics and plotting helpers
free_spoken_digit/  Spectrogram pipeline, configs, training / sampling / evaluation scripts
mnist/              MNIST example of the VAE
logs/               Training logs of the published models
```

## Contributors

- [wajihullahbaig](https://github.com/wajihullahbaig)
- [Claude](https://claude.com/claude-code) (Anthropic): spectrogram pipeline and normalization, DDPM and Flow Matching, neural vocoder, evaluation, training fixes

## Acknowledgements

The VAE part was inspired by Valerio Velardo's [Generating Sound with Neural Networks](https://www.youtube.com/watch?v=Ey8IZQl_lKs&list=PL-wATfeyAMNpEyENTc-tVH5tfLGKtSWPp) series.
