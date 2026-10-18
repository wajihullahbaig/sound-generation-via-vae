"""Train the neural vocoder (mel -> waveform) and compare it with Griffin-Lim on the test set."""
import argparse
import json
import logging
import os
import time
from pathlib import Path
import numpy as np
import torch
import torchaudio
from pesq import pesq
from pystoi import stoi
import configs
from arch.vocoder import MultiResolutionDiscriminator, Vocoder
from audio_renderer import load_vocoder
from common.audio import MelSpectrogram, normalize_mel
from common.logging_utils import setup_logging
from common.seeding import set_seed
from common.visualizations import plot_audio_comparison
from report import update_readme, vocoder_table
from spectrogram_data_module import SpectrogramDataModule
from spectrogram_processor import SpectrogramProcessor
from training.vocoder_trainer import VocoderTrainer

CONFIG = configs.PROCESSING_CONFIGS[configs.SELECTION]


def build_mel() -> MelSpectrogram:
    """Mel transform matching the dataset spectrograms."""
    return MelSpectrogram(CONFIG.sample_rate, CONFIG.n_fft, CONFIG.hop_length, CONFIG.n_mels, CONFIG.f_min, CONFIG.f_max)


def load_split(subset, mel: MelSpectrogram, device: torch.device) -> tuple:
    """Load a split's recordings as (normalised mel, loudness-normalised waveform) tensors on the device."""
    wavs = []
    for i in subset.indices:
        wav, sr = torchaudio.load(str(Path(configs.AUDIO_DIR) / f"{subset.dataset.file_paths[i].stem}.wav"))
        assert sr == CONFIG.sample_rate, f"expected {CONFIG.sample_rate} Hz audio, got {sr}"
        wav = wav.mean(0)[:CONFIG.total_samples]
        wavs.append(torch.nn.functional.pad(wav, (0, CONFIG.total_samples - len(wav))))
    wavs = torch.stack(wavs).to(device)
    wavs = wavs / mel(wavs).amax(dim=(1, 2)).sqrt()[:, None]     # every clip's peak mel power = 0 dB
    return normalize_mel(mel(wavs), CONFIG.top_db), wavs


def load_splits(mel: MelSpectrogram, device: torch.device) -> tuple:
    """Train / val / test splits identical to the generative models' splits."""
    data = SpectrogramDataModule(configs.SPECTROGRAMS_DIR, num_workers=0)
    data.setup()
    return [load_split(s, mel, device) for s in (data.train_dataset, data.val_dataset, data.test_dataset)]


def train(device: torch.device, resume: bool) -> None:
    """Train the vocoder with early stopping on validation mel loss."""
    mel = build_mel().to(device)
    train_set, val_set, _ = load_splits(mel, device)
    cfg = configs.VOCODER_TRAIN_CONFIG
    generator = Vocoder(**configs.VOCODER_CONFIG)
    logging.info(f"Vocoder: {sum(p.numel() for p in generator.parameters()) / 1e6:.1f}M parameters")
    trainer = VocoderTrainer(generator, MultiResolutionDiscriminator(), mel, CONFIG.top_db, device, lr=cfg['lr'],
                             gan_start=cfg['gan_start'], lambda_mel=cfg['lambda_mel'], lambda_fm=cfg['lambda_fm'])
    trainer.fit(train_set, val_set, steps=cfg['steps'], batch_size=cfg['batch_size'], crop_frames=cfg['crop_frames'],
                eval_every=cfg['eval_every'], save_path=configs.VOCODER_CHECKPOINT, patience=cfg['patience'],
                log_every=cfg['log_every'], resume=resume)


def speech_quality(reference: np.ndarray, degraded: np.ndarray) -> tuple:
    """(PESQ narrowband, STOI) of a rendering against the true recording."""
    try:
        quality = pesq(CONFIG.sample_rate, reference, degraded, 'nb')
    except Exception:          # PESQ rejects clips without detectable speech
        quality = np.nan
    return quality, stoi(reference, degraded, CONFIG.sample_rate)


def render_all(mels: torch.Tensor) -> dict:
    """Render every clip with Griffin-Lim and the vocoder; returns {name: (waveforms, ms per clip)}."""
    renderers = {'Griffin-Lim (256 iterations)': SpectrogramProcessor(CONFIG, torch.device('cpu')),
                 'Neural vocoder': SpectrogramProcessor(CONFIG, torch.device('cpu'), vocoder=load_vocoder())}
    outputs = {}
    for name, processor in renderers.items():
        start = time.time()
        wavs = [processor.create_audio_from_spectrogram(m[None], -CONFIG.top_db, 0.0)[0].numpy() for m in mels]
        outputs[name] = (wavs, (time.time() - start) / len(mels) * 1000)
    return outputs


def evaluate(device: torch.device) -> None:
    """Score Griffin-Lim vs vocoder against the true test recordings; write table, figure and example audio."""
    mel = build_mel().to(device)
    data = SpectrogramDataModule(configs.SPECTROGRAMS_DIR, num_workers=0)
    data.setup()
    test_labels = [data.test_dataset[i][1] for i in range(len(data.test_dataset))]
    mels, wavs = load_split(data.test_dataset, mel, device)
    mels, references = mels.cpu(), wavs.cpu().numpy()
    results = []
    for name, (rendered, ms) in render_all(mels).items():
        scores = np.array([speech_quality(r, d) for r, d in zip(references, rendered)])
        results.append({'method': name, 'pesq': float(np.nanmean(scores[:, 0])),
                        'stoi': float(scores[:, 1].mean()), 'ms_per_clip': ms, 'audio': rendered})
        logging.info(f"{name}: PESQ {results[-1]['pesq']:.2f}, STOI {results[-1]['stoi']:.3f}, {ms:.1f} ms/clip")
    save_examples(references, results, test_labels)
    for r in results:
        r.pop('audio')
    with open('assets/vocoder_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    update_readme({'vocoder': vocoder_table(results)})


def save_examples(references: np.ndarray, results: list, test_labels: list, digits: tuple = (0, 5)) -> None:
    """Write recording / Griffin-Lim / vocoder versions of selected test digits and the comparison figure."""
    out_dir = 'free_spoken_digit/samples/vocoder_comparison'
    os.makedirs(out_dir, exist_ok=True)
    first_clip = test_labels.index(digits[0])
    for d in digits:
        c = test_labels.index(d)
        versions = [('1_recording', references[c]), ('2_griffin_lim', results[0]['audio'][c]),
                    ('3_vocoder', results[1]['audio'][c])]
        for suffix, audio in versions:
            audio = 0.9 * audio / max(np.abs(audio).max(), 1e-8)
            torchaudio.save(f"{out_dir}/digit_{d}_{suffix}.wav", torch.from_numpy(audio).float()[None], CONFIG.sample_rate)
    signals = {'Recording': references[first_clip], **{r['method']: r['audio'][first_clip] for r in results}}
    plot_audio_comparison(signals, CONFIG.sample_rate, 'assets/vocoder_comparison.png')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', action='store_true', help='Train the vocoder')
    parser.add_argument('--resume', action='store_true', help='Continue training from the resume checkpoint')
    parser.add_argument('--evaluate', action='store_true', help='Compare with Griffin-Lim on the test set')
    args = parser.parse_args()
    setup_logging('vocoder_train' if args.train else 'vocoder_eval')
    set_seed(111)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.train:
        train(device, args.resume)
    if args.evaluate:
        evaluate(device)
