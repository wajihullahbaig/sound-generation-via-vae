"""Compare VAE, DDPM and Flow Matching samples: Frechet distance, digit accuracy and sampling speed."""
import argparse
import json
import logging
import os
import time
import torch
import torch.nn.functional as F
import configs
from arch.auto_encoder import VAE
from arch.classifier import DigitClassifier
from common.metrics import frechet_distance
from common.logging_utils import setup_logging
from common.seeding import set_seed
from common.visualizations import (plot_confusion_matrices, plot_metric_bars, plot_sample_grid,
                                   plot_training_curves)
from spectrogram_data_module import SpectrogramDataModule
from report import generation_table, update_readme, vae_table
from train_generative import generate, load_method

SAMPLES_PER_DIGIT = 100
NAMES = {'ddpm': 'DDPM', 'flow': 'Flow Matching'}


def train_classifier(train: tuple, val: tuple, device: torch.device, epochs: int = 40,
                     batch_size: int = 64) -> DigitClassifier:
    """Train the evaluation classifier on real spectrograms, keeping the best validation epoch."""
    model = DigitClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_acc, best_state = 0.0, None
    for epoch in range(epochs):
        model.train()
        for idx in torch.randperm(len(train[0]), device=device).split(batch_size):
            loss = F.cross_entropy(model(train[0][idx]), train[1][idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        acc = accuracy(model, *val)
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    logging.info(f"Classifier: best validation accuracy {best_acc:.3f}")
    return model.eval()


@torch.no_grad()
def embed(model: DigitClassifier, x: torch.Tensor, batch_size: int = 200) -> tuple:
    """Return (pooled features, predicted labels) for spectrograms in [0, 1]."""
    model.eval()
    feats = torch.cat([model.features(b) for b in x.split(batch_size)])
    return feats.cpu().numpy(), model.head(feats).argmax(1)


def accuracy(model: DigitClassifier, x: torch.Tensor, y: torch.Tensor) -> float:
    """Fraction of spectrograms classified as their label."""
    return (embed(model, x)[1] == y).float().mean().item()


def timed(fn, n: int) -> tuple:
    """Run fn() and return (result, milliseconds per sample)."""
    torch.cuda.synchronize()
    start = time.time()
    out = fn()
    torch.cuda.synchronize()
    return out, (time.time() - start) / n * 1000


def load_classifier(train: tuple, val: tuple, device: torch.device, retrain: bool) -> DigitClassifier:
    """Load the evaluation classifier, training it first if needed."""
    model = DigitClassifier().to(device)
    if retrain or not os.path.exists(configs.CLASSIFIER_CHECKPOINT):
        model = train_classifier(train, val, device)
        torch.save({'model_state_dict': model.state_dict()}, configs.CLASSIFIER_CHECKPOINT)
    model.load_state_dict(torch.load(configs.CLASSIFIER_CHECKPOINT, map_location=device)['model_state_dict'])
    return model.eval()


def load_vae(device: torch.device) -> VAE:
    """Load the trained VAE."""
    vae = VAE(**configs.build_model_kwargs()).to(device)
    vae.load_state_dict(torch.load(configs.CHECKPOINT_PATH, map_location=device)['model_state_dict'])
    return vae.eval()


@torch.no_grad()
def sample_vae(vae: VAE, n: int, batch_size: int = 100) -> torch.Tensor:
    """Decode n latent vectors z ~ N(0, I) in batches."""
    z = torch.randn(n, vae.latent_dim, device=next(vae.parameters()).device)
    return torch.cat([vae.decoder(b) for b in z.split(batch_size)])


def score(name: str, samples: torch.Tensor, ms: float, classifier: DigitClassifier,
          test_feats, labels: torch.Tensor = None) -> dict:
    """Frechet distance to the test set and, for conditional models, digit accuracy and confusion matrix."""
    feats, preds = embed(classifier, samples)
    result = {'model': name, 'fd': frechet_distance(feats, test_feats), 'accuracy': None, 'ms_per_sample': ms}
    if labels is not None:
        result['accuracy'] = (preds == labels).float().mean().item()
        confusion = torch.zeros(10, 10, device=labels.device).index_put_((labels, preds), torch.tensor(1.0, device=labels.device), accumulate=True)
        result['confusion'] = (confusion / confusion.sum(1, keepdim=True)).tolist()
    logging.info(f"{name}: FD {result['fd']:.2f}, accuracy {_fmt_pct(result['accuracy'])}, "
                 f"{_fmt_ms(ms)} ms/sample")
    return result


def _fmt_pct(value) -> str:
    """Percentage or n/a."""
    return 'n/a' if value is None else f"{value:.1%}"


def _fmt_ms(value) -> str:
    """Milliseconds or n/a."""
    return 'n/a' if value is None else f"{value:.1f}"


def save_outputs(results: list, grid: dict) -> None:
    """Write the results JSON, figures and the README comparison tables."""
    histories = {NAMES[k]: torch.load(configs.GENERATIVE_CHECKPOINTS[k], map_location='cpu')['history'] for k in NAMES}
    plot_training_curves(histories, 'assets/generative_training.png')
    plot_sample_grid(grid, 'assets/generated_samples.png', col_labels=[f'Digit {d}' for d in range(10)])
    plot_metric_bars(results, 'assets/generative_metrics.png')
    plot_confusion_matrices({r['model']: r['confusion'] for r in results if 'confusion' in r},
                            'assets/digit_confusion.png')
    with open('assets/generative_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    sections = {'generation': generation_table(results)}
    if os.path.exists('assets/vae_results.json'):
        with open('assets/vae_results.json') as f:
            sections['vae'] = vae_table(json.load(f))
    update_readme(sections)
    print(sections['generation'])


def main(retrain_classifier: bool) -> None:
    """Score every model against the real test set, then write figures and the README report."""
    device = torch.device('cuda')
    data = SpectrogramDataModule(configs.SPECTROGRAMS_DIR, num_workers=0)
    data.setup()
    train, val, test = [data.load_tensors(s, device) for s in (data.train_dataset, data.val_dataset, data.test_dataset)]
    classifier = load_classifier(train, val, device, retrain_classifier)
    test_feats, _ = embed(classifier, test[0])
    labels = torch.arange(10, device=device).repeat_interleave(SAMPLES_PER_DIGIT)

    results = [score('Real data (train split)', train[0], None, classifier, test_feats)]
    results[0]['accuracy'] = accuracy(classifier, *test)
    vae = load_vae(device)
    samples, ms = timed(lambda: sample_vae(vae, len(labels)), len(labels))
    results.append(score('VAE (prior samples)', samples, ms, classifier, test_feats))
    grid = {'Real': torch.stack([test[0][test[1] == d][0] for d in range(10)]), 'VAE': samples[:10]}

    for key, name in NAMES.items():
        method = load_method(key, device)
        samples, ms = timed(lambda: generate(method, key, labels), len(labels))
        results.append(score(name, samples, ms, classifier, test_feats, labels))
        grid[name] = samples[::SAMPLES_PER_DIGIT]

    save_outputs(results, grid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retrain-classifier', action='store_true')
    args = parser.parse_args()
    setup_logging('evaluate_generative')
    set_seed(111)
    main(args.retrain_classifier)
