import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import torch
from sklearn.decomposition import PCA
import numpy as np
from typing import Optional
from matplotlib.ticker import FormatStrFormatter, LogLocator, NullFormatter

       
def _collect_digit_samples(data_loader, num_images: int):
    """Return one sample per digit 0..9 when available, else the first num_images samples."""
    xs, ys = [], []
    for x, y in data_loader:
        xs.append(x)
        ys.append(y)
    x_all, y_all = torch.cat(xs), torch.cat(ys)
    if num_images == 10 and set(range(10)).issubset(set(y_all.tolist())):
        return torch.stack([x_all[y_all == d][0] for d in range(10)]), [f'Digit {d}' for d in range(10)]
    return x_all[:num_images], [f'Digit {int(y)}' for y in y_all[:num_images]]


def visualize_reconstructions(
    model,
    data_loader,
    device,
    num_images=10,
    display=True,
    save_path=None,
    color_mode='auto',
    spectrogram_mode=False,
    cmap='magma',
    fig_size_multiplier=(1.6, 3.0),
    dataset_type='general',
    filename=None
):
    """Visualize or save original and reconstructed samples in the shared figure style."""
    model.eval()
    save_path = Path(save_path) if save_path else Path('/viz_outputs')
    save_path.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        images, col_labels = _collect_digit_samples(data_loader, num_images)
        images = images.to(device)
        reconstructions, _, _, _ = model(images)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    is_spec = spectrogram_mode or dataset_type == 'spectrogram'
    filename = filename or f"reconstructions_{'spec' if is_spec else 'img'}_{timestamp}.png"
    out_file = save_path / filename

    if is_spec and not display:
        plot_sample_grid({'Original': images.cpu(), 'Reconstructed': reconstructions.cpu()},
                         str(out_file), col_labels=col_labels, cmap=cmap)
        print(f"Saved reconstructions to {out_file}")
        return

    n = len(images)
    fig, axes = plt.subplots(2, n, figsize=(fig_size_multiplier[0] * n, fig_size_multiplier[1]),
                             facecolor=STYLE['surface'], squeeze=False)
    for r, (name, batch) in enumerate([('Original', images.cpu()), ('Reconstructed', reconstructions.cpu())]):
        for c in range(n):
            ax = axes[r, c]
            ax.set_xticks([]), ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            img = batch[c].permute(1, 2, 0).clamp(0, 1) if batch.shape[1] == 3 else batch[c].squeeze()
            ax.imshow(img, cmap='gray' if not is_spec and batch.shape[1] != 3 else cmap,
                      origin='lower' if is_spec else 'upper', aspect='auto' if is_spec else 'equal')
            if c == 0:
                ax.set_ylabel(name, rotation=0, ha='right', va='center', fontsize=10, color=STYLE['text'])
            if r == 0 and col_labels:
                ax.set_title(col_labels[c], fontsize=10, color=STYLE['muted'])
    fig.tight_layout(w_pad=0.3, h_pad=0.4)
    if display:
        plt.show()
    else:
        fig.savefig(out_file, dpi=150, bbox_inches='tight', facecolor=STYLE['surface'])
        plt.close(fig)
        print(f"Saved reconstructions to {out_file}")


def visualize_latent_space(model,
                           data_loader,
                           device,
                           display: bool = True,
                           save_path: Optional[str] = None,
                           filename: Optional[str] = None):
    """Visualize or save the VAE latent space PCA projection and explained variance curve."""
    model.eval()
    save_path = Path(save_path) if save_path else Path('/viz_outputs')
    save_path.mkdir(parents=True, exist_ok=True)

    z_points, labels = [], []
    with torch.no_grad():
        for x, y in data_loader:
            mu, _ = model.encoder(x.to(device))
            z_points.append(mu.cpu())
            labels.extend(y.numpy())

    z_points = torch.cat(z_points, dim=0).numpy()
    labels = np.array(labels)

    n_comp = min(32, z_points.shape[1], z_points.shape[0])
    pca = PCA(n_components=n_comp)
    z_pca = pca.fit_transform(z_points)
    var_ratio = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_ratio)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2), facecolor=STYLE['surface'])

    # Panel 1: 2D PCA scatter by digit
    cmap = plt.get_cmap('tab10', 10)
    for d in sorted(np.unique(labels)):
        mask = labels == d
        ax1.scatter(z_pca[mask, 0], z_pca[mask, 1], color=cmap(int(d)), label=f'{int(d)}', s=22, alpha=0.75, edgecolors='none')
    _style_axis(ax1, f"Latent space PCA (PC1 + PC2 = {var_ratio[:2].sum():.1%} variance)")
    ax1.set_xlabel(f"PC 1 ({var_ratio[0]:.1%} variance)", color=STYLE['muted'])
    ax1.set_ylabel(f"PC 2 ({var_ratio[1]:.1%} variance)", color=STYLE['muted'])
    ax1.grid(True, color=STYLE['grid'], linewidth=0.8)
    ax1.legend(title='Digit', frameon=False, fontsize=8, title_fontsize=8, ncol=2, loc='upper right')

    # Panel 2: Cumulative & individual explained variance across components
    comps = np.arange(1, n_comp + 1)
    ax2.plot(comps, cum_var, 'o-', color=STYLE['series'][0], linewidth=2, markersize=4, label='Cumulative variance')
    ax2.bar(comps, var_ratio, color=STYLE['series'][1], alpha=0.75, width=0.65, label='Per-component variance')
    _style_axis(ax2, f"Explained variance across top {n_comp} latent dimensions ({cum_var[-1]:.1%} total)")
    ax2.set_xlabel('Principal component', color=STYLE['muted'])
    ax2.set_ylabel('Explained variance ratio', color=STYLE['muted'])
    ax2.grid(True, axis='y', color=STYLE['grid'], linewidth=0.8)
    ax2.legend(frameon=False, fontsize=9, labelcolor=STYLE['muted'], loc='center right')

    fig.tight_layout()
    if display:
        plt.show()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = save_path / (filename or f'latent_space_pca_{timestamp}.png')
        fig.savefig(out_file, dpi=150, bbox_inches='tight', facecolor=STYLE['surface'])
        plt.close(fig)
        print(f"Saved latent space plot to {out_file}")


# Shared figure style for README plots (light surface, recessive axes, validated series colours)
STYLE = {'surface': '#fcfcfb', 'text': '#0b0b0b', 'muted': '#52514e', 'grid': '#e4e3df',
         'series': ['#2a78d6', '#eb6834']}


def plot_sample_grid(rows: dict, save_path: str, col_labels: Optional[list] = None, cmap: str = 'magma') -> None:
    """Grid of spectrograms: one row per model ({name: tensor (N, 1, H, W) in [0, 1]})."""
    n_cols = max(len(v) for v in rows.values())
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(1.6 * n_cols, 1.25 * len(rows) + 0.4),
                             facecolor=STYLE['surface'], squeeze=False)
    for r, (name, images) in enumerate(rows.items()):
        for c in range(n_cols):
            ax = axes[r, c]
            ax.set_xticks([]), ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if c < len(images):
                ax.imshow(images[c].squeeze().cpu(), origin='lower', aspect='auto', cmap=cmap, vmin=0, vmax=1)
            if c == 0:
                ax.set_ylabel(name, rotation=0, ha='right', va='center', fontsize=10, color=STYLE['text'])
            if r == 0 and col_labels:
                ax.set_title(col_labels[c], fontsize=10, color=STYLE['muted'])
    fig.tight_layout(w_pad=0.3, h_pad=0.4)
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=STYLE['surface'])
    plt.close(fig)


def plot_training_curves(histories: dict, save_path: str) -> None:
    """One panel per model with train and validation loss ({name: {'step', 'train_loss', 'val_loss'}})."""
    fig, axes = plt.subplots(1, len(histories), figsize=(5.2 * len(histories), 3.4), facecolor=STYLE['surface'])
    for ax, (name, h) in zip(np.atleast_1d(axes), histories.items()):
        ax.set_facecolor(STYLE['surface'])
        for key, label, color in [('train_loss', 'Train', STYLE['series'][0]),
                                  ('val_loss', 'Validation (EMA)', STYLE['series'][1])]:
            ax.plot(h['step'], h[key], color=color, linewidth=2, label=label)
        ax.set_title(name, loc='left', fontsize=12, color=STYLE['text'])
        ax.set_xlabel('Training step', color=STYLE['muted'])
        ax.set_ylabel('Loss (MSE, log scale)', color=STYLE['muted'])
        ax.set_yscale('log')                      # the first validations would otherwise flatten the curve
        ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, axis='y', which='both', color=STYLE['grid'], linewidth=0.8)
        ax.tick_params(colors=STYLE['muted'], length=0)
        for side in ['top', 'right', 'left']:
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color(STYLE['grid'])
        ax.legend(frameon=False, fontsize=9, labelcolor=STYLE['muted'], loc='upper right')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=STYLE['surface'])
    plt.close(fig)


def _style_axis(ax, title: str) -> None:
    """Apply the shared recessive axis style."""
    ax.set_facecolor(STYLE['surface'])
    ax.set_title(title, loc='left', fontsize=11, color=STYLE['text'])
    ax.tick_params(colors=STYLE['muted'], length=0)
    for side in ['top', 'right', 'left']:
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(STYLE['grid'])


def plot_metric_bars(results: list, save_path: str) -> None:
    """One small panel per metric (Frechet distance, digit accuracy, speed) with a bar per model."""
    panels = [('fd', 'Fréchet distance (lower is better)', '{:.1f}'),
              ('accuracy', 'Digit accuracy (higher is better)', '{:.0%}'),
              ('ms_per_sample', 'Generation time, ms/sample (lower is better)', '{:.1f}')]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), facecolor=STYLE['surface'])
    for ax, (key, title, fmt) in zip(axes, panels):
        rows = [r for r in results if r.get(key) is not None]
        names = [r['model'].split(' (')[0] for r in rows]
        values = [r[key] for r in rows]
        colors = [STYLE['muted'] if n.startswith('Real') else STYLE['series'][0] for n in names]
        bars = ax.barh(names, values, color=colors, height=0.6)
        ax.bar_label(bars, labels=[fmt.format(v) for v in values], padding=4, fontsize=9, color=STYLE['text'])
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.margins(x=0.25)
        _style_axis(ax, title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=STYLE['surface'])
    plt.close(fig)


def plot_confusion_matrices(matrices: dict, save_path: str) -> None:
    """Requested digit vs digit heard by the classifier, one row-normalised matrix per model."""
    fig, axes = plt.subplots(1, len(matrices), figsize=(5.2 * len(matrices), 4.6), facecolor=STYLE['surface'])
    for ax, (name, matrix) in zip(np.atleast_1d(axes), matrices.items()):
        m = np.array(matrix)
        image = ax.imshow(m, cmap='Blues', vmin=0, vmax=1)
        for i in range(10):
            for j in range(10):
                if m[i, j] >= 0.05:
                    ax.text(j, i, f"{m[i, j]:.0%}".replace('%', ''), ha='center', va='center', fontsize=7,
                            color='white' if m[i, j] > 0.6 else STYLE['text'])
        ax.set_xticks(range(10)), ax.set_yticks(range(10))
        ax.set_xlabel('Digit heard by classifier', color=STYLE['muted'])
        ax.set_ylabel('Digit requested', color=STYLE['muted'])
        _style_axis(ax, f"{name}: accuracy {np.trace(m) / 10:.0%}")
        ax.spines['bottom'].set_visible(False)
    fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02, label='Share of samples')
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=STYLE['surface'])
    plt.close(fig)


def plot_audio_comparison(signals: dict, sample_rate: int, save_path: str, n_fft: int = 256, hop: int = 16) -> None:
    """High-resolution linear spectrograms of several renderings of one clip ({name: waveform}).

    A finer STFT than the model's reveals phase artefacts (smeared, buzzy harmonics) that the mel hides.
    """
    fig, axes = plt.subplots(1, len(signals), figsize=(5 * len(signals), 3.6), facecolor=STYLE['surface'], sharey=True)
    window = torch.hann_window(n_fft)
    for ax, (name, wav) in zip(axes, signals.items()):
        wav = torch.as_tensor(wav, dtype=torch.float32)
        spec = torch.view_as_real(torch.stft(wav / wav.abs().max(), n_fft, hop, window=window, return_complex=True))
        db = 10 * torch.log10(spec.pow(2).sum(-1).clamp(min=1e-10))
        extent = [0, len(wav) / sample_rate, 0, sample_rate / 2000]
        ax.imshow((db - db.max()).clamp(min=-80), origin='lower', aspect='auto', cmap='magma', extent=extent)
        _style_axis(ax, name)
        ax.set_xlabel('Time (s)', color=STYLE['muted'])
    axes[0].set_ylabel('Frequency (kHz)', color=STYLE['muted'])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=STYLE['surface'])
    plt.close(fig)
