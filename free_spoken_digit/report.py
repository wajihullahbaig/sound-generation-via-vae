"""Markdown comparison tables, written into the README between <!-- results:NAME --> markers."""
import re

# Earlier VAE versions, measured with the same test-set metric before the normalisation changes
VAE_HISTORY = [
    ('v1: 22.05 kHz, per-file min/max normalisation', 4.15),
    ('v2: 8 kHz, dataset-wide dB range', 2.61),
]


def _fmt(value, spec: str) -> str:
    """Format a number, or an em dash when it is not applicable."""
    return '—' if value is None else format(value, spec)


def generation_table(results: list) -> str:
    """Markdown table comparing generated samples against the real test set."""
    lines = ['| Model | Fréchet distance ↓ | Digit accuracy ↑ | ms / sample ↓ |', '|---|---|---|---|']
    best_fd = min(r['fd'] for r in results[1:])
    for r in results:
        fd = f"**{r['fd']:.2f}**" if r['fd'] == best_fd else f"{r['fd']:.2f}"
        lines.append(f"| {r['model']} | {fd} | {_fmt(r['accuracy'], '.1%')} | {_fmt(r['ms_per_sample'], '.1f')} |")
    return '\n'.join(lines)


def vae_table(vae_results: dict) -> str:
    """Markdown table of VAE test-set reconstruction error across versions."""
    lines = ['| VAE version | Spectrogram error on speech (dB) ↓ |', '|---|---|']
    lines += [f"| {name} | {err:.2f} |" for name, err in VAE_HISTORY]
    lines.append(f"| **v3 (current): 8 kHz, per-clip peak normalisation** | **{vae_results['mae_db_speech']:.2f}** |")
    lines.append('')
    lines.append("_The rows are not directly comparable. v1 and v2 clipped quiet speakers' low-level detail to the "
                 "floor, which is trivially easy to reconstruct. v3 keeps the full 80 dB below each clip's own peak, "
                 "so the same 60 dB \"speech\" region now includes much more quiet detail. The v3 number is the honest "
                 "baseline for future changes._")
    return '\n'.join(lines)


def vocoder_table(results: list) -> str:
    """Markdown table comparing spectrogram-to-audio methods against the true recordings."""
    lines = ['| Spectrogram → audio | PESQ (narrowband, 1–4.5) ↑ | STOI (0–1) ↑ | ms / clip (CPU) ↓ |', '|---|---|---|---|']
    lines += [f"| {r['method']} | {r['pesq']:.2f} | {r['stoi']:.3f} | {r['ms_per_clip']:.1f} |" for r in results]
    return '\n'.join(lines)


def update_readme(sections: dict, path: str = 'README.md') -> None:
    """Replace the content between <!-- results:NAME --> and <!-- /results:NAME --> for each section."""
    with open(path, encoding='utf-8') as f:
        text = f.read()
    for name, body in sections.items():
        pattern = re.compile(rf'(<!-- results:{name} -->\n).*?(\n<!-- /results:{name} -->)', re.S)
        assert pattern.search(text), f"README is missing the results:{name} markers"
        text = pattern.sub(lambda m: m.group(1) + body + m.group(2), text)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
