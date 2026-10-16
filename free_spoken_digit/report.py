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
