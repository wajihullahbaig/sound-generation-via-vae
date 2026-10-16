"""Run the full pipeline: VAE, DDPM and Flow Matching training, audio samples and the comparison report."""
import argparse
import logging
import subprocess
import sys
from common.logging_utils import setup_logging

# (name, commands); training stages accept --resume
STAGES = [
    ('vae',        [['free_spoken_digit/main.py', '--train']]),
    ('vae_report', [['free_spoken_digit/main.py', '--test']]),
    ('vocoder',    [['free_spoken_digit/train_vocoder.py', '--train', '--evaluate']]),
    ('vae_audio',  [['free_spoken_digit/create_audios.py']]),
    ('ddpm',       [['free_spoken_digit/train_generative.py', '--method', 'ddpm', '--train']]),
    ('flow',       [['free_spoken_digit/train_generative.py', '--method', 'flow', '--train']]),
    ('samples',    [['free_spoken_digit/train_generative.py', '--method', m, '--sample'] for m in ('ddpm', 'flow')]),
    ('report',     [['free_spoken_digit/evaluate_generative.py']]),
]
TRAINING_STAGES = {'vae', 'vocoder', 'ddpm', 'flow'}


def run(stages: list, resume: bool) -> None:
    """Run the given stages in order; --resume applies to the first (interrupted) training stage only."""
    for i, name in enumerate(stages):
        for command in dict(STAGES)[name]:
            args = command + (['--resume'] if resume and i == 0 and name in TRAINING_STAGES else [])
            logging.info(f"=== Stage '{name}': {' '.join(args)}")
            if subprocess.run([sys.executable, '-u', *args]).returncode != 0:
                logging.error(f"Stage '{name}' failed or was interrupted. Continue with: "
                              f"--from {name}{' --resume' if name in TRAINING_STAGES else ''}")
                sys.exit(1)
    logging.info("Pipeline finished: see assets/, free_spoken_digit/samples/ and the README results")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from', dest='start', default='vae', choices=[name for name, _ in STAGES],
                        help='Stage to start from (earlier stages are skipped)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume the starting training stage from its checkpoint')
    parser.add_argument('--only', nargs='+', choices=[name for name, _ in STAGES],
                        help='Run only these stages (in pipeline order), e.g. --only vocoder vae_audio samples report')
    args = parser.parse_args()
    setup_logging('pipeline')
    names = [name for name, _ in STAGES]
    stages = [n for n in names if n in args.only] if args.only else names[names.index(args.start):]
    run(stages, args.resume)
