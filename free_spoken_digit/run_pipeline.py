"""Run the full pipeline: VAE, DDPM and Flow Matching training, audio samples and the comparison report."""
import argparse
import logging
import subprocess
import sys
from common.logging_utils import setup_logging

# (name, command); training stages accept --resume
STAGES = [
    ('vae',        ['free_spoken_digit/main.py', '--train']),
    ('vae_report', ['free_spoken_digit/main.py', '--test']),
    ('vae_audio',  ['free_spoken_digit/create_audios.py']),
    ('ddpm',       ['free_spoken_digit/train_generative.py', '--method', 'ddpm', '--train', '--sample']),
    ('flow',       ['free_spoken_digit/train_generative.py', '--method', 'flow', '--train', '--sample']),
    ('report',     ['free_spoken_digit/evaluate_generative.py']),
]
TRAINING_STAGES = {'vae', 'ddpm', 'flow'}


def run(start: str, resume: bool) -> None:
    """Run every stage from `start` on; --resume applies to the first (interrupted) training stage only."""
    names = [name for name, _ in STAGES]
    for i, (name, command) in enumerate(STAGES[names.index(start):]):
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
    args = parser.parse_args()
    setup_logging('pipeline')
    run(args.start, args.resume)
