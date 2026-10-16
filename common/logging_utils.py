import logging
from datetime import datetime
from pathlib import Path


def setup_logging(run_name: str, log_dir: str = 'logs') -> Path:
    """Log INFO messages to the console and to logs/<run_name>_<timestamp>.log; returns the file path."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / f"{run_name}_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', force=True,
                        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding='utf-8')])
    logging.info(f"Logging to {log_path}")
    return log_path
