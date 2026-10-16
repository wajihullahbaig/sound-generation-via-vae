import random
import torch
import os
import numpy as np
from typing import Optional

def set_seed(seed: Optional[int] = 42) -> None:
    """
    Set all random seeds for reproducibility.
    
    Args:
        seed: Integer seed for reproducibility. If None, seeds are not set.
    """
    if seed is not None:
        # Python random
        random.seed(seed)
        
        # Numpy
        np.random.seed(seed)
        
        # PyTorch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Set CUDA backend to deterministic mode
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Set Python hash seed
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        print(f"Random seed set to {seed} for reproducibility")

def get_rng_state() -> dict:
    """Capture Python, NumPy and PyTorch RNG states (for exact training resume)."""
    state = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state()}
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    """Restore RNG states captured by get_rng_state()."""
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())   # map_location may have moved it to the GPU
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])
