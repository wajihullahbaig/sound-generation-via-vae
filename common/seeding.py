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