import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from pathlib import Path
from typing import Tuple, Dict, List
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

class MNISTDataModule:
    """Handles MNIST dataset loading and preprocessing with balanced class distribution."""
    
    def __init__(self, data_dir: str = './data', 
                 batch_size: int = 32,
                 train_val_split: float = 0.9,
                 num_workers: int = 4,
                 pin_memory: bool = True):
        """
        Args:
            data_dir: Directory to store the dataset
            batch_size: Batch size for training
            train_val_split: Fraction of training data to use for training
            num_workers: Number of workers for data loading
            pin_memory: Whether to pin memory for GPU training
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(), # Converts to [0,1]            
        ])
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def prepare_data(self) -> None:
        """Download the MNIST dataset if not already present."""
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)
    
    def _create_balanced_split(self, dataset, split_ratio: float) -> Tuple[Subset, Subset]:
        """
        Create a balanced split of the dataset using stratified sampling.
        
        Args:
            dataset: The full dataset to split
            split_ratio: Fraction of data to use for first split
            
        Returns:
            Tuple of (first_split, second_split) as torch Subsets
        """
        # Get all targets
        if isinstance(dataset, Subset):
            # If dataset is already a Subset, get targets from the underlying dataset
            targets = [dataset.dataset.targets[i] for i in dataset.indices]
        else:
            targets = dataset.targets
            
        # Convert targets to numpy array if they're in tensor format
        if torch.is_tensor(targets):
            targets = targets.numpy()
        elif isinstance(targets, list):
            targets = np.array(targets)
            
        # Create stratified split
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            train_size=split_ratio,
            random_state=42
        )
        
        # Get indices for both splits
        train_idx, val_idx = next(splitter.split(np.zeros(len(targets)), targets))
        
        # Create Subset objects
        first_split = Subset(dataset, train_idx)
        second_split = Subset(dataset, val_idx)
        
        return first_split, second_split
    
    def get_class_distribution(self, dataset) -> Dict[int, int]:
        """
        Get the distribution of classes in a dataset.
        
        Args:
            dataset: Dataset to analyze
            
        Returns:
            Dictionary mapping class indices to counts
        """
        if isinstance(dataset, Subset):
            targets = [dataset.dataset.targets[i] for i in dataset.indices]
        else:
            targets = dataset.targets
            
        if torch.is_tensor(targets):
            targets = targets.numpy()
            
        unique, counts = np.unique(targets, return_counts=True)
        return dict(zip(unique, counts))
    
    def setup(self) -> None:
        """Setup train, validation and test datasets with balanced class distribution."""
        # Load full training set
        full_train = datasets.MNIST(
            self.data_dir,
            train=True,
            transform=self.transform
        )
        
        # Create balanced split
        self.train_dataset, self.val_dataset = self._create_balanced_split(
            full_train, 
            self.train_val_split
        )
        
        # Load test set
        self.test_dataset = datasets.MNIST(
            self.data_dir,
            train=False,
            transform=self.transform
        )
        
        # Print class distribution for verification
        print("Class distribution in splits:")
        print("Training set:", self.get_class_distribution(self.train_dataset))
        print("Validation set:", self.get_class_distribution(self.val_dataset))
        print("Test set:", self.get_class_distribution(self.test_dataset))
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )