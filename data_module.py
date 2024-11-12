import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from pathlib import Path
from typing import Tuple, Dict


class MNISTDataModule:
    """Handles MNIST dataset loading and preprocessing."""
    
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
    
    def setup(self) -> None:
        """Setup train, validation and test datasets."""
        # Load full training set
        full_train = datasets.MNIST(
            self.data_dir,
            train=True,
            transform=self.transform
        )
        
        # Calculate split sizes
        train_size = int(len(full_train) * self.train_val_split)
        val_size = len(full_train) - train_size
        
        # Split training set
        self.train_dataset, self.val_dataset = random_split(
            full_train, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Load test set
        self.test_dataset = datasets.MNIST(
            self.data_dir,
            train=False,
            transform=self.transform
        )
    
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




