import torch
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import torchvision.transforms as transforms
from visualizations import plot_spectrogram, plot_batch_spectrograms,plot_spectrogram_tensor_or_array
# Define transform functions outside the class
def squeeze_tensor(x):
    """Remove extra dimension."""
    return x.squeeze(1)

def un_squeeze_tensor(x):
    """Remove extra dimension."""
    return x.unsqueeze(0)

def conditional_permute(x):
    """Reshape the array so that when we do ToTensor we get corrected shape"""
    if x.dim() == 3:
        if x.shape[1] == 1:
            return x.permute(1, 2, 0)
    else:
        return x
            

class SpectrogramDataModule:
    """Handles spectrogram dataset loading and preprocessing with balanced class distribution."""
    
    def __init__(self, 
                 data_dir: str,
                 batch_size: int = 32,
                 train_val_split: float = 0.9,
                 num_workers: int = 8,
                 pin_memory: bool = True):
        """
        Args:
            data_dir: Directory containing spectrogram files
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
        
        # Define transforms using named functions instead of lambdas
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(conditional_permute),
        ])
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    class SpectrogramDataset(Dataset):
        """Inner dataset class for loading spectrograms."""
        
        def __init__(self, data_dir: str, transform: Optional[callable] = None):
            self.data_dir = Path(data_dir)
            self.transform = transform
            self.file_paths = sorted(list(self.data_dir.glob("*.npy")))
            self.targets = [self._extract_label(path.stem) for path in self.file_paths]
        
        def __len__(self) -> int:
            return len(self.file_paths)
        
        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
            spec_path = self.file_paths[idx]
            spec = np.load(spec_path)            
            label = self._extract_label(spec_path.stem)
            if self.transform:
                spec = self.transform(spec)
            return spec, label
        
        @staticmethod
        def _extract_label(filename: str) -> int:
            """Extract digit label from filename (format: {digit}_{speaker}_{index})."""
            return int(filename.split('_')[0])
            
        def get_targets(self) -> List[int]:
            """Get all targets for the dataset."""
            return self.targets
    
    def _create_balanced_split(self, dataset, split_ratio: float) -> Tuple[Subset, Subset]:
        """
        Create a balanced split of the dataset using stratified sampling.
        """
        if isinstance(dataset, Subset):
            # Get the original dataset and indices
            original_dataset = dataset.dataset
            subset_indices = dataset.indices
            # Get targets for the subset
            targets = [original_dataset.targets[i] for i in subset_indices]
        else:
            # For the original dataset, get targets directly
            targets = dataset.targets
            subset_indices = range(len(dataset))
            
        targets = np.array(targets)
        
        # Create stratified split
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            train_size=split_ratio,
            random_state=42
        )
        
        # Get indices for both splits
        split1_idx, split2_idx = next(splitter.split(np.zeros(len(targets)), targets))
        
        # Map indices through subset_indices if needed
        if isinstance(dataset, Subset):
            split1_idx = [subset_indices[i] for i in split1_idx]
            split2_idx = [subset_indices[i] for i in split2_idx]
        
        return Subset(dataset.dataset if isinstance(dataset, Subset) else dataset, split1_idx), \
               Subset(dataset.dataset if isinstance(dataset, Subset) else dataset, split2_idx)
    
    def get_class_distribution(self, dataset) -> Dict[int, int]:
        """Get the distribution of classes in a dataset."""
        if isinstance(dataset, Subset):
            # Get the original dataset and indices
            original_dataset = dataset.dataset
            subset_indices = dataset.indices
            # Get targets for the subset
            targets = [original_dataset.targets[i] for i in subset_indices]
        else:
            targets = dataset.targets
            
        targets = np.array(targets)
        unique, counts = np.unique(targets, return_counts=True)
        return dict(zip(unique, counts))
    
    def setup(self) -> None:
        """Setup train, validation and test datasets with balanced class distribution."""
        # Create full dataset
        full_dataset = self.SpectrogramDataset(
            self.data_dir,
            transform=self.transform
        )
        
        # Split into train and test
        train_full, self.test_dataset = self._create_balanced_split(
            full_dataset,
            split_ratio=0.8  # 80% train+val, 20% test
        )
        
        # Split train into train and validation
        self.train_dataset, self.val_dataset = self._create_balanced_split(
            train_full,
            split_ratio=self.train_val_split
        )
        
        print("\nClass distribution in splits:")
        self.print_class_distribution("Training", self.get_class_distribution(self.train_dataset))
        self.print_class_distribution("Validation", self.get_class_distribution(self.val_dataset))
        self.print_class_distribution("Test", self.get_class_distribution(self.test_dataset))
        
    def print_class_distribution(self,dataset_name, distribution):
        print("\nClass distribution in splits:")        
        print(f"\n{dataset_name} set:")
        print("Class | Count")
        print("-" * 13)
        for class_label, count in distribution.items():
            print(f"{class_label:5d} | {count:5d}")
        print(f"Total | {sum(distribution.values()):5d}")
            
        
    
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

