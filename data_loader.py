"""
Data loader for preprocessed CMAPSS data
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CMAPSSDataset(Dataset):
    """
    PyTorch Dataset for CMAPSS data
    """
    
    def __init__(self, X, y, engine_ids=None):
        """
        Initialize dataset
        
        Parameters:
        -----------
        X : np.array
            Feature sequences [samples, timesteps, features]
        y : np.array
            Target values [samples]
        engine_ids : np.array
            Engine IDs (optional)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.engine_ids = engine_ids
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_preprocessed_data(data_file='data/processed/FD001_preprocessed.npz'):
    """
    Load preprocessed data
    
    Returns:
    --------
    data_dict : dict
        Dictionary containing all data arrays
    """
    data = np.load(data_file, allow_pickle=True)
    
    data_dict = {
        'X_train': data['X_train'],
        'y_train': data['y_train'],
        'train_engines': data['train_engines'],
        'X_val': data['X_val'],
        'y_val': data['y_val'],
        'val_engines': data['val_engines'],
        'X_test': data['X_test'],
        'y_test': data['y_test'],
        'test_engines': data['test_engines'],
        'feature_names': data['feature_names']
    }
    
    print(f"✓ Loaded preprocessed data")
    print(f"  Training: {len(data_dict['X_train'])} samples")
    print(f"  Validation: {len(data_dict['X_val'])} samples")
    print(f"  Test: {len(data_dict['X_test'])} samples")
    
    return data_dict


def create_dataloaders(data_dict, batch_size=64):
    """
    Create PyTorch DataLoaders
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing data arrays
    batch_size : int
        Batch size
        
    Returns:
    --------
    train_loader, val_loader, test_loader : DataLoader
        PyTorch DataLoaders
    """
    # Create datasets
    train_dataset = CMAPSSDataset(
        data_dict['X_train'], 
        data_dict['y_train'],
        data_dict['train_engines']
    )
    
    val_dataset = CMAPSSDataset(
        data_dict['X_val'], 
        data_dict['y_val'],
        data_dict['val_engines']
    )
    
    test_dataset = CMAPSSDataset(
        data_dict['X_test'], 
        data_dict['y_test'],
        data_dict['test_engines']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    print(f"\n✓ Created DataLoaders with batch_size={batch_size}")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Test data loading
    data_dict = load_preprocessed_data()
    train_loader, val_loader, test_loader = create_dataloaders(data_dict, batch_size=64)
    
    # Test one batch
    X_batch, y_batch = next(iter(train_loader))
    print(f"\n✓ Sample batch:")
    print(f"  X shape: {X_batch.shape}")
    print(f"  y shape: {y_batch.shape}")
    print(f"  X range: [{X_batch.min():.4f}, {X_batch.max():.4f}]")
    print(f"  y range: [{y_batch.min():.1f}, {y_batch.max():.1f}]")