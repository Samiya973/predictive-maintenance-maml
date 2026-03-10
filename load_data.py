"""
Load NASA C-MAPSS dataset
"""

import pandas as pd
import numpy as np
import os

def load_cmapss_data(dataset='FD001', data_dir='data/raw'):
    """
    Load NASA C-MAPSS dataset
    
    Parameters:
    -----------
    dataset : str
        Which dataset to load: 'FD001', 'FD002', 'FD003', or 'FD004'
    data_dir : str
        Directory containing the raw data files
    
    Returns:
    --------
    train_df : pd.DataFrame
        Training data
    test_df : pd.DataFrame
        Test data
    rul_df : pd.DataFrame
        RUL labels for test data
    """
    
    # Define column names
    index_names = ['engine_id', 'cycle']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = [f'sensor_{i}' for i in range(1, 22)]  # sensor_1 to sensor_21
    col_names = index_names + setting_names + sensor_names
    
    # File paths
    train_file = os.path.join(data_dir, f'train_{dataset}.txt')
    test_file = os.path.join(data_dir, f'test_{dataset}.txt')
    rul_file = os.path.join(data_dir, f'RUL_{dataset}.txt')
    
    # Load training data
    train_df = pd.read_csv(train_file, sep='\s+', header=None, names=col_names)
    
    # Load test data
    test_df = pd.read_csv(test_file, sep='\s+', header=None, names=col_names)
    
    # Load RUL labels (one value per engine in test set)
    rul_df = pd.read_csv(rul_file, sep='\s+', header=None, names=['RUL'])
    
    print(f"✓ Loaded {dataset}")
    print(f"  Training data: {train_df.shape}")
    print(f"  Test data: {test_df.shape}")
    print(f"  RUL labels: {rul_df.shape}")
    
    return train_df, test_df, rul_df


def add_rul_column(df):
    """
    Add RUL (Remaining Useful Life) column to training data
    
    RUL = max_cycle - current_cycle for each engine
    """
    # Group by engine and get max cycle for each
    max_cycles = df.groupby('engine_id')['cycle'].max().reset_index()
    max_cycles.columns = ['engine_id', 'max_cycle']
    
    # Merge with original data
    df = df.merge(max_cycles, on='engine_id', how='left')
    
    # Calculate RUL
    df['RUL'] = df['max_cycle'] - df['cycle']
    
    # Drop the helper column
    df = df.drop('max_cycle', axis=1)
    
    return df


if __name__ == '__main__':
    # Test the loading
    print("Testing data loading...")
    train_df, test_df, rul_df = load_cmapss_data('FD001')
    
    # Add RUL to training data
    train_df = add_rul_column(train_df)
    
    # Show first few rows
    print("\nFirst 5 rows of training data:")
    print(train_df.head())
    
    print("\nColumn names:")
    print(train_df.columns.tolist())
    
    print("\nBasic statistics:")
    print(train_df.describe())




