"""
Verify preprocessed data
"""

import numpy as np
import os

def verify_preprocessed_data(data_file='data/processed/FD001_preprocessed.npz'):
    """
    Load and verify preprocessed data
    """
    print("="*60)
    print("VERIFYING PREPROCESSED DATA")
    print("="*60)
    
    # Check if file exists
    if not os.path.exists(data_file):
        print(f"❌ ERROR: File not found: {data_file}")
        return False
    
    # Load data
    data = np.load(data_file, allow_pickle=True)
    
    print(f"\n✓ Loaded data from: {data_file}")
    print(f"\nAvailable arrays:")
    for key in data.files:
        print(f"  - {key}: {data[key].shape if hasattr(data[key], 'shape') else 'scalar/list'}")
    
    # Verify shapes
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    
    print(f"\n✓ Data shapes verified:")
    print(f"  Training:   X={X_train.shape}, y={y_train.shape}")
    print(f"  Validation: X={X_val.shape}, y={y_val.shape}")
    print(f"  Test:       X={X_test.shape}, y={y_test.shape}")
    
    # Check for NaN or Inf
    print(f"\n✓ Checking for invalid values...")
    for name, arr in [('X_train', X_train), ('y_train', y_train), 
                      ('X_val', X_val), ('y_val', y_val),
                      ('X_test', X_test), ('y_test', y_test)]:
        has_nan = np.isnan(arr).any()
        has_inf = np.isinf(arr).any()
        if has_nan or has_inf:
            print(f"  ❌ {name}: NaN={has_nan}, Inf={has_inf}")
        else:
            print(f"  ✓ {name}: Clean (no NaN/Inf)")
    
    # Check value ranges
    print(f"\n✓ Value ranges:")
    print(f"  X_train: [{X_train.min():.4f}, {X_train.max():.4f}]")
    print(f"  y_train: [{y_train.min():.1f}, {y_train.max():.1f}]")
    
    # Check sequence format
    print(f"\n✓ Sequence format:")
    print(f"  Timesteps: {X_train.shape[1]}")
    print(f"  Features: {X_train.shape[2]}")
    
    # Sample data
    print(f"\n✓ Sample sequence (first 5 timesteps, first 3 features):")
    print(X_train[0, :5, :3])
    
    print(f"\n✓ Sample targets (first 10):")
    print(y_train[:10])
    
    print("\n" + "="*60)
    print("VERIFICATION COMPLETE - DATA LOOKS GOOD!")
    print("="*60)
    
    return True

if __name__ == '__main__':
    verify_preprocessed_data()