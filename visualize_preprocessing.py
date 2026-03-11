"""
Visualize preprocessed data
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_preprocessed_data(data_file='data/processed/FD001_preprocessed.npz'):
    """
    Create visualizations of preprocessed data
    """
    # Load data
    data = np.load(data_file, allow_pickle=True)
    
    X_train = data['X_train']
    y_train = data['y_train']
    
    print("Creating visualizations...")
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: RUL distribution
    axes[0, 0].hist(y_train, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('RUL (cycles)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Training Set: RUL Distribution')
    axes[0, 0].axvline(y_train.mean(), color='red', linestyle='--', 
                       label=f'Mean: {y_train.mean():.1f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Feature distributions (first 3 features at last timestep)
    for i in range(3):
        feature_data = X_train[:, -1, i]  # Last timestep, feature i
        axes[0, 1].hist(feature_data, bins=30, alpha=0.5, 
                       label=f'Feature {i+1}', edgecolor='black')
    axes[0, 1].set_xlabel('Normalized Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Sample Feature Distributions (Last Timestep)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Sample sequence (first sample, first 5 features)
    for i in range(5):
        axes[1, 0].plot(X_train[0, :, i], label=f'Feature {i+1}', alpha=0.7)
    axes[1, 0].set_xlabel('Timestep')
    axes[1, 0].set_ylabel('Normalized Value')
    axes[1, 0].set_title('Sample Sequence (First 5 Features)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Feature importance by variance
    feature_vars = np.var(X_train[:, -1, :], axis=0)  # Variance at last timestep
    top_10_idx = np.argsort(feature_vars)[-10:][::-1]
    axes[1, 1].bar(range(10), feature_vars[top_10_idx])
    axes[1, 1].set_xlabel('Feature Index (Top 10)')
    axes[1, 1].set_ylabel('Variance')
    axes[1, 1].set_title('Top 10 Features by Variance')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/figures/preprocessing_verification.png', dpi=300, bbox_inches='tight')
    print("✓ Saved visualization: results/figures/preprocessing_verification.png")
    plt.show()
    
    # Print statistics
    print("\n" + "="*60)
    print("PREPROCESSING STATISTICS")
    print("="*60)
    print(f"Sequence shape: {X_train.shape}")
    print(f"Target shape: {y_train.shape}")
    print(f"\nFeature statistics:")
    print(f"  Min: {X_train.min():.6f}")
    print(f"  Max: {X_train.max():.6f}")
    print(f"  Mean: {X_train.mean():.6f}")
    print(f"  Std: {X_train.std():.6f}")
    print(f"\nTarget (RUL) statistics:")
    print(f"  Min: {y_train.min():.1f}")
    print(f"  Max: {y_train.max():.1f}")
    print(f"  Mean: {y_train.mean():.1f}")
    print(f"  Std: {y_train.std():.1f}")
    print("="*60)

if __name__ == '__main__':
    visualize_preprocessed_data()