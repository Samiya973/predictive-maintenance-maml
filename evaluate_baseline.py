"""
Evaluate LSTM baseline model
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Windows
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.models.baselines import LSTMBaseline
from src.data.data_loader import load_preprocessed_data, create_dataloaders
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # NASA scoring function (asymmetric)
    diff = y_pred - y_true
    score = 0
    for d in diff:
        if d < 0:  # Late prediction (more dangerous)
            score += np.exp(-d/13) - 1
        else:  # Early prediction (less dangerous)
            score += np.exp(d/10) - 1
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'NASA_Score': score
    }

def evaluate_baseline():
    """Evaluate trained baseline model"""
    
    print("="*60)
    print("EVALUATING LSTM BASELINE")
    print("="*60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    print("\nLoading data...")
    data_dict = load_preprocessed_data('data/processed/FD001_preprocessed.npz')
    train_loader, val_loader, test_loader = create_dataloaders(data_dict, batch_size=64)
    
    # Load model
    print("Loading trained model...")
    model = LSTMBaseline(input_size=102).to(device)
    checkpoint = torch.load('results/saved_models/lstm_baseline_best.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model trained for {checkpoint['epoch']+1} epochs")
    print(f"Best validation loss: {checkpoint['val_loss']:.4f}")
    
    # Evaluate on all sets
    results = {}
    
    for name, loader in [('Train', train_loader), ('Validation', val_loader), ('Test', test_loader)]:
        y_true_list = []
        y_pred_list = []
        
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(device)
                predictions = model(X_batch).cpu().numpy().flatten()
                y_true_list.extend(y_batch.numpy())
                y_pred_list.extend(predictions)
        
        y_true = np.array(y_true_list)
        y_pred = np.array(y_pred_list)
        
        metrics = calculate_metrics(y_true, y_pred)
        results[name] = {'y_true': y_true, 'y_pred': y_pred, 'metrics': metrics}
        
        print(f"\n{name} Set:")
        print(f"  RMSE: {metrics['RMSE']:.4f} cycles")
        print(f"  MAE:  {metrics['MAE']:.4f} cycles")
        print(f"  R²:   {metrics['R2']:.4f}")
        print(f"  NASA Score: {metrics['NASA_Score']:.2f}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_evaluation_plots(results, checkpoint)
    
    # Save results to text file
    print("\nSaving results summary...")
    save_results_summary(results, checkpoint)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    
    return results

def create_evaluation_plots(results, checkpoint):
    """Create evaluation visualizations"""
    
    print("  Creating plots...")
    
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Model info
    ax1 = plt.subplot(3, 3, 1)
    info_text = f"LSTM Baseline Model\n\n"
    info_text += f"Epochs trained: {checkpoint.get('epoch', 'N/A')+1}\n"
    info_text += f"Best val loss: {checkpoint.get('val_loss', 0):.2f}\n"
    info_text += f"Best val RMSE: {np.sqrt(checkpoint.get('val_loss', 0)):.2f}\n\n"
    info_text += f"Parameters: 168,513\n"
    info_text += f"Architecture:\n"
    info_text += f"  LSTM: 102→128→64\n"
    info_text += f"  Dropout: 0.3\n"
    
    ax1.text(0.5, 0.5, info_text, ha='center', va='center', fontsize=10,
             family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax1.set_title('Model Information', fontsize=11, fontweight='bold')
    ax1.axis('off')
    
    # Plot 2-4: Predictions vs Actual for each set
    for idx, (name, data) in enumerate(results.items(), start=2):
        ax = plt.subplot(3, 3, idx)
        y_true = data['y_true']
        y_pred = data['y_pred']
        
        ax.scatter(y_true, y_pred, alpha=0.3, s=10)
        ax.plot([0, 130], [0, 130], 'r--', linewidth=2, label='Perfect')
        ax.set_xlabel('True RUL (cycles)', fontsize=10)
        ax.set_ylabel('Predicted RUL (cycles)', fontsize=10)
        ax.set_title(f'{name} (RMSE: {data["metrics"]["RMSE"]:.2f})', 
                     fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 5: Error distribution
    ax5 = plt.subplot(3, 3, 5)
    errors = results['Test']['y_pred'] - results['Test']['y_true']
    ax5.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    ax5.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax5.axvline(errors.mean(), color='green', linestyle='--', linewidth=2, 
                label=f'Mean: {errors.mean():.2f}')
    ax5.set_xlabel('Prediction Error (cycles)', fontsize=10)
    ax5.set_ylabel('Frequency', fontsize=10)
    ax5.set_title('Error Distribution', fontsize=11, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Residuals
    ax6 = plt.subplot(3, 3, 6)
    ax6.scatter(results['Test']['y_pred'], errors, alpha=0.3, s=10)
    ax6.axhline(0, color='red', linestyle='--', linewidth=2)
    ax6.set_xlabel('Predicted RUL (cycles)', fontsize=10)
    ax6.set_ylabel('Residual (cycles)', fontsize=10)
    ax6.set_title('Residual Plot', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Sample predictions
    ax7 = plt.subplot(3, 3, 7)
    sample_idx = np.arange(min(100, len(results['Test']['y_true'])))
    ax7.plot(sample_idx, results['Test']['y_true'][:len(sample_idx)], 
             label='True', linewidth=2, marker='o', markersize=3)
    ax7.plot(sample_idx, results['Test']['y_pred'][:len(sample_idx)], 
             label='Predicted', linewidth=2, marker='x', markersize=3)
    ax7.set_xlabel('Sample Index', fontsize=10)
    ax7.set_ylabel('RUL (cycles)', fontsize=10)
    ax7.set_title('Sample Predictions', fontsize=11, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Metrics comparison
    ax8 = plt.subplot(3, 3, 8)
    sets = list(results.keys())
    rmses = [results[s]['metrics']['RMSE'] for s in sets]
    maes = [results[s]['metrics']['MAE'] for s in sets]
    
    x = np.arange(len(sets))
    width = 0.35
    ax8.bar(x - width/2, rmses, width, label='RMSE', alpha=0.8)
    ax8.bar(x + width/2, maes, width, label='MAE', alpha=0.8)
    ax8.set_ylabel('Error (cycles)', fontsize=10)
    ax8.set_title('Metrics Comparison', fontsize=11, fontweight='bold')
    ax8.set_xticks(x)
    ax8.set_xticklabels(sets)
    ax8.legend()
    ax8.grid(True, alpha=0.3, axis='y')
    
    # Plot 9: R² comparison
    ax9 = plt.subplot(3, 3, 9)
    r2_scores = [results[s]['metrics']['R2'] for s in sets]
    colors = ['green' if r2 > 0.9 else 'orange' if r2 > 0.8 else 'red' for r2 in r2_scores]
    ax9.bar(sets, r2_scores, color=colors, alpha=0.7, edgecolor='black')
    ax9.axhline(0.9, color='green', linestyle='--', linewidth=2, label='Good (>0.9)')
    ax9.set_ylabel('R² Score', fontsize=10)
    ax9.set_title('R² Score', fontsize=11, fontweight='bold')
    ax9.set_ylim([0, 1])
    ax9.legend()
    ax9.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('LSTM Baseline - Comprehensive Evaluation', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save
    os.makedirs('results/figures', exist_ok=True)
    plt.savefig('results/figures/lstm_baseline_evaluation.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: results/figures/lstm_baseline_evaluation.png")
    
    plt.close()  # Close instead of show on Windows

def save_results_summary(results, checkpoint):
    """Save results to text file"""
    
    os.makedirs('results/tables', exist_ok=True)
    
    with open('results/tables/lstm_baseline_results.txt', 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("LSTM BASELINE EVALUATION RESULTS\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Model: LSTM (128->64 units)\n")
        f.write(f"Epochs: {checkpoint.get('epoch', 'N/A')+1}\n")
        f.write(f"Parameters: 168,513\n\n")
        
        for name, data in results.items():
            f.write(f"{name} Set:\n")
            f.write(f"  RMSE: {data['metrics']['RMSE']:.4f} cycles\n")
            f.write(f"  MAE:  {data['metrics']['MAE']:.4f} cycles\n")
            f.write(f"  R2:   {data['metrics']['R2']:.4f}\n")
            f.write(f"  NASA Score: {data['metrics']['NASA_Score']:.2f}\n\n")
    
    print("  ✓ Saved: results/tables/lstm_baseline_results.txt")

if __name__ == '__main__':
    evaluate_baseline()