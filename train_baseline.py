"""
Training script for LSTM baseline model
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.models.baselines import LSTMBaseline
from src.data.data_loader import load_preprocessed_data, create_dataloaders

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=15, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)  # [batch, 1]
        
        # Forward pass
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)
            
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def train_baseline_lstm():
    """Main training function"""
    
    print("="*60)
    print("TRAINING LSTM BASELINE")
    print("="*60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load data
    print("\nLoading preprocessed data...")
    data_dict = load_preprocessed_data('data/processed/FD001_preprocessed.npz')
    train_loader, val_loader, test_loader = create_dataloaders(data_dict, batch_size=64)
    
    # Model
    print("\nInitializing model...")
    model = LSTMBaseline(input_size=102).to(device)
    print(f"Total parameters: {model.count_parameters():,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=15)
    
    # Training
    print("\nStarting training...")
    print("-"*60)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(100):  # Max 100 epochs
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs('results/saved_models', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, 'results/saved_models/lstm_baseline_best.pth')
        
        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print("-"*60)
    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation RMSE: {np.sqrt(best_val_loss):.4f} cycles")
    
    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
    }, 'results/saved_models/lstm_baseline_final.pth')
    
    print(f"\nModel saved to: results/saved_models/lstm_baseline_best.pth")
    print("="*60)
    
    return model, train_losses, val_losses

if __name__ == '__main__':
    train_baseline_lstm()