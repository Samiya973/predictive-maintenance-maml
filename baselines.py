"""
Baseline models for predictive maintenance
LSTM implementation based on Zheng et al. (2017)
"""

import torch
import torch.nn as nn

class LSTMBaseline(nn.Module):
    """
    2-layer LSTM for RUL prediction
    
    Based on:
    Zheng et al. (2017) "Long Short-Term Memory Network for 
    Remaining Useful Life estimation" IEEE ICPHM
    
    Architecture:
    - LSTM Layer 1: input_size → 128 units
    - Dropout: 0.3
    - LSTM Layer 2: 128 → 64 units
    - Dropout: 0.3
    - Dense: 64 → 1 (RUL prediction)
    """
    
    def __init__(self, input_size=102, hidden_size_1=128, hidden_size_2=64, dropout=0.3):
        """
        Initialize LSTM model
        
        Parameters:
        -----------
        input_size : int
            Number of features per timestep (default: 102)
        hidden_size_1 : int
            Hidden units in first LSTM layer (default: 128)
        hidden_size_2 : int
            Hidden units in second LSTM layer (default: 64)
        dropout : float
            Dropout probability (default: 0.3)
        """
        super(LSTMBaseline, self).__init__()
        
        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.dropout = dropout
        
        # First LSTM layer
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size_1,
            num_layers=1,
            batch_first=True,
            dropout=0  # Dropout between LSTM layers handled separately
        )
        
        # Dropout after first LSTM
        self.dropout1 = nn.Dropout(dropout)
        
        # Second LSTM layer
        self.lstm2 = nn.LSTM(
            input_size=hidden_size_1,
            hidden_size=hidden_size_2,
            num_layers=1,
            batch_first=True,
            dropout=0
        )
        
        # Dropout after second LSTM
        self.dropout2 = nn.Dropout(dropout)
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size_2, 1)
        
        # ReLU activation to ensure non-negative RUL
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Forward pass
        
        Parameters:
        -----------
        x : torch.Tensor
            Input sequences [batch_size, seq_length, input_size]
            
        Returns:
        --------
        out : torch.Tensor
            RUL predictions [batch_size, 1]
        """
        # x shape: [batch_size, 30, 102]
        
        # First LSTM layer
        lstm1_out, _ = self.lstm1(x)
        # lstm1_out shape: [batch_size, 30, 128]
        
        # Dropout
        lstm1_out = self.dropout1(lstm1_out)
        
        # Second LSTM layer
        lstm2_out, (h_n, c_n) = self.lstm2(lstm1_out)
        # lstm2_out shape: [batch_size, 30, 64]
        # h_n shape: [1, batch_size, 64] - final hidden state
        
        # Use final hidden state for prediction
        final_hidden = h_n.squeeze(0)  # [batch_size, 64]
        
        # Dropout
        final_hidden = self.dropout2(final_hidden)
        
        # Fully connected layer
        out = self.fc(final_hidden)  # [batch_size, 1]
        
        # ReLU to ensure non-negative predictions
        out = self.relu(out)
        
        return out
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Test the model
if __name__ == '__main__':
    # Create model
    model = LSTMBaseline(input_size=102)
    
    print("="*60)
    print("LSTM BASELINE MODEL")
    print("="*60)
    print(f"Architecture: {model}")
    print(f"\nTotal parameters: {model.count_parameters():,}")
    
    # Test forward pass
    batch_size = 64
    seq_length = 30
    input_size = 102
    
    # Dummy input
    x = torch.randn(batch_size, seq_length, input_size)
    
    # Forward pass
    output = model(x)
    
    print(f"\nTest forward pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min().item():.2f}, {output.max().item():.2f}]")
    print("="*60)