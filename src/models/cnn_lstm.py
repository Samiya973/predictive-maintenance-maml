"""
cnn_lstm.py
───────────
CNN + LSTM hybrid model for early fault detection in turbofan engines.

ARCHITECTURE OVERVIEW
─────────────────────
Input window  →  [Conv1D blocks]  →  [LSTM layers]  →  Dual head
                  ↳ extract local          ↳ capture long-      ↳ RUL regression
                    sensor patterns          range degradation      Early-fault
                    across timesteps         trends                 classification

WHY CNN FIRST?
  Each timestep has 14-21 sensor readings. The CNN learns which sensor
  combinations matter (local feature extraction) before the LSTM models
  how those combinations evolve over time. This outperforms pure LSTM on
  CMAPSS because sensors have correlated, structured local patterns.

DUAL HEAD DESIGN
  Head 1 — RUL Regression     : predicts exact remaining cycles
  Head 2 — Early Fault Detect : binary — is the engine in degradation?
    Label definition: RUL <= early_threshold (default 125 cycles for FD001)
    This threshold is the standard CMAPSS piece-wise RUL clipping point.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
#  BUILDING BLOCKS
# ──────────────────────────────────────────────

class ConvBlock(nn.Module):
    """
    Conv1D → BatchNorm → GELU → Dropout

    Uses GELU instead of ReLU: smoother gradient flow, works better with
    the residual connections added at the CNN stack level.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 dilation=1, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2   # same-padding
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size  = kernel_size,
            dilation     = dilation,
            padding      = padding,
            bias         = False       # BatchNorm has its own bias
        )
        self.bn      = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.act(self.bn(self.conv(x))))


class ResidualCNNBlock(nn.Module):
    """
    Two ConvBlocks with a residual skip connection.
    If channel dims differ, a 1×1 conv projects the skip.

    Residual connections help the model learn degradation deltas rather
    than absolute values — important for RUL because the signal is
    mostly flat early in life and changes subtly near failure.
    """
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        self.block1 = ConvBlock(channels, channels, kernel_size, dilation, dropout)
        self.block2 = ConvBlock(channels, channels, kernel_size, dilation, dropout)

    def forward(self, x):
        return x + self.block2(self.block1(x))   # residual add


# ──────────────────────────────────────────────
#  MAIN MODEL
# ──────────────────────────────────────────────

class CNNLSTM(nn.Module):
    """
    CNN + LSTM hybrid for dual-task RUL regression + early fault detection.

    Parameters
    ----------
    input_size      : int   — number of features per timestep (e.g. 14 or 24)
    seq_len         : int   — window length (e.g. 30)
    cnn_channels    : int   — channel width for CNN blocks (default 64)
    cnn_layers      : int   — number of residual CNN blocks (default 3)
    lstm_hidden     : int   — LSTM hidden units (default 128)
    lstm_layers     : int   — stacked LSTM depth (default 2)
    dropout         : float — dropout rate throughout (default 0.3)
    early_threshold : int   — RUL <= this → "early fault" label (default 125)
    """

    def __init__(
        self,
        input_size      = 14,
        seq_len         = 30,
        cnn_channels    = 64,
        cnn_layers      = 3,
        lstm_hidden     = 128,
        lstm_layers     = 2,
        dropout         = 0.3,
        early_threshold = 125,
    ):
        super().__init__()
        self.early_threshold = early_threshold
        self.seq_len         = seq_len
        self.input_size      = input_size

        # ── CNN stack ─────────────────────────────────────────────────
        # Project raw features → cnn_channels, then stack residual blocks
        # with increasing dilation to capture multi-scale patterns:
        #   dilation 1 → short-range sensor coupling
        #   dilation 2 → medium-range trends
        #   dilation 4 → longer degradation patterns
        self.input_proj = ConvBlock(input_size, cnn_channels,
                                    kernel_size=1, dropout=0.0)

        dilations       = [1, 2, 4][:cnn_layers]
        if len(dilations) < cnn_layers:
            dilations  += [4] * (cnn_layers - len(dilations))

        self.cnn_blocks = nn.ModuleList([
            ResidualCNNBlock(cnn_channels,
                             kernel_size = 3,
                             dilation    = dilations[i],
                             dropout     = dropout)
            for i in range(cnn_layers)
        ])

        # ── LSTM stack ────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size   = cnn_channels,
            hidden_size  = lstm_hidden,
            num_layers   = lstm_layers,
            batch_first  = True,
            dropout       = dropout if lstm_layers > 1 else 0.0,
            bidirectional = False     # causal: no future leakage
        )
        self.lstm_dropout = nn.Dropout(dropout)

        # ── Shared feature neck ────────────────────────────────────────
        neck_in = lstm_hidden
        self.neck = nn.Sequential(
            nn.Linear(neck_in, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
        )

        # ── Head 1: RUL regression ────────────────────────────────────
        self.rul_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)        # unbounded output → raw RUL cycles
        )

        # ── Head 2: Early fault detection ────────────────────────────
        # Binary: 0 = healthy, 1 = degrading (RUL <= early_threshold)
        self.fault_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)        # sigmoid applied externally via BCEWithLogitsLoss
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming init for conv, Xavier for linear, orthogonal for LSTM."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.zeros_(param.data)

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor  shape (batch, seq_len, input_size)

        Returns
        -------
        rul_pred   : Tensor  shape (batch, 1)   — predicted RUL in cycles
        fault_logit: Tensor  shape (batch, 1)   — raw logit for fault class
        """
        # CNN expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)          # (B, F, T)
        x = self.input_proj(x)           # (B, C, T)

        for block in self.cnn_blocks:
            x = block(x)                 # (B, C, T)

        # LSTM expects (batch, seq_len, channels)
        x = x.permute(0, 2, 1)          # (B, T, C)
        lstm_out, _ = self.lstm(x)       # (B, T, H)
        x = self.lstm_dropout(lstm_out[:, -1, :])  # take last timestep

        features    = self.neck(x)
        rul_pred    = self.rul_head(features)
        fault_logit = self.fault_head(features)

        return rul_pred, fault_logit

    def predict(self, x, fault_threshold=0.5):
        """
        Convenience method for inference.

        Returns
        -------
        rul   : np.ndarray  shape (N,)
        fault : np.ndarray  shape (N,)  — binary 0/1
        prob  : np.ndarray  shape (N,)  — fault probability
        """
        import numpy as np
        self.eval()
        with torch.no_grad():
            rul_pred, logit = self(x)
            prob  = torch.sigmoid(logit).cpu().numpy().flatten()
            rul   = rul_pred.cpu().numpy().flatten()
            fault = (prob >= fault_threshold).astype(int)
        return rul, fault, prob

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────
#  LOSS FUNCTION
# ──────────────────────────────────────────────

class DualTaskLoss(nn.Module):
    """
    Combined loss for RUL regression + early fault detection.

        L = alpha * L_rul + (1-alpha) * L_fault

    L_rul   = Asymmetric RUL loss (penalises late predictions more than early)
              Standard in CMAPSS literature — better to predict earlier failure.
    L_fault = Binary Cross-Entropy with logits + optional pos_weight for
              class imbalance (healthy windows far outnumber fault windows).

    Parameters
    ----------
    alpha      : float — weight on RUL loss (default 0.7)
    pos_weight : float — upweight fault=1 class in BCE (default 2.0)
    """

    def __init__(self, alpha=0.7, pos_weight=2.0):
        super().__init__()
        self.alpha    = alpha
        self.bce      = nn.BCEWithLogitsLoss(
            pos_weight = torch.tensor([pos_weight])
        )

    def asymmetric_rul_loss(self, pred, target):
        """
        Penalise over-prediction (pred > target, i.e. model thinks more
        life remains than actually does) more than under-prediction.
        Based on Saxena et al. CMAPSS scoring function.
        """
        diff = pred - target
        # Late prediction (positive diff) → exp penalty
        # Early prediction (negative diff) → linear penalty
        loss = torch.where(
            diff > 0,
            torch.exp( diff / 10) - 1,   # late  → /10 is harsh   ✓
            torch.exp(-diff / 13) - 1    # early → /13 is lenient  ✓
        )
        return loss.mean()

    def forward(self, rul_pred, rul_target, fault_logit, fault_target,
                device='cpu'):
        self.bce.pos_weight = self.bce.pos_weight.to(device)

        l_rul   = self.asymmetric_rul_loss(rul_pred.squeeze(), rul_target)
        l_fault = self.bce(fault_logit.squeeze(), fault_target.float())

        return self.alpha * l_rul + (1 - self.alpha) * l_fault, l_rul, l_fault


# ──────────────────────────────────────────────
#  QUICK SMOKE-TEST
# ──────────────────────────────────────────────

if __name__ == '__main__':
    model = CNNLSTM(input_size=14, seq_len=30)
    print(f"Parameters : {model.count_parameters():,}")

    dummy   = torch.randn(8, 30, 14)     # batch=8, seq=30, features=14
    rul, fault = model(dummy)
    print(f"RUL output : {rul.shape}")   # (8, 1)
    print(f"Fault logit: {fault.shape}") # (8, 1)

    criterion = DualTaskLoss()
    rul_t   = torch.rand(8) * 200
    fault_t = (rul_t < 125).float()
    loss, l_r, l_f = criterion(rul, rul_t, fault, fault_t)
    print(f"Total loss : {loss.item():.4f}  "
          f"(RUL={l_r.item():.4f}, Fault={l_f.item():.4f})")
