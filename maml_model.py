"""
MAML Model for Predictive Maintenance — CNN-LSTM hybrid

OUTPUT: sigmoid activation so predictions are always in [0, 1],
matching the normalised labels (y / max_rul).  This eliminates the
negative-output / clipping problem that caused high test RMSE.
"""

import torch
import torch.nn as nn
import copy


class CNNLSTMBase(nn.Module):

    def __init__(self, input_size=102, cnn_filters=[64, 128],
                 lstm_hidden=[128, 64], dropout=0.2):
        super(CNNLSTMBase, self).__init__()

        self.input_size = input_size

        # ── CNN ────────────────────────────────────────────────────────────
        self.conv1 = nn.Conv1d(input_size, cnn_filters[0], kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(cnn_filters[0])
        self.pool1 = nn.MaxPool1d(2)                    # 30 → 15

        self.conv2 = nn.Conv1d(cnn_filters[0], cnn_filters[1], kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(cnn_filters[1])
        self.pool2 = nn.MaxPool1d(2)                    # 15 → 7

        self.dropout_cnn = nn.Dropout(dropout)

        # ── LSTM ───────────────────────────────────────────────────────────
        # CNN output: [batch, 128, 7] → transpose → [batch, 7, 128]
        self.lstm1    = nn.LSTM(cnn_filters[1], lstm_hidden[0], batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        self.lstm2    = nn.LSTM(lstm_hidden[0], lstm_hidden[1], batch_first=True)
        self.dropout2 = nn.Dropout(dropout)

        # ── Output ─────────────────────────────────────────────────────────
        self.fc = nn.Linear(lstm_hidden[1], 1)
        nn.init.zeros_(self.fc.bias)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        # x: [batch, 30, 102]
        x = x.transpose(1, 2)                          # [batch, 102, 30]

        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)                               # [batch, 64, 15]

        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)                               # [batch, 128, 7]
        x = self.dropout_cnn(x)

        x = x.transpose(1, 2)                          # [batch, 7, 128]

        x, _        = self.lstm1(x)
        x           = self.dropout1(x)
        _, (h_n, _) = self.lstm2(x)
        h           = self.dropout2(h_n.squeeze(0))    # [batch, 64]

        # sigmoid keeps output in (0, 1) — matches normalised labels
        return torch.sigmoid(self.fc(h))               # [batch, 1]

    def clone(self):
        return copy.deepcopy(self)


if __name__ == '__main__':
    m = CNNLSTMBase(input_size=102)
    x = torch.randn(32, 30, 102)
    o = m(x)
    print(f"Output shape : {o.shape}")
    print(f"Output range : [{o.min().item():.4f}, {o.max().item():.4f}]  (should be in 0–1)")