import copy
import torch
import torch.nn as nn


class EarlyPredCNNLSTM(nn.Module):
    """
    Separate single-head early prediction model.

    Backbone is kept the SAME as the partner's CNN-LSTM model:
      Conv1d -> BN -> ReLU -> Pool
      Conv1d -> BN -> ReLU -> Pool
      LSTM -> LSTM

    Difference:
      final layer outputs a RAW LOGIT for binary classification,
      so training should use BCEWithLogitsLoss.
    """
    def __init__(self, input_size=102, cnn_filters=[64, 128],
                 lstm_hidden=[128, 64], dropout=0.2):
        super(EarlyPredCNNLSTM, self).__init__()

        self.input_size = input_size

        # same CNN backbone
        self.conv1 = nn.Conv1d(input_size, cnn_filters[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(cnn_filters[0])
        self.pool1 = nn.MaxPool1d(2)      # 30 -> 15

        self.conv2 = nn.Conv1d(cnn_filters[0], cnn_filters[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(cnn_filters[1])
        self.pool2 = nn.MaxPool1d(2)      # 15 -> 7

        self.dropout_cnn = nn.Dropout(dropout)

        # same LSTM backbone
        self.lstm1 = nn.LSTM(cnn_filters[1], lstm_hidden[0], batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(lstm_hidden[0], lstm_hidden[1], batch_first=True)
        self.dropout2 = nn.Dropout(dropout)

        # separate binary classification head (raw logit)
        self.classifier = nn.Linear(lstm_hidden[1], 1)
        nn.init.zeros_(self.classifier.bias)
        nn.init.xavier_uniform_(self.classifier.weight)

    def extract_features(self, x):
        # x: [batch, 30, 102]
        x = x.transpose(1, 2)                  # [batch, 102, 30]

        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)                      # [batch, 64, 15]

        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)                      # [batch, 128, 7]
        x = self.dropout_cnn(x)

        x = x.transpose(1, 2)                  # [batch, 7, 128]

        x, _ = self.lstm1(x)
        x = self.dropout1(x)

        _, (h_n, _) = self.lstm2(x)
        h = self.dropout2(h_n.squeeze(0))      # [batch, 64]

        return h
    def forward(self, x):
        h = self.extract_features(x)
        logit = self.classifier(h)             # raw logit, no sigmoid here
        return logit

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            logit = self.forward(x)
            prob = torch.sigmoid(logit)
        return prob

    def clone(self):
        return copy.deepcopy(self)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def load_backbone_from_rul_checkpoint(self, checkpoint_path, device='cpu'):
        """
        Load matching CNN/LSTM weights from a trained partner CNN-LSTM checkpoint.
        It will ignore unmatched final-layer keys like fc/classifier.
        """
        ckpt = torch.load(checkpoint_path, map_location=device)

        if isinstance(ckpt, dict):
            if 'model_state_dict' in ckpt:
                state_dict = ckpt['model_state_dict']
            elif 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt

        current = self.state_dict()
        matched = {}
        skipped = []

        for k, v in state_dict.items():
            # allow loading from original model keys only when names match exactly
            if k in current and current[k].shape == v.shape:
                matched[k] = v
            else:
                skipped.append(k)

        current.update(matched)
        self.load_state_dict(current)

        return {
            'num_loaded': len(matched),
            'num_skipped': len(skipped),
            'loaded_keys': sorted(list(matched.keys())),
            'skipped_keys': skipped,
        }
if __name__ == '__main__':
    m = EarlyPredCNNLSTM(input_size=102)
    x = torch.randn(32, 30, 102)
    o = m(x)
    print(f"Output shape : {o.shape}")
    print(f"Parameters   : {m.count_parameters():,}")