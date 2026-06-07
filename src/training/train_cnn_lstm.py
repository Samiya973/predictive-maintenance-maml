"""
train_cnn_lstm.py
─────────────────
Training + comprehensive evaluation for the CNN+LSTM early fault detector.

WHAT "EARLY DETECTION" MEANS HERE
───────────────────────────────────
Standard RUL prediction tells you *how many cycles remain*.
Early detection answers: "Has degradation started yet?" as early as possible.

We define fault onset = RUL drops below `early_threshold` (default 125 cycles).
The model must raise an alarm BEFORE the engine actually reaches that threshold.

KEY EVALUATION METRICS (beyond RMSE)
──────────────────────────────────────
  Precision / Recall / F1 — fault detection quality
  AUC-ROC                 — ranking quality across all thresholds
  False Alarm Rate (FAR)  — how often healthy engines are flagged
  Detection Latency       — median cycles *before* threshold that alarm fires
  NASA Scoring Function   — asymmetric cost (late prediction >> early prediction)

USAGE
──────────────────────────────────────
  python src/training/train_cnn_lstm.py
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    roc_auc_score, precision_recall_fscore_support,
    confusion_matrix, average_precision_score
)

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.models.cnn_lstm import CNNLSTM, DualTaskLoss
from src.data.data_loader import load_preprocessed_data

# ──────────────────────────────────────────────
#  DATASET
# ──────────────────────────────────────────────

class RULDataset(Dataset):
    """
    Wraps (X, y_rul) arrays into a PyTorch Dataset.

    Automatically generates binary early-fault labels:
        fault = 1  if  RUL <= early_threshold
        fault = 0  otherwise

    Parameters
    ----------
    X               : np.ndarray  (N, seq_len, features)
    y_rul           : np.ndarray  (N,)  — remaining useful life in cycles
    early_threshold : int         — cycles threshold for fault label
    augment         : bool        — add Gaussian noise for training augmentation
    """

    def __init__(self, X, y_rul, early_threshold=125, augment=False):
        self.X               = torch.FloatTensor(X)
        self.y_rul           = torch.FloatTensor(y_rul)
        self.y_fault         = (self.y_rul <= early_threshold).float()
        self.augment         = augment
        self.early_threshold = early_threshold

        n_fault   = self.y_fault.sum().item()
        n_healthy = len(self.y_fault) - n_fault
        print(f"  Dataset: {len(self.X)} samples | "
              f"fault={int(n_fault)} ({100*n_fault/len(self.X):.1f}%) | "
              f"healthy={int(n_healthy)} ({100*n_healthy/len(self.X):.1f}%)")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment:
            # Small Gaussian noise makes model robust to sensor calibration drift
            x = x + torch.randn_like(x) * 0.02
        return x, self.y_rul[idx], self.y_fault[idx]

    def get_sampler(self):
        """
        WeightedRandomSampler to balance fault/healthy classes per batch.
        Without this, early epochs ignore the rare fault class entirely.
        """
        weights     = torch.where(self.y_fault == 1,
                                  torch.tensor(2.0),
                                  torch.tensor(1.0))
        return WeightedRandomSampler(weights, len(weights), replacement=True)


# ──────────────────────────────────────────────
#  EVALUATION UTILITIES
# ──────────────────────────────────────────────

def nasa_score(rul_pred, rul_true):
    """
    NASA asymmetric scoring function from the CMAPSS challenge.
    Late predictions (pred > true) are penalised exponentially harder.

        s = sum( exp( d/13 ) - 1  if d >= 0  else  exp( -d/10 ) - 1 )

    Lower is better. Perfect prediction → 0.
    """
    d = rul_pred - rul_true
    s = np.where(d >= 0, np.exp(d / 13) - 1, np.exp(-d / 10) - 1)
    return float(np.sum(s))


def detection_latency(y_true_fault, y_pred_fault, y_rul,
                      early_threshold=125):
    """
    For each engine (identified by contiguous fault-label segments),
    measure how many cycles *before* threshold the model first raised
    an alarm.

    Positive latency = alarm before threshold (good — early detection).
    Negative latency = alarm after threshold  (bad  — missed or late).

    Returns
    -------
    median_latency : float  — median cycles before threshold across engines
    latencies      : list   — per-detection latency values
    n_missed       : int    — engines where fault was never detected
    """
    latencies = []
    n_missed  = 0

    # Find contiguous fault regions (one per engine degradation phase)
    in_fault  = False
    fault_start = None

    for i in range(len(y_true_fault)):
        if y_true_fault[i] == 1 and not in_fault:
            in_fault    = True
            fault_start = i

        if (y_true_fault[i] == 0 or i == len(y_true_fault) - 1) and in_fault:
            in_fault    = False
            region_end  = i

            # Find first alarm in this fault region
            region_preds = y_pred_fault[fault_start:region_end]
            alarm_indices = np.where(region_preds == 1)[0]

            if len(alarm_indices) == 0:
                n_missed += 1
            else:
                # Latency = cycles before threshold the alarm fires
                # RUL at fault_start ≈ early_threshold (the onset)
                # Alarm fires at fault_start + alarm_indices[0]
                # Latency = RUL at fault onset - RUL at alarm
                rul_at_alarm = y_rul[fault_start + alarm_indices[0]]
                latency      = float(early_threshold - rul_at_alarm)
                latencies.append(latency)

    median_lat = float(np.median(latencies)) if latencies else float('nan')
    return median_lat, latencies, n_missed


def evaluate_model(model, loader, criterion, device, early_threshold=125,
                   fault_threshold=0.5):
    """
    Full evaluation pass.

    Returns a dict with all metrics:
        rmse, mae, r2, nasa_score
        precision, recall, f1, auc_roc, auc_pr
        false_alarm_rate, detection_latency, n_missed
    """
    model.eval()
    all_rul_pred  = []
    all_rul_true  = []
    all_fault_prob = []
    all_fault_true = []
    total_loss    = 0.0

    with torch.no_grad():
        for X_batch, rul_batch, fault_batch in loader:
            X_batch     = X_batch.to(device)
            rul_batch   = rul_batch.to(device)
            fault_batch = fault_batch.to(device)

            rul_pred, fault_logit = model(X_batch)
            loss, _, _            = criterion(rul_pred, rul_batch,
                                              fault_logit, fault_batch,
                                              device=device)
            total_loss += loss.item()

            prob = torch.sigmoid(fault_logit).cpu().numpy().flatten()
            all_rul_pred.extend(rul_pred.cpu().numpy().flatten())
            all_rul_true.extend(rul_batch.cpu().numpy().flatten())
            all_fault_prob.extend(prob)
            all_fault_true.extend(fault_batch.cpu().numpy().flatten())

    rul_pred_np  = np.array(all_rul_pred)
    rul_true_np  = np.array(all_rul_true)
    fault_prob   = np.array(all_fault_prob)
    fault_true   = np.array(all_fault_true).astype(int)
    fault_pred   = (fault_prob >= fault_threshold).astype(int)

    # ── RUL metrics ──────────────────────────────────────────────────
    rmse    = float(np.sqrt(np.mean((rul_pred_np - rul_true_np) ** 2)))
    mae     = float(np.mean(np.abs(rul_pred_np - rul_true_np)))
    ss_res  = np.sum((rul_true_np - rul_pred_np) ** 2)
    ss_tot  = np.sum((rul_true_np - rul_true_np.mean()) ** 2)
    r2      = float(1 - ss_res / (ss_tot + 1e-8))
    n_score = nasa_score(rul_pred_np, rul_true_np)

    # ── Fault detection metrics ───────────────────────────────────────
    try:
        auc_roc = float(roc_auc_score(fault_true, fault_prob))
        auc_pr  = float(average_precision_score(fault_true, fault_prob))
    except ValueError:
        auc_roc = float('nan')
        auc_pr  = float('nan')

    prec, rec, f1, _ = precision_recall_fscore_support(
        fault_true, fault_pred, average='binary', zero_division=0
    )
    cm  = confusion_matrix(fault_true, fault_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    far = float(fp / (fp + tn + 1e-8))   # false alarm rate

    # ── Detection latency ─────────────────────────────────────────────
    med_lat, latencies, n_missed = detection_latency(
        fault_true, fault_pred, rul_true_np, early_threshold
    )

    return {
        # RUL
        'loss'            : total_loss / max(len(loader), 1),
        'rmse'            : rmse,
        'mae'             : mae,
        'r2'              : r2,
        'nasa_score'      : n_score,
        # Fault detection
        'precision'       : float(prec),
        'recall'          : float(rec),
        'f1'              : float(f1),
        'auc_roc'         : auc_roc,
        'auc_pr'          : auc_pr,
        'false_alarm_rate': far,
        # Early detection
        'detection_latency': med_lat,
        'n_missed'         : n_missed,
        'latencies'        : latencies,
        # Raw arrays for plotting
        'rul_pred'         : rul_pred_np,
        'rul_true'         : rul_true_np,
        'fault_prob'       : fault_prob,
        'fault_true'       : fault_true,
    }


def print_metrics(metrics, split='Val'):
    print(f"\n{'─'*55}")
    print(f"  {split} Results")
    print(f"{'─'*55}")
    print(f"  RUL Regression:")
    print(f"    RMSE              : {metrics['rmse']:.4f} cycles")
    print(f"    MAE               : {metrics['mae']:.4f} cycles")
    print(f"    R²                : {metrics['r2']:.4f}")
    print(f"    NASA Score        : {metrics['nasa_score']:.2f}  (lower=better)")
    print(f"  Early Fault Detection:")
    print(f"    Precision         : {metrics['precision']:.4f}")
    print(f"    Recall            : {metrics['recall']:.4f}")
    print(f"    F1 Score          : {metrics['f1']:.4f}")
    print(f"    AUC-ROC           : {metrics['auc_roc']:.4f}")
    print(f"    AUC-PR            : {metrics['auc_pr']:.4f}")
    print(f"    False Alarm Rate  : {metrics['false_alarm_rate']:.4f}")
    print(f"  Early Detection Quality:")
    print(f"    Median Latency    : {metrics['detection_latency']:.1f} cycles before threshold")
    print(f"    Missed Detections : {metrics['n_missed']}")
    print(f"{'─'*55}")


# ──────────────────────────────────────────────
#  TRAINER
# ──────────────────────────────────────────────

class CNNLSTMTrainer:
    """
    Manages training loop, LR scheduling, early stopping, and checkpointing.

    Early stopping watches F1 score on the val set (not RMSE) because the
    primary goal is early detection quality, not regression accuracy.
    You can change `monitor` to 'rmse' or 'auc_roc' if needed.
    """

    def __init__(self, config):
        self.cfg    = config
        self.device = torch.device(config.get('device', 'cpu'))

        self.model = CNNLSTM(
            input_size      = config.get('input_size', 14),
            seq_len         = config.get('seq_len', 30),
            cnn_channels    = config.get('cnn_channels', 64),
            cnn_layers      = config.get('cnn_layers', 3),
            lstm_hidden     = config.get('lstm_hidden', 128),
            lstm_layers     = config.get('lstm_layers', 2),
            dropout         = config.get('dropout', 0.3),
            early_threshold = config.get('early_threshold', 125),
        ).to(self.device)

        self.criterion = DualTaskLoss(
            alpha      = config.get('loss_alpha', 0.7),
            pos_weight = config.get('pos_weight', 2.0),
        )

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr           = config.get('lr', 1e-3),
            weight_decay = config.get('weight_decay', 1e-4),
        )

        # Cosine annealing: smoothly reduces LR to lr_min over n_epochs
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max = config.get('n_epochs', 100),
            eta_min = config.get('lr_min', 1e-5),
        )

        self.history = {
            'train_loss': [], 'val_loss': [],
            'val_rmse'  : [], 'val_f1'  : [],
            'val_auc'   : [], 'val_latency': [],
        }

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0

        for X_batch, rul_batch, fault_batch in loader:
            X_batch     = X_batch.to(self.device)
            rul_batch   = rul_batch.to(self.device)
            fault_batch = fault_batch.to(self.device)

            self.optimizer.zero_grad()
            rul_pred, fault_logit = self.model(X_batch)
            loss, _, _ = self.criterion(
                rul_pred, rul_batch, fault_logit, fault_batch,
                device=self.device
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(loader)

    def train(self, train_loader, val_loader):
        cfg       = self.cfg
        n_epochs  = cfg.get('n_epochs', 100)
        patience  = cfg.get('patience', 20)
        monitor   = cfg.get('monitor', 'f1')    # 'f1' | 'rmse' | 'auc_roc'
        save_path = cfg.get('save_path',
                            'results/saved_models/cnn_lstm_best.pth')

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Initialise best depending on metric direction
        best_val    = -float('inf') if monitor != 'rmse' else float('inf')
        patience_ct = 0
        best_epoch  = 0

        print("=" * 60)
        print("CNN+LSTM TRAINING  —  Early Fault Detection")
        print("=" * 60)
        print(f"  Parameters    : {self.model.count_parameters():,}")
        print(f"  Device        : {self.device}")
        print(f"  Monitor       : val_{monitor}")
        print(f"  LR            : {cfg.get('lr', 1e-3)}")
        print(f"  Batch size    : {cfg.get('batch_size', 64)}")
        print(f"  Loss alpha    : {cfg.get('loss_alpha', 0.7)} "
              f"(RUL) / {1-cfg.get('loss_alpha',0.7):.1f} (Fault)")
        print("-" * 60)

        for epoch in range(1, n_epochs + 1):
            t0         = time.time()
            train_loss = self.train_epoch(train_loader)
            val_metrics = evaluate_model(
                self.model, val_loader, self.criterion, self.device,
                early_threshold = cfg.get('early_threshold', 125),
                fault_threshold = cfg.get('fault_threshold', 0.5),
            )
            self.scheduler.step()

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_rmse'].append(val_metrics['rmse'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['val_auc'].append(val_metrics['auc_roc'])
            self.history['val_latency'].append(val_metrics['detection_latency'])

            # Print every 5 epochs
            if epoch % 5 == 0 or epoch == 1:
                lr_now = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:4d} | "
                      f"Loss: {train_loss:.4f} | "
                      f"Val RMSE: {val_metrics['rmse']:.2f} | "
                      f"F1: {val_metrics['f1']:.4f} | "
                      f"AUC: {val_metrics['auc_roc']:.4f} | "
                      f"Latency: {val_metrics['detection_latency']:.1f}cy | "
                      f"LR: {lr_now:.2e} | "
                      f"{time.time()-t0:.1f}s")

            # Early stopping check
            current = val_metrics[monitor]
            improved = (current > best_val) if monitor != 'rmse' else (current < best_val)

            if improved:
                best_val    = current
                patience_ct = 0
                best_epoch  = epoch
                torch.save({
                    'epoch'           : epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state' : self.optimizer.state_dict(),
                    'val_metrics'     : val_metrics,
                    'config'          : cfg,
                    'history'         : self.history,
                }, save_path)
            else:
                patience_ct += 1
                if patience_ct >= patience:
                    print(f"\nEarly stopping at epoch {epoch} "
                          f"(best val_{monitor}={best_val:.4f} at epoch {best_epoch})")
                    break

        print("-" * 60)
        print(f"Training complete! Best val_{monitor}: {best_val:.4f} "
              f"(epoch {best_epoch})")
        print(f"Model saved → {save_path}")
        print("=" * 60)
        return self.history


# ──────────────────────────────────────────────
#  THRESHOLD FINDER
# ──────────────────────────────────────────────

def find_optimal_fault_threshold(model, val_loader, device,
                                 early_threshold=125):
    """
    Sweep fault probability thresholds [0.1 … 0.9] and pick the one
    that maximises F1 on the validation set.

    Returns the optimal threshold float.
    """
    model.eval()
    all_prob  = []
    all_true  = []

    with torch.no_grad():
        for X_b, _, fault_b in val_loader:
            _, logit = model(X_b.to(device))
            prob     = torch.sigmoid(logit).cpu().numpy().flatten()
            all_prob.extend(prob)
            all_true.extend(fault_b.numpy())

    all_prob = np.array(all_prob)
    all_true = np.array(all_true).astype(int)

    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(0.1, 0.91, 0.05):
        pred = (all_prob >= thr).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            all_true, pred, average='binary', zero_division=0
        )
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)

    print(f"  Optimal fault threshold : {best_thr:.2f}  (val F1={best_f1:.4f})")
    return best_thr


# ──────────────────────────────────────────────
#  FINAL EVALUATION REPORT
# ──────────────────────────────────────────────

def full_evaluation(model_path, data_dict, config, device):
    """
    Load best checkpoint and run full evaluation on test set.
    Prints comprehensive report and saves results.
    """
    print("\n" + "=" * 60)
    print("CNN+LSTM FINAL EVALUATION  —  Test Set")
    print("=" * 60)

    # ── Rebuild test dataset & loader ────────────────────────────────
    test_ds = RULDataset(
        data_dict['X_test'],
        data_dict['y_test'],
        early_threshold = config.get('early_threshold', 125),
        augment         = False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size = config.get('batch_size', 256),
        shuffle    = False,
        num_workers= 0,
    )

    # ── Load model ────────────────────────────────────────────────────
    model = CNNLSTM(
        input_size      = config.get('input_size', 14),
        seq_len         = config.get('seq_len', 30),
        cnn_channels    = config.get('cnn_channels', 64),
        cnn_layers      = config.get('cnn_layers', 3),
        lstm_hidden     = config.get('lstm_hidden', 128),
        lstm_layers     = config.get('lstm_layers', 2),
        dropout         = config.get('dropout', 0.3),
        early_threshold = config.get('early_threshold', 125),
    ).to(device)

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    criterion = DualTaskLoss(
        alpha      = config.get('loss_alpha', 0.7),
        pos_weight = config.get('pos_weight', 2.0),
    )

    # ── Find optimal threshold on val set first ───────────────────────
    val_ds = RULDataset(
        data_dict['X_val'],
        data_dict['y_val'],
        early_threshold = config.get('early_threshold', 125),
    )
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    opt_thr    = find_optimal_fault_threshold(model, val_loader, device,
                                              config.get('early_threshold', 125))

    # ── Evaluate on test set ──────────────────────────────────────────
    metrics = evaluate_model(
        model, test_loader, criterion, device,
        early_threshold = config.get('early_threshold', 125),
        fault_threshold = opt_thr,
    )

    print_metrics(metrics, split='Test')

    # ── Latency distribution ──────────────────────────────────────────
    if metrics['latencies']:
        lats = np.array(metrics['latencies'])
        print(f"\n  Detection Latency Distribution:")
        print(f"    Mean   : {lats.mean():.1f} cycles")
        print(f"    Median : {np.median(lats):.1f} cycles")
        print(f"    Std    : {lats.std():.1f} cycles")
        print(f"    Min    : {lats.min():.1f} cycles")
        print(f"    Max    : {lats.max():.1f} cycles")
        print(f"    >0 cy  : {(lats>0).mean()*100:.1f}%  (alarm BEFORE threshold)")

    # ── Save results ──────────────────────────────────────────────────
    out_dir = 'results/saved_models'
    os.makedirs(out_dir, exist_ok=True)

    save_dict = {k: v for k, v in metrics.items()
                 if k not in ('rul_pred', 'rul_true', 'fault_prob',
                              'fault_true', 'latencies')}
    save_dict['optimal_threshold'] = opt_thr
    np.save(os.path.join(out_dir, 'cnn_lstm_test_metrics.npy'), save_dict)

    np.save(os.path.join(out_dir, 'cnn_lstm_predictions.npy'), {
        'rul_pred'  : metrics['rul_pred'],
        'rul_true'  : metrics['rul_true'],
        'fault_prob': metrics['fault_prob'],
        'fault_true': metrics['fault_true'],
    })

    print(f"\n✓ Metrics  → {out_dir}/cnn_lstm_test_metrics.npy")
    print(f"✓ Raw preds → {out_dir}/cnn_lstm_predictions.npy")
    print("=" * 60)
    return metrics


# ──────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == '__main__':

    config = {
        # ── Data ──────────────────────────────
        'input_size'      : 14,       # number of sensor features after preprocessing
        'seq_len'         : 30,       # sliding window length (match preprocessing)
        'early_threshold' : 125,      # RUL <= this → fault label = 1

        # ── Model ─────────────────────────────
        'cnn_channels'    : 64,       # CNN feature maps
        'cnn_layers'      : 3,        # residual CNN blocks
        'lstm_hidden'     : 128,      # LSTM hidden units
        'lstm_layers'     : 2,        # stacked LSTM depth
        'dropout'         : 0.3,

        # ── Loss ──────────────────────────────
        'loss_alpha'      : 0.7,      # 0.7 * L_rul + 0.3 * L_fault
        'pos_weight'      : 2.0,      # upweight fault=1 class in BCE

        # ── Training ──────────────────────────
        'lr'              : 1e-3,
        'lr_min'          : 1e-5,
        'weight_decay'    : 1e-4,
        'batch_size'      : 64,
        'n_epochs'        : 100,
        'patience'        : 20,
        'monitor'         : 'f1',     # early stopping metric: 'f1'|'rmse'|'auc_roc'
        'fault_threshold' : 0.5,      # fault probability cutoff (tuned later)

        # ── Misc ──────────────────────────────
        'save_path'       : 'results/saved_models/cnn_lstm_best.pth',
        'device'          : 'cuda' if torch.cuda.is_available() else 'cpu',
    }

    device = torch.device(config['device'])
    print(f"Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────
    print("\nLoading preprocessed data...")
    data_dict = load_preprocessed_data('data/processed/FD001_preprocessed.npz')

    X_train, y_train = data_dict['X_train'], data_dict['y_train']
    X_val,   y_val   = data_dict['X_val'],   data_dict['y_val']

    print(f"  Train : {len(X_train)} samples  |  "
          f"input shape: {X_train.shape[1:]}")

    # ── Auto-detect input_size and seq_len from data ──────────────────
    if X_train.ndim == 3:
        config['seq_len']    = X_train.shape[1]
        config['input_size'] = X_train.shape[2]
    elif X_train.ndim == 2:
        # Flat features: reshape to (N, 1, F) so CNN has a time dimension
        X_train = X_train[:, np.newaxis, :]
        X_val   = X_val[:,   np.newaxis, :]
        data_dict['X_test'] = data_dict['X_test'][:, np.newaxis, :]
        config['seq_len']    = 1
        config['input_size'] = X_train.shape[2]

    print(f"  seq_len={config['seq_len']}  input_size={config['input_size']}")

    # ── Build datasets ────────────────────────────────────────────────
    print("\nBuilding datasets...")
    train_ds = RULDataset(X_train, y_train,
                          early_threshold = config['early_threshold'],
                          augment         = True)
    val_ds   = RULDataset(X_val,   y_val,
                          early_threshold = config['early_threshold'],
                          augment         = False)

    # Balanced sampler prevents model ignoring rare fault class early in training
    train_loader = DataLoader(
        train_ds,
        batch_size  = config['batch_size'],
        sampler     = train_ds.get_sampler(),
        num_workers = 0,
        pin_memory  = (device.type == 'cuda'),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = 256,
        shuffle     = False,
        num_workers = 0,
    )

    # ── Train ─────────────────────────────────────────────────────────
    trainer = CNNLSTMTrainer(config)
    history = trainer.train(train_loader, val_loader)

    # ── Final test evaluation ─────────────────────────────────────────
    full_evaluation(
        model_path = config['save_path'],
        data_dict  = data_dict,
        config     = config,
        device     = device,
    )
