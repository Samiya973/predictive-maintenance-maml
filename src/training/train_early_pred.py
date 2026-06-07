import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.data.data_loader import load_preprocessed_data
from src.models.early_pred_model import EarlyPredCNNLSTM


class EarlyPredDataset(Dataset):
    def __init__(self, X, y_rul_norm, fault_threshold_norm, augment=False):
        self.X = torch.FloatTensor(X)
        self.y_rul = torch.FloatTensor(y_rul_norm)
        self.y_fault = (self.y_rul <= fault_threshold_norm).float()
        self.augment = augment

        n_fault = int(self.y_fault.sum().item())
        n_healthy = len(self.y_fault) - n_fault
        print(
            f"  Dataset: {len(self.X)} samples | "
            f"fault={n_fault} ({100*n_fault/len(self.X):.1f}%) | "
            f"healthy={n_healthy} ({100*n_healthy/len(self.X):.1f}%)"
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment:
            x = x + torch.randn_like(x) * 0.01
        return x, self.y_fault[idx]

    def get_pos_weight(self):
        n_fault = float(self.y_fault.sum().item())
        n_total = float(len(self.y_fault))
        n_healthy = n_total - n_fault
        if n_fault == 0:
            return 1.0
        return max(n_healthy / n_fault, 1.0)


def find_best_threshold(y_true, y_prob):
    best_thr = 0.5
    best_f1 = -1.0

    for thr in np.arange(0.10, 0.91, 0.05):
        y_pred = (y_prob >= thr).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)

    return best_thr, best_f1

def evaluate_model(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0.0
    all_prob = []
    all_true = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch).squeeze(1)
            loss = criterion(logits, y_batch)
            total_loss += loss.item()

            prob = torch.sigmoid(logits).cpu().numpy()
            all_prob.extend(prob)
            all_true.extend(y_batch.cpu().numpy())

    y_prob = np.array(all_prob)
    y_true = np.array(all_true).astype(int)
    y_pred = (y_prob >= threshold).astype(int)

    if len(np.unique(y_true)) >= 2:
        auc_roc = float(roc_auc_score(y_true, y_prob))
        auc_pr = float(average_precision_score(y_true, y_prob))
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        far = float(fp / (fp + tn + 1e-8))
    else:
        auc_roc = float('nan')
        auc_pr = float('nan')
        precision, recall, f1 = 0.0, 0.0, 0.0
        far = float('nan')

    return {
        'loss': total_loss / max(len(loader), 1),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'false_alarm_rate': far,
        'y_true': y_true,
        'y_prob': y_prob,
        'y_pred': y_pred,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='data/processed/FD001_preprocessed.npz')
    parser.add_argument('--save_path', type=str, default='results/saved_models/early_pred_best.pth')
    parser.add_argument('--fault_rul_cycles', type=float, default=30.0,
                        help='RUL threshold in cycles for positive class. 30 is recommended. 15 is stricter.')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--use_pretrained_rul', action='store_true')
    parser.add_argument('--pretrained_rul_path', type=str, default='results/saved_models/cnn_lstm_best.pth')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print('\nLoading preprocessed data...')
    data = load_preprocessed_data(args.dataset)

    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']

    max_rul = float(data['max_rul']) if 'max_rul' in data else 130.0
    fault_threshold_norm = args.fault_rul_cycles / max_rul

    print(f'  Train : {len(X_train)} samples  |  input shape: {X_train.shape[1:]}')
    print(f'  max_rul={max_rul}')
    print(f'  fault_rul_cycles={args.fault_rul_cycles}')
    print(f'  fault_threshold(normalized)={fault_threshold_norm:.4f}')

    print('\nBuilding datasets...')
    train_ds = EarlyPredDataset(X_train, y_train, fault_threshold_norm, augment=True)
    val_ds = EarlyPredDataset(X_val, y_val, fault_threshold_norm, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    model = EarlyPredCNNLSTM(
        input_size=X_train.shape[2],
        dropout=args.dropout,
    ).to(device)

    if args.use_pretrained_rul and os.path.exists(args.pretrained_rul_path):
        print(f'\nLoading backbone from RUL checkpoint: {args.pretrained_rul_path}')
        info = model.load_backbone_from_rul_checkpoint(args.pretrained_rul_path, device=device)
        print(f"  Loaded keys : {info['num_loaded']}")
        print(f"  Skipped keys: {info['num_skipped']}")

    pos_weight = train_ds.get_pos_weight()
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device)
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=4
    )

    best_f1 = -1.0
    best_epoch = 0
    patience_count = 0

    print('=' * 60)
    print('EARLY PREDICTION TRAINING — CNN-LSTM')
    print('=' * 60)
    print(f'  Parameters    : {model.count_parameters():,}')
    print(f'  Device        : {device}')
    print(f'  LR            : {args.lr}')
    print(f'  Batch size    : {args.batch_size}')
    print(f'  pos_weight    : {pos_weight:.4f}')
    print('-' * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch).squeeze(1)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()

        train_loss = total_train_loss / max(len(train_loader), 1)

        val_raw = evaluate_model(model, val_loader, criterion, device, threshold=0.5)
        best_thr, _ = find_best_threshold(val_raw['y_true'], val_raw['y_prob'])
        val_metrics = evaluate_model(model, val_loader, criterion, device, threshold=best_thr)

        scheduler.step(val_metrics['f1'])
        lr_now = optimizer.param_groups[0]['lr']

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Epoch {epoch:4d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f} | "
                f"AUC: {val_metrics['auc_roc']:.4f} | "
                f"FAR: {val_metrics['false_alarm_rate']:.4f} | "
                f"Thr: {best_thr:.2f} | "
                f"LR: {lr_now:.2e} | "
                f"{time.time()-t0:.1f}s"
            )

        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_epoch = epoch
            patience_count = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_val_f1': best_f1,
                'best_threshold': best_thr,
                'fault_rul_cycles': args.fault_rul_cycles,
                'max_rul': max_rul,
                'config': {
                    'input_size': X_train.shape[2],
                    'dropout': args.dropout,
                }
            }, args.save_path)
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (best val_f1={best_f1:.4f} at epoch {best_epoch})")
                break

    print('-' * 60)
    print(f'Best val F1: {best_f1:.4f} (epoch {best_epoch})')
    print(f'Saved -> {args.save_path}')
    print('=' * 60)


if __name__ == '__main__':
    main()
