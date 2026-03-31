import os
import sys
import csv
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
)

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.data.data_loader import load_preprocessed_data
from src.models.early_pred_model import EarlyPredCNNLSTM


class EarlyPredDataset(Dataset):
    def __init__(self, X, y_rul_norm, fault_threshold_norm):
        self.X = torch.FloatTensor(X)
        self.y_rul = torch.FloatTensor(y_rul_norm)
        self.y_fault = (self.y_rul <= fault_threshold_norm).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_rul[idx], self.y_fault[idx]


def find_best_threshold(y_true, y_prob):
    best_thr = 0.5
    best_f1 = -1.0

    for thr in np.arange(0.05, 0.91, 0.05):
        y_pred = (y_prob >= thr).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)

    return best_thr, best_f1


def detection_latency(engine_ids, rul_true_cycles, fault_pred, fault_rul_cycles):
    latencies = []
    missed = 0

    unique_engines = np.unique(engine_ids)

    for eng in unique_engines:
        mask = engine_ids == eng
        rul_eng = rul_true_cycles[mask]
        pred_eng = fault_pred[mask]

        onset_idx = np.where(rul_eng <= fault_rul_cycles)[0]
        if len(onset_idx) == 0:
            continue

        det_idx = np.where(pred_eng == 1)[0]
        if len(det_idx) == 0:
            missed += 1
            continue

        det_idx = det_idx[0]
        rul_at_det = rul_eng[det_idx]

        # positive = early, negative = late
        latencies.append(float(rul_at_det - fault_rul_cycles))

    if len(latencies) == 0:
        return float('nan'), [], missed

    return float(np.median(latencies)), latencies, missed


def build_per_engine_report(engine_ids, rul_true_cycles, y_prob, y_pred, fault_rul_cycles):
    rows = []
    unique_engines = np.unique(engine_ids)

    for eng in unique_engines:
        mask = engine_ids == eng
        rul_eng = rul_true_cycles[mask]
        prob_eng = y_prob[mask]
        pred_eng = y_pred[mask]

        onset_idx = np.where(rul_eng <= fault_rul_cycles)[0]
        if len(onset_idx) == 0:
            continue
        onset_idx = int(onset_idx[0])

        det_idx = np.where(pred_eng == 1)[0]

        if len(det_idx) == 0:
            rows.append({
                'engine_id': int(eng),
                'onset_index': onset_idx,
                'detect_index': -1,
                'rul_at_detection': np.nan,
                'lead_time': np.nan,
                'detected': 0,
                'missed_detection': 1,
                'max_prob': float(np.max(prob_eng)),
            })
            continue

        det_idx = int(det_idx[0])
        rul_at_det = float(rul_eng[det_idx])
        lead_time = float(rul_at_det - fault_rul_cycles)   # positive = early

        rows.append({
            'engine_id': int(eng),
            'onset_index': onset_idx,
            'detect_index': det_idx,
            'rul_at_detection': rul_at_det,
            'lead_time': lead_time,
            'detected': 1,
            'missed_detection': 0,
            'max_prob': float(np.max(prob_eng)),
        })

    return rows


def save_per_engine_csv(rows, out_dir):
    csv_path = os.path.join(out_dir, 'early_pred_per_engine.csv')

    fieldnames = [
        'engine_id',
        'onset_index',
        'detect_index',
        'rul_at_detection',
        'lead_time',
        'detected',
        'missed_detection',
        'max_prob',
    ]

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved -> {csv_path}")


def save_per_engine_boxplot(rows, out_dir):
    lead_times = [r['lead_time'] for r in rows if not np.isnan(r['lead_time'])]

    if len(lead_times) == 0:
        return

    plt.figure(figsize=(5, 5))
    plt.boxplot(lead_times, vert=True, patch_artist=True)
    plt.axhline(0, linestyle='--', linewidth=1)
    plt.ylabel('Lead Time (cycles)')
    plt.title('Per-Engine Lead-Time Box Plot')

    boxplot_path = os.path.join(out_dir, 'early_pred_per_engine_boxplot.png')
    plt.savefig(boxplot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved -> {boxplot_path}")


def save_roc_curve(y_true, y_prob, auc_roc, out_dir):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc_roc:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve — Early Prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)

    roc_png = os.path.join(out_dir, 'early_pred_roc_curve.png')
    plt.savefig(roc_png, dpi=150, bbox_inches='tight')
    plt.close()

    np.save(os.path.join(out_dir, 'early_pred_roc.npy'), {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
    })

    print(f"Saved -> {roc_png}")
    print(f"Saved -> {os.path.join(out_dir, 'early_pred_roc.npy')}")


def save_threshold_tradeoff_plot(y_true, y_prob, out_dir, best_thr):
    thresholds = np.arange(0.05, 0.91, 0.05)

    precisions = []
    recalls = []
    f1s = []
    fprs = []

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fpr = float(fp / (fp + tn + 1e-8))

        precisions.append(float(precision))
        recalls.append(float(recall))
        f1s.append(float(f1))
        fprs.append(float(fpr))

    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, precisions, label='Precision', linewidth=2)
    plt.plot(thresholds, recalls, label='Recall / TPR', linewidth=2)
    plt.plot(thresholds, f1s, label='F1', linewidth=2)
    plt.plot(thresholds, fprs, label='FPR', linewidth=2)
    plt.axvline(best_thr, linestyle='--', linewidth=2, label=f'Selected threshold = {best_thr:.2f}')
    plt.xlabel('Decision Threshold')
    plt.ylabel('Metric Value')
    plt.title('Threshold Trade-off Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)

    tradeoff_png = os.path.join(out_dir, 'early_pred_threshold_tradeoff.png')
    plt.savefig(tradeoff_png, dpi=150, bbox_inches='tight')
    plt.close()

    np.save(os.path.join(out_dir, 'early_pred_threshold_tradeoff.npy'), {
        'thresholds': thresholds,
        'precision': np.array(precisions),
        'recall': np.array(recalls),
        'f1': np.array(f1s),
        'fpr': np.array(fprs),
        'best_threshold': best_thr,
    })

    print(f"Saved -> {tradeoff_png}")
    print(f"Saved -> {os.path.join(out_dir, 'early_pred_threshold_tradeoff.npy')}")


def save_lead_time_histogram(latencies, out_dir):
    if len(latencies) == 0:
        return

    plt.figure(figsize=(6, 5))
    plt.hist(latencies, bins=20, edgecolor='black')
    plt.axvline(np.median(latencies), linestyle='--', linewidth=2,
                label=f'Median = {np.median(latencies):.1f}')
    plt.xlabel('Lead Time (cycles)')
    plt.ylabel('Count')
    plt.title('Lead-Time Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    hist_png = os.path.join(out_dir, 'early_pred_lead_time_hist.png')
    plt.savefig(hist_png, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved -> {hist_png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='data/processed/FD001_preprocessed.npz')
    parser.add_argument('--model_path', type=str, default='results/saved_models/early_pred_best.pth')
    parser.add_argument('--out_dir', type=str, default='results')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print('\nLoading preprocessed data...')
    data = load_preprocessed_data(args.dataset)

    ckpt = torch.load(args.model_path, map_location=device)
    cfg = ckpt.get('config', {})
    max_rul = float(ckpt.get('max_rul', data['max_rul'] if 'max_rul' in data else 130.0))
    fault_rul_cycles = float(ckpt.get('fault_rul_cycles', 30.0))
    fault_threshold_norm = fault_rul_cycles / max_rul

    model = EarlyPredCNNLSTM(
        input_size=cfg.get('input_size', data['X_train'].shape[2]),
        dropout=cfg.get('dropout', 0.2),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    val_ds = EarlyPredDataset(data['X_val'], data['y_val'], fault_threshold_norm)
    test_ds = EarlyPredDataset(data['X_test'], data['y_test'], fault_threshold_norm)

    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    all_val_prob = []
    all_val_true = []

    with torch.no_grad():
        for X_batch, _, y_fault in val_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch).squeeze(1)
            prob = torch.sigmoid(logits).cpu().numpy()
            all_val_prob.extend(prob)
            all_val_true.extend(y_fault.numpy())

    all_val_prob = np.array(all_val_prob)
    all_val_true = np.array(all_val_true).astype(int)

    best_thr, best_f1 = find_best_threshold(all_val_true, all_val_prob)
    print(f'Optimal threshold from validation: {best_thr:.2f} (val F1={best_f1:.4f})')

    all_test_prob = []
    all_test_true = []
    all_test_rul = []

    with torch.no_grad():
        for X_batch, y_rul, y_fault in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch).squeeze(1)
            prob = torch.sigmoid(logits).cpu().numpy()

            all_test_prob.extend(prob)
            all_test_true.extend(y_fault.numpy())
            all_test_rul.extend(y_rul.numpy())

    y_prob = np.array(all_test_prob)
    y_true = np.array(all_test_true).astype(int)
    y_pred = (y_prob >= best_thr).astype(int)
    y_rul_cycles = np.array(all_test_rul) * max_rul

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

    per_engine_rows = []
    if 'test_engine_ids' in data:
        test_engine_ids = np.array(data['test_engine_ids'])
        med_lat, latencies, n_missed = detection_latency(
            test_engine_ids, y_rul_cycles, y_pred, fault_rul_cycles
        )
        per_engine_rows = build_per_engine_report(
            test_engine_ids,
            y_rul_cycles,
            y_prob,
            y_pred,
            fault_rul_cycles
        )
    else:
        med_lat, latencies, n_missed = float('nan'), [], 0

    print('\n' + '=' * 60)
    print('EARLY PREDICTION — TEST RESULTS')
    print('=' * 60)
    print(f'  Precision                     : {precision:.4f}')
    print(f'  Recall / TPR / Sensitivity    : {recall:.4f}')
    print(f'  F1 Score                      : {f1:.4f}')
    print(f'  AUC-ROC                       : {auc_roc:.4f}')
    print(f'  AUC-PR                        : {auc_pr:.4f}')
    print(f'  False Alarm Rate              : {far:.4f}')
    print(f'  Fault RUL cycles              : {fault_rul_cycles:.1f}')
    print(f'  Threshold used                : {best_thr:.2f}')
    if len(latencies) > 0:
        print(f'  Median Lead Time              : {med_lat:.1f} cycles (positive = early, negative = late)')
        print(f'  Missed Detections             : {n_missed}')
    print('=' * 60)

    if len(per_engine_rows) > 0:
        valid_leads = [r['lead_time'] for r in per_engine_rows if not np.isnan(r['lead_time'])]
        missed = sum(r['missed_detection'] for r in per_engine_rows)

        print('\nPer-Engine Summary')
        print('-' * 60)
        print(f'  Engines evaluated             : {len(per_engine_rows)}')
        print(f'  Engines detected              : {len(valid_leads)}')
        print(f'  Engines missed                : {missed}')
        if len(valid_leads) > 0:
            print(f'  Mean lead time                : {np.mean(valid_leads):.2f}')
            print(f'  Median lead time              : {np.median(valid_leads):.2f}')
            print(f'  Std lead time                 : {np.std(valid_leads):.2f}')

    os.makedirs(args.out_dir, exist_ok=True)

    np.save(os.path.join(args.out_dir, 'early_pred_metrics.npy'), {
        'precision': precision,
        'recall_tpr_sensitivity': recall,
        'f1': f1,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'false_alarm_rate': far,
        'fault_rul_cycles': fault_rul_cycles,
        'threshold': best_thr,
        'median_lead_time': med_lat,
        'missed_detections': n_missed,
    })

    np.save(os.path.join(args.out_dir, 'early_pred_outputs.npy'), {
        'y_prob': y_prob,
        'y_pred': y_pred,
        'y_true': y_true,
        'rul_true_cycles': y_rul_cycles,
    })

    print(f"\nSaved -> {os.path.join(args.out_dir, 'early_pred_metrics.npy')}")
    print(f"Saved -> {os.path.join(args.out_dir, 'early_pred_outputs.npy')}")

    save_roc_curve(y_true, y_prob, auc_roc, args.out_dir)
    save_threshold_tradeoff_plot(y_true, y_prob, args.out_dir, best_thr)
    save_lead_time_histogram(latencies, args.out_dir)

    if len(per_engine_rows) > 0:
        save_per_engine_csv(per_engine_rows, args.out_dir)
        save_per_engine_boxplot(per_engine_rows, args.out_dir)


if __name__ == '__main__':
    main()