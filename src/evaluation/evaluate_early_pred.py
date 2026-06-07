# import os
# import sys
# import csv
# import argparse
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# from torch.utils.data import Dataset, DataLoader
# from sklearn.metrics import (
#     roc_auc_score,
#     average_precision_score,
#     precision_recall_fscore_support,
#     confusion_matrix,
#     roc_curve,
# )

# sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# from src.data.data_loader import load_preprocessed_data
# from src.models.early_pred_model import EarlyPredCNNLSTM


# class EarlyPredDataset(Dataset):
#     def __init__(self, X, y_rul_norm, fault_threshold_norm):
#         self.X = torch.FloatTensor(X)
#         self.y_rul = torch.FloatTensor(y_rul_norm)
#         self.y_fault = (self.y_rul <= fault_threshold_norm).float()

#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, idx):
#         return self.X[idx], self.y_rul[idx], self.y_fault[idx]


# def find_best_threshold(y_true, y_prob):
#     best_thr = 0.5
#     best_f1 = -1.0

#     for thr in np.arange(0.05, 0.91, 0.05):
#         y_pred = (y_prob >= thr).astype(int)
#         _, _, f1, _ = precision_recall_fscore_support(
#             y_true, y_pred, average='binary', zero_division=0
#         )
#         if f1 > best_f1:
#             best_f1 = f1
#             best_thr = float(thr)

#     return best_thr, best_f1


# def detection_latency(engine_ids, rul_true_cycles, fault_pred, fault_rul_cycles):
#     latencies = []
#     missed = 0

#     unique_engines = np.unique(engine_ids)

#     for eng in unique_engines:
#         mask = engine_ids == eng
#         rul_eng = rul_true_cycles[mask]
#         pred_eng = fault_pred[mask]

#         onset_idx = np.where(rul_eng <= fault_rul_cycles)[0]
#         if len(onset_idx) == 0:
#             continue

#         det_idx = np.where(pred_eng == 1)[0]
#         if len(det_idx) == 0:
#             missed += 1
#             continue

#         det_idx = det_idx[0]
#         rul_at_det = rul_eng[det_idx]

#         # positive = early, negative = late
#         latencies.append(float(rul_at_det - fault_rul_cycles))

#     if len(latencies) == 0:
#         return float('nan'), [], missed

#     return float(np.median(latencies)), latencies, missed


# def build_per_engine_report(engine_ids, rul_true_cycles, y_prob, y_pred, fault_rul_cycles):
#     rows = []
#     unique_engines = np.unique(engine_ids)

#     for eng in unique_engines:
#         mask = engine_ids == eng
#         rul_eng = rul_true_cycles[mask]
#         prob_eng = y_prob[mask]
#         pred_eng = y_pred[mask]

#         onset_idx = np.where(rul_eng <= fault_rul_cycles)[0]
#         if len(onset_idx) == 0:
#             continue
#         onset_idx = int(onset_idx[0])

#         det_idx = np.where(pred_eng == 1)[0]

#         if len(det_idx) == 0:
#             rows.append({
#                 'engine_id': int(eng),
#                 'onset_index': onset_idx,
#                 'detect_index': -1,
#                 'rul_at_detection': np.nan,
#                 'lead_time': np.nan,
#                 'detected': 0,
#                 'missed_detection': 1,
#                 'max_prob': float(np.max(prob_eng)),
#             })
#             continue

#         det_idx = int(det_idx[0])
#         rul_at_det = float(rul_eng[det_idx])
#         lead_time = float(rul_at_det - fault_rul_cycles)   # positive = early

#         rows.append({
#             'engine_id': int(eng),
#             'onset_index': onset_idx,
#             'detect_index': det_idx,
#             'rul_at_detection': rul_at_det,
#             'lead_time': lead_time,
#             'detected': 1,
#             'missed_detection': 0,
#             'max_prob': float(np.max(prob_eng)),
#         })

#     return rows


# def save_per_engine_csv(rows, out_dir):
#     csv_path = os.path.join(out_dir, 'early_pred_per_engine.csv')

#     fieldnames = [
#         'engine_id',
#         'onset_index',
#         'detect_index',
#         'rul_at_detection',
#         'lead_time',
#         'detected',
#         'missed_detection',
#         'max_prob',
#     ]

#     with open(csv_path, 'w', newline='') as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()
#         writer.writerows(rows)

#     print(f"Saved -> {csv_path}")


# def save_per_engine_boxplot(rows, out_dir):
#     lead_times = [r['lead_time'] for r in rows if not np.isnan(r['lead_time'])]

#     if len(lead_times) == 0:
#         return

#     plt.figure(figsize=(5, 5))
#     plt.boxplot(lead_times, vert=True, patch_artist=True)
#     plt.axhline(0, linestyle='--', linewidth=1)
#     plt.ylabel('Lead Time (cycles)')
#     plt.title('Per-Engine Lead-Time Box Plot')

#     boxplot_path = os.path.join(out_dir, 'early_pred_per_engine_boxplot.png')
#     plt.savefig(boxplot_path, dpi=150, bbox_inches='tight')
#     plt.close()

#     print(f"Saved -> {boxplot_path}")


# def save_roc_curve(y_true, y_prob, auc_roc, out_dir):
#     fpr, tpr, thresholds = roc_curve(y_true, y_prob)

#     plt.figure(figsize=(6, 5))
#     plt.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc_roc:.4f}')
#     plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1)
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('ROC Curve — Early Prediction')
#     plt.legend()
#     plt.grid(True, alpha=0.3)

#     roc_png = os.path.join(out_dir, 'early_pred_roc_curve.png')
#     plt.savefig(roc_png, dpi=150, bbox_inches='tight')
#     plt.close()

#     np.save(os.path.join(out_dir, 'early_pred_roc.npy'), {
#         'fpr': fpr,
#         'tpr': tpr,
#         'thresholds': thresholds,
#     })

#     print(f"Saved -> {roc_png}")
#     print(f"Saved -> {os.path.join(out_dir, 'early_pred_roc.npy')}")


# def save_threshold_tradeoff_plot(y_true, y_prob, out_dir, best_thr):
#     thresholds = np.arange(0.05, 0.91, 0.05)

#     precisions = []
#     recalls = []
#     f1s = []
#     fprs = []

#     for thr in thresholds:
#         y_pred = (y_prob >= thr).astype(int)

#         precision, recall, f1, _ = precision_recall_fscore_support(
#             y_true, y_pred, average='binary', zero_division=0
#         )

#         cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
#         tn, fp, fn, tp = cm.ravel()
#         fpr = float(fp / (fp + tn + 1e-8))

#         precisions.append(float(precision))
#         recalls.append(float(recall))
#         f1s.append(float(f1))
#         fprs.append(float(fpr))

#     plt.figure(figsize=(7, 5))
#     plt.plot(thresholds, precisions, label='Precision', linewidth=2)
#     plt.plot(thresholds, recalls, label='Recall / TPR', linewidth=2)
#     plt.plot(thresholds, f1s, label='F1', linewidth=2)
#     plt.plot(thresholds, fprs, label='FPR', linewidth=2)
#     plt.axvline(best_thr, linestyle='--', linewidth=2, label=f'Selected threshold = {best_thr:.2f}')
#     plt.xlabel('Decision Threshold')
#     plt.ylabel('Metric Value')
#     plt.title('Threshold Trade-off Curve')
#     plt.legend()
#     plt.grid(True, alpha=0.3)

#     tradeoff_png = os.path.join(out_dir, 'early_pred_threshold_tradeoff.png')
#     plt.savefig(tradeoff_png, dpi=150, bbox_inches='tight')
#     plt.close()

#     np.save(os.path.join(out_dir, 'early_pred_threshold_tradeoff.npy'), {
#         'thresholds': thresholds,
#         'precision': np.array(precisions),
#         'recall': np.array(recalls),
#         'f1': np.array(f1s),
#         'fpr': np.array(fprs),
#         'best_threshold': best_thr,
#     })

#     print(f"Saved -> {tradeoff_png}")
#     print(f"Saved -> {os.path.join(out_dir, 'early_pred_threshold_tradeoff.npy')}")


# def save_lead_time_histogram(latencies, out_dir):
#     if len(latencies) == 0:
#         return

#     plt.figure(figsize=(6, 5))
#     plt.hist(latencies, bins=20, edgecolor='black')
#     plt.axvline(np.median(latencies), linestyle='--', linewidth=2,
#                 label=f'Median = {np.median(latencies):.1f}')
#     plt.xlabel('Lead Time (cycles)')
#     plt.ylabel('Count')
#     plt.title('Lead-Time Distribution')
#     plt.legend()
#     plt.grid(True, alpha=0.3)

#     hist_png = os.path.join(out_dir, 'early_pred_lead_time_hist.png')
#     plt.savefig(hist_png, dpi=150, bbox_inches='tight')
#     plt.close()

#     print(f"Saved -> {hist_png}")


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset', type=str, default='data/processed/FD001_preprocessed.npz')
#     parser.add_argument('--model_path', type=str, default='results/saved_models/early_pred_best.pth')
#     parser.add_argument('--out_dir', type=str, default='results')
#     args = parser.parse_args()

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f'Device: {device}')

#     print('\nLoading preprocessed data...')
#     data = load_preprocessed_data(args.dataset)

#     ckpt = torch.load(args.model_path, map_location=device)
#     cfg = ckpt.get('config', {})
#     max_rul = float(ckpt.get('max_rul', data['max_rul'] if 'max_rul' in data else 130.0))
#     fault_rul_cycles = float(ckpt.get('fault_rul_cycles', 30.0))
#     fault_threshold_norm = fault_rul_cycles / max_rul

#     model = EarlyPredCNNLSTM(
#         input_size=cfg.get('input_size', data['X_train'].shape[2]),
#         dropout=cfg.get('dropout', 0.2),
#     ).to(device)
#     model.load_state_dict(ckpt['model_state_dict'])
#     model.eval()

#     val_ds = EarlyPredDataset(data['X_val'], data['y_val'], fault_threshold_norm)
#     test_ds = EarlyPredDataset(data['X_test'], data['y_test'], fault_threshold_norm)

#     val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
#     test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

#     all_val_prob = []
#     all_val_true = []

#     with torch.no_grad():
#         for X_batch, _, y_fault in val_loader:
#             X_batch = X_batch.to(device)
#             logits = model(X_batch).squeeze(1)
#             prob = torch.sigmoid(logits).cpu().numpy()
#             all_val_prob.extend(prob)
#             all_val_true.extend(y_fault.numpy())

#     all_val_prob = np.array(all_val_prob)
#     all_val_true = np.array(all_val_true).astype(int)

#     best_thr, best_f1 = find_best_threshold(all_val_true, all_val_prob)
#     print(f'Optimal threshold from validation: {best_thr:.2f} (val F1={best_f1:.4f})')

#     all_test_prob = []
#     all_test_true = []
#     all_test_rul = []

#     with torch.no_grad():
#         for X_batch, y_rul, y_fault in test_loader:
#             X_batch = X_batch.to(device)
#             logits = model(X_batch).squeeze(1)
#             prob = torch.sigmoid(logits).cpu().numpy()

#             all_test_prob.extend(prob)
#             all_test_true.extend(y_fault.numpy())
#             all_test_rul.extend(y_rul.numpy())

#     y_prob = np.array(all_test_prob)
#     y_true = np.array(all_test_true).astype(int)
#     y_pred = (y_prob >= best_thr).astype(int)
#     y_rul_cycles = np.array(all_test_rul) * max_rul

#     if len(np.unique(y_true)) >= 2:
#         auc_roc = float(roc_auc_score(y_true, y_prob))
#         auc_pr = float(average_precision_score(y_true, y_prob))
#         precision, recall, f1, _ = precision_recall_fscore_support(
#             y_true, y_pred, average='binary', zero_division=0
#         )
#         cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
#         tn, fp, fn, tp = cm.ravel()
#         far = float(fp / (fp + tn + 1e-8))
#     else:
#         auc_roc = float('nan')
#         auc_pr = float('nan')
#         precision, recall, f1 = 0.0, 0.0, 0.0
#         far = float('nan')

#     per_engine_rows = []
#     if 'test_engine_ids' in data:
#         test_engine_ids = np.array(data['test_engine_ids'])
#         med_lat, latencies, n_missed = detection_latency(
#             test_engine_ids, y_rul_cycles, y_pred, fault_rul_cycles
#         )
#         per_engine_rows = build_per_engine_report(
#             test_engine_ids,
#             y_rul_cycles,
#             y_prob,
#             y_pred,
#             fault_rul_cycles
#         )
#     else:
#         med_lat, latencies, n_missed = float('nan'), [], 0

#     print('\n' + '=' * 60)
#     print('EARLY PREDICTION — TEST RESULTS')
#     print('=' * 60)
#     print(f'  Precision                     : {precision:.4f}')
#     print(f'  Recall / TPR / Sensitivity    : {recall:.4f}')
#     print(f'  F1 Score                      : {f1:.4f}')
#     print(f'  AUC-ROC                       : {auc_roc:.4f}')
#     print(f'  AUC-PR                        : {auc_pr:.4f}')
#     print(f'  False Alarm Rate              : {far:.4f}')
#     print(f'  Fault RUL cycles              : {fault_rul_cycles:.1f}')
#     print(f'  Threshold used                : {best_thr:.2f}')
#     if len(latencies) > 0:
#         print(f'  Median Lead Time              : {med_lat:.1f} cycles (positive = early, negative = late)')
#         print(f'  Missed Detections             : {n_missed}')
#     print('=' * 60)

#     if len(per_engine_rows) > 0:
#         valid_leads = [r['lead_time'] for r in per_engine_rows if not np.isnan(r['lead_time'])]
#         missed = sum(r['missed_detection'] for r in per_engine_rows)

#         print('\nPer-Engine Summary')
#         print('-' * 60)
#         print(f'  Engines evaluated             : {len(per_engine_rows)}')
#         print(f'  Engines detected              : {len(valid_leads)}')
#         print(f'  Engines missed                : {missed}')
#         if len(valid_leads) > 0:
#             print(f'  Mean lead time                : {np.mean(valid_leads):.2f}')
#             print(f'  Median lead time              : {np.median(valid_leads):.2f}')
#             print(f'  Std lead time                 : {np.std(valid_leads):.2f}')

#     os.makedirs(args.out_dir, exist_ok=True)

#     np.save(os.path.join(args.out_dir, 'early_pred_metrics.npy'), {
#         'precision': precision,
#         'recall_tpr_sensitivity': recall,
#         'f1': f1,
#         'auc_roc': auc_roc,
#         'auc_pr': auc_pr,
#         'false_alarm_rate': far,
#         'fault_rul_cycles': fault_rul_cycles,
#         'threshold': best_thr,
#         'median_lead_time': med_lat,
#         'missed_detections': n_missed,
#     })

#     np.save(os.path.join(args.out_dir, 'early_pred_outputs.npy'), {
#         'y_prob': y_prob,
#         'y_pred': y_pred,
#         'y_true': y_true,
#         'rul_true_cycles': y_rul_cycles,
#     })

#     print(f"\nSaved -> {os.path.join(args.out_dir, 'early_pred_metrics.npy')}")
#     print(f"Saved -> {os.path.join(args.out_dir, 'early_pred_outputs.npy')}")

#     save_roc_curve(y_true, y_prob, auc_roc, args.out_dir)
#     save_threshold_tradeoff_plot(y_true, y_prob, args.out_dir, best_thr)
#     save_lead_time_histogram(latencies, args.out_dir)

#     if len(per_engine_rows) > 0:
#         save_per_engine_csv(per_engine_rows, args.out_dir)
#         save_per_engine_boxplot(per_engine_rows, args.out_dir)


# if __name__ == '__main__':
#     main()


import os
import sys
import csv
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.data.data_loader import load_preprocessed_data
from src.models.early_pred_model import EarlyPredCNNLSTM


# ── Palette ────────────────────────────────────────────────────────────────────
ACCENT1   = '#1f77b4'   # blue
ACCENT2   = '#2ca02c'   # green
ACCENT3   = '#d62728'   # red
ACCENT4   = '#9467bd'   # purple
MUTED_COL = '#888888'

def apply_light_style():
    plt.rcParams.update({
        'figure.facecolor':  'white',
        'axes.facecolor':    'white',
        'axes.edgecolor':    '#cccccc',
        'axes.labelcolor':   'black',
        'axes.titlecolor':   'black',
        'xtick.color':       'black',
        'ytick.color':       'black',
        'grid.color':        '#dddddd',
        'text.color':        'black',
        'legend.facecolor':  'white',
        'legend.edgecolor':  '#cccccc',
        'font.family':       'sans-serif',
        'figure.dpi':        150,
    })

apply_light_style()


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset / helpers
# ═══════════════════════════════════════════════════════════════════════════════

class EarlyPredDataset(Dataset):
    def __init__(self, X, y_rul_norm, fault_threshold_norm):
        self.X       = torch.FloatTensor(X)
        self.y_rul   = torch.FloatTensor(y_rul_norm)
        self.y_fault = (self.y_rul <= fault_threshold_norm).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_rul[idx], self.y_fault[idx]


def find_best_threshold(y_true, y_prob):
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.arange(0.05, 0.91, 0.05):
        y_pred = (y_prob >= thr).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr, best_f1


def detection_latency(engine_ids, rul_true_cycles, fault_pred, fault_rul_cycles):
    latencies, missed = [], 0
    for eng in np.unique(engine_ids):
        mask    = engine_ids == eng
        rul_eng = rul_true_cycles[mask]
        pred    = fault_pred[mask]
        onset   = np.where(rul_eng <= fault_rul_cycles)[0]
        if len(onset) == 0:
            continue
        det = np.where(pred == 1)[0]
        if len(det) == 0:
            missed += 1
            continue
        latencies.append(float(rul_eng[det[0]] - fault_rul_cycles))
    if not latencies:
        return float('nan'), [], missed
    return float(np.median(latencies)), latencies, missed


def build_per_engine_report(engine_ids, rul_true_cycles, y_prob, y_pred, fault_rul_cycles):
    rows = []
    for eng in np.unique(engine_ids):
        mask    = engine_ids == eng
        rul_eng = rul_true_cycles[mask]
        prob    = y_prob[mask]
        pred    = y_pred[mask]
        onset   = np.where(rul_eng <= fault_rul_cycles)[0]
        if len(onset) == 0:
            continue
        onset = int(onset[0])
        det   = np.where(pred == 1)[0]
        if len(det) == 0:
            rows.append(dict(engine_id=int(eng), onset_index=onset, detect_index=-1,
                             rul_at_detection=np.nan, lead_time=np.nan,
                             detected=0, missed_detection=1, max_prob=float(np.max(prob))))
            continue
        det_i      = int(det[0])
        rul_at_det = float(rul_eng[det_i])
        rows.append(dict(engine_id=int(eng), onset_index=onset, detect_index=det_i,
                         rul_at_detection=rul_at_det, lead_time=rul_at_det - fault_rul_cycles,
                         detected=1, missed_detection=0, max_prob=float(np.max(prob))))
    return rows


def save_per_engine_csv(rows, out_dir):
    path = os.path.join(out_dir, 'early_pred_per_engine.csv')
    fields = ['engine_id','onset_index','detect_index','rul_at_detection',
              'lead_time','detected','missed_detection','max_prob']
    with open(path, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
        csv.DictWriter(f, fieldnames=fields).writerows(rows)
    print(f"Saved -> {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# ── PLOT 1 : Confusion Matrix Heatmap
# ═══════════════════════════════════════════════════════════════════════════════

def save_confusion_matrix(y_true, y_pred, out_dir):
    cm  = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    cmap = LinearSegmentedColormap.from_list('light_blue', ['white', ACCENT1])

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap=cmap, aspect='auto')

    labels = [['TN', 'FP'], ['FN', 'TP']]
    vals   = [[tn, fp], [fn, tp]]
    colors = [[ACCENT2, ACCENT3], [ACCENT3, ACCENT2]]

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{labels[i][j]}\n{vals[i][j]:,}',
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    color=colors[i][j])

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred: Normal', 'Pred: Fault'], fontsize=10)
    ax.set_yticklabels(['True: Normal', 'True: Fault'], fontsize=10)
    ax.set_title('Confusion Matrix', fontsize=13, pad=12)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    path = os.path.join(out_dir, 'plot_confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved -> {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# ── PLOT 2 : ROC Curve
# ═══════════════════════════════════════════════════════════════════════════════

def save_roc_curve(y_true, y_prob, auc_roc, out_dir):
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.fill_between(fpr, tpr, alpha=0.15, color=ACCENT1)
    ax.plot(fpr, tpr, color=ACCENT1, linewidth=2.5, label=f'AUC = {auc_roc:.4f}')
    ax.plot([0, 1], [0, 1], '--', color=MUTED_COL, linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate',  fontsize=11)
    ax.set_title('ROC Curve — Early Prediction', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.4)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

    plt.tight_layout()
    path = os.path.join(out_dir, 'plot_roc_curve.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved -> {path}")

    np.save(os.path.join(out_dir, 'early_pred_roc.npy'),
            {'fpr': fpr, 'tpr': tpr})


# ═══════════════════════════════════════════════════════════════════════════════
# ── PLOT 3 : Precision-Recall Curve
# ═══════════════════════════════════════════════════════════════════════════════

def save_pr_curve(y_true, y_prob, auc_pr, out_dir):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.fill_between(recall, precision, alpha=0.15, color=ACCENT2)
    ax.plot(recall, precision, color=ACCENT2, linewidth=2.5,
            label=f'AUC-PR = {auc_pr:.4f}')
    ax.axhline(baseline, linestyle='--', color=MUTED_COL,
               linewidth=1, label=f'Baseline = {baseline:.3f}')
    ax.set_xlabel('Recall',    fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title('Precision-Recall Curve', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.4)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

    plt.tight_layout()
    path = os.path.join(out_dir, 'plot_pr_curve.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved -> {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# ── PLOT 4 : Probability Distribution (Normal vs Fault)
# ═══════════════════════════════════════════════════════════════════════════════

def save_prob_distribution(y_true, y_prob, best_thr, out_dir):
    prob_normal = y_prob[y_true == 0]
    prob_fault  = y_prob[y_true == 1]
    bins = np.linspace(0, 1, 40)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(prob_normal, bins=bins, alpha=0.65, color=ACCENT2,
            label=f'Normal  (n={len(prob_normal):,})', edgecolor='white', linewidth=0.4)
    ax.hist(prob_fault,  bins=bins, alpha=0.65, color=ACCENT3,
            label=f'Fault   (n={len(prob_fault):,})',  edgecolor='white', linewidth=0.4)
    ax.axvline(best_thr, color=ACCENT1, linestyle='--', linewidth=2,
               label=f'Threshold = {best_thr:.2f}')
    ax.set_xlabel('Predicted Fault Probability', fontsize=11)
    ax.set_ylabel('Sample Count',               fontsize=11)
    ax.set_title('Prediction Probability Distribution', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4, axis='y')

    plt.tight_layout()
    path = os.path.join(out_dir, 'plot_prob_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved -> {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# ── PLOT 5 : Threshold Trade-off Curve
# ═══════════════════════════════════════════════════════════════════════════════

def save_threshold_tradeoff_plot(y_true, y_prob, out_dir, best_thr):
    thresholds = np.arange(0.05, 0.91, 0.05)
    precisions, recalls, f1s, fprs = [], [], [], []

    for thr in thresholds:
        yp = (y_prob >= thr).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y_true, yp, average='binary', zero_division=0)
        cm_ = confusion_matrix(y_true, yp, labels=[0, 1])
        tn_, fp_, _, _ = cm_.ravel()
        precisions.append(float(p)); recalls.append(float(r))
        f1s.append(float(f)); fprs.append(float(fp_ / (fp_ + tn_ + 1e-8)))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thresholds, precisions, color=ACCENT2,  linewidth=2,   label='Precision')
    ax.plot(thresholds, recalls,    color=ACCENT1,  linewidth=2,   label='Recall / TPR')
    ax.plot(thresholds, f1s,        color=ACCENT4,  linewidth=2.5, label='F1')
    ax.plot(thresholds, fprs,       color=ACCENT3,  linewidth=2,   label='FPR')
    ax.axvline(best_thr, color='black', linestyle='--', linewidth=1.8,
               label=f'Threshold = {best_thr:.2f}')
    ax.set_xlabel('Decision Threshold', fontsize=11)
    ax.set_ylabel('Metric Value',       fontsize=11)
    ax.set_title('Threshold Trade-off Curve', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)
    ax.set_xlim(0.05, 0.90); ax.set_ylim(0, 1.05)

    plt.tight_layout()
    path = os.path.join(out_dir, 'plot_threshold_tradeoff.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved -> {path}")

    np.save(os.path.join(out_dir, 'early_pred_threshold_tradeoff.npy'),
            dict(thresholds=thresholds, precision=np.array(precisions),
                 recall=np.array(recalls), f1=np.array(f1s),
                 fpr=np.array(fprs), best_threshold=best_thr))


# ═══════════════════════════════════════════════════════════════════════════════
# ── PLOT 6 : Lead-Time Histogram
# ═══════════════════════════════════════════════════════════════════════════════

def save_lead_time_histogram(latencies, out_dir):
    if not latencies:
        return
    arr    = np.array(latencies)
    median = float(np.median(arr))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(arr, bins=20, color=ACCENT1, alpha=0.75, edgecolor='white', linewidth=0.4)
    ax.axvline(0,      color=ACCENT3, linestyle='--', linewidth=1.5, label='Zero (on-time)')
    ax.axvline(median, color=ACCENT2, linestyle='--', linewidth=2,
               label=f'Median = {median:.1f} cycles')
    ax.set_xlabel('Lead Time (cycles) — positive = early', fontsize=11)
    ax.set_ylabel('Engine Count',                          fontsize=11)
    ax.set_title('Lead-Time Distribution', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4, axis='y')

    plt.tight_layout()
    path = os.path.join(out_dir, 'plot_lead_time_hist.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved -> {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# ── PLOT 7 : Per-Engine Lead-Time Box Plot
# ═══════════════════════════════════════════════════════════════════════════════

def save_per_engine_boxplot(rows, out_dir):
    lead_times = [r['lead_time'] for r in rows if not np.isnan(r['lead_time'])]
    if not lead_times:
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    bp = ax.boxplot(lead_times, vert=True, patch_artist=True,
                    medianprops=dict(color=ACCENT1, linewidth=2.5),
                    whiskerprops=dict(color=MUTED_COL, linewidth=1.5),
                    capprops=dict(color=MUTED_COL, linewidth=1.5),
                    flierprops=dict(marker='o', color=ACCENT3,
                                    markerfacecolor=ACCENT3, markersize=5, alpha=0.6))
    bp['boxes'][0].set_facecolor('#d0e4f7')
    bp['boxes'][0].set_edgecolor(ACCENT1)

    ax.axhline(0, linestyle='--', color=ACCENT3, linewidth=1.5, label='Zero (on-time)')
    ax.set_ylabel('Lead Time (cycles)', fontsize=11)
    ax.set_title('Per-Engine Lead-Time Box Plot', fontsize=13)
    ax.set_xticks([])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4, axis='y')

    plt.tight_layout()
    path = os.path.join(out_dir, 'plot_per_engine_boxplot.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved -> {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# ── PLOT 8 : Lead-Time vs RUL-at-Detection Scatter
# ═══════════════════════════════════════════════════════════════════════════════

def save_lead_time_vs_rul_scatter(rows, fault_rul_cycles, out_dir):
    detected = [r for r in rows if r['detected'] == 1]
    if not detected:
        return

    rul_det = np.array([r['rul_at_detection'] for r in detected])
    lead    = np.array([r['lead_time']        for r in detected])

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(rul_det, lead, c=lead, cmap='RdYlGn',
                    s=60, alpha=0.8, edgecolors='white', linewidths=0.5)
    plt.colorbar(sc, ax=ax, label='Lead Time (cycles)')
    ax.axhline(0, linestyle='--', color=ACCENT3, linewidth=1.5, label='Zero (on-time)')
    ax.axvline(fault_rul_cycles, linestyle=':', color=MUTED_COL,
               linewidth=1.5, label=f'Fault threshold = {fault_rul_cycles:.0f}')
    ax.set_xlabel('RUL at Detection (cycles)', fontsize=11)
    ax.set_ylabel('Lead Time (cycles)',        fontsize=11)
    ax.set_title('Lead Time vs RUL at Detection', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    path = os.path.join(out_dir, 'plot_lead_time_vs_rul_scatter.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved -> {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# ── PLOT 9 : Summary Dashboard
# ═══════════════════════════════════════════════════════════════════════════════

def save_summary_dashboard(y_true, y_prob, y_pred, latencies,
                           auc_roc, auc_pr, best_thr, fault_rul_cycles,
                           per_engine_rows, out_dir):
    fig = plt.figure(figsize=(17, 10), facecolor='white')
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    # ── (0,0) Confusion matrix
    ax0  = fig.add_subplot(gs[0, 0])
    cm_  = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn_, fp_, fn_, tp_ = cm_.ravel()
    cmap = LinearSegmentedColormap.from_list('light_blue', ['white', ACCENT1])
    ax0.imshow(cm_, cmap=cmap, aspect='auto')
    for i, (lrow, vrow, crow) in enumerate(
            zip([['TN','FP'],['FN','TP']],
                [[tn_,fp_],[fn_,tp_]],
                [[ACCENT2,ACCENT3],[ACCENT3,ACCENT2]])):
        for j, (l, v, c) in enumerate(zip(lrow, vrow, crow)):
            ax0.text(j, i, f'{l}\n{v:,}', ha='center', va='center',
                     fontsize=12, fontweight='bold', color=c)
    ax0.set_xticks([0,1]); ax0.set_yticks([0,1])
    ax0.set_xticklabels(['Pred:Normal','Pred:Fault'], fontsize=8)
    ax0.set_yticklabels(['True:Normal','True:Fault'], fontsize=8)
    ax0.set_title('Confusion Matrix', fontsize=11)

    # ── (0,1) ROC
    ax1 = fig.add_subplot(gs[0, 1])
    fpr_, tpr_, _ = roc_curve(y_true, y_prob)
    ax1.fill_between(fpr_, tpr_, alpha=0.12, color=ACCENT1)
    ax1.plot(fpr_, tpr_, color=ACCENT1, linewidth=2, label=f'AUC={auc_roc:.3f}')
    ax1.plot([0,1],[0,1],'--', color=MUTED_COL, linewidth=1)
    ax1.set_xlabel('FPR', fontsize=9); ax1.set_ylabel('TPR', fontsize=9)
    ax1.set_title('ROC Curve', fontsize=11)
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0,1); ax1.set_ylim(0,1.02)

    # ── (0,2) PR Curve
    ax2 = fig.add_subplot(gs[0, 2])
    prec_, rec_, _ = precision_recall_curve(y_true, y_prob)
    ax2.fill_between(rec_, prec_, alpha=0.12, color=ACCENT2)
    ax2.plot(rec_, prec_, color=ACCENT2, linewidth=2, label=f'AUC-PR={auc_pr:.3f}')
    ax2.axhline(y_true.mean(), linestyle='--', color=MUTED_COL, linewidth=1)
    ax2.set_xlabel('Recall', fontsize=9); ax2.set_ylabel('Precision', fontsize=9)
    ax2.set_title('Precision-Recall Curve', fontsize=11)
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0,1); ax2.set_ylim(0,1.02)

    # ── (1,0) Prob distribution
    ax3 = fig.add_subplot(gs[1, 0])
    bins = np.linspace(0, 1, 35)
    ax3.hist(y_prob[y_true==0], bins=bins, alpha=0.65, color=ACCENT2,
             label='Normal', edgecolor='white', linewidth=0.3)
    ax3.hist(y_prob[y_true==1], bins=bins, alpha=0.65, color=ACCENT3,
             label='Fault',  edgecolor='white', linewidth=0.3)
    ax3.axvline(best_thr, color=ACCENT1, linestyle='--', linewidth=1.8,
                label=f'thr={best_thr:.2f}')
    ax3.set_xlabel('Fault Probability', fontsize=9)
    ax3.set_ylabel('Count', fontsize=9)
    ax3.set_title('Probability Distribution', fontsize=11)
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3, axis='y')

    # ── (1,1) Lead-time histogram
    ax4 = fig.add_subplot(gs[1, 1])
    if latencies:
        arr    = np.array(latencies)
        median = float(np.median(arr))
        ax4.hist(arr, bins=18, color=ACCENT1, alpha=0.75, edgecolor='white', linewidth=0.3)
        ax4.axvline(0,      color=ACCENT3, linestyle='--', linewidth=1.5)
        ax4.axvline(median, color=ACCENT2, linestyle='--', linewidth=2,
                    label=f'Median={median:.1f}')
        ax4.legend(fontsize=8)
    ax4.set_xlabel('Lead Time (cycles)', fontsize=9)
    ax4.set_ylabel('Engine Count', fontsize=9)
    ax4.set_title('Lead-Time Distribution', fontsize=11)
    ax4.grid(True, alpha=0.3, axis='y')

    # ── (1,2) Key-metrics bar chart
    ax5 = fig.add_subplot(gs[1, 2])
    precision_, recall_, f1_, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0)
    cm2 = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn2, fp2, _, _ = cm2.ravel()
    far_ = float(fp2 / (fp2 + tn2 + 1e-8))

    metric_names  = ['Precision', 'Recall', 'F1', 'AUC-ROC', 'AUC-PR', '1-FAR']
    metric_values = [precision_, recall_, f1_, auc_roc, auc_pr, 1-far_]
    bar_colors    = [ACCENT2, ACCENT1, ACCENT4, ACCENT1, ACCENT2, ACCENT3]

    bars = ax5.barh(metric_names, metric_values, color=bar_colors,
                    edgecolor='white', height=0.55)
    for bar, val in zip(bars, metric_values):
        ax5.text(min(val + 0.02, 0.97), bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', fontsize=9, color='black')
    ax5.set_xlim(0, 1.12)
    ax5.set_title('Key Metrics', fontsize=11)
    ax5.grid(True, alpha=0.3, axis='x')
    ax5.axvline(1.0, color='#cccccc', linewidth=1, linestyle=':')

    fig.suptitle('Early Fault Detection — Evaluation Summary',
                 fontsize=16, fontweight='bold', y=0.98)

    path = os.path.join(out_dir, 'plot_summary_dashboard.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved -> {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',    type=str, default='data/processed/FD001_preprocessed.npz')
    parser.add_argument('--model_path', type=str, default='results/saved_models/early_pred_best.pth')
    parser.add_argument('--out_dir',    type=str, default='results')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print('\nLoading preprocessed data...')
    data = load_preprocessed_data(args.dataset)

    ckpt             = torch.load(args.model_path, map_location=device)
    cfg              = ckpt.get('config', {})
    max_rul          = float(ckpt.get('max_rul', data.get('max_rul', 130.0)))
    fault_rul_cycles = float(ckpt.get('fault_rul_cycles', 30.0))
    fault_thr_norm   = fault_rul_cycles / max_rul

    model = EarlyPredCNNLSTM(
        input_size=cfg.get('input_size', data['X_train'].shape[2]),
        dropout=cfg.get('dropout', 0.2),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # ── Val → best threshold
    val_ds     = EarlyPredDataset(data['X_val'], data['y_val'], fault_thr_norm)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    all_val_prob, all_val_true = [], []
    with torch.no_grad():
        for X_b, _, y_f in val_loader:
            prob = torch.sigmoid(model(X_b.to(device)).squeeze(1)).cpu().numpy()
            all_val_prob.extend(prob); all_val_true.extend(y_f.numpy())

    best_thr, best_f1 = find_best_threshold(
        np.array(all_val_true).astype(int), np.array(all_val_prob))
    print(f'Optimal threshold: {best_thr:.2f}  (val F1={best_f1:.4f})')

    # ── Test inference
    test_ds     = EarlyPredDataset(data['X_test'], data['y_test'], fault_thr_norm)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    all_prob, all_true, all_rul = [], [], []
    with torch.no_grad():
        for X_b, y_rul, y_f in test_loader:
            prob = torch.sigmoid(model(X_b.to(device)).squeeze(1)).cpu().numpy()
            all_prob.extend(prob); all_true.extend(y_f.numpy()); all_rul.extend(y_rul.numpy())

    y_prob    = np.array(all_prob)
    y_true    = np.array(all_true).astype(int)
    y_pred    = (y_prob >= best_thr).astype(int)
    y_rul_cyc = np.array(all_rul) * max_rul

    # ── Metrics
    if len(np.unique(y_true)) >= 2:
        auc_roc = float(roc_auc_score(y_true, y_prob))
        auc_pr  = float(average_precision_score(y_true, y_prob))
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0)
        cm_  = confusion_matrix(y_true, y_pred, labels=[0,1])
        tn_, fp_, fn_, tp_ = cm_.ravel()
        far  = float(fp_ / (fp_ + tn_ + 1e-8))
    else:
        auc_roc = auc_pr = prec = rec = f1 = far = float('nan')

    # ── Per-engine
    latencies, per_engine_rows = [], []
    if 'test_engine_ids' in data:
        test_eids = np.array(data['test_engine_ids'])
        _, latencies, n_missed = detection_latency(
            test_eids, y_rul_cyc, y_pred, fault_rul_cycles)
        per_engine_rows = build_per_engine_report(
            test_eids, y_rul_cyc, y_prob, y_pred, fault_rul_cycles)
    else:
        n_missed = 0

    # ── Print summary
    print('\n' + '='*60)
    print('EARLY PREDICTION — TEST RESULTS')
    print('='*60)
    print(f'  Precision      : {prec:.4f}')
    print(f'  Recall / TPR   : {rec:.4f}')
    print(f'  F1 Score       : {f1:.4f}')
    print(f'  AUC-ROC        : {auc_roc:.4f}')
    print(f'  AUC-PR         : {auc_pr:.4f}')
    print(f'  False Alarm Rate: {far:.4f}')
    print(f'  Threshold used : {best_thr:.2f}')
    if latencies:
        print(f'  Median Lead Time: {np.median(latencies):.1f} cycles')
        print(f'  Missed Detections: {n_missed}')
    print('='*60)

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Save raw outputs
    np.save(os.path.join(args.out_dir, 'early_pred_metrics.npy'), dict(
        precision=prec, recall_tpr=rec, f1=f1, auc_roc=auc_roc, auc_pr=auc_pr,
        false_alarm_rate=far, fault_rul_cycles=fault_rul_cycles,
        threshold=best_thr,
        median_lead_time=float(np.median(latencies)) if latencies else float('nan'),
        missed_detections=n_missed))

    np.save(os.path.join(args.out_dir, 'early_pred_outputs.npy'), dict(
        y_prob=y_prob, y_pred=y_pred, y_true=y_true, rul_true_cycles=y_rul_cyc))

    if per_engine_rows:
        save_per_engine_csv(per_engine_rows, args.out_dir)

    # ── Plots
    print('\nGenerating plots...')
    save_confusion_matrix(y_true, y_pred, args.out_dir)
    save_roc_curve(y_true, y_prob, auc_roc, args.out_dir)
    save_pr_curve(y_true, y_prob, auc_pr, args.out_dir)
    save_prob_distribution(y_true, y_prob, best_thr, args.out_dir)
    save_threshold_tradeoff_plot(y_true, y_prob, args.out_dir, best_thr)
    save_lead_time_histogram(latencies, args.out_dir)
    if per_engine_rows:
        save_per_engine_boxplot(per_engine_rows, args.out_dir)
        save_lead_time_vs_rul_scatter(per_engine_rows, fault_rul_cycles, args.out_dir)

    save_summary_dashboard(
        y_true, y_prob, y_pred, latencies,
        auc_roc, auc_pr, best_thr, fault_rul_cycles,
        per_engine_rows, args.out_dir)

    print('\nAll plots saved.')


if __name__ == '__main__':
    main()