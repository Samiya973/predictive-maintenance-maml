"""
compare_training_time.py

Reads results/maml_timing.json and results/conventional_timing.json
(written by train_maml_timed.py and train_conventional_timed.py)
and prints a side-by-side comparison table.

Usage:
    python compare_training_time.py

Optional flags:
    --maml_json      PATH   (default: results/maml_timing.json)
    --conv_json      PATH   (default: results/conventional_timing.json)
    --save_csv       PATH   saves comparison table as CSV
"""

import json
import argparse
import os
import sys


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def fmt_time(seconds: float) -> str:
    """Return 'Xs (Y.Y min)' string."""
    return f"{seconds:.2f}s  ({seconds / 60:.1f} min)"


def load_json(path: str) -> dict:
    if not os.path.exists(path):
        print(f"  ✗  File not found: {path}")
        print("     Make sure you have run both training scripts first.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def pct_diff(a: float, b: float) -> str:
    """Return '+X%' or '-X%' of a relative to b."""
    if b == 0:
        return "N/A"
    d = 100.0 * (a - b) / b
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.1f}%"


# ──────────────────────────────────────────────────────────────────────────────
# Main comparison
# ──────────────────────────────────────────────────────────────────────────────

def compare(maml_path: str, conv_path: str, save_csv: str = None):

    print("\n" + "=" * 70)
    print("  TRAINING TIME COMPARISON  —  MAML  vs  Conventional CNN-LSTM")
    print("=" * 70)

    m = load_json(maml_path)
    c = load_json(conv_path)

    # ── Basic sanity ──────────────────────────────────────────────────────
    print(f"\n  MAML timing file        : {maml_path}")
    print(f"  Conventional timing file: {conv_path}\n")

    rows = []   # (label, maml_val, conv_val, note)

    # 1. Total wall time
    t_maml = m['total_wall_time_seconds']
    t_conv = c['total_wall_time_seconds']
    rows.append(("Total wall time",
                 fmt_time(t_maml),
                 fmt_time(t_conv),
                 f"MAML is {t_maml/t_conv:.2f}× slower" if t_maml > t_conv
                 else f"MAML is {t_conv/t_maml:.2f}× faster"))

    # 2. Epochs completed
    rows.append(("Epochs completed",
                 str(m['epochs_completed']),
                 str(c['epochs_completed']),
                 ""))

    # 3. Time per epoch
    e_maml = m['avg_epoch_time_seconds']
    e_conv = c['avg_epoch_time_seconds']
    rows.append(("Avg time / epoch",
                 f"{e_maml:.4f}s",
                 f"{e_conv:.4f}s",
                 f"MAML overhead: {pct_diff(e_maml, e_conv)}"))

    # 4. Median time per epoch
    rows.append(("Median time / epoch",
                 f"{m['median_epoch_time_seconds']:.4f}s",
                 f"{c['median_epoch_time_seconds']:.4f}s",
                 ""))

    # 5. Validation time
    v_maml = m.get('validation_seconds', 0.0)
    v_conv = c.get('validation_seconds', 0.0)
    rows.append(("Total validation time",
                 fmt_time(v_maml),
                 fmt_time(v_conv),
                 ""))

    # 6. Forward/backward (MAML only)
    fb = m.get('forward_backward_seconds', None)
    rows.append(("Forward+backward time",
                 fmt_time(fb) if fb else "N/A",
                 "N/A",
                 "Higher-order grad cost (MAML only)"))

    # 7. Task regen (MAML only)
    tr = m.get('task_regen_seconds', None)
    rows.append(("Task regen time",
                 fmt_time(tr) if tr else "N/A",
                 "N/A",
                 "MAML only"))

    # 8. Best val RMSE
    r_maml = m.get('best_val_rmse', float('inf'))
    r_conv = c.get('best_val_rmse', float('inf'))
    rows.append(("Best val RMSE (cycles)",
                 f"{r_maml:.2f}",
                 f"{r_conv:.2f}",
                 f"Δ = {r_conv - r_maml:+.2f} (−ve = MAML better)"))

    # 9. Time per RMSE point (efficiency)
    eff_maml = t_maml / r_maml if r_maml > 0 else float('inf')
    eff_conv = t_conv / r_conv if r_conv > 0 else float('inf')
    rows.append(("Seconds per RMSE unit",
                 f"{eff_maml:.2f}",
                 f"{eff_conv:.2f}",
                 "Lower = more efficient"))

    # ── Print table ───────────────────────────────────────────────────────
    col0 = max(len(r[0]) for r in rows) + 2
    col1 = max(len(r[1]) for r in rows) + 2
    col2 = max(len(r[2]) for r in rows) + 2

    header = (f"  {'Metric':<{col0}}{'MAML':>{col1}}{'Conventional':>{col2}}  Note")
    sep    = "  " + "-" * (col0 + col1 + col2 + 6)

    print(header)
    print(sep)
    for label, mv, cv, note in rows:
        note_str = f"  ← {note}" if note else ""
        print(f"  {label:<{col0}}{mv:>{col1}}{cv:>{col2}}{note_str}")

    # ── High-level verdict ────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  VERDICT")
    print("─" * 70)
    overhead = t_maml / t_conv if t_conv > 0 else float('inf')
    rmse_gain = r_conv - r_maml   # positive means MAML is better
    print(f"  • MAML training takes {overhead:.2f}× the wall time of conventional training.")
    if rmse_gain > 0:
        print(f"  • MAML achieves {rmse_gain:.2f} fewer RMSE cycles → better accuracy.")
        print(f"  • Trade-off: {overhead:.2f}× cost for {rmse_gain:.2f} RMSE improvement.")
    elif rmse_gain < 0:
        print(f"  • Conventional achieves {-rmse_gain:.2f} fewer RMSE cycles here.")
        print("  • MAML's advantage is in few-shot adaptation, not raw val RMSE.")
    else:
        print("  • Similar val RMSE — MAML value is in fast few-shot adaptation.")
    print("=" * 70 + "\n")

    # ── Optional CSV export ───────────────────────────────────────────────
    if save_csv:
        import csv
        with open(save_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'MAML', 'Conventional', 'Note'])
            writer.writerows(rows)
        print(f"✓ Table saved as CSV → {save_csv}\n")


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--maml_json', default='results/maml_timing.json')
    parser.add_argument('--conv_json', default='results/conventional_timing.json')
    parser.add_argument('--save_csv',  default=None,
                        help='Optional path to save comparison as CSV')
    args = parser.parse_args()

    compare(args.maml_json, args.conv_json, args.save_csv)