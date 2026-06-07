"""
src/scripts/compare_training_time.py
─────────────────────────────────────
Compares training time between:
  1. Full retraining  — train CNN-LSTM from scratch on each new engine
  2. MAML adaptation  — few inner-loop steps from meta-learned weights

WHAT IT MEASURES
────────────────
  - Full retrain time per engine  (epochs × batches)
  - MAML adaptation time per engine (K-shot, N inner steps)
  - Speedup ratio: full_time / maml_time
  - Convergence quality: final loss after each method
  - Plots: time comparison bar chart + convergence curves

USAGE
─────
python src/scripts/compare_training_time.py
python src/scripts/compare_training_time.py --engines 5 --k 5 --steps 5
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import copy

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.data.data_loader import load_preprocessed_data


# ──────────────────────────────────────────────
#  SIMPLE MODEL FOR FAIR COMPARISON
#  (same architecture used for both methods)
# ──────────────────────────────────────────────

class SimpleRULModel(nn.Module):
    """
    Lightweight LSTM RUL predictor used for timing comparison.
    Same architecture for both full retrain and MAML adaptation.
    """
    def __init__(self, input_size=14, hidden_size=64, seq_len=30):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers=2, batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return torch.sigmoid(self.fc(h[-1]))


# ──────────────────────────────────────────────
#  FEATURE SELECTION (raw sensors only)
# ──────────────────────────────────────────────

def get_base_indices(feature_names):
    tags = ('rolling', 'velocity', 'acceleration')
    base = [f for f in feature_names if not any(t in f for t in tags)]
    return [feature_names.index(f) for f in base], base


# ──────────────────────────────────────────────
#  METHOD 1: FULL RETRAINING
# ──────────────────────────────────────────────

def full_retrain(X_engine, y_engine, model_template,
                 epochs=50, lr=1e-3, batch_size=32):
    """
    Train a fresh model from scratch on one engine's data.

    This simulates what you'd have to do WITHOUT meta-learning:
    every new engine requires a full training run.

    Returns
    -------
    elapsed   : float  - wall-clock seconds
    losses    : list   - loss per epoch
    final_loss: float
    """
    model = copy.deepcopy(model_template)   # fresh random weights
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X = torch.FloatTensor(X_engine)
    y = torch.FloatTensor(y_engine).unsqueeze(1)

    losses  = []
    start   = time.perf_counter()

    for epoch in range(epochs):
        # Mini-batch loop
        perm      = torch.randperm(len(X))
        epoch_loss = 0.0
        n_batches  = 0

        for i in range(0, len(X), batch_size):
            idx     = perm[i:i + batch_size]
            x_batch = X[idx]
            y_batch = y[idx]

            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        losses.append(epoch_loss / n_batches)

    elapsed = time.perf_counter() - start
    return elapsed, losses, losses[-1]


# ──────────────────────────────────────────────
#  METHOD 2: MAML ADAPTATION
# ──────────────────────────────────────────────

def maml_adapt(X_engine, y_engine, meta_model,
               k=5, steps=5, lr=0.01):
    """
    Few-shot MAML inner-loop adaptation on one engine.

    Starts from meta-learned weights (not random) and takes
    only `steps` gradient updates on `k` support windows.

    Returns
    -------
    elapsed   : float  - wall-clock seconds
    losses    : list   - loss per step
    final_loss: float
    """
    # Select K support windows spread across engine life
    n = len(X_engine)
    k = min(k, n)
    support_idx = [int(round(i * (n - 1) / (k - 1))) for i in range(k)] \
                  if k > 1 else [0]

    X_support = torch.FloatTensor(X_engine[support_idx])
    y_support = torch.FloatTensor(y_engine[support_idx]).unsqueeze(1)

    adapted   = copy.deepcopy(meta_model)
    adapted.train()
    optimizer = optim.SGD(adapted.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []
    start  = time.perf_counter()

    for step in range(steps):
        optimizer.zero_grad()
        pred = adapted(X_support)
        loss = criterion(pred, y_support)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    elapsed = time.perf_counter() - start
    return elapsed, losses, losses[-1]


# ──────────────────────────────────────────────
#  SIMULATE META-TRAINING
# ──────────────────────────────────────────────

def simulate_meta_training(X_train, y_train, engine_ids,
                            input_size, epochs=20, lr=3e-4):
    """
    Simulate MAML meta-training on training engines.
    Produces a meta-model whose weights are a good starting point
    for any new engine — this is what makes MAML adaptation fast.

    Also times the meta-training itself so you can report it.

    Returns
    -------
    meta_model : trained model
    meta_time  : float  - wall-clock seconds for meta-training
    """
    print("\nSimulating meta-training...")
    meta_model = SimpleRULModel(input_size=input_size)
    optimizer  = optim.Adam(meta_model.parameters(), lr=lr)
    criterion  = nn.MSELoss()

    unique_engines = np.unique(engine_ids)
    meta_model.train()

    start = time.perf_counter()

    for epoch in range(epochs):
        epoch_loss = 0.0
        # Each epoch: sample a few engines, do inner loop, accumulate grads
        sampled = np.random.choice(unique_engines,
                                   size=min(8, len(unique_engines)),
                                   replace=False)
        for eng in sampled:
            mask = engine_ids == eng
            X_e  = torch.FloatTensor(X_train[mask])
            y_e  = torch.FloatTensor(y_train[mask]).unsqueeze(1)

            if len(X_e) < 2:
                continue

            # Inner loop: 3 steps
            fast_weights = copy.deepcopy(meta_model)
            inner_opt    = optim.SGD(fast_weights.parameters(), lr=0.01)
            for _ in range(3):
                idx  = torch.randperm(len(X_e))[:8]
                pred = fast_weights(X_e[idx])
                l    = criterion(pred, y_e[idx])
                inner_opt.zero_grad()
                l.backward()
                inner_opt.step()

            # Outer loop: update meta-model
            optimizer.zero_grad()
            pred = fast_weights(X_e)
            loss = criterion(pred, y_e)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"  Meta epoch {epoch+1:3d} | loss: {epoch_loss/len(sampled):.4f}")

    meta_time = time.perf_counter() - start
    meta_model.eval()
    print(f"  Meta-training done in {meta_time:.2f}s")
    return meta_model, meta_time


# ──────────────────────────────────────────────
#  PLOTS
# ──────────────────────────────────────────────

def plot_time_comparison(results, save_path='results/figures/training_time_comparison.png'):
    """Bar chart: full retrain vs MAML adaptation time per engine."""
    engine_ids    = [r['engine_id'] for r in results]
    full_times    = [r['full_time']  for r in results]
    maml_times    = [r['maml_time']  for r in results]
    speedups      = [r['speedup']    for r in results]

    x   = np.arange(len(engine_ids))
    w   = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Time bars ──────────────────────────────────────────────────
    axes[0].bar(x - w/2, full_times, w, label='Full Retrain', color='tomato',   alpha=0.8)
    axes[0].bar(x + w/2, maml_times, w, label='MAML Adapt',   color='steelblue',alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'Eng {e}' for e in engine_ids], rotation=45)
    axes[0].set_ylabel('Time (seconds)')
    axes[0].set_title('Training Time: Full Retrain vs MAML Adaptation')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # ── Speedup ────────────────────────────────────────────────────
    axes[1].bar(x, speedups, color='mediumseagreen', alpha=0.8, edgecolor='black')
    axes[1].axhline(1.0, color='red', linestyle='--', lw=1.5, label='No speedup (1×)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'Eng {e}' for e in engine_ids], rotation=45)
    axes[1].set_ylabel('Speedup (×)')
    axes[1].set_title('MAML Speedup over Full Retraining')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    for i, s in enumerate(speedups):
        axes[1].text(i, s + 0.1, f'{s:.1f}×', ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Time comparison plot saved → {save_path}")
    plt.close()


def plot_convergence(results, save_path='results/figures/convergence_comparison.png'):
    """Loss curves: full retrain vs MAML adaptation for each engine."""
    n      = len(results)
    ncols  = min(3, n)
    nrows  = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5 * ncols, 4 * nrows),
                              squeeze=False)

    for idx, r in enumerate(results):
        ax  = axes[idx // ncols][idx % ncols]

        # Full retrain curve
        ax.plot(r['full_losses'], color='tomato', lw=2,
                label=f"Full retrain ({r['full_time']:.2f}s)")

        # MAML adaptation curve — plot on secondary x-axis annotation
        maml_x = np.linspace(0, len(r['full_losses']) - 1, len(r['maml_losses']))
        ax.plot(maml_x, r['maml_losses'], color='steelblue', lw=2,
                linestyle='--', label=f"MAML adapt ({r['maml_time']:.3f}s)")

        ax.set_title(f"Engine {r['engine_id']}  |  {r['speedup']:.1f}× speedup")
        ax.set_xlabel('Epoch / Step')
        ax.set_ylabel('MSE Loss')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.suptitle('Convergence: Full Retraining vs MAML Adaptation', fontsize=13, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Convergence plot saved → {save_path}")
    plt.close()


def plot_summary_table(summary, meta_time,
                       save_path='results/figures/timing_summary.png'):
    """Render summary stats as a clean table image."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')

    rows = [
        ['Metric', 'Full Retrain', 'MAML Adaptation'],
        ['Mean time per engine (s)',
         f"{summary['mean_full']:.3f}",
         f"{summary['mean_maml']:.3f}"],
        ['Total time — all engines (s)',
         f"{summary['total_full']:.3f}",
         f"{summary['total_maml']:.3f}"],
        ['Mean final loss',
         f"{summary['mean_full_loss']:.4f}",
         f"{summary['mean_maml_loss']:.4f}"],
        ['Mean speedup',
         '1.0×',
         f"{summary['mean_speedup']:.1f}×"],
        ['Meta-training overhead (s)',
         'N/A  (train from scratch each time)',
         f"{meta_time:.2f}"],
        ['Break-even engines',
         'N/A',
         f"~{summary['breakeven']:.0f}  engines"],
    ]

    tbl = ax.table(cellText=rows[1:], colLabels=rows[0],
                   cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 2)

    # Header styling
    for j in range(3):
        tbl[0, j].set_facecolor('#2c3e50')
        tbl[0, j].set_text_props(color='white', fontweight='bold')

    # Highlight MAML column
    for i in range(1, len(rows)):
        tbl[i, 2].set_facecolor('#d5f5e3')

    plt.title('Training Time Comparison Summary', fontsize=13,
              fontweight='bold', pad=20)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Summary table saved → {save_path}")
    plt.close()


# ──────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────

def main(n_engines=5, k_shot=5, adapt_steps=5, full_epochs=50):

    print("=" * 65)
    print("TRAINING TIME COMPARISON: Full Retrain vs MAML Adaptation")
    print("=" * 65)

    # ── Load data ─────────────────────────────────────────────────────
    data          = load_preprocessed_data()
    feature_names = data['feature_names'].tolist()
    base_indices, base_names = get_base_indices(feature_names)

    # Use base features only
    X_train    = data['X_train'][:, :, base_indices]
    y_train    = data['y_train']
    train_ids  = data['train_engine_ids'].astype(int)

    X_test     = data['X_test'][:, :, base_indices]
    y_test     = data['y_test']
    test_ids   = data['test_engine_ids'].astype(int)

    input_size = X_train.shape[2]
    print(f"\n  Input size (base features): {input_size}")
    print(f"  Test engines available    : {np.unique(test_ids)}")
    print(f"  Evaluating {n_engines} engines")

    # ── Meta-training ─────────────────────────────────────────────────
    meta_model, meta_time = simulate_meta_training(
        X_train, y_train, train_ids, input_size
    )

    # Random init template for full retraining baseline
    random_model = SimpleRULModel(input_size=input_size)

    # ── Per-engine comparison ─────────────────────────────────────────
    unique_test = np.unique(test_ids)[:n_engines]
    results     = []

    print(f"\n{'Engine':>8}  {'Full(s)':>9}  {'MAML(s)':>9}  "
          f"{'Speedup':>9}  {'Full_loss':>10}  {'MAML_loss':>10}")
    print("-" * 65)

    for eng_id in unique_test:
        mask     = test_ids == eng_id
        X_engine = X_test[mask]
        y_engine = y_test[mask]

        if len(X_engine) < k_shot:
            continue

        # Full retrain from scratch
        full_time, full_losses, full_final = full_retrain(
            X_engine, y_engine, random_model,
            epochs=full_epochs, lr=1e-3
        )

        # MAML adaptation from meta weights
        maml_time, maml_losses, maml_final = maml_adapt(
            X_engine, y_engine, meta_model,
            k=k_shot, steps=adapt_steps, lr=0.01
        )

        speedup = full_time / maml_time if maml_time > 0 else float('inf')

        results.append({
            'engine_id'  : eng_id,
            'full_time'  : full_time,
            'maml_time'  : maml_time,
            'speedup'    : speedup,
            'full_losses': full_losses,
            'maml_losses': maml_losses,
            'full_final' : full_final,
            'maml_final' : maml_final,
        })

        print(f"{eng_id:>8}  {full_time:>9.3f}  {maml_time:>9.4f}  "
              f"{speedup:>9.1f}×  {full_final:>10.4f}  {maml_final:>10.4f}")

    # ── Summary ───────────────────────────────────────────────────────
    mean_full     = np.mean([r['full_time']   for r in results])
    mean_maml     = np.mean([r['maml_time']   for r in results])
    total_full    = sum(r['full_time']         for r in results)
    total_maml    = sum(r['maml_time']         for r in results)
    mean_speedup  = np.mean([r['speedup']     for r in results])
    mean_fl       = np.mean([r['full_final']  for r in results])
    mean_ml       = np.mean([r['maml_final']  for r in results])

    # Break-even: how many engines before meta-training overhead is recovered
    time_saved_per_engine = mean_full - mean_maml
    breakeven = meta_time / time_saved_per_engine if time_saved_per_engine > 0 else float('inf')

    summary = {
        'mean_full'      : mean_full,
        'mean_maml'      : mean_maml,
        'total_full'     : total_full,
        'total_maml'     : total_maml,
        'mean_speedup'   : mean_speedup,
        'mean_full_loss' : mean_fl,
        'mean_maml_loss' : mean_ml,
        'breakeven'      : breakeven,
    }

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"  Mean time — full retrain : {mean_full:.3f}s per engine")
    print(f"  Mean time — MAML adapt   : {mean_maml:.4f}s per engine")
    print(f"  Mean speedup             : {mean_speedup:.1f}×")
    print(f"  Mean final loss (full)   : {mean_fl:.4f}")
    print(f"  Mean final loss (MAML)   : {mean_ml:.4f}")
    print(f"  Meta-training overhead   : {meta_time:.2f}s (one-time cost)")
    print(f"  Break-even at            : ~{breakeven:.0f} engines")
    print(f"    (after {breakeven:.0f} engines, MAML has paid back its meta-training cost)")
    print("=" * 65)

    # ── Plots ─────────────────────────────────────────────────────────
    plot_time_comparison(results)
    plot_convergence(results)
    plot_summary_table(summary, meta_time)

    print("\n✓ All plots saved to results/figures/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--engines', type=int, default=5,
                        help='Number of test engines to compare (default: 5)')
    parser.add_argument('--k',       type=int, default=5,
                        help='MAML K-shot support size (default: 5)')
    parser.add_argument('--steps',   type=int, default=5,
                        help='MAML inner loop steps (default: 5)')
    parser.add_argument('--epochs',  type=int, default=50,
                        help='Full retrain epochs (default: 50)')
    args = parser.parse_args()

    main(n_engines=args.engines,
         k_shot=args.k,
         adapt_steps=args.steps,
         full_epochs=args.epochs)