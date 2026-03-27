"""
MAML Meta-Training Script — Fixed

KEY FIXES over previous version:
  1. Validation evaluation now uses the SAME support-selection strategy as
     test time (middle 20-80% of engine life, evenly spaced), so the saved
     "best" checkpoint genuinely corresponds to best test performance.
  2. Early stopping patience added (patience=60 epochs) to catch the real
     minimum instead of epoch-30 flukes.
  3. Increased tasks_per_engine to 5 and meta_epochs to 600 for more stable
     convergence before the cosine LR decays away.
  4. Validation now uses MORE inner steps (15) matching evaluate_maml.py,
     so the val RMSE is a faithful proxy of test-time RMSE.
  5. Meta-batch size raised to 24 for lower gradient variance.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.models.maml_model import CNNLSTMBase
from src.data.data_loader import load_preprocessed_data
import higher


# ──────────────────────────────────────────────────────────────────────────────
# Task creation  (augmented)
# ──────────────────────────────────────────────────────────────────────────────

def create_task_dataset(data_dict, n_support=5, min_query=20, tasks_per_engine=5):
    """
    Build meta-training tasks.

    Each engine produces `tasks_per_engine` tasks by sampling support windows
    from different positions in the degradation timeline.

    Support windows are drawn from the SECOND HALF of each engine's life so
    they contain actual degradation signal (not all-healthy RUL=130 windows).
    """
    X_train    = data_dict['X_train']
    y_train    = data_dict['y_train']
    engine_ids = data_dict['train_engine_ids']

    unique_engines = np.unique(engine_ids)
    tasks = []

    for engine_id in unique_engines:
        indices = np.sort(np.where(engine_ids == engine_id)[0])

        if len(indices) < n_support + min_query:
            continue

        n = len(indices)
        degradation_start = n // 2

        for _ in range(tasks_per_engine):
            max_start = n - n_support - min_query
            if max_start <= degradation_start:
                support_start = degradation_start
            else:
                support_start = np.random.randint(degradation_start, max_start)

            support_idx = indices[support_start: support_start + n_support]
            query_idx   = indices[support_start + n_support:]

            tasks.append({
                'support_X': X_train[support_idx],
                'support_y': y_train[support_idx],
                'query_X':   X_train[query_idx],
                'query_y':   y_train[query_idx],
            })

    print(f"✓ Created {len(tasks)} augmented tasks "
          f"({len(unique_engines)} engines × ~{tasks_per_engine} tasks each)")
    return tasks


# ──────────────────────────────────────────────────────────────────────────────
# MAML outer loop
# ──────────────────────────────────────────────────────────────────────────────

def maml_outer_loop(model, tasks, meta_opt, inner_lr, inner_steps, device):
    criterion = nn.MSELoss()
    meta_losses = []

    for task in tasks:
        support_X = torch.FloatTensor(task['support_X']).to(device)
        support_y = torch.FloatTensor(task['support_y']).unsqueeze(1).to(device)
        query_X   = torch.FloatTensor(task['query_X']).to(device)
        query_y   = torch.FloatTensor(task['query_y']).unsqueeze(1).to(device)

        inner_opt = optim.SGD(model.parameters(), lr=inner_lr)

        with higher.innerloop_ctx(model, inner_opt,
                                  copy_initial_weights=False) as (fmodel, diffopt):
            for _ in range(inner_steps):
                support_pred = fmodel(support_X)
                inner_loss   = criterion(support_pred, support_y)
                diffopt.step(inner_loss)

            query_pred = fmodel(query_X)
            query_loss = criterion(query_pred, query_y)
            meta_losses.append(query_loss)

    meta_opt.zero_grad()
    total_loss = torch.stack(meta_losses).mean()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    meta_opt.step()

    return total_loss.item()


# ──────────────────────────────────────────────────────────────────────────────
# Validation  — FIXED: same support strategy as test time
# ──────────────────────────────────────────────────────────────────────────────

def select_support_indices(n_windows, K):
    """
    FIX: identical to evaluate_maml.py — samples from the MIDDLE 60%
    so val RMSE matches what the test evaluator sees.
    """
    start = int(n_windows * 0.2)
    end   = int(n_windows * 0.8)

    if (end - start) < K:
        return np.linspace(0, n_windows - 1, K, dtype=int)

    return np.linspace(start, end - 1, K, dtype=int)


def evaluate_on_val(model, val_tasks, inner_lr, inner_steps, device, max_rul,
                    K=5):
    """
    FIX: support is now sampled with select_support_indices (middle 60%),
    inner_steps raised to 15 to match evaluate_maml.py.
    """
    criterion = nn.MSELoss()
    rmses = []

    for task in val_tasks[:30]:   # slightly more tasks for stable estimate
        X_engine = task['query_X']   # reuse the full engine window pool
        y_engine = task['query_y']

        n = len(X_engine)
        if n < K + 5:
            continue

        # FIX: same support selection as test time
        support_pos = select_support_indices(n, K)
        query_start = n // 2
        query_pos   = np.arange(query_start, n)
        query_pos   = query_pos[~np.isin(query_pos, support_pos)]

        if len(query_pos) < 5:
            continue

        support_X = torch.FloatTensor(X_engine[support_pos]).to(device)
        support_y = torch.FloatTensor(y_engine[support_pos]).unsqueeze(1).to(device)
        query_X   = torch.FloatTensor(X_engine[query_pos]).to(device)
        query_y   = y_engine[query_pos]

        adapted = model.clone().to(device)
        opt     = optim.SGD(adapted.parameters(), lr=inner_lr)

        adapted.train()
        for _ in range(inner_steps):   # FIX: 15 steps, same as test
            opt.zero_grad()
            loss = criterion(adapted(support_X), support_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapted.parameters(), max_norm=5.0)
            opt.step()

        adapted.eval()
        with torch.no_grad():
            preds = np.clip(
                adapted(query_X).cpu().numpy().flatten(), 0.0, 1.0
            ) * max_rul
            y_act = query_y * max_rul
            rmses.append(np.sqrt(np.mean((preds - y_act) ** 2)))

    return float(np.mean(rmses)) if rmses else float('inf')


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def train_maml():
    print("=" * 60)
    print("MAML META-TRAINING  (fixed)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # ── Hyperparameters ───────────────────────────────────────────────────
    n_support        = 5
    inner_lr         = 0.05
    meta_lr          = 0.001
    inner_steps      = 10      # training inner steps
    val_inner_steps  = 15      # FIX: validation uses same steps as test
    meta_batch_size  = 24      # FIX: raised for lower gradient variance
    meta_epochs      = 600     # FIX: more epochs; early stopping guards against overfit
    tasks_per_engine = 5       # FIX: more tasks for stable gradient
    patience         = 60      # early stopping

    print(f"\nHyperparameters:")
    print(f"  K-shot           : {n_support}")
    print(f"  Inner LR (α)     : {inner_lr}")
    print(f"  Meta LR (β)      : {meta_lr}")
    print(f"  Inner steps      : {inner_steps} (train) / {val_inner_steps} (val/test)")
    print(f"  Meta batch size  : {meta_batch_size}")
    print(f"  Meta epochs      : {meta_epochs}")
    print(f"  Tasks per engine : {tasks_per_engine}")
    print(f"  Early stop pat.  : {patience} epochs")

    # ── Data ──────────────────────────────────────────────────────────────
    print("\nLoading data...")
    data_dict = load_preprocessed_data('data/processed/FD001_preprocessed.npz')
    max_rul   = float(data_dict['max_rul'])
    print(f"  max_rul          : {max_rul}")
    print(f"  X_train shape    : {data_dict['X_train'].shape}")

    # ── Tasks ─────────────────────────────────────────────────────────────
    print("\nCreating augmented task datasets...")
    all_tasks = create_task_dataset(
        data_dict, n_support=n_support, tasks_per_engine=tasks_per_engine
    )

    split       = int(0.8 * len(all_tasks))
    train_tasks = all_tasks[:split]
    val_tasks   = all_tasks[split:]
    print(f"  Meta-train tasks : {len(train_tasks)}")
    print(f"  Meta-val tasks   : {len(val_tasks)}")

    # ── Model ─────────────────────────────────────────────────────────────
    print("\nInitialising MAML model...")
    model    = CNNLSTMBase(input_size=102).to(device)
    meta_opt = optim.Adam(model.parameters(), lr=meta_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        meta_opt, T_max=meta_epochs, eta_min=1e-5
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters : {total_params:,}")

    # ── Meta-training loop ────────────────────────────────────────────────
    print("\nStarting meta-training...")
    print("-" * 60)

    best_val_rmse   = float('inf')
    no_improve_count = 0
    os.makedirs('results/saved_models', exist_ok=True)

    for epoch in range(1, meta_epochs + 1):

        # Regenerate tasks every 50 epochs for fresh support sampling
        if epoch % 50 == 1 and epoch > 1:
            all_tasks   = create_task_dataset(
                data_dict, n_support=n_support, tasks_per_engine=tasks_per_engine
            )
            train_tasks = all_tasks[:int(0.8 * len(all_tasks))]
            val_tasks   = all_tasks[int(0.8 * len(all_tasks)):]

        idx        = np.random.choice(len(train_tasks),
                                      size=min(meta_batch_size, len(train_tasks)),
                                      replace=False)
        meta_batch = [train_tasks[i] for i in idx]

        meta_loss  = maml_outer_loop(
            model, meta_batch, meta_opt, inner_lr, inner_steps, device
        )
        scheduler.step()

        if epoch % 10 == 0:
            # FIX: val eval uses same inner_steps and support strategy as test
            val_rmse = evaluate_on_val(
                model, val_tasks, inner_lr, val_inner_steps, device, max_rul
            )
            is_best = val_rmse < best_val_rmse
            marker  = " ◀ best" if is_best else ""
            print(f"Epoch {epoch:3d} | Meta-Loss: {meta_loss:.4f} | "
                  f"Val RMSE: {val_rmse:.2f} cycles{marker}")

            if is_best:
                best_val_rmse    = val_rmse
                no_improve_count = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'val_rmse': val_rmse,
                    'hyperparameters': {
                        'n_support':   n_support,
                        'inner_lr':    inner_lr,
                        'meta_lr':     meta_lr,
                        'inner_steps': val_inner_steps,
                    }
                }, 'results/saved_models/maml_meta_model_best.pth')
            else:
                no_improve_count += 10   # epochs between checks

            # Early stopping
            if no_improve_count >= patience:
                print(f"\n⚠  Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

        elif epoch % 5 == 0:
            print(f"Epoch {epoch:3d} | Meta-Loss: {meta_loss:.4f}")

    print("-" * 60)
    print(f"\n✓ Best Val RMSE : {best_val_rmse:.2f} cycles")

    torch.save({
        'model_state_dict': model.state_dict(),
        'hyperparameters': {
            'n_support':   n_support,
            'inner_lr':    inner_lr,
            'meta_lr':     meta_lr,
            'inner_steps': val_inner_steps,
        }
    }, 'results/saved_models/maml_meta_model.pth')

    print("✓ Final model saved  →  results/saved_models/maml_meta_model.pth")
    print("✓ Best  model saved  →  results/saved_models/maml_meta_model_best.pth")
    print("=" * 60)

    return model


if __name__ == '__main__':
    train_maml()