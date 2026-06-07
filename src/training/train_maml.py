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
# Task creation — split-aware and time-series safe
# ──────────────────────────────────────────────────────────────────────────────

def create_task_dataset_from_split(X_data, y_data, engine_ids,
                                   n_support=5, min_query=20, tasks_per_engine=5):
    """
    Build MAML tasks from a single split only.

    Time-series rule:
    - support comes from an earlier degradation region
    - query comes strictly after support
    So no future window is used to adapt before predicting earlier windows.
    """
    unique_engines = np.unique(engine_ids)
    tasks = []

    for engine_id in unique_engines:
        indices = np.sort(np.where(engine_ids == engine_id)[0])

        if len(indices) < n_support + min_query:
            continue

        n = len(indices)

        # safer time-series regions
        support_region_start = int(n * 0.30)
        support_region_end   = int(n * 0.50)
        query_region_start   = int(n * 0.50)

        if support_region_end - support_region_start < n_support:
            support_region_start = max(0, n // 3)
            support_region_end   = min(n - min_query, support_region_start + n_support)

        if support_region_end - support_region_start < n_support:
            continue

        if n - query_region_start < 5:
            continue

        for _ in range(tasks_per_engine):
            max_start = support_region_end - n_support

            if max_start <= support_region_start:
                support_start = support_region_start
            else:
                support_start = np.random.randint(support_region_start, max_start + 1)

            support_idx = indices[support_start:support_start + n_support]

            # strictly later query pool
            query_pool = indices[query_region_start:]

            # remove overlap if any
            query_pool = query_pool[~np.isin(query_pool, support_idx)]

            if len(query_pool) < 5:
                continue

            # keep query size controlled for lower variance
            if len(query_pool) > 25:
                chosen = np.linspace(0, len(query_pool) - 1, 25, dtype=int)
                query_idx = query_pool[chosen]
            else:
                query_idx = query_pool

            tasks.append({
                'support_X': X_data[support_idx],
                'support_y': y_data[support_idx],
                'query_X':   X_data[query_idx],
                'query_y':   y_data[query_idx],
            })

    np.random.shuffle(tasks)
    print(f"✓ Created {len(tasks)} tasks from {len(unique_engines)} engines")
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
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    meta_opt.step()

    return total_loss.item()


# ──────────────────────────────────────────────────────────────────────────────
# Validation — true validation engines only, causal support/query
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_on_val(model, val_tasks, inner_lr, inner_steps, device, max_rul):
    criterion = nn.MSELoss()
    rmses = []

    for task in val_tasks[:30]:
        support_X = torch.FloatTensor(task['support_X']).to(device)
        support_y = torch.FloatTensor(task['support_y']).unsqueeze(1).to(device)
        query_X   = torch.FloatTensor(task['query_X']).to(device)
        query_y   = task['query_y']

        if len(query_y) < 5:
            continue

        adapted = model.clone().to(device)
        opt = optim.SGD(adapted.parameters(), lr=inner_lr)

        adapted.train()
        for _ in range(inner_steps):
            opt.zero_grad()
            pred = adapted(support_X)
            loss = criterion(pred, support_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapted.parameters(), max_norm=5.0)
            opt.step()

        adapted.eval()
        with torch.no_grad():
            preds = np.clip(
                adapted(query_X).cpu().numpy().flatten(), 0.0, 1.0
            ) * max_rul
            y_act = query_y * max_rul
            rmse = np.sqrt(np.mean((preds - y_act) ** 2))
            rmses.append(rmse)

    return float(np.mean(rmses)) if rmses else float('inf')


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def train_maml():
    print("=" * 60)
    print("MAML META-TRAINING  (time-series safe, no leakage)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # safer for K=5
    n_support        = 5
    inner_lr         = 0.005
    meta_lr          = 0.0003
    inner_steps      = 3
    val_inner_steps  = 5
    meta_batch_size  = 16
    meta_epochs      = 300
    tasks_per_engine = 5
    patience         = 40

    print(f"\nHyperparameters:")
    print(f"  K-shot           : {n_support}")
    print(f"  Inner LR (α)     : {inner_lr}")
    print(f"  Meta LR (β)      : {meta_lr}")
    print(f"  Inner steps      : {inner_steps} (train) / {val_inner_steps} (val)")
    print(f"  Meta batch size  : {meta_batch_size}")
    print(f"  Meta epochs      : {meta_epochs}")
    print(f"  Tasks per engine : {tasks_per_engine}")
    print(f"  Early stop pat.  : {patience} epochs")

    print("\nLoading data...")
    data_dict = load_preprocessed_data('data/processed/FD001_preprocessed.npz')
    max_rul   = float(data_dict['max_rul'])

    print(f"  max_rul          : {max_rul}")
    print(f"  X_train shape    : {data_dict['X_train'].shape}")
    print(f"  X_val shape      : {data_dict['X_val'].shape}")

    print("\nCreating split-aware task datasets...")
    train_tasks = create_task_dataset_from_split(
        data_dict['X_train'],
        data_dict['y_train'],
        data_dict['train_engine_ids'],
        n_support=n_support,
        tasks_per_engine=tasks_per_engine
    )

    val_tasks = create_task_dataset_from_split(
        data_dict['X_val'],
        data_dict['y_val'],
        data_dict['val_engine_ids'],
        n_support=n_support,
        tasks_per_engine=3
    )

    print(f"  Meta-train tasks : {len(train_tasks)}")
    print(f"  Meta-val tasks   : {len(val_tasks)}")

    print("\nInitialising MAML model...")
    model = CNNLSTMBase(input_size=102).to(device)

    meta_opt = optim.AdamW(model.parameters(), lr=meta_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        meta_opt, T_max=meta_epochs, eta_min=1e-5
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters : {total_params:,}")

    print("\nStarting meta-training...")
    print("-" * 60)

    best_val_rmse = float('inf')
    no_improve_count = 0
    os.makedirs('results/saved_models', exist_ok=True)

    for epoch in range(1, meta_epochs + 1):

        if epoch % 50 == 1 and epoch > 1:
            train_tasks = create_task_dataset_from_split(
                data_dict['X_train'],
                data_dict['y_train'],
                data_dict['train_engine_ids'],
                n_support=n_support,
                tasks_per_engine=tasks_per_engine
            )

            val_tasks = create_task_dataset_from_split(
                data_dict['X_val'],
                data_dict['y_val'],
                data_dict['val_engine_ids'],
                n_support=n_support,
                tasks_per_engine=3
            )

        idx = np.random.choice(
            len(train_tasks),
            size=min(meta_batch_size, len(train_tasks)),
            replace=False
        )
        meta_batch = [train_tasks[i] for i in idx]

        meta_loss = maml_outer_loop(
            model, meta_batch, meta_opt, inner_lr, inner_steps, device
        )
        scheduler.step()

        if epoch % 10 == 0:
            val_rmse = evaluate_on_val(
                model, val_tasks, inner_lr, val_inner_steps, device, max_rul
            )

            is_best = val_rmse < best_val_rmse
            marker = " ◀ best" if is_best else ""

            print(f"Epoch {epoch:3d} | Meta-Loss: {meta_loss:.4f} | "
                  f"Val RMSE: {val_rmse:.2f} cycles{marker}")

            if is_best:
                best_val_rmse = val_rmse
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
                no_improve_count += 10

            if no_improve_count >= patience:
                print(f"\n⚠ Early stopping at epoch {epoch} "
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