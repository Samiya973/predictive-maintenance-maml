# import copy
# import os,sys
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import higher

# sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


# from src.models.baselines import LSTMBaseline
# from src.data.data_loader import load_preprocessed_data


# LSTMBaseline.clone = lambda self: copy.deepcopy(self)


# # ─────────────────────────────────────────────────────────────────────────────
# # Task creation from a single already-separated split
# # ─────────────────────────────────────────────────────────────────────────────

# def create_task_dataset_from_split(X, y, engine_ids, n_support=5, min_query=20, tasks_per_engine=5):
#     """
#     Build meta-learning tasks from one split only.
#     This keeps engines disjoint across train/val/test.
#     """
#     unique_engines = np.unique(engine_ids)
#     tasks = []

#     for engine_id in unique_engines:
#         indices = np.sort(np.where(engine_ids == engine_id)[0])

#         if len(indices) < n_support + min_query:
#             continue

#         n = len(indices)
#         degradation_start = n // 2

#         for _ in range(tasks_per_engine):
#             max_start = n - n_support - min_query

#             support_start = (
#                 degradation_start
#                 if max_start <= degradation_start
#                 else np.random.randint(degradation_start, max_start)
#             )

#             support_idx = indices[support_start : support_start + n_support]
#             query_idx   = indices[support_start + n_support :]

#             tasks.append({
#                 'support_X': X[support_idx],
#                 'support_y': y[support_idx],
#                 'query_X':   X[query_idx],
#                 'query_y':   y[query_idx],
#             })

#     print(f"✓ Created {len(tasks)} tasks ({len(unique_engines)} engines × ~{tasks_per_engine} each)")
#     return tasks


# # ─────────────────────────────────────────────────────────────────────────────
# # MAML outer loop
# # ─────────────────────────────────────────────────────────────────────────────

# def maml_outer_loop(model, tasks, meta_opt, inner_lr, inner_steps, device):
#     criterion = nn.MSELoss()
#     meta_losses = []

#     for task in tasks:
#         support_X = torch.FloatTensor(task['support_X']).to(device)
#         support_y = torch.FloatTensor(task['support_y']).unsqueeze(1).to(device)

#         query_X = torch.FloatTensor(task['query_X']).to(device)
#         query_y = torch.FloatTensor(task['query_y']).unsqueeze(1).to(device)

#         inner_opt = optim.SGD(model.parameters(), lr=inner_lr)

#         with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (fmodel, diffopt):
#             for _ in range(inner_steps):
#                 support_pred = fmodel(support_X)
#                 inner_loss = criterion(support_pred, support_y)
#                 diffopt.step(inner_loss)

#             query_pred = fmodel(query_X)
#             meta_losses.append(criterion(query_pred, query_y))

#     meta_opt.zero_grad()
#     total_loss = torch.stack(meta_losses).mean()
#     total_loss.backward()
#     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
#     meta_opt.step()

#     return total_loss.item()


# # ─────────────────────────────────────────────────────────────────────────────
# # Support index selection
# # ─────────────────────────────────────────────────────────────────────────────

# def select_support_indices(n_windows, K):
#     start = int(n_windows * 0.2)
#     end = int(n_windows * 0.8)

#     if (end - start) < K:
#         return np.linspace(0, n_windows - 1, K, dtype=int)

#     return np.linspace(start, end - 1, K, dtype=int)


# # ─────────────────────────────────────────────────────────────────────────────
# # Validation
# # ─────────────────────────────────────────────────────────────────────────────

# def evaluate_on_val(model, val_tasks, inner_lr, inner_steps, device, max_rul, K=5):
#     criterion = nn.MSELoss()
#     rmses = []

#     for task in val_tasks[:30]:
#         X_engine = task['query_X']
#         y_engine = task['query_y']

#         n = len(X_engine)
#         if n < K + 5:
#             continue

#         support_pos = select_support_indices(n, K)
#         query_start = n // 2
#         query_pos = np.arange(query_start, n)
#         query_pos = query_pos[~np.isin(query_pos, support_pos)]

#         if len(query_pos) < 5:
#             continue

#         support_X = torch.FloatTensor(X_engine[support_pos]).to(device)
#         support_y = torch.FloatTensor(y_engine[support_pos]).unsqueeze(1).to(device)

#         query_X = torch.FloatTensor(X_engine[query_pos]).to(device)
#         query_y = y_engine[query_pos]

#         adapted = model.clone().to(device)
#         opt = optim.SGD(adapted.parameters(), lr=inner_lr)

#         adapted.train()
#         for _ in range(inner_steps):
#             opt.zero_grad()
#             loss = criterion(adapted(support_X), support_y)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(adapted.parameters(), max_norm=5.0)
#             opt.step()

#         adapted.eval()
#         with torch.no_grad():
#             preds = np.clip(
#                 adapted(query_X).cpu().numpy().flatten(), 0.0, 1.0
#             ) * max_rul

#             y_true = query_y * max_rul
#             rmse = np.sqrt(np.mean((preds - y_true) ** 2))
#             rmses.append(rmse)

#     return float(np.mean(rmses)) if rmses else float('inf')


# # ─────────────────────────────────────────────────────────────────────────────
# # Main
# # ─────────────────────────────────────────────────────────────────────────────

# def train_maml_lstm():
#     print("=" * 60)
#     print("MAML META-TRAINING  —  LSTMBaseline backbone")
#     print("=" * 60)

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"\nDevice: {device}")

#     n_support = 5
#     inner_lr = 0.05
#     meta_lr = 0.001
#     inner_steps = 10
#     val_inner_steps = 15
#     meta_batch_size = 24
#     meta_epochs = 600
#     tasks_per_engine = 5
#     patience = 60

#     print(f"\nHyperparameters:")
#     print(f"  K-shot           : {n_support}")
#     print(f"  Inner LR (α)     : {inner_lr}")
#     print(f"  Meta  LR (β)     : {meta_lr}")
#     print(f"  Inner steps      : {inner_steps} (train) / {val_inner_steps} (val/test)")
#     print(f"  Meta batch size  : {meta_batch_size}")
#     print(f"  Meta epochs      : {meta_epochs}")
#     print(f"  Tasks per engine : {tasks_per_engine}")
#     print(f"  Early-stop pat.  : {patience} epochs")

#     print("\nLoading data...")
#     data_dict = load_preprocessed_data('data/processed/FD001_preprocessed.npz')

#     max_rul = float(data_dict['max_rul'])
#     n_features = data_dict['X_train'].shape[2]

#     print(f"  max_rul          : {max_rul}")
#     print(f"  input features   : {n_features}")
#     print(f"  X_train shape    : {data_dict['X_train'].shape}")

#     # Check required keys
#     required_keys = [
#         'X_train', 'y_train', 'train_engine_ids',
#         'X_val', 'y_val', 'val_engine_ids'
#     ]
#     for key in required_keys:
#         if key not in data_dict:
#             raise KeyError(f"Missing key in .npz file: {key}")

#     print("\nCreating engine-disjoint meta-task datasets...")
#     train_tasks = create_task_dataset_from_split(
#         data_dict['X_train'],
#         data_dict['y_train'],
#         data_dict['train_engine_ids'],
#         n_support=n_support,
#         tasks_per_engine=tasks_per_engine
#     )

#     val_tasks = create_task_dataset_from_split(
#         data_dict['X_val'],
#         data_dict['y_val'],
#         data_dict['val_engine_ids'],
#         n_support=n_support,
#         tasks_per_engine=tasks_per_engine
#     )

#     print(f"  Meta-train tasks : {len(train_tasks)}")
#     print(f"  Meta-val   tasks : {len(val_tasks)}")

#     print("\nInitialising LSTMBaseline for MAML...")
#     model = LSTMBaseline(input_size=n_features).to(device)

#     meta_opt = optim.Adam(model.parameters(), lr=meta_lr)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#         meta_opt, T_max=meta_epochs, eta_min=1e-5
#     )

#     print(f"  Total parameters : {model.count_parameters():,}")

#     print("\nStarting meta-training...")
#     print("-" * 60)

#     best_val_rmse = float('inf')
#     no_improve_count = 0
#     os.makedirs('results/saved_models', exist_ok=True)

#     for epoch in range(1, meta_epochs + 1):

#         if epoch % 50 == 1 and epoch > 1:
#             print("\nRefreshing meta-task datasets...")
#             train_tasks = create_task_dataset_from_split(
#                 data_dict['X_train'],
#                 data_dict['y_train'],
#                 data_dict['train_engine_ids'],
#                 n_support=n_support,
#                 tasks_per_engine=tasks_per_engine
#             )

#             val_tasks = create_task_dataset_from_split(
#                 data_dict['X_val'],
#                 data_dict['y_val'],
#                 data_dict['val_engine_ids'],
#                 n_support=n_support,
#                 tasks_per_engine=tasks_per_engine
#             )

#         idx = np.random.choice(
#             len(train_tasks),
#             size=min(meta_batch_size, len(train_tasks)),
#             replace=False
#         )

#         meta_batch = [train_tasks[i] for i in idx]

#         meta_loss = maml_outer_loop(
#             model, meta_batch, meta_opt, inner_lr, inner_steps, device
#         )

#         scheduler.step()

#         if epoch % 10 == 0:
#             val_rmse = evaluate_on_val(
#                 model, val_tasks, inner_lr, val_inner_steps, device, max_rul, K=n_support
#             )

#             is_best = val_rmse < best_val_rmse
#             marker = " ◀ best" if is_best else ""

#             print(f"Epoch {epoch:3d} | Meta-Loss: {meta_loss:.4f} | Val RMSE: {val_rmse:.2f} cycles{marker}")

#             if is_best:
#                 best_val_rmse = val_rmse
#                 no_improve_count = 0

#                 torch.save(
#                     {
#                         'model_state_dict': model.state_dict(),
#                         'epoch': epoch,
#                         'val_rmse': val_rmse,
#                         'hyperparameters': {
#                             'n_support': n_support,
#                             'inner_lr': inner_lr,
#                             'meta_lr': meta_lr,
#                             'inner_steps': val_inner_steps,
#                             'backbone': 'LSTMBaseline',
#                         },
#                     },
#                     'results/saved_models/maml_lstm_best.pth',
#                 )
#             else:
#                 no_improve_count += 10

#             if no_improve_count >= patience:
#                 print(f"\n⚠ Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
#                 break

#         elif epoch % 5 == 0:
#             print(f"Epoch {epoch:3d} | Meta-Loss: {meta_loss:.4f}")

#     print("-" * 60)
#     print(f"\n✓ Best Val RMSE : {best_val_rmse:.2f} cycles")

#     torch.save(
#         {
#             'model_state_dict': model.state_dict(),
#             'hyperparameters': {
#                 'n_support': n_support,
#                 'inner_lr': inner_lr,
#                 'meta_lr': meta_lr,
#                 'inner_steps': val_inner_steps,
#                 'backbone': 'LSTMBaseline',
#             },
#         },
#         'results/saved_models/maml_lstm_final.pth',
#     )

#     print("✓ Final model → results/saved_models/maml_lstm_final.pth")
#     print("✓ Best  model → results/saved_models/maml_lstm_best.pth")
#     print("=" * 60)

#     return model


# if __name__ == '__main__':
#     train_maml_lstm()

"""
MAML on LSTMBaseline
Uses the LSTMBaseline (2-layer LSTM + ReLU output) as the meta-learner backbone.

Leakage-free preprocessing assumed (split engines before scaling).

NOTE on output activation:
  LSTMBaseline uses ReLU → outputs in [0, +∞).
  Labels are normalised to [0, 1] (y / 130).
  MSE pushes outputs into [0,1] during training.
  At eval time predictions are clipped to [0,1] before rescaling.
"""

import copy
import os,sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import higher
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.models.baselines import LSTMBaseline
from src.data.data_loader import load_preprocessed_data


# ─────────────────────────────────────────────────────────────────────────────
# Monkey-patch .clone() onto LSTMBaseline (mirrors CNNLSTMBase.clone)
# ─────────────────────────────────────────────────────────────────────────────

LSTMBaseline.clone = lambda self: copy.deepcopy(self)


# ─────────────────────────────────────────────────────────────────────────────
# Task creation
# ─────────────────────────────────────────────────────────────────────────────

def create_task_dataset(data_dict, n_support=5, min_query=20, tasks_per_engine=5):
    """
    Build meta-training tasks from training engines.

    Each engine contributes `tasks_per_engine` tasks. Support windows are
    drawn from the SECOND HALF of each engine's life so they carry real
    degradation signal rather than all-healthy RUL=130 windows.

    Parameters
    ----------
    data_dict       : dict   Output of load_preprocessed_data.
    n_support       : int    K-shot support size per task.
    min_query       : int    Minimum query sequences required per task.
    tasks_per_engine: int    How many tasks to sample per engine.

    Returns
    -------
    tasks : list of dicts, each with keys
        support_X, support_y, query_X, query_y  (numpy arrays)
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

        n                 = len(indices)
        degradation_start = n // 2   # only sample from second half

        for _ in range(tasks_per_engine):
            max_start = n - n_support - min_query

            support_start = (
                degradation_start
                if max_start <= degradation_start
                else np.random.randint(degradation_start, max_start)
            )

            support_idx = indices[support_start : support_start + n_support]
            query_idx   = indices[support_start + n_support :]

            tasks.append({
                'support_X': X_train[support_idx],
                'support_y': y_train[support_idx],
                'query_X':   X_train[query_idx],
                'query_y':   y_train[query_idx],
            })

    print(f"✓ Created {len(tasks)} tasks "
          f"({len(unique_engines)} engines × ~{tasks_per_engine} each)")
    return tasks


# ─────────────────────────────────────────────────────────────────────────────
# MAML outer loop
# ─────────────────────────────────────────────────────────────────────────────

def maml_outer_loop(model, tasks, meta_opt, inner_lr, inner_steps, device):
    """
    One meta-update step over a batch of tasks.

    For each task:
      - Run `inner_steps` SGD steps on support set  (via higher differentiable opt)
      - Evaluate adapted model on query set
    Accumulate query losses, average, backprop through inner loop into meta-params.

    Parameters
    ----------
    model       : LSTMBaseline   Meta-initialisation (θ).
    tasks       : list           Batch of task dicts.
    meta_opt    : Optimizer      Outer (meta) optimiser.
    inner_lr    : float          Inner-loop learning rate (α).
    inner_steps : int            Number of inner gradient steps.
    device      : torch.device

    Returns
    -------
    float  Mean query loss over the batch.
    """
    criterion   = nn.MSELoss()
    meta_losses = []

    for task in tasks:
        support_X = torch.FloatTensor(task['support_X']).to(device)
        support_y = torch.FloatTensor(task['support_y']).unsqueeze(1).to(device)
        query_X   = torch.FloatTensor(task['query_X']).to(device)
        query_y   = torch.FloatTensor(task['query_y']).unsqueeze(1).to(device)

        inner_opt = optim.SGD(model.parameters(), lr=inner_lr)

        with higher.innerloop_ctx(
            model, inner_opt, copy_initial_weights=False
        ) as (fmodel, diffopt):

            # Inner loop: adapt on support set
            for _ in range(inner_steps):
                support_pred = fmodel(support_X)
                inner_loss   = criterion(support_pred, support_y)
                diffopt.step(inner_loss)

            # Outer loss: evaluate adapted model on query set
            query_pred = fmodel(query_X)
            meta_losses.append(criterion(query_pred, query_y))

    meta_opt.zero_grad()
    total_loss = torch.stack(meta_losses).mean()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    meta_opt.step()

    return total_loss.item()


# ─────────────────────────────────────────────────────────────────────────────
# Support-index selection  (identical strategy to evaluate_maml.py)
# ─────────────────────────────────────────────────────────────────────────────

def select_support_indices(n_windows, K):
    """
    Sample K evenly-spaced indices from the middle 60% of the engine timeline
    [0.2*n, 0.8*n). This mirrors what evaluate_maml.py does at test time so
    val RMSE is a faithful proxy of test RMSE.

    Parameters
    ----------
    n_windows : int   Total number of sequence windows for this engine.
    K         : int   Number of support samples to select.

    Returns
    -------
    np.ndarray of int, shape (K,)
    """
    start = int(n_windows * 0.2)
    end   = int(n_windows * 0.8)

    if (end - start) < K:
        # Fallback: spread across the full timeline
        return np.linspace(0, n_windows - 1, K, dtype=int)

    return np.linspace(start, end - 1, K, dtype=int)


# ─────────────────────────────────────────────────────────────────────────────
# Validation evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_on_val(model, val_tasks, inner_lr, inner_steps, device, max_rul, K=5):
    """
    Estimate RMSE on validation tasks using the same adapt-then-predict
    protocol as evaluate_maml.py.

    Support is selected with select_support_indices (middle 60%).
    Query is the second half of the engine's windows, excluding support.
    inner_steps should match the test-time value (15).

    Meta-weights are NEVER modified — each engine gets a temporary clone.

    Parameters
    ----------
    model       : LSTMBaseline   Current meta-initialisation (read-only).
    val_tasks   : list           Validation task dicts.
    inner_lr    : float          Inner adaptation LR.
    inner_steps : int            Inner steps (should equal test-time value).
    device      : torch.device
    max_rul     : float          Denormalisation ceiling (130).
    K           : int            Support size.

    Returns
    -------
    float  Mean RMSE in cycles across evaluated tasks.
    """
    criterion = nn.MSELoss()
    rmses     = []

    for task in val_tasks[:30]:   # cap at 30 for speed
        X_engine = task['query_X']
        y_engine = task['query_y']

        n = len(X_engine)
        if n < K + 5:
            continue

        # Select support and query indices
        support_pos = select_support_indices(n, K)
        query_start = n // 2
        query_pos   = np.arange(query_start, n)
        query_pos   = query_pos[~np.isin(query_pos, support_pos)]

        if len(query_pos) < 5:
            continue

        support_X = torch.FloatTensor(X_engine[support_pos]).to(device)
        support_y = torch.FloatTensor(y_engine[support_pos]).unsqueeze(1).to(device)
        query_X   = torch.FloatTensor(X_engine[query_pos]).to(device)
        query_y   = y_engine[query_pos]   # stays numpy for RMSE calc

        # Clone meta-model so original weights are untouched
        adapted = model.clone().to(device)
        opt     = optim.SGD(adapted.parameters(), lr=inner_lr)

        adapted.train()
        for _ in range(inner_steps):
            opt.zero_grad()
            loss = criterion(adapted(support_X), support_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapted.parameters(), max_norm=5.0)
            opt.step()

        adapted.eval()
        with torch.no_grad():
            # Clip to [0,1]: ReLU can exceed 1.0 early in training
            preds = np.clip(
                adapted(query_X).cpu().numpy().flatten(), 0.0, 1.0
            ) * max_rul
            y_act = query_y * max_rul
            rmses.append(np.sqrt(np.mean((preds - y_act) ** 2)))

    return float(np.mean(rmses)) if rmses else float('inf')


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train_maml_lstm():
    """
    Meta-train LSTMBaseline with MAML.

    Saves two checkpoints:
      results/saved_models/maml_lstm_best.pth   ← lowest val RMSE
      results/saved_models/maml_lstm_final.pth  ← weights at end of training

    Returns
    -------
    model : LSTMBaseline  Final meta-trained model (last epoch weights).
    """
    print("=" * 60)
    print("MAML META-TRAINING  —  LSTMBaseline backbone")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # ── Hyperparameters ───────────────────────────────────────────────────
    n_support        = 5
    inner_lr         = 0.05
    meta_lr          = 0.001
    inner_steps      = 10    # steps used during meta-training inner loop
    val_inner_steps  = 15    # steps at val/test time — matches evaluate_maml.py
    meta_batch_size  = 24
    meta_epochs      = 600
    tasks_per_engine = 5
    patience         = 60    # early stopping: epochs without val improvement

    print(f"\nHyperparameters:")
    print(f"  K-shot           : {n_support}")
    print(f"  Inner LR (α)     : {inner_lr}")
    print(f"  Meta  LR (β)     : {meta_lr}")
    print(f"  Inner steps      : {inner_steps} (train) / {val_inner_steps} (val/test)")
    print(f"  Meta batch size  : {meta_batch_size}")
    print(f"  Meta epochs      : {meta_epochs}")
    print(f"  Tasks per engine : {tasks_per_engine}")
    print(f"  Early-stop pat.  : {patience} epochs")

    # ── Load data ─────────────────────────────────────────────────────────
    print("\nLoading data...")
    data_dict  = load_preprocessed_data('data/processed/FD001_preprocessed.npz')
    max_rul    = float(data_dict['max_rul'])
    n_features = data_dict['X_train'].shape[2]

    print(f"  max_rul          : {max_rul}")
    print(f"  Input features   : {n_features}")
    print(f"  X_train shape    : {data_dict['X_train'].shape}")
    print(f"  X_val   shape    : {data_dict['X_val'].shape}")
    print(f"  X_test  shape    : {data_dict['X_test'].shape}")

    # ── Build task pool ───────────────────────────────────────────────────
    print("\nCreating augmented task datasets...")
    all_tasks   = create_task_dataset(
        data_dict, n_support=n_support, tasks_per_engine=tasks_per_engine
    )
    split       = int(0.8 * len(all_tasks))
    train_tasks = all_tasks[:split]
    val_tasks   = all_tasks[split:]

    print(f"  Meta-train tasks : {len(train_tasks)}")
    print(f"  Meta-val   tasks : {len(val_tasks)}")

    # ── Model, optimiser, scheduler ───────────────────────────────────────
    print("\nInitialising LSTMBaseline for MAML...")
    model     = LSTMBaseline(input_size=n_features).to(device)
    meta_opt  = optim.Adam(model.parameters(), lr=meta_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        meta_opt, T_max=meta_epochs, eta_min=1e-5
    )

    print(f"  Total parameters : {model.count_parameters():,}")

    # ── Meta-training loop ────────────────────────────────────────────────
    print("\nStarting meta-training...")
    print("-" * 60)

    best_val_rmse    = float('inf')
    no_improve_count = 0
    os.makedirs('results/saved_models', exist_ok=True)

    for epoch in range(1, meta_epochs + 1):

        # Refresh task pool every 50 epochs for fresh support sampling
        if epoch % 50 == 1 and epoch > 1:
            all_tasks   = create_task_dataset(
                data_dict, n_support=n_support, tasks_per_engine=tasks_per_engine
            )
            split       = int(0.8 * len(all_tasks))
            train_tasks = all_tasks[:split]
            val_tasks   = all_tasks[split:]

        # Sample meta-batch
        idx        = np.random.choice(
            len(train_tasks),
            size=min(meta_batch_size, len(train_tasks)),
            replace=False,
        )
        meta_batch = [train_tasks[i] for i in idx]

        # Outer loop update
        meta_loss = maml_outer_loop(
            model, meta_batch, meta_opt, inner_lr, inner_steps, device
        )
        scheduler.step()

        # ── Validation (every 10 epochs) ──────────────────────────────────
        if epoch % 10 == 0:
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
                torch.save(
                    {
                        'model_state_dict': model.state_dict(),
                        'epoch':            epoch,
                        'val_rmse':         val_rmse,
                        'hyperparameters': {
                            'backbone':    'LSTMBaseline',
                            'n_support':   n_support,
                            'inner_lr':    inner_lr,
                            'meta_lr':     meta_lr,
                            'inner_steps': val_inner_steps,
                            'n_features':  n_features,
                        },
                    },
                    'results/saved_models/maml_lstm_best.pth',
                )
            else:
                no_improve_count += 10   # counted in epochs

            # Early stopping check
            if no_improve_count >= patience:
                print(f"\n⚠  Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

        elif epoch % 5 == 0:
            print(f"Epoch {epoch:3d} | Meta-Loss: {meta_loss:.4f}")

    # ── End of training ───────────────────────────────────────────────────
    print("-" * 60)
    print(f"\n✓ Best Val RMSE : {best_val_rmse:.2f} cycles")

    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'hyperparameters': {
                'backbone':    'LSTMBaseline',
                'n_support':   n_support,
                'inner_lr':    inner_lr,
                'meta_lr':     meta_lr,
                'inner_steps': val_inner_steps,
                'n_features':  n_features,
            },
        },
        'results/saved_models/maml_lstm_final.pth',
    )

    print("✓ Final model → results/saved_models/maml_lstm_final.pth")
    print("✓ Best  model → results/saved_models/maml_lstm_best.pth")
    print("=" * 60)

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helper  (mirrors evaluate_maml.py for the LSTM backbone)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_maml_lstm(
    checkpoint_path='results/saved_models/maml_lstm_best.pth',
    data_file='data/processed/FD001_preprocessed.npz',
    inner_steps=15,
    K=5,
):
    """
    Load a saved MAML-LSTM checkpoint and evaluate on the held-out test split.

    Adapt-then-predict protocol:
      For each test engine:
        1. Select K support windows from the middle 60% of its timeline.
        2. Fine-tune a clone of the meta-model for `inner_steps` SGD steps.
        3. Predict RUL on remaining (query) windows.
        4. Compute RMSE in cycles.

    Parameters
    ----------
    checkpoint_path : str    Path to saved .pth file.
    data_file       : str    Path to preprocessed .npz file.
    inner_steps     : int    Adaptation steps (default 15, matches training).
    K               : int    Support size (default 5).

    Returns
    -------
    results : dict
        'mean_rmse'   : float   Mean RMSE across test engines (cycles)
        'median_rmse' : float
        'std_rmse'    : float
        'per_engine'  : list of (engine_id, rmse) pairs
    """
    print("=" * 60)
    print("MAML-LSTM EVALUATION")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice     : {device}")
    print(f"Checkpoint : {checkpoint_path}")

    # ── Load checkpoint ───────────────────────────────────────────────────
    ckpt       = torch.load(checkpoint_path, map_location=device)
    hparams    = ckpt['hyperparameters']
    n_features = hparams.get('n_features', 102)
    inner_lr   = hparams['inner_lr']

    print(f"Checkpoint info:")
    print(f"  Backbone    : {hparams.get('backbone', 'LSTMBaseline')}")
    print(f"  n_features  : {n_features}")
    print(f"  inner_lr    : {inner_lr}")
    print(f"  inner_steps : {inner_steps}  (override; trained with {hparams['inner_steps']})")

    model = LSTMBaseline(input_size=n_features).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # ── Load data ─────────────────────────────────────────────────────────
    data_dict = load_preprocessed_data(data_file)
    max_rul   = float(data_dict['max_rul'])

    X_test     = data_dict['X_test']
    y_test     = data_dict['y_test']
    engine_ids = data_dict['test_engine_ids']

    print(f"\nTest engines : {len(np.unique(engine_ids))}")
    print(f"Test samples : {len(X_test)}")

    # ── Per-engine adapt + predict ─────────────────────────────────────────
    criterion    = nn.MSELoss()
    per_engine   = []
    all_preds    = []
    all_actuals  = []

    for engine_id in np.unique(engine_ids):
        mask    = engine_ids == engine_id
        X_eng   = X_test[mask]
        y_eng   = y_test[mask]
        n       = len(X_eng)

        if n < K + 5:
            print(f"  Engine {engine_id:3d}: skipped (only {n} windows)")
            continue

        # Support: middle 60% of the timeline, K evenly spaced
        support_pos = select_support_indices(n, K)

        # Query: second half, excluding support positions
        query_start = n // 2
        query_pos   = np.arange(query_start, n)
        query_pos   = query_pos[~np.isin(query_pos, support_pos)]

        if len(query_pos) < 2:
            continue

        support_X = torch.FloatTensor(X_eng[support_pos]).to(device)
        support_y = torch.FloatTensor(y_eng[support_pos]).unsqueeze(1).to(device)
        query_X   = torch.FloatTensor(X_eng[query_pos]).to(device)
        query_y   = y_eng[query_pos]

        # Adapt on support — clone so meta-weights are never mutated
        adapted = model.clone().to(device)
        opt     = optim.SGD(adapted.parameters(), lr=inner_lr)

        adapted.train()
        for _ in range(inner_steps):
            opt.zero_grad()
            loss = criterion(adapted(support_X), support_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapted.parameters(), max_norm=5.0)
            opt.step()

        # Predict on query
        adapted.eval()
        with torch.no_grad():
            preds  = np.clip(
                adapted(query_X).cpu().numpy().flatten(), 0.0, 1.0
            ) * max_rul
            actual = query_y * max_rul

        rmse = float(np.sqrt(np.mean((preds - actual) ** 2)))
        per_engine.append((engine_id, rmse))
        all_preds.extend(preds.tolist())
        all_actuals.extend(actual.tolist())

    # ── Aggregate metrics ─────────────────────────────────────────────────
    rmse_values = [r for _, r in per_engine]
    mean_rmse   = float(np.mean(rmse_values))
    median_rmse = float(np.median(rmse_values))
    std_rmse    = float(np.std(rmse_values))

    # Overall RMSE across all query predictions
    all_preds   = np.array(all_preds)
    all_actuals = np.array(all_actuals)
    overall_rmse = float(np.sqrt(np.mean((all_preds - all_actuals) ** 2)))

    print("\n" + "-" * 60)
    print("RESULTS")
    print("-" * 60)
    print(f"  Engines evaluated : {len(per_engine)}")
    print(f"  Mean RMSE         : {mean_rmse:.2f} cycles")
    print(f"  Median RMSE       : {median_rmse:.2f} cycles")
    print(f"  Std  RMSE         : {std_rmse:.2f} cycles")
    print(f"  Overall RMSE      : {overall_rmse:.2f} cycles")
    print("-" * 60)

    # Per-engine breakdown
    print("\nPer-engine RMSE (sorted):")
    for eid, rmse in sorted(per_engine, key=lambda x: x[1]):
        bar = "█" * int(rmse / 5)
        print(f"  Engine {eid:3d} : {rmse:6.2f}  {bar}")

    print("=" * 60)

    return {
        'mean_rmse':    mean_rmse,
        'median_rmse':  median_rmse,
        'std_rmse':     std_rmse,
        'overall_rmse': overall_rmse,
        'per_engine':   per_engine,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='MAML on LSTMBaseline')
    parser.add_argument('--mode', choices=['train', 'eval', 'both'],
                        default='both',
                        help='train only, eval only, or train then eval')
    parser.add_argument('--checkpoint',
                        default='results/saved_models/maml_lstm_best.pth',
                        help='Checkpoint path for eval mode')
    parser.add_argument('--data',
                        default='data/processed/FD001_preprocessed.npz',
                        help='Path to preprocessed .npz file')
    args = parser.parse_args()

    if args.mode in ('train', 'both'):
        train_maml_lstm()

    if args.mode in ('eval', 'both'):
        evaluate_maml_lstm(
            checkpoint_path=args.checkpoint,
            data_file=args.data,
        )