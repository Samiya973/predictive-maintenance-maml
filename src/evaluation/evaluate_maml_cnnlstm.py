import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.models.maml_model import CNNLSTMBase
from src.data.data_loader import load_preprocessed_data


# ============================================================
# Utility metrics
# ============================================================

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def r2_score_np(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)

def score_phm08(y_true, y_pred):
    d = y_pred - y_true
    score = 0.0
    for x in d:
        if x < 0:
            score += np.exp(-x / 13.0) - 1.0
        else:
            score += np.exp(x / 10.0) - 1.0
    return score


# ============================================================
# Time-series-safe support/query split per engine
# ============================================================

def split_engine_support_query(indices, n_support=5, min_query=10):
    """
    Time-series safe:
    - support from middle region
    - query strictly after support
    """
    n = len(indices)

    if n < n_support + min_query:
        return None, None

    support_start = int(n * 0.35)
    support_end   = int(n * 0.55)

    if support_end - support_start < n_support:
        support_start = max(0, n // 3)
        support_end   = min(n - min_query, support_start + n_support)

    if support_end - support_start < n_support:
        return None, None

    # evenly spaced support points inside support zone
    support_local = np.linspace(support_start, support_end - 1, n_support, dtype=int)
    support_idx = indices[support_local]

    # query strictly after last support position
    query_start = max(support_local[-1] + 1, int(n * 0.55))
    query_local = np.arange(query_start, n)

    if len(query_local) < min_query:
        return None, None

    query_idx = indices[query_local]
    return support_idx, query_idx


# ============================================================
# Adapt one engine and evaluate
# ============================================================

def adapt_and_predict_engine(model, X_engine, y_engine, support_idx_local, query_idx_local,
                             inner_lr, inner_steps, device):
    criterion = nn.MSELoss()

    support_X = torch.FloatTensor(X_engine[support_idx_local]).to(device)
    support_y = torch.FloatTensor(y_engine[support_idx_local]).unsqueeze(1).to(device)

    query_X = torch.FloatTensor(X_engine[query_idx_local]).to(device)

    # before adaptation
    model.eval()
    with torch.no_grad():
        pred_before = model(query_X).cpu().numpy().flatten()

    # clone and adapt
    adapted = model.clone().to(device)
    opt = optim.SGD(adapted.parameters(), lr=inner_lr)

    adapted.train()
    for _ in range(inner_steps):
        opt.zero_grad()
        pred_support = adapted(support_X)
        loss = criterion(pred_support, support_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapted.parameters(), max_norm=5.0)
        opt.step()

    adapted.eval()
    with torch.no_grad():
        pred_after = adapted(query_X).cpu().numpy().flatten()

    return pred_before, pred_after


# ============================================================
# Full evaluation on test split only
# ============================================================

def evaluate_maml_model(
    model_path="results/saved_models/maml_meta_model_best.pth",
    data_path="data/processed/FD001_preprocessed.npz",
    save_dir="results/figures/maml_eval",
    n_support=5,
    inner_lr=0.005,
    inner_steps=5,
    max_engines_to_plot=6
):
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("EVALUATING MAML CNN-LSTM (TIME-SERIES SAFE)")
    print("=" * 70)
    print(f"Device: {device}")

    # -----------------------------
    # Load data
    # -----------------------------
    data_dict = load_preprocessed_data(data_path)
    X_test = data_dict["X_test"]
    y_test = data_dict["y_test"]
    test_engine_ids = data_dict["test_engine_ids"]
    max_rul = float(data_dict["max_rul"])

    print(f"X_test shape      : {X_test.shape}")
    print(f"y_test shape      : {y_test.shape}")
    print(f"Test engines      : {len(np.unique(test_engine_ids))}")
    print(f"max_rul           : {max_rul}")

    # -----------------------------
    # Load model
    # -----------------------------
    model = CNNLSTMBase(input_size=X_test.shape[2]).to(device)

    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint  : {model_path}")
    if "hyperparameters" in ckpt:
        print(f"Checkpoint hparams : {ckpt['hyperparameters']}")

    # -----------------------------
    # Evaluate engine by engine
    # -----------------------------
    all_true = []
    all_pred_before = []
    all_pred_after = []
    all_engine_marks = []

    engine_curves = []

    unique_engines = np.unique(test_engine_ids)

    for engine_id in unique_engines:
        idx = np.where(test_engine_ids == engine_id)[0]
        idx = np.sort(idx)

        support_idx, query_idx = split_engine_support_query(idx, n_support=n_support, min_query=10)
        if support_idx is None:
            continue

        # local arrays for this engine only
        X_engine = X_test
        y_engine = y_test

        pred_before, pred_after = adapt_and_predict_engine(
            model=model,
            X_engine=X_engine,
            y_engine=y_engine,
            support_idx_local=support_idx,
            query_idx_local=query_idx,
            inner_lr=inner_lr,
            inner_steps=inner_steps,
            device=device
        )

        true_vals = y_test[query_idx]

        all_true.extend(true_vals.tolist())
        all_pred_before.extend(pred_before.tolist())
        all_pred_after.extend(pred_after.tolist())
        all_engine_marks.extend([engine_id] * len(query_idx))

        engine_curves.append({
            "engine_id": engine_id,
            "support_idx": support_idx,
            "query_idx": query_idx,
            "true": true_vals * max_rul,
            "pred_before": np.clip(pred_before, 0.0, 1.0) * max_rul,
            "pred_after": np.clip(pred_after, 0.0, 1.0) * max_rul
        })

    all_true = np.array(all_true) * max_rul
    all_pred_before = np.clip(np.array(all_pred_before), 0.0, 1.0) * max_rul
    all_pred_after = np.clip(np.array(all_pred_after), 0.0, 1.0) * max_rul
    all_engine_marks = np.array(all_engine_marks)

    # -----------------------------
    # Metrics
    # -----------------------------
    metrics_before = {
        "RMSE": rmse(all_true, all_pred_before),
        "MAE": mae(all_true, all_pred_before),
        "R2": r2_score_np(all_true, all_pred_before),
        "PHM08": score_phm08(all_true, all_pred_before)
    }

    metrics_after = {
        "RMSE": rmse(all_true, all_pred_after),
        "MAE": mae(all_true, all_pred_after),
        "R2": r2_score_np(all_true, all_pred_after),
        "PHM08": score_phm08(all_true, all_pred_after)
    }

    print("\n" + "=" * 70)
    print("METRICS")
    print("=" * 70)
    print("Before adaptation:")
    for k, v in metrics_before.items():
        print(f"  {k:6s}: {v:.4f}")
    print("After adaptation:")
    for k, v in metrics_after.items():
        print(f"  {k:6s}: {v:.4f}")

    # save metrics
    metrics_file = os.path.join(save_dir, "metrics_maml.txt")
    with open(metrics_file, "w") as f:
        f.write("MAML CNN-LSTM Evaluation\n")
        f.write("=" * 50 + "\n\n")
        f.write("Before adaptation\n")
        for k, v in metrics_before.items():
            f.write(f"{k}: {v:.6f}\n")
        f.write("\nAfter adaptation\n")
        for k, v in metrics_after.items():
            f.write(f"{k}: {v:.6f}\n")

    # ============================================================
    # Plot 1: Sorted actual vs prediction like paper figure
    # ============================================================
    sort_idx = np.argsort(all_true)

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(all_true)), all_true[sort_idx], lw=2, label="Ground-Truth RUL")
    plt.scatter(np.arange(len(all_true)), all_pred_after[sort_idx], s=18, label="MAML CNN-LSTM")
    plt.scatter(np.arange(len(all_true)), all_pred_before[sort_idx], s=18, marker="x", label="Before Adaptation")
    plt.xlabel("Test Instances (sorted by actual RUL)")
    plt.ylabel("Remaining Useful Life (cycles)")
    plt.title("Sorted Actual vs Predicted RUL")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "01_sorted_actual_vs_pred.png"), dpi=300)
    plt.close()

    # ============================================================
    # Plot 2: Parity plot
    # ============================================================
    min_v = min(all_true.min(), all_pred_after.min(), all_pred_before.min())
    max_v = max(all_true.max(), all_pred_after.max(), all_pred_before.max())

    plt.figure(figsize=(7, 7))
    plt.scatter(all_true, all_pred_before, s=18, alpha=0.6, label="Before Adaptation")
    plt.scatter(all_true, all_pred_after, s=18, alpha=0.6, label="After Adaptation")
    plt.plot([min_v, max_v], [min_v, max_v], "--", lw=2, label="Ideal")
    plt.xlabel("Actual RUL")
    plt.ylabel("Predicted RUL")
    plt.title("Parity Plot")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "02_parity_plot.png"), dpi=300)
    plt.close()

    # ============================================================
    # Plot 3: Residual vs actual
    # ============================================================
    residual_before = all_pred_before - all_true
    residual_after = all_pred_after - all_true

    plt.figure(figsize=(9, 6))
    plt.scatter(all_true, residual_before, s=18, alpha=0.6, label="Before Adaptation")
    plt.scatter(all_true, residual_after, s=18, alpha=0.6, label="After Adaptation")
    plt.axhline(0, linestyle="--", linewidth=2)
    plt.xlabel("Actual RUL")
    plt.ylabel("Residual (Pred - Actual)")
    plt.title("Residual Plot")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "03_residual_vs_actual.png"), dpi=300)
    plt.close()

    # ============================================================
    # Plot 4: Residual histogram
    # ============================================================
    plt.figure(figsize=(9, 6))
    plt.hist(residual_before, bins=30, alpha=0.6, label="Before Adaptation")
    plt.hist(residual_after, bins=30, alpha=0.6, label="After Adaptation")
    plt.xlabel("Residual (Pred - Actual)")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "04_residual_histogram.png"), dpi=300)
    plt.close()

    # ============================================================
    # Plot 5: Absolute error vs actual RUL
    # ============================================================
    abs_err_before = np.abs(residual_before)
    abs_err_after = np.abs(residual_after)

    plt.figure(figsize=(9, 6))
    plt.scatter(all_true, abs_err_before, s=18, alpha=0.6, label="Before Adaptation")
    plt.scatter(all_true, abs_err_after, s=18, alpha=0.6, label="After Adaptation")
    plt.xlabel("Actual RUL")
    plt.ylabel("Absolute Error")
    plt.title("Absolute Error vs Actual RUL")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "05_abs_error_vs_actual.png"), dpi=300)
    plt.close()

    # ============================================================
    # Plot 6: Per-engine trajectories
    # ============================================================
    engines_to_plot = engine_curves[:max_engines_to_plot]

    for item in engines_to_plot:
        engine_id = item["engine_id"]
        true_vals = item["true"]
        pred_before = item["pred_before"]
        pred_after = item["pred_after"]

        x = np.arange(len(true_vals))

        plt.figure(figsize=(10, 5))
        plt.plot(x, true_vals, lw=2, label="Actual RUL")
        plt.plot(x, pred_before, "--", lw=2, label="Before Adaptation")
        plt.plot(x, pred_after, lw=2, label="After Adaptation")
        plt.xlabel("Query Windows (later timeline)")
        plt.ylabel("RUL (cycles)")
        plt.title(f"Engine {engine_id}: Query Trajectory")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"06_engine_{engine_id}_trajectory.png"), dpi=300)
        plt.close()

    # ============================================================
    # Plot 7: Engine-wise RMSE improvement
    # ============================================================
    engine_ids_unique = []
    engine_rmse_before = []
    engine_rmse_after = []

    for item in engine_curves:
        engine_id = item["engine_id"]
        y_t = item["true"]
        y_b = item["pred_before"]
        y_a = item["pred_after"]

        engine_ids_unique.append(engine_id)
        engine_rmse_before.append(rmse(y_t, y_b))
        engine_rmse_after.append(rmse(y_t, y_a))

    x = np.arange(len(engine_ids_unique))
    width = 0.38

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, engine_rmse_before, width, label="Before Adaptation")
    plt.bar(x + width/2, engine_rmse_after, width, label="After Adaptation")
    plt.xlabel("Engine ID")
    plt.ylabel("RMSE")
    plt.title("Engine-wise RMSE Comparison")
    plt.xticks(x, engine_ids_unique, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "07_enginewise_rmse.png"), dpi=300)
    plt.close()

    # ============================================================
    # Save predictions
    # ============================================================
    pred_file = os.path.join(save_dir, "predictions_maml.npz")
    np.savez_compressed(
        pred_file,
        y_true=all_true,
        y_pred_before=all_pred_before,
        y_pred_after=all_pred_after,
        engine_ids=all_engine_marks
    )

    print("\nSaved files:")
    print(f"  Metrics      : {metrics_file}")
    print(f"  Predictions  : {pred_file}")
    print(f"  Plots        : {save_dir}")

    print("\nDone.")
    return {
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "save_dir": save_dir
    }


if __name__ == "__main__":
    evaluate_maml_model(
        model_path="results/saved_models/maml_meta_model_best.pth",
        data_path="data/processed/FD001_preprocessed.npz",
        save_dir="results/figures/maml_eval",
        n_support=5,
        inner_lr=0.005,
        inner_steps=5,
        max_engines_to_plot=6
    )