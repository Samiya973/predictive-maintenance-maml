"""
Training Script — Random Forest Baseline
Mirrors src/training/train_baseline.py

Place at: src/training/train_rf_baseline.py
Run with: python src/training/train_rf_baseline.py
"""

import sys, os, time, pickle
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.rf_baseline  import RFBaseline
from src.data.data_loader    import load_preprocessed_data

os.makedirs("results/saved_models", exist_ok=True)


def train_rf():
    print("=" * 60)
    print("TRAINING RANDOM FOREST BASELINE")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────────────
    print("\nLoading preprocessed data...")
    data   = load_preprocessed_data()
    X_train, y_train = data['X_train'], data['y_train']
    X_val,   y_val   = data['X_val'],   data['y_val']
    X_test,  y_test  = data['X_test'],  data['y_test']
    print(f"  Training   : {len(X_train):,} samples")
    print(f"  Validation : {len(X_val):,} samples")
    print(f"  Test       : {len(X_test):,} samples")

    # ── Init model ───────────────────────────────────────────────────
    print("\nInitializing model...")
    rf = RFBaseline(
        n_estimators      = 200,
        max_depth         = 20,
        min_samples_split = 5,
        min_samples_leaf  = 2,
        max_features      = 0.5,
        random_state      = 42,
    )
    rf.summary()

    # ── Train ────────────────────────────────────────────────────────
    print("\nFitting Random Forest...")
    t0 = time.time()
    rf.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"  Training time: {elapsed:.1f}s")

    # ── Quick val check ──────────────────────────────────────────────
    val_preds = rf.predict(X_val)
    val_rmse  = float(np.sqrt(np.mean((val_preds - y_val) ** 2)))
    print(f"\n  Validation RMSE: {val_rmse:.4f} cycles")

    # ── Save model ───────────────────────────────────────────────────
    model_path = "results/saved_models/rf_baseline_best.pkl"
    rf.save(model_path)

    # Save metadata (mirrors LSTM checkpoint dict)
    meta = {
        "val_rmse"            : val_rmse,
        "train_samples"       : len(X_train),
        "params"              : rf.params,
        "feature_importances" : rf.feature_importances_,
        "training_time_s"     : elapsed,
    }
    meta_path = "results/saved_models/rf_baseline_meta.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    print(f"  Metadata saved -> {meta_path}")

    # ── Summary ──────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Training complete!")
    print(f"  Best validation RMSE : {val_rmse:.4f} cycles")
    print(f"  Model saved to       : {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    train_rf()