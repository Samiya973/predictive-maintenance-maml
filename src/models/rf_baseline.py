"""
Random Forest Baseline Model
Mirrors src/models/baselines.py

Place at: src/models/rf_baseline.py
Run with: python src/models/rf_baseline.py  (sanity check)
"""

import sys, os, pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ══════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION  (N, 30, 102) -> (N, 408)
# ══════════════════════════════════════════════════════════════════════

def extract_features(X: np.ndarray) -> np.ndarray:
    """
    Convert 3-D sequences (N, 30, 102) to flat feature matrix (N, 408).
    For each of the 102 features: mean, std, min, max over 30 timesteps.
    This lets RF capture temporal information without recurrence.
    """
    return np.concatenate([
        X.mean(axis=1),
        X.std(axis=1),
        X.min(axis=1),
        X.max(axis=1),
    ], axis=1)


# ══════════════════════════════════════════════════════════════════════
#  MODEL
# ══════════════════════════════════════════════════════════════════════

class RFBaseline:
    def __init__(
        self,
        n_estimators      = 200,
        max_depth         = 20,
        min_samples_split = 5,
        min_samples_leaf  = 2,
        max_features      = 0.5,
        random_state      = 42,
    ):
        self.params = dict(
            n_estimators      = n_estimators,
            max_depth         = max_depth,
            min_samples_split = min_samples_split,
            min_samples_leaf  = min_samples_leaf,
            max_features      = max_features,
            n_jobs            = -1,
            random_state      = random_state,
        )
        self.model                = RandomForestRegressor(**self.params)
        self.feature_importances_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """X shape: (N, 30, 102) — feature extraction done internally."""
        print("  Extracting temporal features ...", end=" ", flush=True)
        Xf = extract_features(X)
        print(f"done  ->  {Xf.shape}")
        self.model.fit(Xf, y)
        self.feature_importances_ = self.model.feature_importances_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X shape: (N, 30, 102)  ->  predictions (N,) clipped to [0, 130]."""
        return np.clip(self.model.predict(extract_features(X)), 0, 130)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"  Model saved -> {path}")

    @staticmethod
    def load(path: str) -> "RFBaseline":
        with open(path, "rb") as f:
            return pickle.load(f)

    def summary(self):
        p      = self.params
        fitted = hasattr(self.model, "estimators_")
        nodes  = (
            sum(e.tree_.node_count for e in self.model.estimators_)
            if fitted else "N/A (not fitted yet)"
        )
        print("=" * 60)
        print("RANDOM FOREST BASELINE MODEL")
        print("=" * 60)
        print("Architecture: RFBaseline(")
        print(f"  n_estimators     = {p['n_estimators']}")
        print(f"  max_depth        = {p['max_depth']}")
        print(f"  max_features     = {p['max_features']}")
        print(f"  min_samples_leaf = {p['min_samples_leaf']}")
        print(f"  n_jobs           = {p['n_jobs']}")
        print(")")
        print(f"Feature extraction : (N, 30, 102) -> (N, 408)")
        print(f"  mean + std + min + max over 30 timesteps")
        print(f"Total tree nodes   : {nodes}")
        print("=" * 60)


# ══════════════════════════════════════════════════════════════════════
#  SANITY CHECK  (mirrors baselines.py test forward pass)
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("RANDOM FOREST BASELINE MODEL")
    print("=" * 60)

    rf = RFBaseline()
    rf.summary()

    X_dummy = np.random.rand(64, 30, 102).astype(np.float32)
    y_dummy = np.random.uniform(0, 130, 64).astype(np.float32)

    print("\nTest forward pass:")
    print(f"  Input shape  : {X_dummy.shape}")
    rf.fit(X_dummy, y_dummy)
    preds = rf.predict(X_dummy[:8])
    print(f"  Output shape : {preds.shape}")
    print(f"  Output range : [{preds.min():.2f}, {preds.max():.2f}]")
    print("=" * 60)