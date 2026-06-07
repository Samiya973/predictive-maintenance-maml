"""
src/scripts/sanity_test_inference.py
Run the inference wrapper on one FD001 engine and print a cycle-by-cycle table.
"""
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.inference.predictor import PredictiveMaintenancePredictor
from src.data.data_loader import load_preprocessed_data

CNN_CKPT   = "results/saved_models/early_pred_best.pth"
MAML_CKPT  = "results/saved_models/maml_meta_model_best.pth"
VAE_CKPT   = "results/saved_models/vae_early_warning.pth"
STATS_PATH = "results/saved_models/baseline_stats.npz"
DATASET    = "data/processed/FD001_preprocessed.npz"

TARGET_ENGINE = 3   # change to any engine id
K_SHOT        = 5   # support windows for MAML adaptation

predictor = PredictiveMaintenancePredictor(CNN_CKPT, MAML_CKPT, VAE_CKPT, STATS_PATH)
data      = load_preprocessed_data(DATASET)

# ── Print loaded model architectures ─────────────────────────────────────────
print("\n" + "=" * 60)
print("CNN-LSTM MODEL ARCHITECTURE")
print("=" * 60)
print(predictor.cnn_model)

print("\n" + "=" * 60)
print("MAML MODEL ARCHITECTURE")
print("=" * 60)
print(predictor._maml_meta_model)

print("\n" + "=" * 60)
print("VAE MODEL ARCHITECTURE")
print("=" * 60)
print(predictor.ews.vae)

# ── Pull windows belonging to this engine ─────────────────────────────────────
test_ids = np.array(data["test_engine_ids"]).astype(int)
mask     = test_ids == TARGET_ENGINE

# ── Debug: inspect what IDs actually exist ───────────────────────────────────
print(f"\ntest_engine_ids dtype : {test_ids.dtype}")
print(f"Unique engine IDs     : {np.unique(test_ids)}")
print(f"TARGET_ENGINE         : {TARGET_ENGINE}  (type: {type(TARGET_ENGINE)})")
print(f"Mask hits             : {mask.sum()}")

X_eng    = data["X_test"][mask]                         # (N, 30, 102)
y_eng    = data["y_test"][mask]                         # normalised, shape (N,)
y_eng_cycles = y_eng * data["max_rul"]                  # de-normalised for display

print(f"\nEngine {TARGET_ENGINE}  ({len(X_eng)} windows)")

# ── MAML adaptation — must happen before any predict() calls ─────────────────
# Uses the same engine windows + normalised RUL labels (0-1) as support
predictor.adapt_to_engine(X_eng, y_eng, K=K_SHOT)

"""
src/scripts/sanity_test_inference.py
"""
import numpy as np
import torch
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.inference.predictor import PredictiveMaintenancePredictor, _select_support_indices
from src.data.data_loader import load_preprocessed_data

CNN_CKPT   = "results/saved_models/early_pred_best.pth"
MAML_CKPT  = "results/saved_models/maml_meta_model_best.pth"
VAE_CKPT   = "results/saved_models/vae_early_warning.pth"
STATS_PATH = "results/saved_models/baseline_stats.npz"
DATASET    = "data/processed/FD001_preprocessed.npz"

TARGET_ENGINE = 3
K_SHOT        = 5

predictor = PredictiveMaintenancePredictor(CNN_CKPT, MAML_CKPT, VAE_CKPT, STATS_PATH)
data      = load_preprocessed_data(DATASET)

test_ids     = np.array(data["test_engine_ids"]).astype(int)
mask         = test_ids == TARGET_ENGINE
X_eng        = data["X_test"][mask]
y_eng        = data["y_test"][mask]
y_eng_cycles = y_eng * data["max_rul"]

# ── MAML DIAGNOSTIC ───────────────────────────────────────────────────────────
print("\n--- MAML DIAGNOSTIC ---")

# 1. Raw meta-model output (no adaptation)
predictor._maml_meta_model.eval()
with torch.no_grad():
    sample = torch.FloatTensor(X_eng[:10]).to(predictor.device)
    raw    = predictor._maml_meta_model(sample).cpu().numpy().flatten()
print(f"Raw meta output (first 10): {np.round(raw, 4)}")
print(f"Raw output range           : [{raw.min():.4f}, {raw.max():.4f}]")

# 2. After adaptation
predictor.adapt_to_engine(X_eng, y_eng, K=K_SHOT)
with torch.no_grad():
    adapted_out = predictor._maml_adapted(sample).cpu().numpy().flatten()
print(f"Adapted output (first 10)  : {np.round(adapted_out, 4)}")
print(f"Adapted output range       : [{adapted_out.min():.4f}, {adapted_out.max():.4f}]")

# 3. Ground truth
print(f"y_eng normalised (first 10): {np.round(y_eng[:10], 4)}")
print(f"y_eng_cycles     (first 10): {np.round(y_eng_cycles[:10], 1)}")

# 4. Support indices
support_pos = _select_support_indices(len(X_eng), K_SHOT)
print(f"\nSupport indices : {support_pos}")
print(f"Support RUL_true: {np.round(y_eng_cycles[support_pos], 1)}")
print(f"Support y_norm  : {np.round(y_eng[support_pos], 4)}")
print("--- END DIAGNOSTIC ---\n")

# ── Prediction loop ───────────────────────────────────────────────────────────
# adapt_to_engine already called above — no need to call again
print(f"\n{'Cyc':>4}  {'RUL_true':>8}  {'RUL_pred':>8}  "
      f"{'FaultP':>7}  {'CUSUM':>7}  {'State':<15}  Action")
print("-" * 85)

for i, (window, rul_true) in enumerate(zip(X_eng, y_eng_cycles)):
    out = predictor.predict(window, cycle=i + 1)
    print(f"{i+1:4d}  {rul_true:8.1f}  {out['rul_cycles']:8.1f}  "
          f"{out['fault_prob']:7.4f}  {out['vae_cusum']:7.3f}  "
          f"{out['state']:<15}  {out['action']}")