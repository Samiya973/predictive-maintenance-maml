# """
# src/inference/predictor.py
# ──────────────────────────
# Unified inference wrapper: loads CNN-LSTM, MAML, and VAE+CUSUM checkpoints.
# Runs a window through all three branches and fuses into one output dict.

# MAML inference follows the same adapt-then-predict protocol as evaluate_maml.py:
#   1. adapt_to_engine()  — call once per new engine with K support windows
#   2. predict()          — call per cycle; uses the adapted MAML model for RUL
# """

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import copy

# from src.models.early_pred_model import EarlyPredCNNLSTM
# from src.models.early_warning_vae import TimeSeriesVAE, EarlyWarningSystem
# from src.models.maml_model import CNNLSTMBase

# USEFUL_SENSORS = [
#     "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_9",
#     "sensor_11", "sensor_12", "sensor_14", "sensor_17", "sensor_20", "sensor_21",
# ]

# # MAML inner-loop defaults — match evaluate_maml.py exactly
# _MAML_INNER_LR    = 0.05
# _MAML_INNER_STEPS = 15


# # ── Support index selection (identical to evaluate_maml.py) ──────────────────

# def _select_support_indices(n_windows: int, K: int) -> np.ndarray:
#     """
#     K evenly-spaced indices from the middle 60% of engine life.
#     K=1 special case: 75% point (early-degradation, better gradient signal).
#     """
#     if K == 1:
#         return np.array([int(n_windows * 0.75)], dtype=int)

#     start = int(n_windows * 0.2)
#     end   = int(n_windows * 0.8)

#     if (end - start) < K:
#         return np.linspace(0, n_windows - 1, K, dtype=int)

#     return np.linspace(start, end - 1, K, dtype=int)


# def _adapt_hyperparams_for_k(K: int) -> dict:
#     """
#     Scale inner-loop hyperparams by support size — matches evaluate_maml.py.
#     Low K → fewer steps + lower lr to avoid catastrophic overfitting.
#     """
#     if K == 1:
#         return dict(inner_lr=0.01, inner_steps=10)
#     if K == 2:
#         return dict(inner_lr=0.02, inner_steps=8)
#     return dict(inner_lr=_MAML_INNER_LR, inner_steps=_MAML_INNER_STEPS)


# # ── Alert fusion ──────────────────────────────────────────────────────────────

# def _fuse_alerts(cnn_fault: bool, rul_cycles: float,
#                  vae_alarm: bool, vae_z: float) -> dict:
#     """
#     Four named states, deterministic rules.

#     CRITICAL      : CNN says fault AND RUL ≤ 30
#     EARLY WARNING : CNN says fault OR VAE alarm fires
#     MONITOR       : RUL ≤ 50 but neither model alarming yet
#     HEALTHY       : everything clear
#     """
#     if cnn_fault and rul_cycles <= 30:
#         state  = "CRITICAL"
#         color  = "#EF4444"
#         action = "Immediate intervention — schedule unplanned maintenance"
#     elif cnn_fault or vae_alarm:
#         state  = "EARLY WARNING"
#         color  = "#F59E0B"
#         action = "Elevated risk detected — plan inspection within 5 cycles"
#     elif rul_cycles <= 50:
#         state  = "MONITOR"
#         color  = "#3B82F6"
#         action = "Degradation trending — increase monitoring frequency"
#     else:
#         state  = "HEALTHY"
#         color  = "#10B981"
#         action = "Normal operation"

#     return {
#         "state":       state,
#         "color":       color,
#         "action":      action,
#         "cnn_fault":   bool(cnn_fault),
#         "vae_alarm":   bool(vae_alarm),
#         "vae_z_score": float(vae_z),
#     }


# # ── Main predictor ────────────────────────────────────────────────────────────

# class PredictiveMaintenancePredictor:
#     """
#     Loads all three checkpoints once; exposes adapt_to_engine() + predict().

#     Parameters
#     ----------
#     cnn_ckpt_path   : path to early_pred_best.pth         (CNN-LSTM fault classifier)
#     maml_ckpt_path  : path to maml_meta_model_best.pth    (MAML RUL regression)
#     vae_ckpt_path   : path to vae_early_warning.pth       (VAE + CUSUM)
#     stats_path      : path to baseline_stats.npz
#     max_rul         : float, de-normalisation factor for RUL (default 130)
#     device          : 'cpu' or 'cuda'

#     Typical usage
#     -------------
#     predictor = PredictiveMaintenancePredictor(...)

#     # Once per engine — provide K support windows
#     predictor.adapt_to_engine(support_windows, support_ruls_norm, K=5)

#     # Per cycle
#     result = predictor.predict(window, cycle)
#     """

#     def __init__(self, cnn_ckpt_path: str, maml_ckpt_path: str,
#                  vae_ckpt_path: str, stats_path: str,
#                  max_rul: float = 130.0, device: str = "cpu"):

#         self.device  = torch.device(device)
#         self.max_rul = max_rul

#         # ── 1. CNN-LSTM fault classifier ─────────────────────────────────
#         cnn_ckpt = torch.load(cnn_ckpt_path, map_location=self.device)
#         cfg      = cnn_ckpt.get("config", {})

#         self.fault_rul_cycles     = float(cnn_ckpt.get("fault_rul_cycles", 30.0))
#         self.fault_threshold_norm = self.fault_rul_cycles / self.max_rul

#         self.cnn_model = EarlyPredCNNLSTM(
#             input_size=cfg.get("input_size", 102),
#             dropout=cfg.get("dropout", 0.2),
#         ).to(self.device)
#         self.cnn_model.load_state_dict(cnn_ckpt["model_state_dict"])
#         self.cnn_model.eval()

#         self.cnn_threshold = float(cnn_ckpt.get("best_threshold", 0.5))
#         print(f"[Predictor] CNN-LSTM loaded   threshold={self.cnn_threshold:.2f}  "
#               f"fault_rul={self.fault_rul_cycles}")

#         # ── 2. MAML meta-model (CNNLSTMBase) ─────────────────────────────
#         # Checkpoint keys: model_state_dict, epoch, val_rmse, hyperparameters
#         # config dict is empty → architecture is always CNNLSTMBase(input_size=102)
#         maml_ckpt  = torch.load(maml_ckpt_path, map_location=self.device)
#         self._maml_meta_model = CNNLSTMBase(input_size=102).to(self.device)
#         self._maml_meta_model.load_state_dict(maml_ckpt["model_state_dict"])
#         self._maml_meta_model.eval()

#         # Restore inner-loop hyperparams saved at training time (fallback to defaults)
#         saved_hp = maml_ckpt.get("hyperparameters", {})
#         self._maml_default_inner_lr    = float(saved_hp.get("inner_lr",    _MAML_INNER_LR))
#         self._maml_default_inner_steps = int(saved_hp.get("inner_steps", _MAML_INNER_STEPS))

#         # Adapted model starts as None — must call adapt_to_engine() before predict()
#         self._maml_adapted: CNNLSTMBase | None = None

#         val_rmse_str = f"  val_rmse={maml_ckpt['val_rmse']:.2f}" if "val_rmse" in maml_ckpt else ""
#         print(f"[Predictor] MAML loaded       inner_lr={self._maml_default_inner_lr}  "
#               f"inner_steps={self._maml_default_inner_steps}{val_rmse_str}")

#         # ── 3. VAE + CUSUM early-warning system ───────────────────────────
#         stats          = np.load(stats_path, allow_pickle=True)
#         baseline_mean  = float(stats["baseline_mean"])
#         baseline_std   = float(stats["baseline_std"])
#         threshold_h    = float(stats["threshold_h"])
#         drift_k        = float(stats["drift_k"])
#         decay          = float(stats["decay"])
#         warmup         = int(stats["warmup"])
#         self.sensor_indices = stats["sensor_indices"].tolist()

#         vae_ckpt = torch.load(vae_ckpt_path, map_location=self.device)
#         vcfg     = vae_ckpt.get("config", {})
#         vae      = TimeSeriesVAE(
#             input_size  = vcfg.get("input_size",   len(USEFUL_SENSORS)),
#             seq_len     = vcfg.get("seq_len",       30),
#             hidden_size = vcfg.get("hidden_size",   64),
#             latent_dim  = vcfg.get("latent_dim",    16),
#         )
#         vae.load_state_dict(vae_ckpt["model_state_dict"])

#         self.ews = EarlyWarningSystem(
#             vae=vae, baseline_mean=baseline_mean, baseline_std=baseline_std,
#             drift_k=drift_k, threshold_h=threshold_h,
#             warmup=warmup, decay=decay, device=device,
#         )
#         print(f"[Predictor] VAE+CUSUM loaded  h={threshold_h:.4f}  "
#               f"drift_k={drift_k}  warmup={warmup}")

#     # ── Engine-level adaptation ───────────────────────────────────────────────

#     def adapt_to_engine(self, engine_windows: np.ndarray,
#                         engine_ruls_norm: np.ndarray,
#                         K: int = 5) -> None:
#         """
#         Adapt the MAML meta-model to a new engine using K support windows.

#         Call this ONCE per engine before the first predict() call.
#         Also resets the VAE CUSUM state for the new engine.

#         Parameters
#         ----------
#         engine_windows   : np.ndarray, shape (N, seq_len, 102)
#                            All available windows for this engine so far.
#         engine_ruls_norm : np.ndarray, shape (N,)
#                            Normalised RUL labels (0–1) for each window.
#         K                : int, number of support examples (default 5).
#                            More = better accuracy, fewer = faster adaptation.
#         """
#         n = len(engine_windows)

#         if n < K:
#             raise ValueError(
#                 f"adapt_to_engine() needs at least K={K} windows, got {n}. "
#                 f"Collect more cycles before calling this method, or reduce K."
#             )

#         # Select support indices — same strategy as evaluate_maml.py
#         support_pos = _select_support_indices(n, K)
#         hp          = _adapt_hyperparams_for_k(K)

#         X_support = torch.FloatTensor(engine_windows[support_pos]).to(self.device)
#         y_support = torch.FloatTensor(engine_ruls_norm[support_pos]).unsqueeze(1).to(self.device)

#         # Deep-copy meta-model and fine-tune on support set
#         adapted = copy.deepcopy(self._maml_meta_model).to(self.device)
#         for param in adapted.parameters():
#             param.requires_grad = True

#         criterion = nn.MSELoss()
#         optimizer = optim.SGD(adapted.parameters(), lr=hp["inner_lr"])

#         adapted.train()
#         for _ in range(hp["inner_steps"]):
#             optimizer.zero_grad()
#             loss = criterion(adapted(X_support), y_support)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(adapted.parameters(), max_norm=5.0)
#             optimizer.step()

#         adapted.eval()
#         self._maml_adapted = adapted

#         # Reset CUSUM state for new engine
#         self.ews.reset()

#         print(f"[Predictor] MAML adapted to engine  K={K}  "
#               f"inner_lr={hp['inner_lr']}  inner_steps={hp['inner_steps']}")

#     # ── Per-cycle prediction ──────────────────────────────────────────────────

#     def reset_engine(self) -> None:
#         """
#         Clear adapted MAML model and VAE CUSUM state between engines.
#         You must call adapt_to_engine() again before the next predict().
#         """
#         self._maml_adapted = None
#         self.ews.reset()

#     def predict(self, window: np.ndarray, cycle: int) -> dict:
#         """
#         Run one cycle window through all three branches and return a fused dict.

#         Parameters
#         ----------
#         window  : np.ndarray, shape (seq_len, 102)
#                   Full-feature normalised window from your data loader.
#         cycle   : int, current cycle number (used by CUSUM tracker).

#         Returns
#         -------
#         dict with keys:
#             cycle, rul_cycles, fault_prob, cnn_fault,
#             vae_alarm, vae_z_score, vae_cusum,
#             state, color, action

#         Raises
#         ------
#         RuntimeError  if adapt_to_engine() has not been called yet.
#         """
#         if self._maml_adapted is None:
#             raise RuntimeError(
#                 "adapt_to_engine() must be called before predict(). "
#                 "Provide at least K support windows for this engine."
#             )

#         x_full = torch.FloatTensor(window).unsqueeze(0).to(self.device)  # (1, 30, 102)

#         # ── MAML branch — primary RUL source ─────────────────────────────
#         with torch.no_grad():
#             maml_out   = self._maml_adapted(x_full).squeeze()            # scalar
#             rul_norm   = float(torch.clamp(maml_out, 0.0, 1.0).cpu())    # sigmoid already in [0,1]
#         rul_cycles = rul_norm * self.max_rul

#         # ── CNN-LSTM branch — fault probability ──────────────────────────
#         with torch.no_grad():
#             logit      = self.cnn_model(x_full).squeeze()
#             fault_prob = float(torch.sigmoid(logit).cpu())
#         cnn_fault = fault_prob >= self.cnn_threshold

#         # ── VAE + CUSUM branch — anomaly detection ────────────────────────
#         x_vae   = window[:, self.sensor_indices]                          # (seq_len, 11)
#         x_vae_t = torch.FloatTensor(x_vae).unsqueeze(0)                  # (1, seq_len, 11)
#         vae_result = self.ews.monitor(x_vae_t, cycle)

#         # ── Alert fusion ──────────────────────────────────────────────────
#         fused = _fuse_alerts(
#             cnn_fault  = cnn_fault,
#             rul_cycles = rul_cycles,
#             vae_alarm  = vae_result["alarm"],
#             vae_z      = vae_result["z_score"],
#         )

#         return {
#             "cycle":       cycle,
#             "rul_cycles":  round(rul_cycles, 1),
#             "rul_norm":    round(rul_norm, 4),
#             "fault_prob":  round(fault_prob, 4),
#             "cnn_fault":   bool(cnn_fault),
#             "vae_alarm":   bool(vae_result["alarm"]),
#             "vae_z_score": round(vae_result["z_score"], 3),
#             "vae_cusum":   round(vae_result["cusum"], 3),
#             **fused,
#         }

"""
src/inference/predictor.py
──────────────────────────
Unified inference wrapper: loads CNN-LSTM, MAML, and VAE+CUSUM checkpoints.
Runs a window through all three branches and fuses into one output dict.

MAML inference follows the same adapt-then-predict protocol as evaluate_maml.py:
  1. adapt_to_engine()  — call once per new engine with K support windows
  2. predict()          — call per cycle; uses the adapted MAML model for RUL
"""

import numpy as np
import os,sys
import torch
import torch.nn as nn
import torch.optim as optim
import copy

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.models.early_pred_model import EarlyPredCNNLSTM
from src.models.early_warning_vae import TimeSeriesVAE, EarlyWarningSystem
from src.models.maml_model import CNNLSTMBase

USEFUL_SENSORS = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_9",
    "sensor_11", "sensor_12", "sensor_14", "sensor_17", "sensor_20", "sensor_21",
]

# MAML inner-loop defaults — match evaluate_maml.py exactly
_MAML_INNER_LR    = 0.05
_MAML_INNER_STEPS = 15


# ── Support index selection (identical to evaluate_maml.py) ──────────────────

def _select_support_indices(n_windows: int, K: int) -> np.ndarray:
    """
    K evenly-spaced indices from the middle 60% of engine life.
    K=1 special case: 75% point (early-degradation, better gradient signal).
    """
    if K == 1:
        return np.array([int(n_windows * 0.75)], dtype=int)

    start = int(n_windows * 0.2)
    end   = int(n_windows * 0.8)

    if (end - start) < K:
        return np.linspace(0, n_windows - 1, K, dtype=int)

    return np.linspace(start, end - 1, K, dtype=int)


def _adapt_hyperparams_for_k(K: int) -> dict:
    """
    Scale inner-loop hyperparams by support size — matches evaluate_maml.py.
    Low K → fewer steps + lower lr to avoid catastrophic overfitting.
    """
    if K == 1:
        return dict(inner_lr=0.01, inner_steps=10)
    if K == 2:
        return dict(inner_lr=0.02, inner_steps=8)
    return dict(inner_lr=_MAML_INNER_LR, inner_steps=_MAML_INNER_STEPS)


# ── Alert fusion ──────────────────────────────────────────────────────────────

def _fuse_alerts(cnn_fault: bool, rul_cycles: float,
                 vae_alarm: bool, vae_z: float) -> dict:
    """
    Four named states, deterministic rules.

    CRITICAL      : CNN says fault AND RUL ≤ 30
    EARLY WARNING : CNN says fault OR VAE alarm fires
    MONITOR       : RUL ≤ 50 but neither model alarming yet
    HEALTHY       : everything clear
    """
    if cnn_fault and rul_cycles <= 30:
        state  = "CRITICAL"
        color  = "#EF4444"
        action = "Immediate intervention — schedule unplanned maintenance"
    elif cnn_fault or vae_alarm:
        state  = "EARLY WARNING"
        color  = "#F59E0B"
        action = "Elevated risk detected — plan inspection within 5 cycles"
    elif rul_cycles <= 50:
        state  = "MONITOR"
        color  = "#3B82F6"
        action = "Degradation trending — increase monitoring frequency"
    else:
        state  = "HEALTHY"
        color  = "#10B981"
        action = "Normal operation"

    return {
        "state":       state,
        "color":       color,
        "action":      action,
        "cnn_fault":   bool(cnn_fault),
        "vae_alarm":   bool(vae_alarm),
        "vae_z_score": float(vae_z),
    }


# ── Main predictor ────────────────────────────────────────────────────────────

class PredictiveMaintenancePredictor:
    """
    Loads all three checkpoints once; exposes adapt_to_engine() + predict().

    Parameters
    ----------
    cnn_ckpt_path   : path to early_pred_best.pth         (CNN-LSTM fault classifier)
    maml_ckpt_path  : path to maml_meta_model_best.pth    (MAML RUL regression)
    vae_ckpt_path   : path to vae_early_warning.pth       (VAE + CUSUM)
    stats_path      : path to baseline_stats.npz
    max_rul         : float, de-normalisation factor for RUL (default 130)
    device          : 'cpu' or 'cuda'

    Typical usage
    -------------
    predictor = PredictiveMaintenancePredictor(...)

    # Once per engine — provide K support windows
    predictor.adapt_to_engine(support_windows, support_ruls_norm, K=5)

    # Per cycle
    result = predictor.predict(window, cycle)
    """

    def __init__(self, cnn_ckpt_path: str, maml_ckpt_path: str,
                 vae_ckpt_path: str, stats_path: str,
                 max_rul: float = 130.0, device: str = "cpu"):

        self.device  = torch.device(device)
        self.max_rul = max_rul

        # ── 1. CNN-LSTM fault classifier ─────────────────────────────────
        cnn_ckpt = torch.load(cnn_ckpt_path, map_location=self.device)
        cfg      = cnn_ckpt.get("config", {})

        self.fault_rul_cycles     = float(cnn_ckpt.get("fault_rul_cycles", 30.0))
        self.fault_threshold_norm = self.fault_rul_cycles / self.max_rul

        self.cnn_model = EarlyPredCNNLSTM(
            input_size=cfg.get("input_size", 102),
            dropout=cfg.get("dropout", 0.2),
        ).to(self.device)
        self.cnn_model.load_state_dict(cnn_ckpt["model_state_dict"])
        self.cnn_model.eval()

        self.cnn_threshold = float(cnn_ckpt.get("best_threshold", 0.5))
        print(f"[Predictor] CNN-LSTM loaded   threshold={self.cnn_threshold:.2f}  "
              f"fault_rul={self.fault_rul_cycles}")

        # ── 2. MAML meta-model (CNNLSTMBase) ─────────────────────────────
        maml_ckpt  = torch.load(maml_ckpt_path, map_location=self.device)
        self._maml_meta_model = CNNLSTMBase(input_size=102).to(self.device)
        self._maml_meta_model.load_state_dict(maml_ckpt["model_state_dict"])
        self._maml_meta_model.eval()

        saved_hp = maml_ckpt.get("hyperparameters", {})
        self._maml_default_inner_lr    = float(saved_hp.get("inner_lr",    _MAML_INNER_LR))
        self._maml_default_inner_steps = int(saved_hp.get("inner_steps", _MAML_INNER_STEPS))

        self._maml_adapted: CNNLSTMBase | None = None

        val_rmse_str = f"  val_rmse={maml_ckpt['val_rmse']:.2f}" if "val_rmse" in maml_ckpt else ""
        print(f"[Predictor] MAML loaded       inner_lr={self._maml_default_inner_lr}  "
              f"inner_steps={self._maml_default_inner_steps}{val_rmse_str}")

        # ── 3. VAE + CUSUM early-warning system ───────────────────────────
        stats          = np.load(stats_path, allow_pickle=True)
        baseline_mean  = float(stats["baseline_mean"])
        baseline_std   = float(stats["baseline_std"])
        threshold_h    = float(stats["threshold_h"])
        drift_k        = float(stats["drift_k"])
        decay          = float(stats["decay"])
        warmup         = int(stats["warmup"])
        self.sensor_indices = stats["sensor_indices"].tolist()

        vae_ckpt = torch.load(vae_ckpt_path, map_location=self.device)
        vcfg     = vae_ckpt.get("config", {})
        vae      = TimeSeriesVAE(
            input_size  = vcfg.get("input_size",   len(USEFUL_SENSORS)),
            seq_len     = vcfg.get("seq_len",       30),
            hidden_size = vcfg.get("hidden_size",   64),
            latent_dim  = vcfg.get("latent_dim",    16),
        )
        vae.load_state_dict(vae_ckpt["model_state_dict"])

        self.ews = EarlyWarningSystem(
            vae=vae, baseline_mean=baseline_mean, baseline_std=baseline_std,
            drift_k=drift_k, threshold_h=threshold_h,
            warmup=warmup, decay=decay, device=device,
        )
        print(f"[Predictor] VAE+CUSUM loaded  h={threshold_h:.4f}  "
              f"drift_k={drift_k}  warmup={warmup}")

    # ── Engine-level adaptation ───────────────────────────────────────────────

    def adapt_to_engine(self, engine_windows: np.ndarray,
                        engine_ruls_norm: np.ndarray,
                        K: int = 5) -> None:
        """
        Adapt the MAML meta-model to a new engine using K support windows.

        Call this ONCE per engine before the first predict() call.
        Also resets the VAE CUSUM state for the new engine.

        Parameters
        ----------
        engine_windows   : np.ndarray, shape (N, seq_len, 102)
                           All available windows for this engine so far.
        engine_ruls_norm : np.ndarray, shape (N,)
                           Normalised RUL labels (0–1) for each window.
        K                : int, number of support examples (default 5).
        """
        n = len(engine_windows)

        if n < K:
            raise ValueError(
                f"adapt_to_engine() needs at least K={K} windows, got {n}. "
                f"Collect more cycles before calling this method, or reduce K."
            )

        support_pos = _select_support_indices(n, K)
        hp          = _adapt_hyperparams_for_k(K)

        X_support = torch.FloatTensor(engine_windows[support_pos]).to(self.device)
        y_support = torch.FloatTensor(engine_ruls_norm[support_pos]).unsqueeze(1).to(self.device)

        adapted = copy.deepcopy(self._maml_meta_model).to(self.device)
        for param in adapted.parameters():
            param.requires_grad = True

        criterion = nn.MSELoss()
        optimizer = optim.SGD(adapted.parameters(), lr=hp["inner_lr"])

        adapted.train()
        for _ in range(hp["inner_steps"]):
            optimizer.zero_grad()
            loss = criterion(adapted(X_support), y_support)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapted.parameters(), max_norm=5.0)
            optimizer.step()

        adapted.eval()
        self._maml_adapted = adapted

        # Reset CUSUM state for new engine
        self.ews.reset()

        print(f"[Predictor] MAML adapted to engine  K={K}  "
              f"inner_lr={hp['inner_lr']}  inner_steps={hp['inner_steps']}")

    # ── Per-cycle prediction ──────────────────────────────────────────────────

    def reset_engine(self) -> None:
        """
        Clear adapted MAML model and VAE CUSUM state between engines.
        You must call adapt_to_engine() again before the next predict().
        """
        self._maml_adapted = None
        self.ews.reset()

    def predict(self, window: np.ndarray, cycle: int) -> dict:
        """
        Run one cycle window through all three branches and return a fused dict.

        Parameters
        ----------
        window  : np.ndarray, shape (seq_len, 102)
        cycle   : int, current cycle number (used by CUSUM tracker).

        Returns
        -------
        dict with keys:
            cycle, rul_cycles, fault_prob, cnn_fault,
            vae_alarm, vae_z_score, vae_cusum,
            state, color, action
        """
        if self._maml_adapted is None:
            raise RuntimeError(
                "adapt_to_engine() must be called before predict(). "
                "Provide at least K support windows for this engine."
            )

        x_full = torch.FloatTensor(window).unsqueeze(0).to(self.device)  # (1, 30, 102)

        # ── MAML branch — primary RUL source ─────────────────────────────
        # BUG FIX #1: CNNLSTMBase outputs a raw regression value, NOT a
        # sigmoid-bounded [0,1] value. Do NOT clamp to [0,1] before scaling.
        # The old code applied torch.clamp(maml_out, 0.0, 1.0) which crushed
        # any prediction > 1.0 (i.e. > ~8 cycles worth) to exactly 1.0,
        # making the predicted RUL flat at max_rul whenever the engine was
        # healthy, and snapping it to a single value when it degraded.
        #
        # Correct approach: clamp AFTER de-normalisation so the floor/ceiling
        # is in meaningful cycle space, not in normalised [0,1] space.
        with torch.no_grad():
            maml_out   = self._maml_adapted(x_full).squeeze()   # raw regression scalar
            rul_norm   = float(maml_out.cpu())                   # may be outside [0,1] — that is fine
        rul_cycles = rul_norm * self.max_rul
        rul_cycles = float(np.clip(rul_cycles, 0.0, self.max_rul))   # clamp in cycle space

        # ── CNN-LSTM branch — fault probability ──────────────────────────
        with torch.no_grad():
            logit      = self.cnn_model(x_full).squeeze()
            fault_prob = float(torch.sigmoid(logit).cpu())
        cnn_fault = fault_prob >= self.cnn_threshold

        # ── VAE + CUSUM branch — anomaly detection ────────────────────────
        # BUG FIX #2: sensor_indices contains integer column positions.
        # The old slice window[:, self.sensor_indices] works correctly only
        # when sensor_indices are plain Python ints. If they were loaded from
        # npz as numpy int64 scalars, advanced indexing still works — but if
        # the list is accidentally a list-of-arrays (shape mismatch from npz
        # load), the slice silently returns wrong shape. Force to a plain
        # Python list of ints here to be safe.
        sensor_idx = [int(i) for i in self.sensor_indices]
        x_vae   = window[:, sensor_idx]                              # (seq_len, 11)
        x_vae_t = torch.FloatTensor(x_vae).unsqueeze(0)             # (1, seq_len, 11)
        vae_result = self.ews.monitor(x_vae_t, cycle)

        # ── Alert fusion ──────────────────────────────────────────────────
        fused = _fuse_alerts(
            cnn_fault  = cnn_fault,
            rul_cycles = rul_cycles,
            vae_alarm  = vae_result["alarm"],
            vae_z      = vae_result["z_score"],
        )

        return {
            "cycle":       cycle,
            "rul_cycles":  round(rul_cycles, 1),
            "rul_norm":    round(rul_norm, 4),
            "fault_prob":  round(fault_prob, 4),
            "cnn_fault":   bool(cnn_fault),
            "vae_alarm":   bool(vae_result["alarm"]),
            "vae_z_score": round(vae_result["z_score"], 3),
            "vae_cusum":   round(vae_result["cusum"], 3),
            **fused,
        }