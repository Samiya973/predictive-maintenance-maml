"""
Enhanced Evaluation Script — Random Forest Baseline
PHM-grade plots for research and industry.

Place at: src/evaluation/evaluate_rf_baseline.py
Run with: python src/evaluation/evaluate_rf_baseline.py

Extra dependency for SHAP plots (optional):
    pip install shap
"""

import sys, os, pickle, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import matplotlib.cm as cm

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

os.makedirs("results/figures", exist_ok=True)

from src.models.rf_baseline  import RFBaseline, extract_features
from src.data.data_loader    import load_preprocessed_data


# ══════════════════════════════════════════════════════════════════════
#  METRICS  (identical to LSTM evaluator)
# ══════════════════════════════════════════════════════════════════════

def nasa_score(y_true, y_pred):
    d = y_pred - y_true
    return float(np.where(d < 0, np.exp(-d/13)-1, np.exp(d/10)-1).sum())

def compute_metrics(y_true, y_pred):
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae  = float(np.mean(np.abs(y_pred - y_true)))
    r2   = float(1 - np.sum((y_true-y_pred)**2) / np.sum((y_true-y_true.mean())**2))
    return {"rmse": rmse, "mae": mae, "r2": r2, "nasa": nasa_score(y_true, y_pred)}

def print_metrics(label, m):
    print(f"{label}:")
    print(f"  RMSE       : {m['rmse']:.4f} cycles")
    print(f"  MAE        : {m['mae']:.4f} cycles")
    print(f"  R2         : {m['r2']:.4f}")
    print(f"  NASA Score : {m['nasa']:.2f}")


# ══════════════════════════════════════════════════════════════════════
#  PLOT 1 — Standard 4-panel (Predicted vs Actual, Error dist,
#            Feature importance, Metrics comparison)
# ══════════════════════════════════════════════════════════════════════

def plot_standard(results, fi):
    """Original 4-panel evaluation plot."""
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Random Forest Baseline — Evaluation Overview",
                 fontsize=16, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    y_test = results["test"]["y_true"]
    p_test = results["test"]["y_pred"]
    errors = p_test - y_test

    # Predicted vs Actual
    ax1 = fig.add_subplot(gs[0, 0])
    idx = np.random.choice(len(y_test), min(1000, len(y_test)), replace=False)
    ax1.scatter(y_test[idx], p_test[idx], alpha=0.4, s=12, color="steelblue")
    ax1.plot([0,130],[0,130], "r--", lw=1.5, label="Perfect")
    ax1.set_xlim([0,130]); ax1.set_ylim([0,130])
    ax1.set_xlabel("Actual RUL (cycles)"); ax1.set_ylabel("Predicted RUL (cycles)")
    ax1.set_title(f"Predicted vs Actual (Test)\n"
                  f"RMSE={results['test']['rmse']:.2f}  R²={results['test']['r2']:.4f}")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    # Error Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(errors, bins=50, color="coral", edgecolor="white", alpha=0.85)
    ax2.axvline(0,             color="black", lw=1.5, label="Zero error")
    ax2.axvline(errors.mean(), color="red",   lw=1.5, ls="--",
                label=f"Mean={errors.mean():.2f}")
    ax2.set_xlabel("Prediction Error (cycles)"); ax2.set_ylabel("Frequency")
    ax2.set_title(f"Error Distribution (Test)\nMAE={results['test']['mae']:.2f} cycles")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    # Top-20 Feature Importances
    ax3 = fig.add_subplot(gs[1, 0])
    if fi is not None:
        top_k   = 20
        top_idx = np.argsort(fi)[-top_k:][::-1]
        stat_labels = (
            [f"mean_f{i}" for i in range(102)] +
            [f"std_f{i}"  for i in range(102)] +
            [f"min_f{i}"  for i in range(102)] +
            [f"max_f{i}"  for i in range(102)]
        )
        lbl = [stat_labels[i] for i in top_idx]
        ax3.barh(range(top_k), fi[top_idx][::-1], color="mediumseagreen")
        ax3.set_yticks(range(top_k))
        ax3.set_yticklabels(lbl[::-1], fontsize=7)
        ax3.set_xlabel("Importance")
        ax3.set_title(f"Top-{top_k} Feature Importances")
        ax3.grid(True, alpha=0.3, axis="x")

    # RMSE & MAE by split
    ax4 = fig.add_subplot(gs[1, 1])
    splits = ["Train", "Validation", "Test"]
    rmses  = [results[k]["rmse"] for k in ["train","val","test"]]
    maes   = [results[k]["mae"]  for k in ["train","val","test"]]
    x, w   = np.arange(3), 0.35
    b1 = ax4.bar(x-w/2, rmses, w, label="RMSE", color="steelblue", alpha=0.85)
    b2 = ax4.bar(x+w/2, maes,  w, label="MAE",  color="coral",     alpha=0.85)
    for b in [*b1, *b2]:
        ax4.text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
                 f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    ax4.set_xticks(x); ax4.set_xticklabels(splits)
    ax4.set_ylabel("Cycles")
    ax4.set_title("RMSE & MAE Across Splits")
    ax4.legend(); ax4.grid(True, alpha=0.3, axis="y")

    out = "results/figures/rf_01_standard_evaluation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  [Plot 1] Saved -> {out}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
#  PLOT 2 — Engine-wise Degradation Trajectories
#  Most cited plot in PHM literature. Shows actual vs predicted RUL
#  over time for individual engines. Reveals where model struggles.
# ══════════════════════════════════════════════════════════════════════

def plot_degradation_trajectories(X_test, y_test, y_pred, engine_ids, n_engines=6):
    """
    For selected test engines: plot actual RUL and predicted RUL
    against cycle number side by side.
    """
    unique_engines = np.unique(engine_ids)
    # Pick engines with diverse lifetimes for interesting plots
    engine_seq_counts = {e: np.sum(engine_ids == e) for e in unique_engines}
    sorted_engines    = sorted(unique_engines, key=lambda e: engine_seq_counts[e])
    # Pick spread: shortest, 2 mid, longest + 2 random
    pick = [sorted_engines[0],
            sorted_engines[len(sorted_engines)//4],
            sorted_engines[len(sorted_engines)//2],
            sorted_engines[3*len(sorted_engines)//4],
            sorted_engines[-1]]
    remaining = [e for e in unique_engines if e not in pick]
    if remaining:
        pick.append(np.random.choice(remaining))
    selected = pick[:n_engines]

    cols = 3
    rows = (len(selected) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, 4 * rows))
    fig.suptitle("Engine-wise Degradation Trajectories\n"
                 "Actual vs Predicted RUL over Operational Cycles",
                 fontsize=14, fontweight="bold")
    axes = axes.flatten()

    for ax_idx, eng_id in enumerate(selected):
        mask   = engine_ids == eng_id
        y_true = y_test[mask]
        y_pr   = y_pred[mask]
        cycles = np.arange(1, len(y_true) + 1)   # relative cycle index

        ax = axes[ax_idx]
        ax.plot(cycles, y_true, "b-",  lw=2,   label="Actual RUL",    alpha=0.9)
        ax.plot(cycles, y_pr,  "r--", lw=1.5, label="Predicted RUL", alpha=0.85)

        # Shade the error region
        ax.fill_between(cycles, y_true, y_pr,
                        alpha=0.15, color="orange", label="Error region")

        # Mark failure zone
        ax.axhline(30, color="gray", ls=":", lw=1, alpha=0.7)
        ax.text(cycles[-1]*0.02, 32, "Failure zone <30", fontsize=7, color="gray")

        rmse_eng = float(np.sqrt(np.mean((y_pr - y_true)**2)))
        ax.set_title(f"Engine {eng_id}  |  "
                     f"Lifetime: {len(y_true)} cycles  |  RMSE: {rmse_eng:.2f}",
                     fontsize=9)
        ax.set_xlabel("Relative Cycle"); ax.set_ylabel("RUL (cycles)")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    # Hide unused subplots
    for ax_idx in range(len(selected), len(axes)):
        axes[ax_idx].set_visible(False)

    plt.tight_layout()
    out = "results/figures/rf_02_degradation_trajectories.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  [Plot 2] Saved -> {out}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
#  PLOT 3 — Residual vs Actual RUL
#  Reveals systematic bias: does model over/under-predict at different
#  life stages? Critical for maintenance scheduling decisions.
# ══════════════════════════════════════════════════════════════════════

def plot_residuals(y_true, y_pred):
    """
    Residual (predicted - actual) vs actual RUL.
    Binned median line shows systematic over/under-prediction trend.
    """
    residuals = y_pred - y_true

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Residual Analysis — Test Set",
                 fontsize=14, fontweight="bold")

    # Left: scatter with binned median
    ax1 = axes[0]
    ax1.scatter(y_true, residuals, alpha=0.3, s=10, color="steelblue",
                label="Residuals")
    ax1.axhline(0, color="red", lw=2, ls="--", label="Zero residual")

    # Binned median line (10 bins)
    bins    = np.linspace(0, 130, 11)
    centers = (bins[:-1] + bins[1:]) / 2
    medians = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_true >= lo) & (y_true < hi)
        medians.append(np.median(residuals[mask]) if mask.sum() > 0 else 0)
    ax1.plot(centers, medians, "orange", lw=2.5, marker="o",
             markersize=5, label="Binned median", zorder=5)

    ax1.set_xlabel("Actual RUL (cycles)")
    ax1.set_ylabel("Residual: Predicted - Actual (cycles)")
    ax1.set_title("Residual vs Actual RUL\n"
                  "Orange line = systematic bias per life stage")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 130])

    # Annotate regions
    ax1.axvspan(0,  30, alpha=0.05, color="red",    label="Near failure")
    ax1.axvspan(30, 80, alpha=0.05, color="orange")
    ax1.axvspan(80,130, alpha=0.05, color="green")
    ax1.text(5,  ax1.get_ylim()[1]*0.90, "Near\nfailure", fontsize=7, color="red")
    ax1.text(45, ax1.get_ylim()[1]*0.90, "Degrading",    fontsize=7, color="darkorange")
    ax1.text(95, ax1.get_ylim()[1]*0.90, "Healthy",      fontsize=7, color="green")

    # Right: residual distribution split by life stage
    ax2 = axes[1]
    stages = {
        "Near failure (0-30)":  (y_true <  30),
        "Degrading (30-80)":    (y_true >= 30) & (y_true < 80),
        "Healthy (80-130)":     (y_true >= 80),
    }
    colors = ["red", "orange", "green"]
    for (label, mask), color in zip(stages.items(), colors):
        ax2.hist(residuals[mask], bins=40, alpha=0.55,
                 label=f"{label}\n  n={mask.sum()}", color=color,
                 edgecolor="white", density=True)
    ax2.axvline(0, color="black", lw=2, ls="--")
    ax2.set_xlabel("Residual (cycles)")
    ax2.set_ylabel("Density")
    ax2.set_title("Residual Distribution by Life Stage\n"
                  "Skew reveals over/under-prediction tendency")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = "results/figures/rf_03_residual_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  [Plot 3] Saved -> {out}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
#  PLOT 4 — Prediction Uncertainty Bands (RF-specific)
#  RF gives free uncertainty via tree variance. Invaluable in industry:
#  a confident wrong prediction is worse than an uncertain correct one.
# ══════════════════════════════════════════════════════════════════════

def plot_uncertainty_bands(rf_model, X_test, y_test, engine_ids, n_engines=4):
    """
    For selected engines: plot predicted RUL mean ± std across trees.
    Wide bands = model uncertain (act conservatively).
    Narrow bands = model confident.
    """
    print("    Computing per-tree predictions (this may take ~30s)...")

    Xf   = extract_features(X_test)
    # Get prediction from every tree in the forest
    tree_preds = np.array([
        np.clip(tree.predict(Xf), 0, 130)
        for tree in rf_model.model.estimators_
    ])                                          # shape: (n_trees, N)
    mean_pred = tree_preds.mean(axis=0)
    std_pred  = tree_preds.std(axis=0)

    unique_engines = np.unique(engine_ids)
    # Pick engines with varying uncertainty profiles
    engine_seq_counts = {e: np.sum(engine_ids == e) for e in unique_engines}
    sorted_eng = sorted(unique_engines, key=lambda e: engine_seq_counts[e])
    selected   = [sorted_eng[0],
                  sorted_eng[len(sorted_eng)//3],
                  sorted_eng[2*len(sorted_eng)//3],
                  sorted_eng[-1]][:n_engines]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Prediction Uncertainty Bands (RF Tree Variance)\n"
                 "Shaded region = Mean ± 1 Std Dev across all trees",
                 fontsize=14, fontweight="bold")
    axes = axes.flatten()

    for ax_idx, eng_id in enumerate(selected):
        mask   = engine_ids == eng_id
        cycles = np.arange(1, mask.sum() + 1)
        m_pred = mean_pred[mask]
        s_pred = std_pred[mask]
        y_true = y_test[mask]

        ax = axes[ax_idx]
        ax.plot(cycles, y_true,  "b-",  lw=2,   label="Actual RUL", alpha=0.9)
        ax.plot(cycles, m_pred,  "r--", lw=1.5, label="Mean prediction", alpha=0.9)

        # ±1 std band
        ax.fill_between(cycles,
                        np.clip(m_pred - s_pred, 0, 130),
                        np.clip(m_pred + s_pred, 0, 130),
                        alpha=0.3, color="red", label="±1 std (uncertainty)")

        # ±2 std band (wider, lighter)
        ax.fill_between(cycles,
                        np.clip(m_pred - 2*s_pred, 0, 130),
                        np.clip(m_pred + 2*s_pred, 0, 130),
                        alpha=0.12, color="red", label="±2 std")

        # Mean uncertainty across engine life
        mean_unc = s_pred.mean()
        ax.set_title(f"Engine {eng_id}  |  "
                     f"Avg uncertainty: ±{mean_unc:.1f} cycles", fontsize=10)
        ax.set_xlabel("Relative Cycle"); ax.set_ylabel("RUL (cycles)")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3); ax.set_ylim(bottom=0)

    plt.tight_layout()
    out = "results/figures/rf_04_uncertainty_bands.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  [Plot 4] Saved -> {out}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
#  PLOT 5 — SHAP Explainability (optional, needs pip install shap)
#  Gold standard for model explainability in industry + research.
# ══════════════════════════════════════════════════════════════════════

def plot_shap(rf_model, X_test, feature_names=None):
    """
    SHAP summary (beeswarm) + bar plot for RF.
    Requires: pip install shap
    """
    try:
        import shap
    except ImportError:
        print("  [Plot 5] Skipped — run: pip install shap")
        return

    print("    Computing SHAP values (this may take 1-2 min)...")

    Xf = extract_features(X_test)
    # Use a background sample for speed (200 samples is enough)
    bg_idx = np.random.choice(len(Xf), min(200, len(Xf)), replace=False)
    explainer  = shap.TreeExplainer(rf_model.model)
    # Explain a sample (500 instances max for speed)
    exp_idx    = np.random.choice(len(Xf), min(500, len(Xf)), replace=False)
    shap_vals  = explainer.shap_values(Xf[exp_idx])  # (n_samples, n_features)

    # Build feature names (mean_f0 … max_f101)
    if feature_names is None:
        feature_names = (
            [f"mean_f{i}" for i in range(102)] +
            [f"std_f{i}"  for i in range(102)] +
            [f"min_f{i}"  for i in range(102)] +
            [f"max_f{i}"  for i in range(102)]
        )

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("SHAP Explainability — Random Forest Baseline\n"
                 "Feature contribution to RUL predictions",
                 fontsize=14, fontweight="bold")

    # SHAP bar plot (mean |SHAP|) — top 20
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    top_k  = 20
    top_idx = np.argsort(mean_abs_shap)[-top_k:][::-1]
    top_vals = mean_abs_shap[top_idx]
    top_lbl  = [feature_names[i] for i in top_idx]

    ax1 = axes[0]
    bars = ax1.barh(range(top_k), top_vals[::-1], color="steelblue", alpha=0.85)
    ax1.set_yticks(range(top_k))
    ax1.set_yticklabels(top_lbl[::-1], fontsize=8)
    ax1.set_xlabel("Mean |SHAP value|  (impact on RUL prediction)")
    ax1.set_title(f"Top-{top_k} Features by SHAP Importance")
    ax1.grid(True, alpha=0.3, axis="x")

    # SHAP beeswarm (summary plot) — top 15 features
    ax2 = axes[1]
    top15_idx = np.argsort(mean_abs_shap)[-15:][::-1]
    shap_top  = shap_vals[:, top15_idx]
    feat_top  = Xf[exp_idx][:, top15_idx]
    lbl_top   = [feature_names[i] for i in top15_idx]

    # Manual beeswarm-style plot
    for row_idx, feat_idx in enumerate(range(15)):
        sv  = shap_top[:, feat_idx]
        fv  = feat_top[:, feat_idx]
        # Normalize feature value for colour
        fv_norm = (fv - fv.min()) / (fv.max() - fv.min() + 1e-9)
        # Jitter y
        jitter = (np.random.rand(len(sv)) - 0.5) * 0.4
        scatter = ax2.scatter(sv, np.full_like(sv, 14 - row_idx) + jitter,
                              c=fv_norm, cmap="RdBu_r", s=6, alpha=0.5,
                              vmin=0, vmax=1)
    ax2.axvline(0, color="gray", lw=1)
    ax2.set_yticks(range(15))
    ax2.set_yticklabels(lbl_top[::-1], fontsize=8)
    ax2.set_xlabel("SHAP value  (positive = increases predicted RUL)")
    ax2.set_title("SHAP Beeswarm — Top 15 Features\n"
                  "Color: feature value (blue=low, red=high)")
    cbar = plt.colorbar(scatter, ax=ax2, fraction=0.03, pad=0.04)
    cbar.set_label("Feature value (normalised)", fontsize=8)
    ax2.grid(True, alpha=0.2, axis="x")

    plt.tight_layout()
    out = "results/figures/rf_05_shap_explainability.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  [Plot 5] Saved -> {out}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
#  PLOT 6 — Early vs Late Prediction Cost Analysis
#  Most practically relevant plot for maintenance decision-making.
#  Early prediction (under-estimate RUL) → unnecessary downtime cost.
#  Late prediction  (over-estimate RUL)  → catastrophic failure risk.
# ══════════════════════════════════════════════════════════════════════

def plot_cost_analysis(y_true, y_pred):
    """
    Splits errors into early (conservative) and late (dangerous)
    predictions and shows the cost-weighted distribution.
    """
    residuals  = y_pred - y_true           # positive = over-predicted (late)
    early_mask = residuals < 0             # predicted < actual (early, safe)
    late_mask  = residuals > 0             # predicted > actual (late, risky)

    early_err  = residuals[early_mask]
    late_err   = residuals[late_mask]

    # Asymmetric cost weights (industry standard: late is ~3x costly)
    EARLY_COST_PER_CYCLE = 1.0
    LATE_COST_PER_CYCLE  = 3.0

    early_cost = np.abs(early_err) * EARLY_COST_PER_CYCLE
    late_cost  = np.abs(late_err)  * LATE_COST_PER_CYCLE

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Early vs Late Prediction Cost Analysis\n"
                 "Early = conservative (safe) | Late = dangerous (risky)",
                 fontsize=14, fontweight="bold")

    # Panel 1: Counts
    ax1 = axes[0, 0]
    counts  = [early_mask.sum(), late_mask.sum(), (residuals == 0).sum()]
    labels  = [f"Early\n(n={counts[0]})",
               f"Late\n(n={counts[1]})",
               f"Exact\n(n={counts[2]})"]
    colors  = ["#2196F3", "#F44336", "#4CAF50"]
    wedges, texts, autotexts = ax1.pie(
        counts, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 9}
    )
    ax1.set_title("Prediction Direction\n(Early / On-time / Late)", fontsize=10)

    # Panel 2: Error magnitude distributions
    ax2 = axes[0, 1]
    ax2.hist(early_err, bins=40, color="#2196F3", alpha=0.7,
             label=f"Early (under-predict)\nMean={early_err.mean():.1f} cycles",
             edgecolor="white")
    ax2.hist(late_err,  bins=40, color="#F44336", alpha=0.7,
             label=f"Late (over-predict)\nMean={late_err.mean():.1f} cycles",
             edgecolor="white")
    ax2.axvline(0, color="black", lw=2, ls="--")
    ax2.set_xlabel("Residual (cycles)"); ax2.set_ylabel("Frequency")
    ax2.set_title("Error Magnitude: Early vs Late Predictions")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    # Panel 3: Cost-weighted distribution
    ax3 = axes[1, 0]
    ax3.hist(early_cost, bins=40, color="#2196F3", alpha=0.7,
             label=f"Early cost (×{EARLY_COST_PER_CYCLE:.0f}/cycle)\nTotal={early_cost.sum():.0f}",
             edgecolor="white")
    ax3.hist(late_cost,  bins=40, color="#F44336", alpha=0.7,
             label=f"Late cost (×{LATE_COST_PER_CYCLE:.0f}/cycle)\nTotal={late_cost.sum():.0f}",
             edgecolor="white")
    ax3.set_xlabel("Weighted Cost"); ax3.set_ylabel("Frequency")
    ax3.set_title(f"Cost-Weighted Distribution\n"
                  f"Late prediction penalised {LATE_COST_PER_CYCLE:.0f}× more")
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

    # Panel 4: Error type vs actual RUL (binned)
    ax4 = axes[1, 1]
    bins    = np.linspace(0, 130, 14)
    centers = (bins[:-1] + bins[1:]) / 2
    early_counts_bin = []
    late_counts_bin  = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_true >= lo) & (y_true < hi)
        e = early_mask[mask].sum()
        l = late_mask[mask].sum()
        early_counts_bin.append(e)
        late_counts_bin.append(l)

    w = (bins[1]-bins[0]) * 0.42
    ax4.bar(centers - w/2, early_counts_bin, w,
            color="#2196F3", alpha=0.8, label="Early predictions")
    ax4.bar(centers + w/2, late_counts_bin,  w,
            color="#F44336", alpha=0.8, label="Late predictions")
    ax4.set_xlabel("Actual RUL (cycles)")
    ax4.set_ylabel("Count")
    ax4.set_title("Early vs Late Predictions by RUL Stage\n"
                  "Are late predictions concentrated near failure?")
    ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3, axis="y")
    ax4.axvline(30, color="gray", ls=":", lw=1.5)
    ax4.text(31, ax4.get_ylim()[1]*0.9, "Failure zone", fontsize=8, color="gray")

    # Summary text box
    total_cost = early_cost.sum() + late_cost.sum()
    summary = (
        f"Early predictions : {early_mask.sum():4d}  ({100*early_mask.mean():.1f}%)\n"
        f"Late  predictions : {late_mask.sum():4d}  ({100*late_mask.mean():.1f}%)\n"
        f"Total early cost  : {early_cost.sum():.0f}\n"
        f"Total late cost   : {late_cost.sum():.0f}\n"
        f"Total weighted    : {total_cost:.0f}"
    )
    fig.text(0.5, 0.01, summary, ha="center", va="bottom",
             fontsize=9, family="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    out = "results/figures/rf_06_cost_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  [Plot 6] Saved -> {out}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def evaluate_rf():
    print("=" * 60)
    print("EVALUATING RANDOM FOREST BASELINE")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────────────
    print("\nLoading data...")
    data     = load_preprocessed_data()
    X_train  = data['X_train']; y_train = data['y_train']
    X_val    = data['X_val'];   y_val   = data['y_val']
    X_test   = data['X_test'];  y_test  = data['y_test']
    print(f"  Training   : {len(X_train):,} samples")
    print(f"  Validation : {len(X_val):,} samples")
    print(f"  Test       : {len(X_test):,} samples")

    # Engine IDs per sequence (needed for trajectory + uncertainty plots)
    # Try to load from the .npz; fall back to sequential numbering
    npz_path = "data/processed/FD001_preprocessed.npz"
    npz      = np.load(npz_path, allow_pickle=True)
    if "test_engine_ids" in npz:
        test_engine_ids = npz["test_engine_ids"]
    elif "engine_ids" in npz:
        # Some versions store a single 'engine_ids' for full dataset
        n_test = len(X_test)
        test_engine_ids = npz["engine_ids"][-n_test:]
    else:
        # Fallback: assign a dummy engine id per sequence block
        print("  Warning: engine_ids not found in .npz — using sequential IDs")
        n_test  = len(X_test)
        n_eng   = 15
        # np.array_split guarantees total length == n_test (no truncation)
        splits  = np.array_split(np.arange(n_test), n_eng)
        test_engine_ids = np.zeros(n_test, dtype=int)
        for eng_idx, idx_arr in enumerate(splits):
            test_engine_ids[idx_arr] = eng_idx

    # ── Load model ───────────────────────────────────────────────────
    print("\nLoading trained model...")
    model_path = "results/saved_models/rf_baseline_best.pkl"
    rf = RFBaseline.load(model_path)
    print(f"  Loaded from {model_path}")

    # ── Predictions ──────────────────────────────────────────────────
    print("\nRunning predictions...")
    p_train = rf.predict(X_train)
    p_val   = rf.predict(X_val)
    p_test  = rf.predict(X_test)

    m_train = compute_metrics(y_train, p_train)
    m_val   = compute_metrics(y_val,   p_val)
    m_test  = compute_metrics(y_test,  p_test)

    print()
    print_metrics("Train Set",      m_train)
    print_metrics("Validation Set", m_val)
    print_metrics("Test Set",       m_test)

    results = {
        "train": {**m_train, "y_true": y_train, "y_pred": p_train},
        "val"  : {**m_val,   "y_true": y_val,   "y_pred": p_val  },
        "test" : {**m_test,  "y_true": y_test,  "y_pred": p_test },
    }

    # ── All plots ────────────────────────────────────────────────────
    print("\nCreating visualizations...")

    print("  Generating Plot 1: Standard evaluation...")
    plot_standard(results, rf.feature_importances_)

    print("  Generating Plot 2: Degradation trajectories...")
    plot_degradation_trajectories(X_test, y_test, p_test, test_engine_ids)

    print("  Generating Plot 3: Residual analysis...")
    plot_residuals(y_test, p_test)

    print("  Generating Plot 4: Uncertainty bands...")
    plot_uncertainty_bands(rf, X_test, y_test, test_engine_ids)

    print("  Generating Plot 5: SHAP explainability...")
    plot_shap(rf, X_test)

    print("  Generating Plot 6: Cost analysis...")
    plot_cost_analysis(y_test, p_test)

    # ── Final summary ────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"  Train RMSE : {m_train['rmse']:.4f} cycles")
    print(f"  Val   RMSE : {m_val['rmse']:.4f} cycles")
    print(f"  Test  RMSE : {m_test['rmse']:.4f} cycles")
    print()
    print("  Figures saved to results/figures/:")
    print("    rf_01_standard_evaluation.png")
    print("    rf_02_degradation_trajectories.png")
    print("    rf_03_residual_analysis.png")
    print("    rf_04_uncertainty_bands.png")
    print("    rf_05_shap_explainability.png  (if shap installed)")
    print("    rf_06_cost_analysis.png")
    print("=" * 60)


if __name__ == "__main__":
    evaluate_rf()
