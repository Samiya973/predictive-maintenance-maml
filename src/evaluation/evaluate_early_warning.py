
"""
evaluate_early_warning.py
─────────────────────────
Evaluation suite for VAE Early Warning Detector.
Window-level as primary metrics. Honest engine-level with adaptive CUSUM.
"""

import os, sys, argparse, csv
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                              average_precision_score, confusion_matrix)

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.models.early_warning_vae import (
    TimeSeriesVAE, ChangePointDetector, EarlyWarningSystem,
    classify_severity, USEFUL_SENSORS, N_SENSORS,
)

SEVERITY_ORDER = ["HEALTHY", "MILD", "MODERATE", "SEVERE"]
SEVERITY_COLORS = {"HEALTHY": "#2ecc71", "MILD": "#f1c40f",
                    "MODERATE": "#e67e22", "SEVERE": "#e74c3c"}
MAX_VALID_EARLY = 90
POLICY_SWEEP = [55, 70, 90, 100]

def load_model_and_baseline(model_path, baseline_path, device):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = TimeSeriesVAE(config["input_size"], config["seq_len"],
                          config["hidden_size"], config["latent_dim"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    st = np.load(baseline_path, allow_pickle=True)
    mu = float(st["baseline_mean"])
    sigma = float(st["baseline_std"])
    h = float(st["threshold_h"]) if "threshold_h" in st else 4.0
    dk = float(st["drift_k"]) if "drift_k" in st else 0.5
    decay = float(st["decay"]) if "decay" in st else 0.95
    wu = int(st["warmup"]) if "warmup" in st else 20
    si = (st["sensor_indices"].tolist() if "sensor_indices" in st
          else list(range(N_SENSORS)))

    print(f"VAE: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Baseline: mu={mu:.6f} sigma={sigma:.6f}")
    print(f"CUSUM: h={h:.4f} drift_k={dk} decay={decay} warmup={wu}")
    return model, mu, sigma, h, dk, decay, wu, si


def run_engine_evaluation(engine_windows, engine_rul, engine_id,
                           model, mu, sigma, h, dk, decay, wu,
                           healthy_ratio, device):
    ews = EarlyWarningSystem(
        vae=model, baseline_mean=mu, baseline_std=sigma,
        drift_k=dk, threshold_h=h, warmup=wu,
        decay=decay, device=str(device))

    true_onset = None
    for i, rul in enumerate(engine_rul):
        if rul <= (1.0 - healthy_ratio):
            true_onset = i
            break

    errors, zs, cs, als, sevs = [], [], [], [], []
    for idx, x_np in enumerate(engine_windows):
        r = ews.monitor(torch.FloatTensor(x_np).unsqueeze(0), cycle=idx)
        errors.append(r["error"]); zs.append(r["z_score"])
        cs.append(r["cusum"]); als.append(r["alarm"])
        sevs.append(r["severity"])

    det = ews.get_detection_cycle()
    lat = ((true_onset - det) if (true_onset is not None and
            det is not None) else None)

    return {"engine_id": engine_id, "n_windows": len(engine_windows),
            "true_onset_idx": true_onset, "detection_idx": det,
            "latency": lat, "errors": np.array(errors),
            "z_scores": np.array(zs), "cusums": np.array(cs),
            "alarms": np.array(als), "severities": sevs,
            "rul": engine_rul}


def build_window_labels(results, hr):
    yt, ys = [], []
    for r in results:
        for rul, z in zip(r["rul"], r["z_scores"]):
            yt.append(1 if rul <= (1.0 - hr) else 0)
            ys.append(float(z))
    return np.array(yt, dtype=int), np.array(ys, dtype=float)


def compute_engine_metrics(results, healthy_ratio,
                           max_early=MAX_VALID_EARLY):
    tp = fn = fp = tn = 0
    valid_lats = []
    all_lats = []
    degraded = 0

    for r in results:
        has_on = r["true_onset_idx"] is not None
        has_det = r["detection_idx"] is not None
        lat = r["latency"]

        if has_on:
            degraded += 1
            if has_det and lat is not None:
                all_lats.append(lat)
                if -15 <= lat <= max_early:
                    tp += 1
                    valid_lats.append(lat)
                elif lat > max_early:
                    fp += 1
                else:
                    fn += 1
            else:
                fn += 1
        else:
            if has_det:
                fp += 1
            else:
                tn += 1

    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)   # fixed
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * tpr / max(prec + tpr, 1e-9)

    vl = np.array(valid_lats, dtype=float)
    al = np.array(all_lats, dtype=float)

    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "tpr": tpr, "fpr_engine": fpr, "precision": prec, "f1": f1,
        "degraded_engines": degraded,
        "valid_detections": tp,
        "too_early_count": fp,
        "latencies": vl,
        "mean_latency": float(vl.mean()) if len(vl) > 0 else float("nan"),
        "median_latency": float(np.median(vl)) if len(vl) > 0 else float("nan"),
        "std_latency": float(vl.std()) if len(vl) > 0 else float("nan"),
        "min_latency": float(vl.min()) if len(vl) > 0 else float("nan"),
        "max_latency": float(vl.max()) if len(vl) > 0 else float("nan"),
        "early_count": int((vl > 0).sum()) if len(vl) > 0 else 0,
        "late_count": int((vl < 0).sum()) if len(vl) > 0 else 0,
        "exact_count": int((vl == 0).sum()) if len(vl) > 0 else 0,
        "all_latencies": al,
        "all_mean_latency": float(al.mean()) if len(al) > 0 else float("nan"),
    }

def compute_roc(yt, ys):
    fpr, tpr, th = roc_curve(yt, ys, drop_intermediate=False)
    a = auc(fpr, tpr); j = tpr - fpr; oi = np.argmax(j)
    return {"fpr": fpr, "tpr": tpr, "thresholds": th, "auc_roc": a,
            "opt_threshold": float(th[oi]), "opt_tpr": float(tpr[oi]),
            "opt_fpr": float(fpr[oi]), "opt_j": float(j[oi])}


def compute_pr(yt, ys):
    p, r, th = precision_recall_curve(yt, ys)
    ap = average_precision_score(yt, ys)
    f1 = 2 * p * r / np.maximum(p + r, 1e-9); bi = np.argmax(f1)
    return {"precision": p, "recall": r, "thresholds": th,
            "avg_precision": ap, "best_f1": float(f1[bi]),
            "best_threshold": (float(th[bi]) if bi < len(th)
                               else float("nan")),
            "best_precision": float(p[bi]),
            "best_recall": float(r[bi])}


def cm_stats(yt, ys, thr):
    yp = (ys >= thr).astype(int)
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    tpr = tp / max(tp + fn, 1); fpr = fp / max(fp + tn, 1)
    ppv = tp / max(tp + fp, 1); f1 = 2 * ppv * tpr / max(ppv + tpr, 1e-9)
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "tpr": tpr, "fpr": fpr, "precision": ppv, "f1": f1}


def diagnose(results, hr, max_early=MAX_VALID_EARLY):
    diags = []
    for r in results:
        to = r["true_onset_idx"]; det = r["detection_idx"]
        lat = r["latency"]
        if to is None: continue
        zo = float(r["z_scores"][to]); co = float(r["cusums"][to])
        if det is None:
            diags.append({"eid": r["engine_id"], "issue": "MISSED",
                          "desc": "Never detected", "to": to, "det": None,
                          "lat": None, "zo": zo, "co": co,
                          "rec": "Lower threshold_h"})
        elif lat is not None and lat < -15:
            diags.append({"eid": r["engine_id"], "issue": "LATE",
                          "desc": f"{abs(lat)} cycles after onset",
                          "to": to, "det": det, "lat": lat, "zo": zo,
                          "co": co, "rec": "Lower threshold_h"})
        elif lat is not None and lat > max_early:
            diags.append({"eid": r["engine_id"], "issue": "TOO_EARLY",
                          "desc": f"{lat} cycles before onset",
                          "to": to, "det": det, "lat": lat, "zo": zo,
                          "co": co, "rec": "Check VAE generalization"})
    return diags

def apply_alarm_policy(alarms, cooldown=None):
    """
    Convert raw per-window alarms into deployable alert episodes.

    cooldown = None  -> one-shot policy (only first alert kept)
    cooldown = int   -> keep first alert, then suppress next `cooldown` windows
    """
    alarms = np.asarray(alarms, dtype=int)
    policy_alarms = np.zeros_like(alarms)

    fired_once = False
    cooldown_left = 0

    for i, a in enumerate(alarms):
        if cooldown_left > 0:
            cooldown_left -= 1
            continue

        if a == 1:
            if cooldown is None:
                if not fired_once:
                    policy_alarms[i] = 1
                    fired_once = True
            else:
                policy_alarms[i] = 1
                cooldown_left = int(cooldown)

    return policy_alarms

def compute_deployability_metrics(results, hr, max_early=MAX_VALID_EARLY):
    useful = too_early = late = missed = 0
    degraded = 0

    useful_lats = []
    total_alert_episodes = 0
    healthy_alert_episodes = 0
    healthy_window_count = 0
    pre_onset_alert_episodes = 0

    first_alert_rows = []

    # None = one-shot first alert only
    # set something like 20 if you want cooldown mode instead
    ALARM_COOLDOWN = None

    for r in results:
        eid = r["engine_id"]
        to = r["true_onset_idx"]
        det = r["detection_idx"]
        lat = r["latency"]

        raw_alarms = np.asarray(r["alarms"], dtype=int)
        rul = np.asarray(r["rul"], dtype=float)

        # convert raw alarm windows into deployable alert episodes
        alarm_events = apply_alarm_policy(raw_alarms, cooldown=ALARM_COOLDOWN)

        total_alarm_count = int(alarm_events.sum())
        total_alert_episodes += total_alarm_count

        healthy_mask = rul > (1.0 - hr)
        healthy_window_count += int(healthy_mask.sum())
        healthy_alert_episodes += int(alarm_events[healthy_mask].sum())

        pre_onset_n = int(alarm_events[:to].sum()) if to is not None and to > 0 else 0

        if to is not None:
            degraded += 1
            pre_onset_alert_episodes += pre_onset_n

            if det is None or lat is None:
                status = "MISSED"
                missed += 1
            elif lat > max_early:
                status = "TOO_EARLY"
                too_early += 1
            elif lat < 0:
                status = "LATE"
                late += 1
            else:
                status = "USEFUL"
                useful += 1
                useful_lats.append(lat)
        else:
            status = "NO_ONSET"

        first_alert_rows.append({
            "engine_id": int(eid),
            "onset_idx": (int(to) if to is not None else ""),
            "detection_idx": (int(det) if det is not None else ""),
            "latency": (int(lat) if lat is not None else ""),
            "status": status,
            "pre_onset_alarm_count": int(pre_onset_n),
            "total_alarm_count": int(total_alarm_count),
        })

    useful_lats = np.array(useful_lats, dtype=float)
    detected = useful + too_early + late

    return {
        "degraded_engines": degraded,
        "useful_count": useful,
        "too_early_count": too_early,
        "late_count": late,
        "missed_count": missed,
        "actionable_rate": useful / max(degraded, 1),
        "too_early_rate": too_early / max(degraded, 1),
        "late_rate": late / max(degraded, 1),
        "missed_rate": missed / max(degraded, 1),
        "first_alert_precision": useful / max(detected, 1),
        "mean_actionable_latency": (
            float(useful_lats.mean()) if len(useful_lats) > 0 else float("nan")
        ),
        "median_actionable_latency": (
            float(np.median(useful_lats)) if len(useful_lats) > 0 else float("nan")
        ),
        "false_alarms_per_100_healthy_windows": (
            100.0 * healthy_alert_episodes / max(healthy_window_count, 1)
        ),
        "avg_alarms_per_engine": total_alert_episodes / max(len(results), 1),
        "avg_pre_onset_alarms_per_degraded_engine": (
            pre_onset_alert_episodes / max(degraded, 1)
        ),
        "first_alert_rows": first_alert_rows,
    }

def compute_policy_sensitivity(results, policy_values=POLICY_SWEEP):
    rows = []
    degraded = sum(1 for r in results if r["true_onset_idx"] is not None)

    for max_early in policy_values:
        useful = too_early = late = missed = 0

        for r in results:
            if r["true_onset_idx"] is None:
                continue

            det = r["detection_idx"]
            lat = r["latency"]

            if det is None or lat is None:
                missed += 1
            elif lat > max_early:
                too_early += 1
            elif lat < 0:
                late += 1
            else:
                useful += 1

        precision = useful / max(useful + too_early + late, 1)
        recall = useful / max(degraded, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)

        rows.append({
            "max_valid_early": int(max_early),
            "useful_count": int(useful),
            "too_early_count": int(too_early),
            "late_count": int(late),
            "missed_count": int(missed),
            "actionable_rate": recall,
            "too_early_rate": too_early / max(degraded, 1),
            "late_rate": late / max(degraded, 1),
            "missed_rate": missed / max(degraded, 1),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    return rows


def save_csv_rows(rows, path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] {path}")

def sev_dist(results):
    c = defaultdict(int)
    for r in results:
        for s in r["severities"]: c[s] += 1
    return dict(c)


# ── PLOTS ──────────────────────────────────────────────────────────────
def plot_roc(roc, pr, out):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    ax.plot(roc["fpr"], roc["tpr"], color="#2980b9", lw=2.5,
            label=f"AUC={roc['auc_roc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.scatter(roc["opt_fpr"], roc["opt_tpr"], color="#e74c3c", s=120,
               zorder=5, label=f"Opt z={roc['opt_threshold']:.2f}")
    ax.axhline(0.9, color="#2ecc71", ls=":", lw=1.5, alpha=0.8)
    ax.axvline(0.1, color="#e67e22", ls=":", lw=1.5, alpha=0.8)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(pr["recall"], pr["precision"], color="#8e44ad", lw=2.5,
            label=f"AP={pr['avg_precision']:.3f}")
    ax.scatter(pr["best_recall"], pr["best_precision"], color="#e74c3c",
               s=120, zorder=5, label=f"F1={pr['best_f1']:.3f}")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("PR", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p = os.path.join(out, "roc_pr_curves.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[OK] {p}")


def plot_latency(m, out):
    al = m.get("all_latencies", m["latencies"])
    if len(al) == 0: return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    if (al > 0).any():
        ax.hist(al[al > 0], bins=20, color="#2ecc71", alpha=0.8,
                label=f"Early ({(al > 0).sum()})")
    if (al <= 0).any():
        ax.hist(al[al <= 0], bins=20, color="#e74c3c", alpha=0.8,
                label=f"Late ({(al <= 0).sum()})")
    ax.axvline(0, color="black", lw=2)
    ax.axvline(30, color="#f39c12", lw=2, ls="--", label="Target +30")
    ax.axvline(50, color="#f39c12", lw=2, ls=":", label="Target +50")
    ax.axvspan(30, 50, alpha=0.12, color="#f39c12")
    ax.axvline(MAX_VALID_EARLY, color="purple", lw=1.5, ls="-.",
               label=f"Max valid (+{MAX_VALID_EARLY})")
    ax.set_xlabel("Latency (cycles)"); ax.set_ylabel("Count")
    ax.set_title(f"All Latencies (mean={m['all_mean_latency']:.1f})",
                 fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1]
    sl = np.sort(al); cdf = np.arange(1, len(sl) + 1) / len(sl)
    ax.plot(sl, cdf, color="#2980b9", lw=2.5)
    ax.axvline(0, color="black", lw=1.5)
    ax.axvline(30, color="#f39c12", lw=1.5, ls="--")
    ax.axvline(50, color="#f39c12", lw=1.5, ls=":")
    ax.axvspan(30, 50, alpha=0.12, color="#f39c12")
    ax.set_xlabel("Latency"); ax.set_ylabel("CDF")
    ax.set_ylim([0, 1.05]); ax.set_title("CDF", fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p = os.path.join(out, "detection_latency.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[OK] {p}")


def plot_cusum(results, out, h, n=6):
    det = [r for r in results if r["detection_idx"] is not None]
    mis = [r for r in results if r["detection_idx"] is None
           and r["true_onset_idx"] is not None]
    sample = (det[:n // 2] + mis[:n // 2])[:n]
    if not sample: sample = results[:n]

    nc = 2; nr = (len(sample) + 1) // nc
    fig, axes = plt.subplots(nr, nc, figsize=(14, 4 * nr))
    axes = np.array(axes).flatten()

    for i, r in enumerate(sample):
        ax = axes[i]; cu = r["cusums"]; nn = len(cu)
        xs = np.arange(nn)
        if r["true_onset_idx"] is not None:
            o = r["true_onset_idx"]
            ax.axvspan(0, o, alpha=0.08, color="#2ecc71")
            ax.axvspan(o, nn, alpha=0.08, color="#e74c3c")
        ax.plot(xs, cu, color="#2980b9", lw=1.8, label="CUSUM", zorder=3)
        ax.axhline(h, color="#e74c3c", lw=1.5, ls="--",
                   label=f"h={h:.1f}")
        if r["true_onset_idx"] is not None:
            ax.axvline(r["true_onset_idx"], color="#e74c3c", lw=2,
                       ls="-.", label="Onset")
        if r["detection_idx"] is not None:
            ax.axvline(r["detection_idx"], color="#f39c12", lw=2,
                       label=f"Det(lat={r['latency']:+d})")
        sf = (f" [{r['latency']:+d}]" if r["latency"] is not None
              else " [MISS]")
        ax.set_title(f"Eng {r['engine_id']}{sf}", fontsize=10,
                     fontweight="bold")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3); ax.set_xlim([0, nn])

    for j in range(len(sample), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("CUSUM Traces (Engine-Adaptive)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    p = os.path.join(out, "cusum_traces.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[OK] {p}")


def plot_errors(results, mu, sigma, out):
    he, de = [], []
    for r in results:
        o = r["true_onset_idx"]; e = r["errors"]
        if o is not None:
            he.extend(e[:o].tolist()); de.extend(e[o:].tolist())
        else:
            he.extend(e.tolist())
    fig, ax = plt.subplots(figsize=(10, 5))
    kw = dict(bins=60, density=True, alpha=0.7)
    if he: ax.hist(he, **kw, color="#2980b9", label="Healthy")
    if de: ax.hist(de, **kw, color="#e74c3c", label="Degraded")
    ax.axvline(mu, color="navy", lw=2, label=f"mu={mu:.4f}")
    for n, c in [(1, "#f1c40f"), (2, "#e67e22"), (3, "#e74c3c")]:
        ax.axvline(mu + n * sigma, color=c, lw=1.5, ls="--")
    ax.set_xlabel("MSE"); ax.set_ylabel("Density")
    ax.set_title("Error Distribution", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(out, "error_distribution.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[OK] {p}")


def plot_severity(results, out, n=4):
    sample = results[:n]
    fig, axes = plt.subplots(len(sample), 1,
                             figsize=(14, 3 * len(sample)))
    if len(sample) == 1: axes = [axes]
    sv = {"HEALTHY": 0, "MILD": 1, "MODERATE": 2, "SEVERE": 3}
    for ax, r in zip(axes, sample):
        vals = np.array([sv.get(s, 0) for s in r["severities"]])
        xs = np.arange(len(vals))
        cols = [SEVERITY_COLORS.get(s, "gray") for s in r["severities"]]
        ax.bar(xs, vals + 0.5, color=cols, width=1, align="edge",
               alpha=0.85)
        if r["true_onset_idx"] is not None:
            ax.axvline(r["true_onset_idx"], color="black", lw=2,
                       ls="-.", label="Onset")
        if r["detection_idx"] is not None:
            ax.axvline(r["detection_idx"], color="#f39c12", lw=2,
                       label=f"Det(lat={r['latency']:+d})")
        ax.set_yticks([0.25, 1.25, 2.25, 3.25])
        ax.set_yticklabels(SEVERITY_ORDER, fontsize=9)
        ax.set_xlim([0, len(vals)]); ax.set_ylim([0, 4])
        ax.set_title(f"Engine {r['engine_id']}", fontweight="bold",
                     fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.2, axis="x")
    plt.tight_layout()
    p = os.path.join(out, "severity_timeline.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[OK] {p}")


def plot_thresh(yt, ys, out):
    ths = np.linspace(ys.min(), ys.max(), 200)
    tprs, fprs, f1s, precs = [], [], [], []
    for t in ths:
        yp = (ys >= t).astype(int)
        tp = int(((yp == 1) & (yt == 1)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())
        tn = int(((yp == 0) & (yt == 0)).sum())
        tr = tp / max(tp + fn, 1); fr = fp / max(fp + tn, 1)
        pr = tp / max(tp + fp, 1)
        f1 = 2 * pr * tr / max(pr + tr, 1e-9)
        tprs.append(tr); fprs.append(fr)
        f1s.append(f1); precs.append(pr)
    tprs = np.array(tprs); fprs = np.array(fprs)
    f1s = np.array(f1s); precs = np.array(precs)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ths, tprs, color="#2ecc71", lw=2, label="TPR")
    ax.plot(ths, fprs, color="#e74c3c", lw=2, label="FPR")
    ax.plot(ths, f1s, color="#2980b9", lw=2, label="F1")
    ax.plot(ths, precs, color="#8e44ad", lw=2, label="Prec", alpha=0.7)
    bi = np.argmax(f1s)
    ax.axvline(ths[bi], color="#f39c12", lw=1.5, ls="--",
               label=f"Best F1 @ z={ths[bi]:.2f}")
    ax.set_xlabel("Z-threshold"); ax.set_ylabel("Metric")
    ax.set_title("Threshold Sensitivity", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.02, 1.05])
    plt.tight_layout()
    p = os.path.join(out, "threshold_sensitivity.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[OK] {p}")

def plot_policy_sensitivity(rows, out):
    if not rows:
        return

    xs = [r["max_valid_early"] for r in rows]
    actionable = [r["actionable_rate"] for r in rows]
    too_early = [r["too_early_rate"] for r in rows]
    late = [r["late_rate"] for r in rows]
    missed = [r["missed_rate"] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xs, actionable, marker="o", lw=2, label="Actionable rate")
    ax.plot(xs, too_early, marker="o", lw=2, label="Too-early rate")
    ax.plot(xs, late, marker="o", lw=2, label="Late rate")
    ax.plot(xs, missed, marker="o", lw=2, label="Missed rate")

    ax.set_xlabel("MAX_VALID_EARLY")
    ax.set_ylabel("Rate")
    ax.set_title("Policy Sensitivity", fontweight="bold")
    ax.set_ylim([-0.02, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    p = os.path.join(out, "policy_sensitivity.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {p}")

# ── REPORT ─────────────────────────────────────────────────────────────
def write_report(em, wm, roc, pr, diags, sev, dep, policy_rows, cfg, out):
    L = []
    S = "=" * 70
    S2 = "-" * 70

    def s(t=""):
        L.append(t)

    def h(t):
        L.extend([S, f"  {t}", S])

    def h2(t):
        L.extend([S2, f"  {t}", S2])

    h("EARLY WARNING DETECTOR -- EVALUATION REPORT")
    s(f"  h={cfg['threshold_h']:.4f} drift_k={cfg['drift_k']} "
      f"decay={cfg['decay']} warmup={cfg['warmup']} "
      f"healthy_ratio={cfg['healthy_ratio']} engines={cfg['n_engines']}")
    s(f"  Engine-adaptive baseline: first {cfg['warmup']} windows per engine")
    s()

    h2("PRIMARY METRICS (Window-Level)")
    total_w = wm['tn'] + wm['fp'] + wm['fn'] + wm['tp']
    s(f"  Total windows: {total_w:,}")
    s()
    s(f"  Window TPR       : {wm['tpr']:.3f}  (target >= 0.90) "
      f"[{'PASS' if wm['tpr'] >= 0.90 else 'FAIL'}]")
    s(f"  Window FPR       : {wm['fpr']:.3f}  (target <= 0.15) "
      f"[{'PASS' if wm['fpr'] <= 0.15 else 'FAIL'}]")
    s(f"  Window Precision : {wm['precision']:.3f}")
    s(f"  Window F1        : {wm['f1']:.3f}  (target >= 0.70) "
      f"[{'PASS' if wm['f1'] >= 0.70 else 'FAIL'}]")
    s()
    s(f"  Confusion Matrix (z={roc['opt_threshold']:.3f}):")
    s(f"    TN={wm['tn']:>6,}  FP={wm['fp']:>5,}")
    s(f"    FN={wm['fn']:>6,}  TP={wm['tp']:>5,}")
    s()

    h2("SECONDARY METRICS (Engine-Level)")
    s(f"  Detections > {MAX_VALID_EARLY} cycles before onset = FP")
    s(f"  Degraded engines : {em['degraded_engines']}")
    s(f"  Valid detections : {em['valid_detections']}")
    s(f"  Too-early (FP)   : {em['too_early_count']}")
    s(f"  Missed (FN)      : {em['fn']}")
    s()
    s(f"  Engine TPR  : {em['tpr']:.3f}  "
      f"[{'PASS' if em['tpr'] >= 0.85 else 'FAIL'}]")
    s(f"  Engine FPR  : {em['fpr_engine']:.3f}  "
      f"[{'PASS' if em['fpr_engine'] <= 0.20 else 'FAIL'}]")
    s(f"  Engine Prec : {em['precision']:.3f}")
    s(f"  Engine F1   : {em['f1']:.3f}  "
      f"[{'PASS' if em['f1'] >= 0.75 else 'FAIL'}]")
    s()

    h2("DETECTION LATENCY (existing engine metric)")
    if len(em["latencies"]) > 0:
        lat_ok = ("PASS" if 20 <= em["mean_latency"] <= 55 else "~")
        s(f"  Mean   : {em['mean_latency']:+.1f} (target +20..+55) [{lat_ok}]")
        s(f"  Median : {em['median_latency']:+.1f}")
        s(f"  Std    : {em['std_latency']:.1f}")
        s(f"  Range  : [{em['min_latency']:+.0f}, {em['max_latency']:+.0f}]")
        s(f"  Early={em['early_count']} Late={em['late_count']} Exact={em['exact_count']}")
    else:
        s("  No valid detections.")
    if len(em.get("all_latencies", [])) > 0:
        s(f"  All latencies mean: {em['all_mean_latency']:+.1f}")
    s()

    h2("OPERATIONAL USEFULNESS (strict deployability view)")
    s(f"  Actionable first alerts [0..{MAX_VALID_EARLY}] : "
      f"{dep['useful_count']} / {dep['degraded_engines']} "
      f"({100*dep['actionable_rate']:.1f}%)")
    s(f"  Too-early alerts  (>{MAX_VALID_EARLY})         : "
      f"{dep['too_early_count']} / {dep['degraded_engines']} "
      f"({100*dep['too_early_rate']:.1f}%)")
    s(f"  Late alerts       (<0)                         : "
      f"{dep['late_count']} / {dep['degraded_engines']} "
      f"({100*dep['late_rate']:.1f}%)")
    s(f"  Missed detections                              : "
      f"{dep['missed_count']} / {dep['degraded_engines']} "
      f"({100*dep['missed_rate']:.1f}%)")
    if not np.isnan(dep["mean_actionable_latency"]):
        s(f"  Mean actionable lead time                     : "
          f"{dep['mean_actionable_latency']:+.1f}")
        s(f"  Median actionable lead time                   : "
          f"{dep['median_actionable_latency']:+.1f}")
    s()

    h2("ALARM TRUST")
    s(f"  First-alert precision                         : "
      f"{dep['first_alert_precision']:.3f}")
    s(f"  False alarms per 100 healthy windows          : "
      f"{dep['false_alarms_per_100_healthy_windows']:.2f}")
    s(f"  Avg alarms per engine                         : "
      f"{dep['avg_alarms_per_engine']:.2f}")
    s(f"  Avg pre-onset alarms / degraded engine        : "
      f"{dep['avg_pre_onset_alarms_per_degraded_engine']:.2f}")
    s()

    h2("POLICY SENSITIVITY")
    for row in policy_rows:
        s(f"  max_valid_early={row['max_valid_early']:>3d} | "
          f"useful={row['useful_count']:>2d} "
          f"too_early={row['too_early_count']:>2d} "
          f"late={row['late_count']:>2d} "
          f"missed={row['missed_count']:>2d} | "
          f"Prec={row['precision']:.3f} "
          f"Rec={row['recall']:.3f} "
          f"F1={row['f1']:.3f}")
    s()

    h2("ROC / PR")
    s(f"  AUC={roc['auc_roc']:.4f}  "
      f"[{'PASS' if roc['auc_roc'] >= 0.85 else 'FAIL'}]  "
      f"Opt z={roc['opt_threshold']:.3f} J={roc['opt_j']:.3f}")
    s(f"  AP={pr['avg_precision']:.4f}  "
      f"Best F1={pr['best_f1']:.4f} (z={pr['best_threshold']:.3f})")
    s()

    h2("SEVERITY")
    tw = sum(sev.values())
    for sv in SEVERITY_ORDER:
        c = sev.get(sv, 0)
        s(f"  {sv:10s}: {c:6,} ({100 * c / max(tw, 1):5.1f}%)")
    s()

    h2("DIAGNOSIS")
    if not diags:
        s("  All detections within acceptable range.")
    else:
        for d in diags:
            s(f"  Eng {d['eid']:3d} [{d['issue']}] {d['desc']}")
            if d["lat"] is not None:
                s(f"    lat={d['lat']:+d} z@onset={d['zo']:.3f} "
                  f"cusum@onset={d['co']:.3f}")
            s(f"    -> {d['rec']}")
            s()

    h("TARGET SUMMARY")
    tgts = [
        ("Window TPR >= 0.90", wm["tpr"] >= 0.90, f"{wm['tpr']:.3f}"),
        ("Window FPR <= 0.15", wm["fpr"] <= 0.15, f"{wm['fpr']:.3f}"),
        ("Window F1  >= 0.70", wm["f1"] >= 0.70, f"{wm['f1']:.3f}"),
        ("AUC-ROC    >= 0.85", roc["auc_roc"] >= 0.85, f"{roc['auc_roc']:.3f}"),
        ("Engine TPR >= 0.85", em["tpr"] >= 0.85, f"{em['tpr']:.3f}"),
        ("Engine F1  >= 0.75", em["f1"] >= 0.75, f"{em['f1']:.3f}"),
    ]
    if len(em["latencies"]) > 0:
        tgts.append(("Latency +20..+55",
                     20 <= em["mean_latency"] <= 55,
                     f"{em['mean_latency']:+.1f}"))

    ap = True
    for nm, ok, v in tgts:
        ap = ap and ok
        s(f"  [{'PASS' if ok else 'FAIL'}] {nm:<24s} = {v}")
    s()
    s(f"  Overall: {'ALL PASS' if ap else 'SOME FAILED'}")
    s()

    report = "\n".join(L)
    p = os.path.join(out, "metrics_summary.txt")
    with open(p, "w", encoding="utf-8", errors="replace") as f:
        f.write(report)
    print(f"[OK] Report: {p}\n")
    print(report)
    return report

def split_by_engine(X, y, eids):
    engines = []
    for eid in np.unique(eids):
        m = eids == eid; idx = np.where(m)[0]
        o = np.argsort(y[idx])[::-1]
        engines.append((X[idx[o]], y[idx[o]], int(eid)))
    return engines


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, mu, sigma, h, dk, decay, wu, si = \
        load_model_and_baseline(args.model, args.baseline, device)

    print(f"\nLoading: {args.dataset}")
    data = np.load(args.dataset, allow_pickle=True)
    X_test, y_test = data["X_test"], data["y_test"]
    fnames = data["feature_names"].tolist()

    if "engine_ids_test" in data:
        eids = data["engine_ids_test"]
    else:
        eids = np.zeros(len(y_test), dtype=int); eid = 0
        for i in range(1, len(y_test)):
            if y_test[i] > y_test[i - 1]: eid += 1
            eids[i] = eid

    Xsel = X_test[:, :, si] if len(si) > 0 else X_test
    print(f"Test: {Xsel.shape}  Engines: {len(np.unique(eids))}")
    engines = split_by_engine(Xsel, y_test, eids)
    hr = args.healthy_ratio

    print(f"\nEvaluating {len(engines)} engines...")
    results = []
    for Xe, re, eid in engines:
        r = run_engine_evaluation(
            Xe, re, eid, model, mu, sigma,
            h, dk, decay, wu, hr, device)
        results.append(r)
        st = (f"lat={r['latency']:+d}" if r['latency'] is not None
              else "MISS" if r['true_onset_idx'] is not None
              else "no_onset")
        print(f"  Eng {eid:3d}: {len(Xe):4d}w "
              f"onset={r['true_onset_idx']} "
              f"det={r['detection_idx']} {st}")

    print("\nMetrics...")
    em = compute_engine_metrics(results, hr, max_early=MAX_VALID_EARLY)
    dep = compute_deployability_metrics(results, hr, max_early=MAX_VALID_EARLY)
    policy_rows = compute_policy_sensitivity(results, policy_values=POLICY_SWEEP)

    yt, ys = build_window_labels(results, hr)
    print(f"Windows: {(yt == 1).sum()} degraded / "
          f"{(yt == 0).sum()} healthy")

    roc = compute_roc(yt, ys)
    prs = compute_pr(yt, ys)
    wm = cm_stats(yt, ys, roc["opt_threshold"])
    diags = diagnose(results, hr, max_early=MAX_VALID_EARLY)
    sv = sev_dist(results)

    print(f"Actionable first alerts [0..{MAX_VALID_EARLY}] : "
          f"{dep['useful_count']}/{dep['degraded_engines']}")
    print(f"Too-early / Late / Missed                     : "
          f"{dep['too_early_count']}/{dep['late_count']}/{dep['missed_count']}")
    print(f"False alarms per 100 healthy windows          : "
          f"{dep['false_alarms_per_100_healthy_windows']:.2f}")

    print("\nPlots...")
    plot_roc(roc, prs, args.out_dir)
    plot_latency(em, args.out_dir)
    plot_cusum(results, args.out_dir, h, n=args.n_trace_engines)
    plot_errors(results, mu, sigma, args.out_dir)
    plot_severity(results, args.out_dir, n=min(4, len(results)))
    plot_thresh(yt, ys, args.out_dir)
    plot_policy_sensitivity(policy_rows, args.out_dir)

    save_csv_rows(
        dep["first_alert_rows"],
        os.path.join(args.out_dir, "first_alert_table.csv")
    )
    save_csv_rows(
        policy_rows,
        os.path.join(args.out_dir, "policy_sensitivity.csv")
    )

    write_report(
        em, wm, roc, prs, diags, sv, dep, policy_rows,
        {"threshold_h": h, "drift_k": dk, "decay": decay,
         "warmup": wu, "healthy_ratio": hr,
         "n_engines": len(engines)},
        args.out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",
                   default="data/processed/FD001_preprocessed.npz")
    p.add_argument("--model",
                   default="results/saved_models/vae_early_warning.pth")
    p.add_argument("--baseline",
                   default="results/saved_models/baseline_stats.npz")
    p.add_argument("--out_dir", default="results/evaluation")
    p.add_argument("--healthy_ratio", type=float, default=0.80)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--n_trace_engines", type=int, default=6)
    main(p.parse_args())
    