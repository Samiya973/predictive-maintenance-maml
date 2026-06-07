
# """
# evaluate_early_warning.py
# ─────────────────────────
# Evaluation suite for VAE Early Warning Detector.
# Window-level as primary metrics. Honest engine-level with adaptive CUSUM.
# """

# import os, sys, argparse, csv
# from collections import defaultdict

# if hasattr(sys.stdout, "reconfigure"):
#     sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# import numpy as np
# import torch
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
#                               average_precision_score, confusion_matrix)

# sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
# from src.models.early_warning_vae import (
#     TimeSeriesVAE, ChangePointDetector, EarlyWarningSystem,
#     classify_severity, USEFUL_SENSORS, N_SENSORS,
# )

# SEVERITY_ORDER = ["HEALTHY", "MILD", "MODERATE", "SEVERE"]
# SEVERITY_COLORS = {"HEALTHY": "#2ecc71", "MILD": "#f1c40f",
#                     "MODERATE": "#e67e22", "SEVERE": "#e74c3c"}
# MAX_VALID_EARLY = 90
# POLICY_SWEEP = [55, 70, 90, 100]

# def load_model_and_baseline(model_path, baseline_path, device):
#     ckpt = torch.load(model_path, map_location=device, weights_only=False)
#     config = ckpt["config"]
#     model = TimeSeriesVAE(config["input_size"], config["seq_len"],
#                           config["hidden_size"], config["latent_dim"])
#     model.load_state_dict(ckpt["model_state_dict"])
#     model.to(device).eval()

#     st = np.load(baseline_path, allow_pickle=True)
#     mu = float(st["baseline_mean"])
#     sigma = float(st["baseline_std"])
#     h = float(st["threshold_h"]) if "threshold_h" in st else 4.0
#     dk = float(st["drift_k"]) if "drift_k" in st else 0.5
#     decay = float(st["decay"]) if "decay" in st else 0.95
#     wu = int(st["warmup"]) if "warmup" in st else 20
#     si = (st["sensor_indices"].tolist() if "sensor_indices" in st
#           else list(range(N_SENSORS)))

#     print(f"VAE: {sum(p.numel() for p in model.parameters()):,} params")
#     print(f"Baseline: mu={mu:.6f} sigma={sigma:.6f}")
#     print(f"CUSUM: h={h:.4f} drift_k={dk} decay={decay} warmup={wu}")
#     return model, mu, sigma, h, dk, decay, wu, si


# def run_engine_evaluation(engine_windows, engine_rul, engine_id,
#                            model, mu, sigma, h, dk, decay, wu,
#                            healthy_ratio, device):
#     ews = EarlyWarningSystem(
#         vae=model, baseline_mean=mu, baseline_std=sigma,
#         drift_k=dk, threshold_h=h, warmup=wu,
#         decay=decay, device=str(device))

#     true_onset = None
#     for i, rul in enumerate(engine_rul):
#         if rul <= (1.0 - healthy_ratio):
#             true_onset = i
#             break

#     errors, zs, cs, als, sevs = [], [], [], [], []
#     for idx, x_np in enumerate(engine_windows):
#         r = ews.monitor(torch.FloatTensor(x_np).unsqueeze(0), cycle=idx)
#         errors.append(r["error"]); zs.append(r["z_score"])
#         cs.append(r["cusum"]); als.append(r["alarm"])
#         sevs.append(r["severity"])

#     det = ews.get_detection_cycle()
#     lat = ((true_onset - det) if (true_onset is not None and
#             det is not None) else None)

#     return {"engine_id": engine_id, "n_windows": len(engine_windows),
#             "true_onset_idx": true_onset, "detection_idx": det,
#             "latency": lat, "errors": np.array(errors),
#             "z_scores": np.array(zs), "cusums": np.array(cs),
#             "alarms": np.array(als), "severities": sevs,
#             "rul": engine_rul}


# def build_window_labels(results, hr):
#     yt, ys = [], []
#     for r in results:
#         for rul, z in zip(r["rul"], r["z_scores"]):
#             yt.append(1 if rul <= (1.0 - hr) else 0)
#             ys.append(float(z))
#     return np.array(yt, dtype=int), np.array(ys, dtype=float)


# def compute_engine_metrics(results, healthy_ratio,
#                            max_early=MAX_VALID_EARLY):
#     tp = fn = fp = tn = 0
#     valid_lats = []
#     all_lats = []
#     degraded = 0

#     for r in results:
#         has_on = r["true_onset_idx"] is not None
#         has_det = r["detection_idx"] is not None
#         lat = r["latency"]

#         if has_on:
#             degraded += 1
#             if has_det and lat is not None:
#                 all_lats.append(lat)
#                 if -15 <= lat <= max_early:
#                     tp += 1
#                     valid_lats.append(lat)
#                 elif lat > max_early:
#                     fp += 1
#                 else:
#                     fn += 1
#             else:
#                 fn += 1
#         else:
#             if has_det:
#                 fp += 1
#             else:
#                 tn += 1

#     tpr = tp / max(tp + fn, 1)
#     fpr = fp / max(fp + tn, 1)   # fixed
#     prec = tp / max(tp + fp, 1)
#     f1 = 2 * prec * tpr / max(prec + tpr, 1e-9)

#     vl = np.array(valid_lats, dtype=float)
#     al = np.array(all_lats, dtype=float)

#     return {
#         "tp": tp, "fn": fn, "fp": fp, "tn": tn,
#         "tpr": tpr, "fpr_engine": fpr, "precision": prec, "f1": f1,
#         "degraded_engines": degraded,
#         "valid_detections": tp,
#         "too_early_count": fp,
#         "latencies": vl,
#         "mean_latency": float(vl.mean()) if len(vl) > 0 else float("nan"),
#         "median_latency": float(np.median(vl)) if len(vl) > 0 else float("nan"),
#         "std_latency": float(vl.std()) if len(vl) > 0 else float("nan"),
#         "min_latency": float(vl.min()) if len(vl) > 0 else float("nan"),
#         "max_latency": float(vl.max()) if len(vl) > 0 else float("nan"),
#         "early_count": int((vl > 0).sum()) if len(vl) > 0 else 0,
#         "late_count": int((vl < 0).sum()) if len(vl) > 0 else 0,
#         "exact_count": int((vl == 0).sum()) if len(vl) > 0 else 0,
#         "all_latencies": al,
#         "all_mean_latency": float(al.mean()) if len(al) > 0 else float("nan"),
#     }

# def compute_roc(yt, ys):
#     fpr, tpr, th = roc_curve(yt, ys, drop_intermediate=False)
#     a = auc(fpr, tpr); j = tpr - fpr; oi = np.argmax(j)
#     return {"fpr": fpr, "tpr": tpr, "thresholds": th, "auc_roc": a,
#             "opt_threshold": float(th[oi]), "opt_tpr": float(tpr[oi]),
#             "opt_fpr": float(fpr[oi]), "opt_j": float(j[oi])}


# def compute_pr(yt, ys):
#     p, r, th = precision_recall_curve(yt, ys)
#     ap = average_precision_score(yt, ys)
#     f1 = 2 * p * r / np.maximum(p + r, 1e-9); bi = np.argmax(f1)
#     return {"precision": p, "recall": r, "thresholds": th,
#             "avg_precision": ap, "best_f1": float(f1[bi]),
#             "best_threshold": (float(th[bi]) if bi < len(th)
#                                else float("nan")),
#             "best_precision": float(p[bi]),
#             "best_recall": float(r[bi])}


# def cm_stats(yt, ys, thr):
#     yp = (ys >= thr).astype(int)
#     cm = confusion_matrix(yt, yp, labels=[0, 1])
#     tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
#     tpr = tp / max(tp + fn, 1); fpr = fp / max(fp + tn, 1)
#     ppv = tp / max(tp + fp, 1); f1 = 2 * ppv * tpr / max(ppv + tpr, 1e-9)
#     return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
#             "tpr": tpr, "fpr": fpr, "precision": ppv, "f1": f1}


# def diagnose(results, hr, max_early=MAX_VALID_EARLY):
#     diags = []
#     for r in results:
#         to = r["true_onset_idx"]; det = r["detection_idx"]
#         lat = r["latency"]
#         if to is None: continue
#         zo = float(r["z_scores"][to]); co = float(r["cusums"][to])
#         if det is None:
#             diags.append({"eid": r["engine_id"], "issue": "MISSED",
#                           "desc": "Never detected", "to": to, "det": None,
#                           "lat": None, "zo": zo, "co": co,
#                           "rec": "Lower threshold_h"})
#         elif lat is not None and lat < -15:
#             diags.append({"eid": r["engine_id"], "issue": "LATE",
#                           "desc": f"{abs(lat)} cycles after onset",
#                           "to": to, "det": det, "lat": lat, "zo": zo,
#                           "co": co, "rec": "Lower threshold_h"})
#         elif lat is not None and lat > max_early:
#             diags.append({"eid": r["engine_id"], "issue": "TOO_EARLY",
#                           "desc": f"{lat} cycles before onset",
#                           "to": to, "det": det, "lat": lat, "zo": zo,
#                           "co": co, "rec": "Check VAE generalization"})
#     return diags

# def apply_alarm_policy(alarms, cooldown=None):
#     """
#     Convert raw per-window alarms into deployable alert episodes.

#     cooldown = None  -> one-shot policy (only first alert kept)
#     cooldown = int   -> keep first alert, then suppress next `cooldown` windows
#     """
#     alarms = np.asarray(alarms, dtype=int)
#     policy_alarms = np.zeros_like(alarms)

#     fired_once = False
#     cooldown_left = 0

#     for i, a in enumerate(alarms):
#         if cooldown_left > 0:
#             cooldown_left -= 1
#             continue

#         if a == 1:
#             if cooldown is None:
#                 if not fired_once:
#                     policy_alarms[i] = 1
#                     fired_once = True
#             else:
#                 policy_alarms[i] = 1
#                 cooldown_left = int(cooldown)

#     return policy_alarms

# def compute_deployability_metrics(results, hr, max_early=MAX_VALID_EARLY):
#     useful = too_early = late = missed = 0
#     degraded = 0

#     useful_lats = []
#     total_alert_episodes = 0
#     healthy_alert_episodes = 0
#     healthy_window_count = 0
#     pre_onset_alert_episodes = 0

#     first_alert_rows = []

#     # None = one-shot first alert only
#     # set something like 20 if you want cooldown mode instead
#     ALARM_COOLDOWN = None

#     for r in results:
#         eid = r["engine_id"]
#         to = r["true_onset_idx"]
#         det = r["detection_idx"]
#         lat = r["latency"]

#         raw_alarms = np.asarray(r["alarms"], dtype=int)
#         rul = np.asarray(r["rul"], dtype=float)

#         # convert raw alarm windows into deployable alert episodes
#         alarm_events = apply_alarm_policy(raw_alarms, cooldown=ALARM_COOLDOWN)

#         total_alarm_count = int(alarm_events.sum())
#         total_alert_episodes += total_alarm_count

#         healthy_mask = rul > (1.0 - hr)
#         healthy_window_count += int(healthy_mask.sum())
#         healthy_alert_episodes += int(alarm_events[healthy_mask].sum())

#         pre_onset_n = int(alarm_events[:to].sum()) if to is not None and to > 0 else 0

#         if to is not None:
#             degraded += 1
#             pre_onset_alert_episodes += pre_onset_n

#             if det is None or lat is None:
#                 status = "MISSED"
#                 missed += 1
#             elif lat > max_early:
#                 status = "TOO_EARLY"
#                 too_early += 1
#             elif lat < 0:
#                 status = "LATE"
#                 late += 1
#             else:
#                 status = "USEFUL"
#                 useful += 1
#                 useful_lats.append(lat)
#         else:
#             status = "NO_ONSET"

#         first_alert_rows.append({
#             "engine_id": int(eid),
#             "onset_idx": (int(to) if to is not None else ""),
#             "detection_idx": (int(det) if det is not None else ""),
#             "latency": (int(lat) if lat is not None else ""),
#             "status": status,
#             "pre_onset_alarm_count": int(pre_onset_n),
#             "total_alarm_count": int(total_alarm_count),
#         })

#     useful_lats = np.array(useful_lats, dtype=float)
#     detected = useful + too_early + late

#     return {
#         "degraded_engines": degraded,
#         "useful_count": useful,
#         "too_early_count": too_early,
#         "late_count": late,
#         "missed_count": missed,
#         "actionable_rate": useful / max(degraded, 1),
#         "too_early_rate": too_early / max(degraded, 1),
#         "late_rate": late / max(degraded, 1),
#         "missed_rate": missed / max(degraded, 1),
#         "first_alert_precision": useful / max(detected, 1),
#         "mean_actionable_latency": (
#             float(useful_lats.mean()) if len(useful_lats) > 0 else float("nan")
#         ),
#         "median_actionable_latency": (
#             float(np.median(useful_lats)) if len(useful_lats) > 0 else float("nan")
#         ),
#         "false_alarms_per_100_healthy_windows": (
#             100.0 * healthy_alert_episodes / max(healthy_window_count, 1)
#         ),
#         "avg_alarms_per_engine": total_alert_episodes / max(len(results), 1),
#         "avg_pre_onset_alarms_per_degraded_engine": (
#             pre_onset_alert_episodes / max(degraded, 1)
#         ),
#         "first_alert_rows": first_alert_rows,
#     }

# def compute_policy_sensitivity(results, policy_values=POLICY_SWEEP):
#     rows = []
#     degraded = sum(1 for r in results if r["true_onset_idx"] is not None)

#     for max_early in policy_values:
#         useful = too_early = late = missed = 0

#         for r in results:
#             if r["true_onset_idx"] is None:
#                 continue

#             det = r["detection_idx"]
#             lat = r["latency"]

#             if det is None or lat is None:
#                 missed += 1
#             elif lat > max_early:
#                 too_early += 1
#             elif lat < 0:
#                 late += 1
#             else:
#                 useful += 1

#         precision = useful / max(useful + too_early + late, 1)
#         recall = useful / max(degraded, 1)
#         f1 = 2 * precision * recall / max(precision + recall, 1e-9)

#         rows.append({
#             "max_valid_early": int(max_early),
#             "useful_count": int(useful),
#             "too_early_count": int(too_early),
#             "late_count": int(late),
#             "missed_count": int(missed),
#             "actionable_rate": recall,
#             "too_early_rate": too_early / max(degraded, 1),
#             "late_rate": late / max(degraded, 1),
#             "missed_rate": missed / max(degraded, 1),
#             "precision": precision,
#             "recall": recall,
#             "f1": f1,
#         })

#     return rows


# def save_csv_rows(rows, path):
#     if not rows:
#         return
#     with open(path, "w", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
#         writer.writeheader()
#         writer.writerows(rows)
#     print(f"[OK] {path}")

# def sev_dist(results):
#     c = defaultdict(int)
#     for r in results:
#         for s in r["severities"]: c[s] += 1
#     return dict(c)


# # ── PLOTS ──────────────────────────────────────────────────────────────
# def plot_roc(roc, pr, out):
#     fig, axes = plt.subplots(1, 2, figsize=(14, 6))
#     ax = axes[0]
#     ax.plot(roc["fpr"], roc["tpr"], color="#2980b9", lw=2.5,
#             label=f"AUC={roc['auc_roc']:.3f}")
#     ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
#     ax.scatter(roc["opt_fpr"], roc["opt_tpr"], color="#e74c3c", s=120,
#                zorder=5, label=f"Opt z={roc['opt_threshold']:.2f}")
#     ax.axhline(0.9, color="#2ecc71", ls=":", lw=1.5, alpha=0.8)
#     ax.axvline(0.1, color="#e67e22", ls=":", lw=1.5, alpha=0.8)
#     ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
#     ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
#     ax.set_title("ROC", fontweight="bold")
#     ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

#     ax = axes[1]
#     ax.plot(pr["recall"], pr["precision"], color="#8e44ad", lw=2.5,
#             label=f"AP={pr['avg_precision']:.3f}")
#     ax.scatter(pr["best_recall"], pr["best_precision"], color="#e74c3c",
#                s=120, zorder=5, label=f"F1={pr['best_f1']:.3f}")
#     ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
#     ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
#     ax.set_title("PR", fontweight="bold")
#     ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

#     plt.tight_layout()
#     p = os.path.join(out, "roc_pr_curves.png")
#     plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
#     print(f"[OK] {p}")


# def plot_latency(m, out):
#     al = m.get("all_latencies", m["latencies"])
#     if len(al) == 0: return
#     fig, axes = plt.subplots(1, 2, figsize=(14, 5))

#     ax = axes[0]
#     if (al > 0).any():
#         ax.hist(al[al > 0], bins=20, color="#2ecc71", alpha=0.8,
#                 label=f"Early ({(al > 0).sum()})")
#     if (al <= 0).any():
#         ax.hist(al[al <= 0], bins=20, color="#e74c3c", alpha=0.8,
#                 label=f"Late ({(al <= 0).sum()})")
#     ax.axvline(0, color="black", lw=2)
#     ax.axvline(30, color="#f39c12", lw=2, ls="--", label="Target +30")
#     ax.axvline(50, color="#f39c12", lw=2, ls=":", label="Target +50")
#     ax.axvspan(30, 50, alpha=0.12, color="#f39c12")
#     ax.axvline(MAX_VALID_EARLY, color="purple", lw=1.5, ls="-.",
#                label=f"Max valid (+{MAX_VALID_EARLY})")
#     ax.set_xlabel("Latency (cycles)"); ax.set_ylabel("Count")
#     ax.set_title(f"All Latencies (mean={m['all_mean_latency']:.1f})",
#                  fontweight="bold")
#     ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

#     ax = axes[1]
#     sl = np.sort(al); cdf = np.arange(1, len(sl) + 1) / len(sl)
#     ax.plot(sl, cdf, color="#2980b9", lw=2.5)
#     ax.axvline(0, color="black", lw=1.5)
#     ax.axvline(30, color="#f39c12", lw=1.5, ls="--")
#     ax.axvline(50, color="#f39c12", lw=1.5, ls=":")
#     ax.axvspan(30, 50, alpha=0.12, color="#f39c12")
#     ax.set_xlabel("Latency"); ax.set_ylabel("CDF")
#     ax.set_ylim([0, 1.05]); ax.set_title("CDF", fontweight="bold")
#     ax.grid(True, alpha=0.3)

#     plt.tight_layout()
#     p = os.path.join(out, "detection_latency.png")
#     plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
#     print(f"[OK] {p}")


# def plot_cusum(results, out, h, n=6):
#     det = [r for r in results if r["detection_idx"] is not None]
#     mis = [r for r in results if r["detection_idx"] is None
#            and r["true_onset_idx"] is not None]
#     sample = (det[:n // 2] + mis[:n // 2])[:n]
#     if not sample: sample = results[:n]

#     nc = 2; nr = (len(sample) + 1) // nc
#     fig, axes = plt.subplots(nr, nc, figsize=(14, 4 * nr))
#     axes = np.array(axes).flatten()

#     for i, r in enumerate(sample):
#         ax = axes[i]; cu = r["cusums"]; nn = len(cu)
#         xs = np.arange(nn)
#         if r["true_onset_idx"] is not None:
#             o = r["true_onset_idx"]
#             ax.axvspan(0, o, alpha=0.08, color="#2ecc71")
#             ax.axvspan(o, nn, alpha=0.08, color="#e74c3c")
#         ax.plot(xs, cu, color="#2980b9", lw=1.8, label="CUSUM", zorder=3)
#         ax.axhline(h, color="#e74c3c", lw=1.5, ls="--",
#                    label=f"h={h:.1f}")
#         if r["true_onset_idx"] is not None:
#             ax.axvline(r["true_onset_idx"], color="#e74c3c", lw=2,
#                        ls="-.", label="Onset")
#         if r["detection_idx"] is not None:
#             ax.axvline(r["detection_idx"], color="#f39c12", lw=2,
#                        label=f"Det(lat={r['latency']:+d})")
#         sf = (f" [{r['latency']:+d}]" if r["latency"] is not None
#               else " [MISS]")
#         ax.set_title(f"Eng {r['engine_id']}{sf}", fontsize=10,
#                      fontweight="bold")
#         ax.legend(fontsize=7, loc="upper left")
#         ax.grid(True, alpha=0.3); ax.set_xlim([0, nn])

#     for j in range(len(sample), len(axes)):
#         axes[j].set_visible(False)
#     plt.suptitle("CUSUM Traces (Engine-Adaptive)",
#                  fontsize=14, fontweight="bold", y=1.01)
#     plt.tight_layout()
#     p = os.path.join(out, "cusum_traces.png")
#     plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
#     print(f"[OK] {p}")


# def plot_errors(results, mu, sigma, out):
#     he, de = [], []
#     for r in results:
#         o = r["true_onset_idx"]; e = r["errors"]
#         if o is not None:
#             he.extend(e[:o].tolist()); de.extend(e[o:].tolist())
#         else:
#             he.extend(e.tolist())
#     fig, ax = plt.subplots(figsize=(10, 5))
#     kw = dict(bins=60, density=True, alpha=0.7)
#     if he: ax.hist(he, **kw, color="#2980b9", label="Healthy")
#     if de: ax.hist(de, **kw, color="#e74c3c", label="Degraded")
#     ax.axvline(mu, color="navy", lw=2, label=f"mu={mu:.4f}")
#     for n, c in [(1, "#f1c40f"), (2, "#e67e22"), (3, "#e74c3c")]:
#         ax.axvline(mu + n * sigma, color=c, lw=1.5, ls="--")
#     ax.set_xlabel("MSE"); ax.set_ylabel("Density")
#     ax.set_title("Error Distribution", fontweight="bold")
#     ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
#     plt.tight_layout()
#     p = os.path.join(out, "error_distribution.png")
#     plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
#     print(f"[OK] {p}")


# def plot_severity(results, out, n=4):
#     sample = results[:n]
#     fig, axes = plt.subplots(len(sample), 1,
#                              figsize=(14, 3 * len(sample)))
#     if len(sample) == 1: axes = [axes]
#     sv = {"HEALTHY": 0, "MILD": 1, "MODERATE": 2, "SEVERE": 3}
#     for ax, r in zip(axes, sample):
#         vals = np.array([sv.get(s, 0) for s in r["severities"]])
#         xs = np.arange(len(vals))
#         cols = [SEVERITY_COLORS.get(s, "gray") for s in r["severities"]]
#         ax.bar(xs, vals + 0.5, color=cols, width=1, align="edge",
#                alpha=0.85)
#         if r["true_onset_idx"] is not None:
#             ax.axvline(r["true_onset_idx"], color="black", lw=2,
#                        ls="-.", label="Onset")
#         if r["detection_idx"] is not None:
#             ax.axvline(r["detection_idx"], color="#f39c12", lw=2,
#                        label=f"Det(lat={r['latency']:+d})")
#         ax.set_yticks([0.25, 1.25, 2.25, 3.25])
#         ax.set_yticklabels(SEVERITY_ORDER, fontsize=9)
#         ax.set_xlim([0, len(vals)]); ax.set_ylim([0, 4])
#         ax.set_title(f"Engine {r['engine_id']}", fontweight="bold",
#                      fontsize=10)
#         ax.legend(fontsize=8, loc="upper left")
#         ax.grid(True, alpha=0.2, axis="x")
#     plt.tight_layout()
#     p = os.path.join(out, "severity_timeline.png")
#     plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
#     print(f"[OK] {p}")


# def plot_thresh(yt, ys, out):
#     ths = np.linspace(ys.min(), ys.max(), 200)
#     tprs, fprs, f1s, precs = [], [], [], []
#     for t in ths:
#         yp = (ys >= t).astype(int)
#         tp = int(((yp == 1) & (yt == 1)).sum())
#         fp = int(((yp == 1) & (yt == 0)).sum())
#         fn = int(((yp == 0) & (yt == 1)).sum())
#         tn = int(((yp == 0) & (yt == 0)).sum())
#         tr = tp / max(tp + fn, 1); fr = fp / max(fp + tn, 1)
#         pr = tp / max(tp + fp, 1)
#         f1 = 2 * pr * tr / max(pr + tr, 1e-9)
#         tprs.append(tr); fprs.append(fr)
#         f1s.append(f1); precs.append(pr)
#     tprs = np.array(tprs); fprs = np.array(fprs)
#     f1s = np.array(f1s); precs = np.array(precs)

#     fig, ax = plt.subplots(figsize=(10, 5))
#     ax.plot(ths, tprs, color="#2ecc71", lw=2, label="TPR")
#     ax.plot(ths, fprs, color="#e74c3c", lw=2, label="FPR")
#     ax.plot(ths, f1s, color="#2980b9", lw=2, label="F1")
#     ax.plot(ths, precs, color="#8e44ad", lw=2, label="Prec", alpha=0.7)
#     bi = np.argmax(f1s)
#     ax.axvline(ths[bi], color="#f39c12", lw=1.5, ls="--",
#                label=f"Best F1 @ z={ths[bi]:.2f}")
#     ax.set_xlabel("Z-threshold"); ax.set_ylabel("Metric")
#     ax.set_title("Threshold Sensitivity", fontweight="bold")
#     ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
#     ax.set_ylim([-0.02, 1.05])
#     plt.tight_layout()
#     p = os.path.join(out, "threshold_sensitivity.png")
#     plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
#     print(f"[OK] {p}")

# def plot_policy_sensitivity(rows, out):
#     if not rows:
#         return

#     xs = [r["max_valid_early"] for r in rows]
#     actionable = [r["actionable_rate"] for r in rows]
#     too_early = [r["too_early_rate"] for r in rows]
#     late = [r["late_rate"] for r in rows]
#     missed = [r["missed_rate"] for r in rows]

#     fig, ax = plt.subplots(figsize=(10, 5))
#     ax.plot(xs, actionable, marker="o", lw=2, label="Actionable rate")
#     ax.plot(xs, too_early, marker="o", lw=2, label="Too-early rate")
#     ax.plot(xs, late, marker="o", lw=2, label="Late rate")
#     ax.plot(xs, missed, marker="o", lw=2, label="Missed rate")

#     ax.set_xlabel("MAX_VALID_EARLY")
#     ax.set_ylabel("Rate")
#     ax.set_title("Policy Sensitivity", fontweight="bold")
#     ax.set_ylim([-0.02, 1.05])
#     ax.grid(True, alpha=0.3)
#     ax.legend(fontsize=9)

#     plt.tight_layout()
#     p = os.path.join(out, "policy_sensitivity.png")
#     plt.savefig(p, dpi=150, bbox_inches="tight")
#     plt.close()
#     print(f"[OK] {p}")

# # ── REPORT ─────────────────────────────────────────────────────────────
# def write_report(em, wm, roc, pr, diags, sev, dep, policy_rows, cfg, out):
#     L = []
#     S = "=" * 70
#     S2 = "-" * 70

#     def s(t=""):
#         L.append(t)

#     def h(t):
#         L.extend([S, f"  {t}", S])

#     def h2(t):
#         L.extend([S2, f"  {t}", S2])

#     h("EARLY WARNING DETECTOR -- EVALUATION REPORT")
#     s(f"  h={cfg['threshold_h']:.4f} drift_k={cfg['drift_k']} "
#       f"decay={cfg['decay']} warmup={cfg['warmup']} "
#       f"healthy_ratio={cfg['healthy_ratio']} engines={cfg['n_engines']}")
#     s(f"  Engine-adaptive baseline: first {cfg['warmup']} windows per engine")
#     s()

#     h2("PRIMARY METRICS (Window-Level)")
#     total_w = wm['tn'] + wm['fp'] + wm['fn'] + wm['tp']
#     s(f"  Total windows: {total_w:,}")
#     s()
#     s(f"  Window TPR       : {wm['tpr']:.3f}  (target >= 0.90) "
#       f"[{'PASS' if wm['tpr'] >= 0.90 else 'FAIL'}]")
#     s(f"  Window FPR       : {wm['fpr']:.3f}  (target <= 0.15) "
#       f"[{'PASS' if wm['fpr'] <= 0.15 else 'FAIL'}]")
#     s(f"  Window Precision : {wm['precision']:.3f}")
#     s(f"  Window F1        : {wm['f1']:.3f}  (target >= 0.70) "
#       f"[{'PASS' if wm['f1'] >= 0.70 else 'FAIL'}]")
#     s()
#     s(f"  Confusion Matrix (z={roc['opt_threshold']:.3f}):")
#     s(f"    TN={wm['tn']:>6,}  FP={wm['fp']:>5,}")
#     s(f"    FN={wm['fn']:>6,}  TP={wm['tp']:>5,}")
#     s()

#     h2("SECONDARY METRICS (Engine-Level)")
#     s(f"  Detections > {MAX_VALID_EARLY} cycles before onset = FP")
#     s(f"  Degraded engines : {em['degraded_engines']}")
#     s(f"  Valid detections : {em['valid_detections']}")
#     s(f"  Too-early (FP)   : {em['too_early_count']}")
#     s(f"  Missed (FN)      : {em['fn']}")
#     s()
#     s(f"  Engine TPR  : {em['tpr']:.3f}  "
#       f"[{'PASS' if em['tpr'] >= 0.85 else 'FAIL'}]")
#     s(f"  Engine FPR  : {em['fpr_engine']:.3f}  "
#       f"[{'PASS' if em['fpr_engine'] <= 0.20 else 'FAIL'}]")
#     s(f"  Engine Prec : {em['precision']:.3f}")
#     s(f"  Engine F1   : {em['f1']:.3f}  "
#       f"[{'PASS' if em['f1'] >= 0.75 else 'FAIL'}]")
#     s()

#     h2("DETECTION LATENCY (existing engine metric)")
#     if len(em["latencies"]) > 0:
#         lat_ok = ("PASS" if 20 <= em["mean_latency"] <= 55 else "~")
#         s(f"  Mean   : {em['mean_latency']:+.1f} (target +20..+55) [{lat_ok}]")
#         s(f"  Median : {em['median_latency']:+.1f}")
#         s(f"  Std    : {em['std_latency']:.1f}")
#         s(f"  Range  : [{em['min_latency']:+.0f}, {em['max_latency']:+.0f}]")
#         s(f"  Early={em['early_count']} Late={em['late_count']} Exact={em['exact_count']}")
#     else:
#         s("  No valid detections.")
#     if len(em.get("all_latencies", [])) > 0:
#         s(f"  All latencies mean: {em['all_mean_latency']:+.1f}")
#     s()

#     h2("OPERATIONAL USEFULNESS (strict deployability view)")
#     s(f"  Actionable first alerts [0..{MAX_VALID_EARLY}] : "
#       f"{dep['useful_count']} / {dep['degraded_engines']} "
#       f"({100*dep['actionable_rate']:.1f}%)")
#     s(f"  Too-early alerts  (>{MAX_VALID_EARLY})         : "
#       f"{dep['too_early_count']} / {dep['degraded_engines']} "
#       f"({100*dep['too_early_rate']:.1f}%)")
#     s(f"  Late alerts       (<0)                         : "
#       f"{dep['late_count']} / {dep['degraded_engines']} "
#       f"({100*dep['late_rate']:.1f}%)")
#     s(f"  Missed detections                              : "
#       f"{dep['missed_count']} / {dep['degraded_engines']} "
#       f"({100*dep['missed_rate']:.1f}%)")
#     if not np.isnan(dep["mean_actionable_latency"]):
#         s(f"  Mean actionable lead time                     : "
#           f"{dep['mean_actionable_latency']:+.1f}")
#         s(f"  Median actionable lead time                   : "
#           f"{dep['median_actionable_latency']:+.1f}")
#     s()

#     h2("ALARM TRUST")
#     s(f"  First-alert precision                         : "
#       f"{dep['first_alert_precision']:.3f}")
#     s(f"  False alarms per 100 healthy windows          : "
#       f"{dep['false_alarms_per_100_healthy_windows']:.2f}")
#     s(f"  Avg alarms per engine                         : "
#       f"{dep['avg_alarms_per_engine']:.2f}")
#     s(f"  Avg pre-onset alarms / degraded engine        : "
#       f"{dep['avg_pre_onset_alarms_per_degraded_engine']:.2f}")
#     s()

#     h2("POLICY SENSITIVITY")
#     for row in policy_rows:
#         s(f"  max_valid_early={row['max_valid_early']:>3d} | "
#           f"useful={row['useful_count']:>2d} "
#           f"too_early={row['too_early_count']:>2d} "
#           f"late={row['late_count']:>2d} "
#           f"missed={row['missed_count']:>2d} | "
#           f"Prec={row['precision']:.3f} "
#           f"Rec={row['recall']:.3f} "
#           f"F1={row['f1']:.3f}")
#     s()

#     h2("ROC / PR")
#     s(f"  AUC={roc['auc_roc']:.4f}  "
#       f"[{'PASS' if roc['auc_roc'] >= 0.85 else 'FAIL'}]  "
#       f"Opt z={roc['opt_threshold']:.3f} J={roc['opt_j']:.3f}")
#     s(f"  AP={pr['avg_precision']:.4f}  "
#       f"Best F1={pr['best_f1']:.4f} (z={pr['best_threshold']:.3f})")
#     s()

#     h2("SEVERITY")
#     tw = sum(sev.values())
#     for sv in SEVERITY_ORDER:
#         c = sev.get(sv, 0)
#         s(f"  {sv:10s}: {c:6,} ({100 * c / max(tw, 1):5.1f}%)")
#     s()

#     h2("DIAGNOSIS")
#     if not diags:
#         s("  All detections within acceptable range.")
#     else:
#         for d in diags:
#             s(f"  Eng {d['eid']:3d} [{d['issue']}] {d['desc']}")
#             if d["lat"] is not None:
#                 s(f"    lat={d['lat']:+d} z@onset={d['zo']:.3f} "
#                   f"cusum@onset={d['co']:.3f}")
#             s(f"    -> {d['rec']}")
#             s()

#     h("TARGET SUMMARY")
#     tgts = [
#         ("Window TPR >= 0.90", wm["tpr"] >= 0.90, f"{wm['tpr']:.3f}"),
#         ("Window FPR <= 0.15", wm["fpr"] <= 0.15, f"{wm['fpr']:.3f}"),
#         ("Window F1  >= 0.70", wm["f1"] >= 0.70, f"{wm['f1']:.3f}"),
#         ("AUC-ROC    >= 0.85", roc["auc_roc"] >= 0.85, f"{roc['auc_roc']:.3f}"),
#         ("Engine TPR >= 0.85", em["tpr"] >= 0.85, f"{em['tpr']:.3f}"),
#         ("Engine F1  >= 0.75", em["f1"] >= 0.75, f"{em['f1']:.3f}"),
#     ]
#     if len(em["latencies"]) > 0:
#         tgts.append(("Latency +20..+55",
#                      20 <= em["mean_latency"] <= 55,
#                      f"{em['mean_latency']:+.1f}"))

#     ap = True
#     for nm, ok, v in tgts:
#         ap = ap and ok
#         s(f"  [{'PASS' if ok else 'FAIL'}] {nm:<24s} = {v}")
#     s()
#     s(f"  Overall: {'ALL PASS' if ap else 'SOME FAILED'}")
#     s()

#     report = "\n".join(L)
#     p = os.path.join(out, "metrics_summary.txt")
#     with open(p, "w", encoding="utf-8", errors="replace") as f:
#         f.write(report)
#     print(f"[OK] Report: {p}\n")
#     print(report)
#     return report

# def split_by_engine(X, y, eids):
#     engines = []
#     for eid in np.unique(eids):
#         m = eids == eid; idx = np.where(m)[0]
#         o = np.argsort(y[idx])[::-1]
#         engines.append((X[idx[o]], y[idx[o]], int(eid)))
#     return engines


# def main(args):
#     os.makedirs(args.out_dir, exist_ok=True)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Device: {device}")

#     model, mu, sigma, h, dk, decay, wu, si = \
#         load_model_and_baseline(args.model, args.baseline, device)

#     print(f"\nLoading: {args.dataset}")
#     data = np.load(args.dataset, allow_pickle=True)
#     X_test, y_test = data["X_test"], data["y_test"]
#     fnames = data["feature_names"].tolist()

#     if "engine_ids_test" in data:
#         eids = data["engine_ids_test"]
#     else:
#         eids = np.zeros(len(y_test), dtype=int); eid = 0
#         for i in range(1, len(y_test)):
#             if y_test[i] > y_test[i - 1]: eid += 1
#             eids[i] = eid

#     Xsel = X_test[:, :, si] if len(si) > 0 else X_test
#     print(f"Test: {Xsel.shape}  Engines: {len(np.unique(eids))}")
#     engines = split_by_engine(Xsel, y_test, eids)
#     hr = args.healthy_ratio

#     print(f"\nEvaluating {len(engines)} engines...")
#     results = []
#     for Xe, re, eid in engines:
#         r = run_engine_evaluation(
#             Xe, re, eid, model, mu, sigma,
#             h, dk, decay, wu, hr, device)
#         results.append(r)
#         st = (f"lat={r['latency']:+d}" if r['latency'] is not None
#               else "MISS" if r['true_onset_idx'] is not None
#               else "no_onset")
#         print(f"  Eng {eid:3d}: {len(Xe):4d}w "
#               f"onset={r['true_onset_idx']} "
#               f"det={r['detection_idx']} {st}")

#     print("\nMetrics...")
#     em = compute_engine_metrics(results, hr, max_early=MAX_VALID_EARLY)
#     dep = compute_deployability_metrics(results, hr, max_early=MAX_VALID_EARLY)
#     policy_rows = compute_policy_sensitivity(results, policy_values=POLICY_SWEEP)

#     yt, ys = build_window_labels(results, hr)
#     print(f"Windows: {(yt == 1).sum()} degraded / "
#           f"{(yt == 0).sum()} healthy")

#     roc = compute_roc(yt, ys)
#     prs = compute_pr(yt, ys)
#     wm = cm_stats(yt, ys, roc["opt_threshold"])
#     diags = diagnose(results, hr, max_early=MAX_VALID_EARLY)
#     sv = sev_dist(results)

#     print(f"Actionable first alerts [0..{MAX_VALID_EARLY}] : "
#           f"{dep['useful_count']}/{dep['degraded_engines']}")
#     print(f"Too-early / Late / Missed                     : "
#           f"{dep['too_early_count']}/{dep['late_count']}/{dep['missed_count']}")
#     print(f"False alarms per 100 healthy windows          : "
#           f"{dep['false_alarms_per_100_healthy_windows']:.2f}")

#     print("\nPlots...")
#     plot_roc(roc, prs, args.out_dir)
#     plot_latency(em, args.out_dir)
#     plot_cusum(results, args.out_dir, h, n=args.n_trace_engines)
#     plot_errors(results, mu, sigma, args.out_dir)
#     plot_severity(results, args.out_dir, n=min(4, len(results)))
#     plot_thresh(yt, ys, args.out_dir)
#     plot_policy_sensitivity(policy_rows, args.out_dir)

#     save_csv_rows(
#         dep["first_alert_rows"],
#         os.path.join(args.out_dir, "first_alert_table.csv")
#     )
#     save_csv_rows(
#         policy_rows,
#         os.path.join(args.out_dir, "policy_sensitivity.csv")
#     )

#     write_report(
#         em, wm, roc, prs, diags, sv, dep, policy_rows,
#         {"threshold_h": h, "drift_k": dk, "decay": decay,
#          "warmup": wu, "healthy_ratio": hr,
#          "n_engines": len(engines)},
#         args.out_dir)


# if __name__ == "__main__":
#     p = argparse.ArgumentParser()
#     p.add_argument("--dataset",
#                    default="data/processed/FD001_preprocessed.npz")
#     p.add_argument("--model",
#                    default="results/saved_models/vae_early_warning.pth")
#     p.add_argument("--baseline",
#                    default="results/saved_models/baseline_stats.npz")
#     p.add_argument("--out_dir", default="results/evaluation")
#     p.add_argument("--healthy_ratio", type=float, default=0.80)
#     p.add_argument("--warmup", type=int, default=20)
#     p.add_argument("--n_trace_engines", type=int, default=6)
#     main(p.parse_args())
    

"""
plots_extended.py
─────────────────
Drop-in replacement for all plot_* functions in evaluate_early_warning.py.
Each function adds extra metric panels for industry-level reporting.

USAGE – paste this file next to evaluate_early_warning.py and add at the top:
    from plots_extended import (
        plot_roc, plot_latency, plot_cusum,
        plot_errors, plot_severity, plot_thresh,
        plot_policy_sensitivity,
    )
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats as scipy_stats

# ── shared style ──────────────────────────────────────────────────────
SEVERITY_ORDER  = ["HEALTHY", "MILD", "MODERATE", "SEVERE"]
SEVERITY_COLORS = {
    "HEALTHY":  "#2ecc71",
    "MILD":     "#f1c40f",
    "MODERATE": "#e67e22",
    "SEVERE":   "#e74c3c",
}
MAX_VALID_EARLY = 90

_PALETTE = {
    "blue":   "#2980b9",
    "red":    "#e74c3c",
    "green":  "#2ecc71",
    "orange": "#f39c12",
    "purple": "#8e44ad",
    "navy":   "#1a252f",
    "gray":   "#95a5a6",
}

def _style(fig, axes_flat):
    """Apply consistent dark-grid publication style to every axis."""
    for ax in axes_flat:
        ax.set_facecolor("#fafafa")
        ax.grid(True, color="#dddddd", linewidth=0.8, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        ax.tick_params(labelsize=9)
    fig.patch.set_facecolor("white")


def _savefig(fig, path):
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] {path}")


# ══════════════════════════════════════════════════════════════════════
# 1. ROC / PR  →  adds DET curve + Reliability (calibration) curve
# ══════════════════════════════════════════════════════════════════════
def plot_roc(roc, pr, out):
    """
    Row 1 : ROC | PR
    Row 2 : DET curve | Calibration / reliability curve
    """
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]

    # ── panel 1: ROC ──────────────────────────────────────────────────
    ax = axes[0]
    ax.fill_between(roc["fpr"], roc["tpr"], alpha=0.10, color=_PALETTE["blue"])
    ax.plot(roc["fpr"], roc["tpr"],
            color=_PALETTE["blue"], lw=2.5,
            label=f"AUC = {roc['auc_roc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    ax.scatter(roc["opt_fpr"], roc["opt_tpr"],
               color=_PALETTE["red"], s=120, zorder=6,
               label=f"Youden J={roc['opt_j']:.3f}  z={roc['opt_threshold']:.2f}")
    # operating-region shading
    ax.axhspan(0.90, 1.02, alpha=0.06, color=_PALETTE["green"],
               label="TPR ≥ 0.90 target")
    ax.axvspan(0, 0.10, alpha=0.06, color=_PALETTE["green"])
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontweight="bold", fontsize=11)
    ax.legend(fontsize=8); ax.text(0.60, 0.08, "Random", fontsize=8,
                                    color="gray", rotation=35)

    # ── panel 2: PR ───────────────────────────────────────────────────
    ax = axes[1]
    # iso-F1 contours
    f1_levels = [0.3, 0.5, 0.7, 0.9]
    _r = np.linspace(0.01, 1, 300)
    for f1v in f1_levels:
        _p = f1v * _r / (2 * _r - f1v)
        _p = np.where(_p > 0, _p, np.nan)
        ax.plot(_r, _p, color="#cccccc", lw=0.9, ls="--", zorder=1)
        ok = np.where((_p > 0.02) & (_p <= 1))[0]
        if len(ok):
            xi = ok[len(ok) // 2]
            ax.text(_r[xi], _p[xi] + 0.02, f"F₁={f1v}", fontsize=7,
                    color="#aaaaaa", ha="center")
    # baseline (random)
    pos_rate = pr["recall"][0] if len(pr["recall"]) else 0.5
    ax.axhline(pos_rate, color="#aaaaaa", lw=1, ls=":", label="Baseline")
    ax.fill_between(pr["recall"], pr["precision"],
                    alpha=0.10, color=_PALETTE["purple"])
    ax.plot(pr["recall"], pr["precision"],
            color=_PALETTE["purple"], lw=2.5,
            label=f"AP = {pr['avg_precision']:.3f}")
    ax.scatter(pr["best_recall"], pr["best_precision"],
               color=_PALETTE["red"], s=120, zorder=6,
               label=f"Best F₁={pr['best_f1']:.3f}  z={pr['best_threshold']:.2f}")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve", fontweight="bold", fontsize=11)
    ax.legend(fontsize=8)

    # ── panel 3: DET curve ────────────────────────────────────────────
    ax = axes[2]
    # DET = FPR vs FNR (miss rate) — both on probit scale
    fnr = 1.0 - roc["tpr"]
    fpr_d = roc["fpr"]
    # avoid log(0)
    eps = 1e-6
    fnr_c  = np.clip(fnr,  eps, 1 - eps)
    fpr_c  = np.clip(fpr_d, eps, 1 - eps)
    xd = scipy_stats.norm.ppf(fpr_c)
    yd = scipy_stats.norm.ppf(fnr_c)
    ax.plot(xd, yd, color=_PALETTE["blue"], lw=2.5, label="VAE detector")
    # equal-error-rate line
    lims = [max(xd.min(), yd.min()), min(xd.max(), yd.max())]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.4, label="EER line")
    # mark optimal
    oi = np.argmin(np.abs(fpr_d - roc["opt_fpr"]))
    ax.scatter(xd[oi], yd[oi], color=_PALETTE["red"], s=100, zorder=6,
               label=f"Opt operating point")
    # probit tick labels
    prob_ticks = [0.001, 0.01, 0.05, 0.10, 0.20, 0.40]
    tick_vals  = [scipy_stats.norm.ppf(t) for t in prob_ticks]
    tick_labs  = [f"{int(t*100)}%" for t in prob_ticks]
    ax.set_xticks(tick_vals); ax.set_xticklabels(tick_labs, fontsize=7)
    ax.set_yticks(tick_vals); ax.set_yticklabels(tick_labs, fontsize=7)
    ax.set_xlabel("False Positive Rate (probit scale)")
    ax.set_ylabel("Miss Rate / FNR (probit scale)")
    ax.set_title("DET Curve  (Detection Error Tradeoff)",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=8)

    # ── panel 4: Reliability / calibration curve ──────────────────────
    ax = axes[3]
    # Use z-score thresholds to build calibration bins
    thresholds = roc["thresholds"]
    tpr_arr    = roc["tpr"]
    fpr_arr    = roc["fpr"]
    # predicted "probability" proxy: linearly map z-score to [0,1]
    # We build calibration from (fpr, tpr) at each threshold
    # fraction of positives in bin ≈ tpr; fraction predicted ≈ 1-fpr
    n_bins = 10
    bin_edges   = np.linspace(0, 1, n_bins + 1)
    frac_pos    = []
    mean_pred   = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (fpr_arr >= lo) & (fpr_arr < hi)
        if mask.sum() > 0:
            mean_pred.append(float(fpr_arr[mask].mean()))
            frac_pos.append(float(tpr_arr[mask].mean()))
    mean_pred = np.array(mean_pred)
    frac_pos  = np.array(frac_pos)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Perfect calibration")
    ax.fill_between(mean_pred, frac_pos, mean_pred,
                    alpha=0.15, color=_PALETTE["orange"],
                    label="Calibration gap")
    ax.plot(mean_pred, frac_pos,
            color=_PALETTE["orange"], lw=2.5, marker="o", ms=5,
            label="VAE (z-score proxy)")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    ax.set_xlabel("Mean predicted score (FPR proxy)")
    ax.set_ylabel("Fraction of positives (TPR proxy)")
    ax.set_title("Reliability Diagram  (Calibration)",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=8)
    ax.text(0.05, 0.88, "Above diagonal = over-sensitive",
            fontsize=7, color=_PALETTE["gray"], transform=ax.transAxes)
    ax.text(0.05, 0.82, "Below diagonal = under-sensitive",
            fontsize=7, color=_PALETTE["gray"], transform=ax.transAxes)

    _style(fig, axes)
    fig.suptitle("ROC / PR / DET / Calibration", fontsize=14,
                 fontweight="bold", y=1.01)
    _savefig(fig, os.path.join(out, "roc_pr_curves.png"))


# ══════════════════════════════════════════════════════════════════════
# 2. Detection Latency  →  adds per-engine box-whisker + latency-vs-RUL
# ══════════════════════════════════════════════════════════════════════
def plot_latency(m, out):
    al = m.get("all_latencies", m["latencies"])
    if len(al) == 0:
        return

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]

    # ── panel 1: latency histogram ────────────────────────────────────
    ax = axes[0]
    bins = np.linspace(al.min() - 1, al.max() + 1, 35)
    early = al[al > 0]; late = al[al <= 0]
    if len(early):
        ax.hist(early, bins=bins, color=_PALETTE["green"],
                alpha=0.75, label=f"Early ({len(early)})")
    if len(late):
        ax.hist(late, bins=bins, color=_PALETTE["red"],
                alpha=0.75, label=f"Late / at-onset ({len(late)})")
    ax.axvline(0,  color="black", lw=2, label="Onset = 0")
    ax.axvline(30, color=_PALETTE["orange"], lw=2, ls="--", label="Target +30")
    ax.axvline(50, color=_PALETTE["orange"], lw=2, ls=":",  label="Target +50")
    ax.axvspan(30, 50, alpha=0.10, color=_PALETTE["orange"])
    ax.axvline(MAX_VALID_EARLY, color="purple", lw=1.5, ls="-.",
               label=f"Max valid (+{MAX_VALID_EARLY})")
    ax.axvline(float(al.mean()), color=_PALETTE["blue"], lw=1.5, ls="--",
               label=f"Mean={al.mean():.1f}")
    ax.set_xlabel("Latency (cycles)"); ax.set_ylabel("Count")
    ax.set_title(f"Latency Distribution  (n={len(al)})",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=7, ncol=2)

    # ── panel 2: CDF ──────────────────────────────────────────────────
    ax = axes[1]
    sl  = np.sort(al)
    cdf = np.arange(1, len(sl) + 1) / len(sl)
    ax.plot(sl, cdf, color=_PALETTE["blue"], lw=2.5, label="Empirical CDF")
    ax.fill_betweenx(cdf, sl, 0, where=(sl > 0),
                     alpha=0.08, color=_PALETTE["green"], label="Early zone")
    ax.fill_betweenx(cdf, sl, 0, where=(sl <= 0),
                     alpha=0.08, color=_PALETTE["red"], label="Late zone")
    ax.axvline(0,  color="black", lw=1.5)
    ax.axvline(30, color=_PALETTE["orange"], lw=1.5, ls="--")
    ax.axvline(50, color=_PALETTE["orange"], lw=1.5, ls=":")
    ax.axvspan(30, 50, alpha=0.10, color=_PALETTE["orange"])
    # percentile annotations
    for pct in [25, 50, 75]:
        v = np.percentile(al, pct)
        ax.axvline(v, color=_PALETTE["gray"], lw=1, ls=":")
        ax.text(v, 0.02, f"P{pct}\n{v:.0f}", fontsize=7,
                ha="center", color=_PALETTE["gray"])
    ax.set_xlabel("Latency"); ax.set_ylabel("Cumulative Proportion")
    ax.set_ylim([0, 1.05])
    ax.set_title("Latency CDF  with Percentiles", fontweight="bold", fontsize=11)
    ax.legend(fontsize=8)

    # ── panel 3: violin + box ─────────────────────────────────────────
    ax = axes[2]
    vp = ax.violinplot([al], positions=[0], showmedians=False,
                       showextrema=False)
    for body in vp["bodies"]:
        body.set_facecolor(_PALETTE["blue"])
        body.set_alpha(0.35)
    bp = ax.boxplot([al], positions=[0], widths=0.25,
                    patch_artist=True, notch=True,
                    boxprops=dict(facecolor=_PALETTE["blue"], alpha=0.5),
                    medianprops=dict(color="black", lw=2),
                    flierprops=dict(marker=".", color=_PALETTE["red"],
                                    alpha=0.5, ms=4))
    ax.axhline(0,  color="black", lw=1.5, ls="--", label="Onset = 0")
    ax.axhline(30, color=_PALETTE["orange"], lw=1.5, ls="--",
               label="Target +30")
    ax.axhline(50, color=_PALETTE["orange"], lw=1.5, ls=":",
               label="Target +50")
    ax.axhline(MAX_VALID_EARLY, color="purple", lw=1.2, ls="-.",
               label=f"Max valid +{MAX_VALID_EARLY}")
    ax.set_xticks([0]); ax.set_xticklabels(["All engines"])
    ax.set_ylabel("Latency (cycles)")
    ax.set_title("Violin + Box  (Latency)", fontweight="bold", fontsize=11)
    ax.legend(fontsize=8)
    # IQR annotation
    q1, q3 = np.percentile(al, [25, 75])
    ax.text(0.55, float(q1), f"Q1={q1:.0f}", fontsize=8,
            color=_PALETTE["blue"], transform=ax.get_yaxis_transform(),
            va="bottom")
    ax.text(0.55, float(q3), f"Q3={q3:.0f}", fontsize=8,
            color=_PALETTE["blue"], transform=ax.get_yaxis_transform(),
            va="top")

    # ── panel 4: latency statistics summary table ──────────────────────
    ax = axes[3]
    ax.axis("off")
    stats = [
        ("N detections",         f"{len(al)}"),
        ("Mean latency",         f"{al.mean():+.1f} cycles"),
        ("Median latency",       f"{np.median(al):+.1f} cycles"),
        ("Std deviation",        f"{al.std():.1f} cycles"),
        ("Min / Max",            f"{al.min():+.0f} / {al.max():+.0f}"),
        ("P25 / P75",            f"{np.percentile(al,25):+.0f} / {np.percentile(al,75):+.0f}"),
        ("Early (lat > 0)",      f"{(al>0).sum()}  ({100*(al>0).mean():.0f}%)"),
        ("Late (lat ≤ 0)",       f"{(al<=0).sum()}  ({100*(al<=0).mean():.0f}%)"),
        ("In target [+30..+50]", f"{((al>=30)&(al<=50)).sum()}  ({100*((al>=30)&(al<=50)).mean():.0f}%)"),
        ("Beyond max valid",     f"{(al>MAX_VALID_EARLY).sum()}"),
    ]
    col_labels = ["Metric", "Value"]
    table_data = [[k, v] for k, v in stats]
    tbl = ax.table(cellText=table_data, colLabels=col_labels,
                   cellLoc="left", loc="center",
                   colWidths=[0.58, 0.38])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    tbl.scale(1, 1.55)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2980b9"); cell.set_text_props(color="white",
                                                                fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#eaf2fb")
        cell.set_edgecolor("#cccccc")
    ax.set_title("Latency Summary Statistics", fontweight="bold", fontsize=11)

    _style(fig, [axes[0], axes[1], axes[2]])
    fig.suptitle("Detection Latency Analysis", fontsize=14,
                 fontweight="bold", y=1.01)
    _savefig(fig, os.path.join(out, "detection_latency.png"))


# ══════════════════════════════════════════════════════════════════════
# 3. CUSUM Traces  →  adds z-score subplot + severity heatmap per engine
# ══════════════════════════════════════════════════════════════════════
def plot_cusum(results, out, h, n=6):
    det = [r for r in results if r["detection_idx"] is not None]
    mis = [r for r in results if
           r["detection_idx"] is None and r["true_onset_idx"] is not None]
    sample = (det[:n // 2] + mis[:n // 2])[:n]
    if not sample:
        sample = results[:n]

    sv_map = {"HEALTHY": 0, "MILD": 1, "MODERATE": 2, "SEVERE": 3}

    fig = plt.figure(figsize=(18, 5 * len(sample)))
    # Each engine: 3 rows (CUSUM | z-score | severity heatmap)
    outer = gridspec.GridSpec(len(sample), 1,
                              hspace=0.55, figure=fig)

    all_axes = []
    for i, r in enumerate(sample):
        inner = gridspec.GridSpecFromSubplotSpec(
            3, 1, subplot_spec=outer[i],
            hspace=0.08, height_ratios=[3, 2, 0.6])
        ax_cu  = fig.add_subplot(inner[0])
        ax_z   = fig.add_subplot(inner[1], sharex=ax_cu)
        ax_sev = fig.add_subplot(inner[2], sharex=ax_cu)
        all_axes += [ax_cu, ax_z, ax_sev]

        nn  = len(r["cusums"])
        xs  = np.arange(nn)
        to  = r["true_onset_idx"]
        det_idx = r["detection_idx"]
        lat = r["latency"]

        # background shading
        for ax in [ax_cu, ax_z]:
            if to is not None:
                ax.axvspan(0,  to, alpha=0.06, color=_PALETTE["green"])
                ax.axvspan(to, nn, alpha=0.06, color=_PALETTE["red"])

        # ── CUSUM ────────────────────────────────────────────────────
        ax_cu.plot(xs, r["cusums"],
                   color=_PALETTE["blue"], lw=1.8, label="CUSUM", zorder=3)
        ax_cu.axhline(h, color=_PALETTE["red"], lw=1.5, ls="--",
                      label=f"h = {h:.2f}")
        ax_cu.fill_between(xs, 0, r["cusums"],
                           where=(np.array(r["cusums"]) >= h),
                           color=_PALETTE["red"], alpha=0.25,
                           label="Alarm zone")
        if to is not None:
            ax_cu.axvline(to, color=_PALETTE["red"], lw=2, ls="-.",
                          label="True onset")
        if det_idx is not None:
            ax_cu.axvline(det_idx, color=_PALETTE["orange"], lw=2,
                          label=f"Detection (lat={lat:+d})")
        tag = f" [lat={lat:+d}]" if lat is not None else " [MISS]"
        ax_cu.set_title(f"Engine {r['engine_id']}{tag}",
                        fontsize=10, fontweight="bold")
        ax_cu.set_ylabel("CUSUM", fontsize=8)
        ax_cu.legend(fontsize=7, loc="upper left", ncol=3)
        ax_cu.set_xlim([0, nn])
        plt.setp(ax_cu.get_xticklabels(), visible=False)

        # ── Z-score ──────────────────────────────────────────────────
        zs = r["z_scores"]
        ax_z.plot(xs, zs, color=_PALETTE["purple"],
                  lw=1.5, label="Z-score", zorder=3)
        ax_z.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
        for zv, col in [(1.5, _PALETTE["orange"]),
                         (2.5, _PALETTE["red"])]:
            ax_z.axhline( zv, color=col, lw=1, ls=":", alpha=0.7)
            ax_z.axhline(-zv, color=col, lw=1, ls=":", alpha=0.4)
        # rolling mean overlay
        win = max(5, nn // 20)
        if nn > win:
            roll = np.convolve(zs, np.ones(win) / win, mode="same")
            ax_z.plot(xs, roll, color=_PALETTE["navy"],
                      lw=1.5, ls="--", alpha=0.7, label=f"Rolling mean (w={win})")
        if to is not None:
            ax_z.axvline(to, color=_PALETTE["red"], lw=1.5, ls="-.", alpha=0.7)
        if det_idx is not None:
            ax_z.axvline(det_idx, color=_PALETTE["orange"], lw=1.5, alpha=0.7)
        ax_z.set_ylabel("Z-score", fontsize=8)
        ax_z.legend(fontsize=7, loc="upper left", ncol=2)
        plt.setp(ax_z.get_xticklabels(), visible=False)

        # ── Severity heatmap ─────────────────────────────────────────
        sev_vals = np.array([sv_map.get(s, 0) for s in r["severities"]])
        img_data = sev_vals.reshape(1, -1)
        cmap     = matplotlib.colors.ListedColormap(
            [SEVERITY_COLORS[s] for s in SEVERITY_ORDER])
        ax_sev.imshow(img_data, aspect="auto", cmap=cmap,
                      vmin=0, vmax=3, extent=[0, nn, 0, 1])
        if to is not None:
            ax_sev.axvline(to, color="black", lw=2, ls="-.")
        if det_idx is not None:
            ax_sev.axvline(det_idx, color=_PALETTE["orange"], lw=2)
        ax_sev.set_yticks([]); ax_sev.set_ylabel("Sev.", fontsize=8)
        ax_sev.set_xlabel("Cycle", fontsize=8)

    # shared legend for severity heatmap
    legend_patches = [Patch(facecolor=SEVERITY_COLORS[s], label=s)
                      for s in SEVERITY_ORDER]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.01),
               title="Severity")

    _style(fig, all_axes)
    fig.suptitle("CUSUM + Z-Score + Severity  (Engine-Adaptive)",
                 fontsize=14, fontweight="bold", y=1.01)
    _savefig(fig, os.path.join(out, "cusum_traces.png"))


# ══════════════════════════════════════════════════════════════════════
# 4. Error Distribution  →  adds Q-Q plot + error-vs-cycle scatter
# ══════════════════════════════════════════════════════════════════════
def plot_errors(results, mu, sigma, out):
    he, de = [], []
    he_cycle, de_cycle = [], []   # (cycle_index, error)
    for r in results:
        o   = r["true_onset_idx"]
        err = r["errors"]
        nn  = len(err)
        if o is not None:
            he.extend(err[:o].tolist())
            de.extend(err[o:].tolist())
            # normalised cycle position [0,1]
            he_cycle.extend([(i / max(nn - 1, 1), e)
                              for i, e in enumerate(err[:o])])
            de_cycle.extend([(i / max(nn - 1, 1), e)
                              for i, e in enumerate(err[o:])])
        else:
            he.extend(err.tolist())
            he_cycle.extend([(i / max(nn - 1, 1), e)
                              for i, e in enumerate(err)])

    he = np.array(he, dtype=float)
    de = np.array(de, dtype=float)
    all_e = np.concatenate([he, de]) if len(de) else he

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]

    # ── panel 1: histogram ────────────────────────────────────────────
    ax = axes[0]
    kw = dict(bins=60, density=True, alpha=0.65)
    if len(he): ax.hist(he, **kw, color=_PALETTE["blue"],  label=f"Healthy (n={len(he):,})")
    if len(de): ax.hist(de, **kw, color=_PALETTE["red"],   label=f"Degraded (n={len(de):,})")
    xs_fit = np.linspace(all_e.min(), all_e.max(), 300)
    ax.plot(xs_fit, scipy_stats.norm.pdf(xs_fit, mu, sigma),
            color=_PALETTE["navy"], lw=2, ls="--", label=f"Fitted N(μ={mu:.4f}, σ={sigma:.4f})")
    ax.axvline(mu, color=_PALETTE["navy"], lw=2, label=f"μ = {mu:.4f}")
    for n, c in [(1, _PALETTE["orange"]), (2, _PALETTE["orange"]),
                 (3, _PALETTE["red"])]:
        ax.axvline(mu + n * sigma, color=c, lw=1.5, ls="--", alpha=0.7,
                   label=f"μ+{n}σ = {mu+n*sigma:.4f}")
    ax.set_xlabel("Reconstruction MSE"); ax.set_ylabel("Density")
    ax.set_title("Error Distribution  (Healthy vs Degraded)",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=7, ncol=2)

    # ── panel 2: Q-Q plot (all errors vs. normal) ─────────────────────
    ax = axes[1]
    qq_data = (all_e - all_e.mean()) / max(all_e.std(), 1e-12)
    qq_data_sorted = np.sort(qq_data)
    n_pts = len(qq_data_sorted)
    theoretical = scipy_stats.norm.ppf(
        np.linspace(1 / (n_pts + 1), n_pts / (n_pts + 1), n_pts))
    ax.scatter(theoretical, qq_data_sorted,
               s=4, alpha=0.35, color=_PALETTE["blue"], label="Observed")
    ax.plot(theoretical[[0, -1]], theoretical[[0, -1]],
            color=_PALETTE["red"], lw=2, label="Normal reference")
    ax.set_xlabel("Theoretical Normal Quantiles")
    ax.set_ylabel("Sample Quantiles (standardised)")
    ax.set_title("Q-Q Plot  (All Errors vs. Normal)",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=8)
    # annotate tail deviation
    tail_idx = n_pts - n_pts // 20
    x_ann, y_ann = float(theoretical[tail_idx]), float(qq_data_sorted[tail_idx])
    if y_ann > x_ann:
        ax.annotate("Heavy tail\n(degraded MSE)",
                    xy=(x_ann, y_ann), xytext=(x_ann - 0.8, y_ann + 0.5),
                    arrowprops=dict(arrowstyle="->", color=_PALETTE["red"]),
                    fontsize=8, color=_PALETTE["red"])

    # ── panel 3: error vs. normalised cycle (scatter density) ─────────
    ax = axes[2]
    if he_cycle:
        hcx, hcy = zip(*he_cycle)
        ax.scatter(hcx, hcy, s=2, alpha=0.18,
                   color=_PALETTE["blue"], label="Healthy", rasterized=True)
    if de_cycle:
        dcx, dcy = zip(*de_cycle)
        ax.scatter(dcx, dcy, s=2, alpha=0.18,
                   color=_PALETTE["red"], label="Degraded", rasterized=True)
    ax.axhline(mu,              color=_PALETTE["navy"],  lw=1.5, ls="--", label="μ")
    ax.axhline(mu + sigma,      color=_PALETTE["orange"], lw=1, ls=":",  label="μ+1σ")
    ax.axhline(mu + 2 * sigma,  color=_PALETTE["orange"], lw=1, ls=":")
    ax.axhline(mu + 3 * sigma,  color=_PALETTE["red"],   lw=1.5, ls="--",label="μ+3σ")
    ax.set_xlabel("Normalised Cycle Position [0 = start, 1 = end]")
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title("Error vs. Cycle Position",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=8, ncol=2)

    # ── panel 4: box comparison healthy vs degraded ───────────────────
    ax = axes[3]
    data_to_plot = [d for d in [he, de] if len(d)]
    labels       = ["Healthy", "Degraded"][:len(data_to_plot)]
    colors_bp    = [_PALETTE["blue"], _PALETTE["red"]][:len(data_to_plot)]
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                    notch=True, widths=0.4,
                    medianprops=dict(color="black", lw=2),
                    flierprops=dict(marker=".", ms=3, alpha=0.3))
    for patch, col in zip(bp["boxes"], colors_bp):
        patch.set_facecolor(col); patch.set_alpha(0.55)
    ax.axhline(mu + 3 * sigma, color=_PALETTE["red"],
               lw=1.5, ls="--", label="μ+3σ alarm")
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title("Box Plot  (Healthy vs Degraded)",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=8)
    # print summary stats as text
    if len(he) and len(de):
        ks_stat, ks_p = scipy_stats.ks_2samp(he, de)
        ax.text(0.97, 0.97,
                f"KS test: stat={ks_stat:.3f}  p={ks_p:.2e}",
                transform=ax.transAxes, fontsize=8,
                ha="right", va="top", color=_PALETTE["navy"],
                bbox=dict(boxstyle="round,pad=0.3",
                          fc="white", ec=_PALETTE["gray"]))

    _style(fig, axes)
    fig.suptitle("Reconstruction Error Analysis", fontsize=14,
                 fontweight="bold", y=1.01)
    _savefig(fig, os.path.join(out, "error_distribution.png"))


# ══════════════════════════════════════════════════════════════════════
# 5. Severity Timeline  →  adds stacked-area fleet view + pie
# ══════════════════════════════════════════════════════════════════════
def plot_severity(results, out, n=4):
    sv_map = {"HEALTHY": 0, "MILD": 1, "MODERATE": 2, "SEVERE": 3}
    sv_int = {"HEALTHY": 0, "MILD": 1, "MODERATE": 2, "SEVERE": 3}
    sample = results[:n]

    # ── figure layout: n individual timelines + 2 fleet panels ────────
    total_rows = len(sample) + 1   # +1 for fleet stacked + pie side-by-side
    fig = plt.figure(figsize=(16, 3.2 * total_rows))
    gs  = gridspec.GridSpec(total_rows, 2,
                            hspace=0.50, wspace=0.30,
                            height_ratios=[2] * len(sample) + [3])

    all_axes = []

    # ── per-engine bar timelines ──────────────────────────────────────
    for i, r in enumerate(sample):
        ax = fig.add_subplot(gs[i, :])
        all_axes.append(ax)
        vals = np.array([sv_int.get(s, 0) for s in r["severities"]])
        xs   = np.arange(len(vals))
        cols = [SEVERITY_COLORS.get(s, "gray") for s in r["severities"]]
        ax.bar(xs, vals + 0.5, color=cols, width=1,
               align="edge", alpha=0.85)
        if r["true_onset_idx"] is not None:
            ax.axvline(r["true_onset_idx"], color="black", lw=2.0,
                       ls="-.", label="True onset")
        if r["detection_idx"] is not None:
            ax.axvline(r["detection_idx"], color=_PALETTE["orange"], lw=2,
                       label=f"Detection (lat={r['latency']:+d})")
        ax.set_yticks([0.25, 1.25, 2.25, 3.25])
        ax.set_yticklabels(SEVERITY_ORDER, fontsize=8)
        ax.set_xlim([0, len(vals)]); ax.set_ylim([0, 4])
        ax.set_title(f"Engine {r['engine_id']}", fontweight="bold",
                     fontsize=10)
        ax.legend(fontsize=8, loc="upper left")

    # ── fleet stacked-area (all engines, resampled to 200 ticks) ─────
    ax_fleet = fig.add_subplot(gs[len(sample), 0])
    all_axes.append(ax_fleet)
    N_TICKS = 200
    sev_acc = {s: np.zeros(N_TICKS) for s in SEVERITY_ORDER}
    for r in results:
        sv_arr = np.array([sv_int.get(s, 0) for s in r["severities"]])
        if len(sv_arr) == 0:
            continue
        # interpolate to common length
        x_orig = np.linspace(0, 1, len(sv_arr))
        x_new  = np.linspace(0, 1, N_TICKS)
        sv_interp = np.round(np.interp(x_new, x_orig, sv_arr)).astype(int)
        for t_idx, sv_val in enumerate(sv_interp):
            label = SEVERITY_ORDER[sv_val]
            sev_acc[label][t_idx] += 1
    xs_fleet = np.linspace(0, 100, N_TICKS)
    stk_vals = [sev_acc[s] for s in SEVERITY_ORDER]
    ax_fleet.stackplot(xs_fleet, stk_vals,
                       labels=SEVERITY_ORDER,
                       colors=[SEVERITY_COLORS[s] for s in SEVERITY_ORDER],
                       alpha=0.80)
    ax_fleet.set_xlabel("Normalised Engine Lifecycle (%)")
    ax_fleet.set_ylabel("Engine Count")
    ax_fleet.set_title("Fleet Severity Composition  (all engines)",
                       fontweight="bold", fontsize=11)
    ax_fleet.legend(loc="upper left", fontsize=8)

    # ── severity pie ──────────────────────────────────────────────────
    ax_pie = fig.add_subplot(gs[len(sample), 1])
    all_axes.append(ax_pie)
    counts = {s: 0 for s in SEVERITY_ORDER}
    for r in results:
        for s in r["severities"]:
            counts[s] += 1
    pie_vals   = [counts[s] for s in SEVERITY_ORDER]
    pie_labels = [f"{s}\n({counts[s]:,})" for s in SEVERITY_ORDER]
    pie_colors = [SEVERITY_COLORS[s] for s in SEVERITY_ORDER]
    wedges, _, autotexts = ax_pie.pie(
        pie_vals, labels=pie_labels, colors=pie_colors,
        autopct="%1.1f%%", startangle=140, pctdistance=0.75,
        wedgeprops=dict(edgecolor="white", linewidth=1.5))
    for at in autotexts:
        at.set_fontsize(8)
    ax_pie.set_title("Severity Class Distribution  (all windows)",
                     fontweight="bold", fontsize=11)

    _style(fig, all_axes)
    fig.suptitle("Severity Timeline & Fleet View", fontsize=14,
                 fontweight="bold", y=1.01)
    _savefig(fig, os.path.join(out, "severity_timeline.png"))


# ══════════════════════════════════════════════════════════════════════
# 6. Threshold Sensitivity  →  adds PR iso-F1 grid + ECE vs. threshold
# ══════════════════════════════════════════════════════════════════════
def plot_thresh(yt, ys, out):
    ths = np.linspace(ys.min(), ys.max(), 300)
    tprs, fprs, f1s, precs, recs = [], [], [], [], []
    for t in ths:
        yp = (ys >= t).astype(int)
        tp = int(((yp == 1) & (yt == 1)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())
        tn = int(((yp == 0) & (yt == 0)).sum())
        tr = tp / max(tp + fn, 1)
        fr = fp / max(fp + tn, 1)
        pr = tp / max(tp + fp, 1)
        f1 = 2 * pr * tr / max(pr + tr, 1e-9)
        tprs.append(tr); fprs.append(fr)
        f1s.append(f1); precs.append(pr); recs.append(tr)
    tprs  = np.array(tprs);  fprs  = np.array(fprs)
    f1s   = np.array(f1s);   precs = np.array(precs)
    recs  = np.array(recs)
    bi    = np.argmax(f1s)

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]

    # ── panel 1: metric vs. threshold ─────────────────────────────────
    ax = axes[0]
    ax.plot(ths, tprs, color=_PALETTE["green"],  lw=2, label="TPR / Recall")
    ax.plot(ths, fprs, color=_PALETTE["red"],    lw=2, label="FPR")
    ax.plot(ths, f1s,  color=_PALETTE["blue"],   lw=2, label="F₁")
    ax.plot(ths, precs,color=_PALETTE["purple"], lw=2, label="Precision", alpha=0.8)
    ax.axvline(ths[bi], color=_PALETTE["orange"], lw=2, ls="--",
               label=f"Best F₁ @ z={ths[bi]:.2f}")
    ax.fill_betweenx([0, 1], ths[bi] - 0.2, ths[bi] + 0.2,
                     alpha=0.10, color=_PALETTE["orange"])
    ax.set_xlabel("Z-score Threshold")
    ax.set_ylabel("Metric Value")
    ax.set_title("Metrics vs. Threshold", fontweight="bold", fontsize=11)
    ax.legend(fontsize=8); ax.set_ylim([-0.02, 1.05])

    # ── panel 2: precision-recall tradeoff with iso-F1 contours ───────
    ax = axes[1]
    iso_f1 = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
    _r = np.linspace(0.01, 1, 300)
    for f1v in iso_f1:
        _p = f1v * _r / (2 * _r - f1v)
        _p = np.where(_p > 0, _p, np.nan)
        ax.plot(_r, _p, color="#cccccc", lw=0.9, ls="--", zorder=1)
        ok = np.where((_p > 0.02) & (_p <= 1.0))[0]
        if len(ok):
            xi = ok[len(ok) // 2]
            ax.text(_r[xi], float(_p[xi]) + 0.02, f"F₁={f1v}",
                    fontsize=7, color="#aaaaaa", ha="center")
    sc = ax.scatter(recs, precs, c=ths, cmap="viridis",
                    s=12, alpha=0.7, zorder=3)
    plt.colorbar(sc, ax=ax, label="Z-score threshold", fraction=0.035)
    ax.scatter(recs[bi], precs[bi],
               color=_PALETTE["red"], s=150, zorder=6,
               marker="*", label=f"Best F₁={f1s[bi]:.3f}")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("P-R Space  with Iso-F₁ Contours",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=8)

    # ── panel 3: TPR-FPR operating envelope ──────────────────────────
    ax = axes[2]
    ax.fill_between(ths, tprs, fprs,
                    where=(tprs >= fprs), alpha=0.15,
                    color=_PALETTE["green"], label="TPR > FPR zone")
    ax.fill_between(ths, tprs, fprs,
                    where=(tprs < fprs), alpha=0.15,
                    color=_PALETTE["red"], label="FPR > TPR zone")
    ax.plot(ths, tprs, color=_PALETTE["green"], lw=2, label="TPR")
    ax.plot(ths, fprs, color=_PALETTE["red"],   lw=2, label="FPR")
    ax.plot(ths, tprs - fprs, color=_PALETTE["blue"],
            lw=2, ls="--", label="Youden J = TPR − FPR")
    ax.axvline(ths[bi], color=_PALETTE["orange"], lw=2, ls="--",
               label=f"Best @ z={ths[bi]:.2f}")
    ax.set_xlabel("Z-score Threshold"); ax.set_ylabel("Rate")
    ax.set_title("TPR / FPR Operating Envelope",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=8); ax.set_ylim([-0.05, 1.05])

    # ── panel 4: threshold recommendation table ───────────────────────
    ax = axes[3]
    ax.axis("off")
    # pick 5 representative thresholds
    targets = [
        ("Max F₁",             ths[bi],                f1s[bi]),
        ("TPR ≥ 0.90",         ths[np.where(tprs >= 0.90)[0][-1]] if (tprs >= 0.90).any() else float("nan"), None),
        ("FPR ≤ 0.10",         ths[np.where(fprs <= 0.10)[0][0]]  if (fprs <= 0.10).any() else float("nan"), None),
        ("Youden J",           ths[np.argmax(tprs - fprs)],       None),
        ("Balanced (TPR=FPR)", ths[np.argmin(np.abs(tprs - fprs))], None),
    ]
    rows = []
    for name, t, f1_v in targets:
        if np.isnan(t):
            rows.append([name, "N/A", "–", "–", "–", "–"])
            continue
        idx = np.argmin(np.abs(ths - t))
        rows.append([
            name,
            f"{t:.3f}",
            f"{tprs[idx]:.3f}",
            f"{fprs[idx]:.3f}",
            f"{precs[idx]:.3f}",
            f"{f1s[idx]:.3f}",
        ])
    col_labels = ["Criterion", "z-thresh", "TPR", "FPR", "Prec", "F₁"]
    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    tbl.scale(1.1, 1.8)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor(_PALETTE["blue"])
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#eaf2fb")
        cell.set_edgecolor("#cccccc")
    ax.set_title("Threshold Selection Guide", fontweight="bold", fontsize=11)

    _style(fig, axes[:3])
    fig.suptitle("Threshold Sensitivity Analysis", fontsize=14,
                 fontweight="bold", y=1.01)
    _savefig(fig, os.path.join(out, "threshold_sensitivity.png"))


# ══════════════════════════════════════════════════════════════════════
# 7. Policy Sensitivity  →  adds grouped bar chart + radar chart
# ══════════════════════════════════════════════════════════════════════
def plot_policy_sensitivity(rows, out):
    if not rows:
        return

    xs         = [r["max_valid_early"] for r in rows]
    actionable = [r["actionable_rate"]  for r in rows]
    too_early  = [r["too_early_rate"]   for r in rows]
    late_r     = [r["late_rate"]        for r in rows]
    missed_r   = [r["missed_rate"]      for r in rows]
    prec       = [r["precision"]        for r in rows]
    rec        = [r["recall"]           for r in rows]
    f1v        = [r["f1"]               for r in rows]

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.32)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]

    # ── panel 1: line chart (original) ────────────────────────────────
    ax = axes[0]
    ax.plot(xs, actionable, marker="o", lw=2.5, color=_PALETTE["green"],
            label="Actionable rate")
    ax.plot(xs, too_early,  marker="s", lw=2.5, color=_PALETTE["orange"],
            label="Too-early rate")
    ax.plot(xs, late_r,     marker="^", lw=2.5, color=_PALETTE["blue"],
            label="Late rate")
    ax.plot(xs, missed_r,   marker="D", lw=2.5, color=_PALETTE["red"],
            label="Missed rate")
    ax.set_xlabel("MAX_VALID_EARLY  (cycles)")
    ax.set_ylabel("Rate")
    ax.set_title("Policy Sensitivity  – Rate vs. Policy",
                 fontweight="bold", fontsize=11)
    ax.set_ylim([-0.02, 1.05])
    ax.legend(fontsize=8)
    # annotate each point
    for xi, ai in zip(xs, actionable):
        ax.text(xi, ai + 0.02, f"{ai:.2f}", fontsize=7,
                ha="center", color=_PALETTE["green"])

    # ── panel 2: grouped bar chart ────────────────────────────────────
    ax = axes[1]
    n_grp  = len(rows)
    n_bars = 4
    bw     = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * bw
    bar_data = [
        ("Actionable",  actionable, _PALETTE["green"]),
        ("Too-early",   too_early,  _PALETTE["orange"]),
        ("Late",        late_r,     _PALETTE["blue"]),
        ("Missed",      missed_r,   _PALETTE["red"]),
    ]
    xi = np.arange(n_grp)
    for (label, vals, col), offset in zip(bar_data, offsets):
        bars = ax.bar(xi + offset, vals, width=bw,
                      label=label, color=col, alpha=0.80)
        for bar, v in zip(bars, vals):
            if v > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{v:.2f}", ha="center", va="bottom",
                        fontsize=6.5, rotation=45)
    ax.set_xticks(xi)
    ax.set_xticklabels([f"max={x}" for x in xs], fontsize=8)
    ax.set_ylabel("Rate"); ax.set_ylim([0, 1.15])
    ax.set_title("Grouped Bar  – Outcome Breakdown per Policy",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=8, ncol=2)

    # ── panel 3: Precision / Recall / F1 vs. policy ───────────────────
    ax = axes[2]
    ax.plot(xs, prec, marker="o", lw=2.5, color=_PALETTE["purple"],
            label="Precision")
    ax.plot(xs, rec,  marker="s", lw=2.5, color=_PALETTE["blue"],
            label="Recall")
    ax.plot(xs, f1v,  marker="D", lw=2.5, color=_PALETTE["orange"],
            label="F₁")
    ax.set_xlabel("MAX_VALID_EARLY  (cycles)")
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F₁  vs. Policy",
                 fontweight="bold", fontsize=11)
    ax.set_ylim([-0.02, 1.05])
    ax.legend(fontsize=8)
    bi = int(np.argmax(f1v))
    ax.scatter([xs[bi]], [f1v[bi]], color=_PALETTE["red"],
               s=120, zorder=6, label=f"Best F₁@{xs[bi]}")
    ax.axvline(xs[bi], color=_PALETTE["red"], lw=1.2, ls=":",
               alpha=0.6)

    # ── panel 4: summary table ────────────────────────────────────────
    ax = axes[3]
    ax.axis("off")
    tbl_rows = []
    for r in rows:
        tbl_rows.append([
            str(r["max_valid_early"]),
            str(r["useful_count"]),
            str(r["too_early_count"]),
            str(r["late_count"]),
            str(r["missed_count"]),
            f"{r['precision']:.3f}",
            f"{r['recall']:.3f}",
            f"{r['f1']:.3f}",
        ])
    col_labels = ["MaxEarly", "Useful", "TooEarly",
                  "Late", "Missed", "Prec", "Rec", "F₁"]
    tbl = ax.table(cellText=tbl_rows, colLabels=col_labels,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    tbl.scale(1.05, 1.9)
    for (row_i, col_i), cell in tbl.get_celld().items():
        if row_i == 0:
            cell.set_facecolor(_PALETTE["blue"])
            cell.set_text_props(color="white", fontweight="bold")
        elif row_i == bi + 1:           # highlight best F1 row
            cell.set_facecolor("#d5f5e3")
        elif row_i % 2 == 0:
            cell.set_facecolor("#eaf2fb")
        cell.set_edgecolor("#cccccc")
    ax.set_title("Policy Sensitivity  – Summary Table\n"
                 "(highlighted = best F₁)",
                 fontweight="bold", fontsize=11)

    _style(fig, [axes[0], axes[1], axes[2]])
    fig.suptitle("Policy Sensitivity Analysis", fontsize=14,
                 fontweight="bold", y=1.01)
    _savefig(fig, os.path.join(out, "policy_sensitivity.png"))