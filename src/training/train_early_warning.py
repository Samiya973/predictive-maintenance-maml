"""
train_early_warning.py
"""

import os, sys, copy, argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.models.early_warning_vae import (
    TimeSeriesVAE, vae_loss, ChangePointDetector,
    USEFUL_SENSORS, N_SENSORS,
)


def select_useful_sensors(X, feature_names):
    selected_names = [f for f in USEFUL_SENSORS if f in feature_names]
    selected_idx = [feature_names.index(f) for f in selected_names]
    if not selected_idx:
        raise ValueError(f"No useful sensors found. Expected: {USEFUL_SENSORS}")
    print(f"Selected {len(selected_idx)} useful sensors: {selected_names}")
    return X[:, :, selected_idx], selected_names, selected_idx


class HealthyDataset(Dataset):
    def __init__(self, X, y_rul, healthy_ratio=0.80):
        mask = y_rul > healthy_ratio
        self.X = torch.FloatTensor(X[mask])
        self.y = torch.FloatTensor(y_rul[mask])
        total, kept = len(X), len(self.X)
        print(f"Healthy windows (RUL > {healthy_ratio}): {kept}/{total} "
              f"({100 * kept / max(total, 1):.1f}%)")

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def train_vae(model, train_loader, val_loader, config, device):
    opt = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-5)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min",
                                                  factor=0.5, patience=15)
    best_val = float("inf"); best_ep = 0; pat = 0; best_st = None
    hist = {k: [] for k in ["train_loss", "train_recon", "train_kl",
                             "val_loss", "val_recon", "val_kl", "beta"]}

    print("\n" + "=" * 90)
    print("TRAINING VAE (healthy windows only)")
    print("=" * 90)
    print(f"{'Ep':>5} {'Beta':>6} {'TrL':>10} {'VaL':>10} "
          f"{'TrR':>10} {'TrKL':>10} {'VaKL':>10}")
    print("-" * 90)

    for ep in range(1, config["n_epochs"] + 1):
        beta = config["beta_start"] + (
            (config["beta_end"] - config["beta_start"]) *
            min(1.0, ep / config["warmup_epochs"]))

        model.train()
        tl = tr = tk = 0.0; nb = 0
        for xb, _ in train_loader:
            xb = xb.to(device); opt.zero_grad()
            xr, mu, lv, _ = model(xb)
            loss, rec, kl = vae_loss(xr, xb, mu, lv, beta=beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item(); tr += rec.item(); tk += kl.item(); nb += 1
        tl /= max(nb, 1); tr /= max(nb, 1); tk /= max(nb, 1)

        model.eval()
        vl = vr = vk = 0.0; nv = 0
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(device)
                xr, mu, lv, _ = model(xb)
                loss, rec, kl = vae_loss(xr, xb, mu, lv, beta=beta)
                vl += loss.item(); vr += rec.item(); vk += kl.item(); nv += 1
        vl /= max(nv, 1); vr /= max(nv, 1); vk /= max(nv, 1)

        for k, v in [("train_loss", tl), ("train_recon", tr), ("train_kl", tk),
                      ("val_loss", vl), ("val_recon", vr), ("val_kl", vk),
                      ("beta", beta)]:
            hist[k].append(v)

        sched.step(vl)
        if ep <= 5 or ep % 10 == 0:
            print(f"{ep:5d} {beta:6.3f} {tl:10.6f} {vl:10.6f} "
                  f"{tr:10.6f} {tk:10.6f} {vk:10.6f}")

        if vl < best_val:
            best_val = vl; best_ep = ep; pat = 0
            best_st = copy.deepcopy(model.state_dict())
        else:
            pat += 1
            if pat >= config["patience"]:
                print(f"\nEarly stopping at epoch {ep} (best={best_ep})")
                break

    if best_st is not None:
        os.makedirs(os.path.dirname(config["save_path"]), exist_ok=True)
        torch.save({"epoch": best_ep, "model_state_dict": best_st,
                     "val_loss": best_val, "config": config,
                     "history": hist}, config["save_path"])
        print(f"\nModel saved: {config['save_path']}  "
              f"Best epoch: {best_ep}  Val loss: {best_val:.6f}")
    return hist, best_st


def compute_baseline(model, val_loader, device):
    model.eval()
    errs = []
    with torch.no_grad():
        for xb, _ in val_loader:
            errs.extend(model.get_reconstruction_error(xb.to(device)).tolist())
    errs = np.array(errs, dtype=np.float32)
    mu, sigma = float(errs.mean()), float(errs.std())
    print(f"\nBaseline: mu={mu:.6f}  sigma={sigma:.6f}  n={len(errs)}")
    return mu, sigma, errs


def calibrate_cusum(val_errors, mu, sigma, drift_k, target_fpr, warmup, decay):
    h = ChangePointDetector.calibrate_threshold(
        val_errors, mu, sigma, drift_k=drift_k,
        target_fpr=target_fpr, warmup=warmup, decay=decay)
    print(f"\nCUSUM calibration (engine-adaptive decayed z-score):")
    print(f"  drift_k={drift_k}  decay={decay}  warmup={warmup}  "
          f"FPR<={100 * target_fpr:.0f}%")
    print(f"  threshold_h = {h:.4f}")
    return h


def plot_history(hist, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    eps = range(1, len(hist["train_loss"]) + 1)
    for ax, tk, vk, t in [(axes[0, 0], "train_loss", "val_loss", "Total Loss"),
                           (axes[0, 1], "train_recon", "val_recon", "Recon Loss"),
                           (axes[1, 0], "train_kl", "val_kl", "KL Divergence")]:
        ax.plot(eps, hist[tk], label="Train")
        ax.plot(eps, hist[vk], label="Val")
        ax.set_title(t); ax.legend(); ax.grid(True, alpha=0.3)
    axes[1, 1].plot(eps, hist["beta"])
    axes[1, 1].set_title("Beta Schedule"); axes[1, 1].grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"Training plot: {path}")


def plot_baseline(errors, mu, sigma, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors, bins=50, density=True, alpha=0.7, color="steelblue",
            label="Healthy")
    for n, c in [(1, "gold"), (2, "orange"), (3, "red")]:
        ax.axvline(mu + n * sigma, color=c, linestyle="--", label=f"{n}sig")
    ax.axvline(mu, color="green", lw=2, label="mu")
    ax.set_xlabel("MSE"); ax.set_ylabel("Density")
    ax.set_title("Baseline Distribution"); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"Baseline plot: {path}")


def main(args):
    config = {
        "hidden_size": 64, "latent_dim": 16,
        "lr": 1e-3, "batch_size": 128, "n_epochs": 100, "patience": 40,
        "beta_start": 0.0, "beta_end": 1.0, "warmup_epochs": 50,
        "healthy_ratio": 0.80,
        # ── Engine-adaptive CUSUM with per-engine STANDARDIZATION ──
        "drift_k": 0.5,
        "decay": 0.95,
        "target_fpr": 0.10,
        "warmup": 30,  # 30 windows for mean+std estimation
        "save_path": "results/saved_models/vae_early_warning.pth",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    device = torch.device(config["device"])
    print(f"Device: {device}  Sensors: {N_SENSORS}")

    data = np.load(args.dataset, allow_pickle=True)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    feature_names = data["feature_names"].tolist()
    print(f"Train: {X_train.shape}  Val: {X_val.shape}")

    X_train, snames, sidx = select_useful_sensors(X_train, feature_names)
    X_val, _, _ = select_useful_sensors(X_val, feature_names)
    config["input_size"] = X_train.shape[2]
    config["seq_len"] = X_train.shape[1]
    assert config["input_size"] == N_SENSORS

    train_ds = HealthyDataset(X_train, y_train, config["healthy_ratio"])
    val_ds = HealthyDataset(X_val, y_val, config["healthy_ratio"])
    train_ld = DataLoader(train_ds, batch_size=config["batch_size"],
                          shuffle=True, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)

    model = TimeSeriesVAE(config["input_size"], config["seq_len"],
                          config["hidden_size"], config["latent_dim"]).to(device)
    print(f"\nVAE params: {sum(p.numel() for p in model.parameters()):,}")

    hist, best_st = train_vae(model, train_ld, val_ld, config, device)
    if best_st:
        model.load_state_dict(best_st)

    mu, sigma, val_errs = compute_baseline(model, val_ld, device)
    h = calibrate_cusum(val_errs, mu, sigma,
                         config["drift_k"], config["target_fpr"],
                         config["warmup"], config["decay"])

    stats_path = "results/saved_models/baseline_stats.npz"
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    np.savez(stats_path,
             baseline_mean=mu, baseline_std=sigma,
             threshold_h=h, drift_k=config["drift_k"],
             decay=config["decay"], warmup=config["warmup"],
             errors=val_errs,
             sensor_indices=np.array(sidx),
             sensor_names=np.array(snames),
             healthy_ratio=np.array(config["healthy_ratio"]))
    print(f"Baseline saved: {stats_path}")

    os.makedirs("results/figures", exist_ok=True)
    plot_history(hist, "results/figures/vae_training.png")
    plot_baseline(val_errs, mu, sigma,
                  "results/figures/baseline_distribution.png")

    print(f"\nDone! mu={mu:.6f} sigma={sigma:.6f}")
    print(f"CUSUM: h={h:.4f} drift_k={config['drift_k']} "
          f"decay={config['decay']} warmup={config['warmup']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",
                   default="data/processed/FD001_preprocessed.npz")
    main(p.parse_args())