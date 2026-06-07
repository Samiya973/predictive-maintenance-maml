"""
early_warning_vae.py
────────────────────
VAE-based Early Warning Detector for turbofan engine degradation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

USEFUL_SENSORS = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_9",
    "sensor_11", "sensor_12", "sensor_14", "sensor_17",
    "sensor_20", "sensor_21",
]
N_SENSORS = len(USEFUL_SENSORS)


class TimeSeriesVAE(nn.Module):
    def __init__(self, input_size=N_SENSORS, seq_len=30,
                 hidden_size=64, latent_dim=16):
        super().__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim

        self.encoder_lstm = nn.LSTM(input_size, hidden_size, 1, batch_first=True)
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, latent_dim)
        self.decoder_lstm = nn.LSTM(latent_dim, hidden_size, 1, batch_first=True)
        self.output_fc = nn.Linear(hidden_size, input_size)

    def encode(self, x):
        _, (h_n, _) = self.encoder_lstm(x)
        h = h_n.squeeze(0)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), -10.0, 2.0)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return mu

    def decode(self, z):
        h = torch.relu(self.decoder_input(z))
        h = h.unsqueeze(1).expand(-1, self.seq_len, -1)
        out, _ = self.decoder_lstm(h)
        return self.output_fc(out)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

    def get_reconstruction_error(self, x):
        self.eval()
        with torch.no_grad():
            x_r, _, _, _ = self(x)
            err = F.mse_loss(x_r, x, reduction="none").mean(dim=(1, 2))
        return err.cpu().numpy()


def vae_loss(x_recon, x, mu, logvar, beta=1.0):
    recon = F.mse_loss(x_recon, x, reduction="mean")
    kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, recon, kl


# ──────────────────────────────────────────────────────────────────────────────
# CHANGE POINT DETECTOR — Engine-Adaptive Decayed z-score CUSUM
# ──────────────────────────────────────────────────────────────────────────────
class ChangePointDetector:
    """
    Engine-Adaptive Decayed CUSUM with per-engine STANDARDIZATION.

    KEY FIX: Uses both mean AND std from warmup period.
    
    Problem with mean-only offset:
      Engine warmup z-scores: [3, 5, 7, 4, 6, 8, 3, 5, ...]
      offset_mean = 5.0
      Post-warmup z=7 → corrected = 7-5 = 2.0 → breaches h=1.5 immediately!
    
    Fix with mean+std standardization:
      offset_mean = 5.0, offset_std = 1.8
      Post-warmup z=7 → standardized = (7-5)/1.8 = 1.1
      increment = 1.1 - 0.5 = 0.6 → needs ~3 consecutive to breach h=1.5
      Only SUSTAINED worsening triggers detection, not normal variation.

    Update rule:
      raw_z = (error - mu_global) / sigma_global
      engine_z = (raw_z - warmup_mean) / max(warmup_std, 1.0)
      S_t = max(0, decay * S_{t-1} + (engine_z - drift_k))

    The max(warmup_std, 1.0) floor prevents division by tiny std
    (which would amplify normal noise into false alarms).
    """

    def __init__(self, baseline_mean, baseline_std,
                 drift_k=0.5, threshold_h=4.0, warmup=30, decay=0.95):
        self.baseline_mean = float(baseline_mean)
        self.baseline_std = max(float(baseline_std), 1e-8)
        self.drift_k = float(drift_k)
        self.threshold = float(threshold_h)
        self.warmup = int(warmup)
        self.decay = float(decay)
        self.reset()

    def reset(self):
        self.cusum = 0.0
        self.values = []
        self.z_history = []
        self.cusum_history = []
        self.detected = False
        self.detection_idx = None
        self._warmup_zs = []
        self._engine_mean = 0.0
        self._engine_std = 1.0

    def update(self, error):
        error = float(error)
        self.values.append(error)
        idx = len(self.values) - 1

        raw_z = (error - self.baseline_mean) / self.baseline_std

        if idx < self.warmup:
            # Collect warmup z-scores
            self._warmup_zs.append(raw_z)
            self.cusum = 0.0

            # At end of warmup, compute engine-specific mean AND std
            if idx == self.warmup - 1:
                arr = np.array(self._warmup_zs)
                self._engine_mean = float(arr.mean())
                # Floor at 1.0: prevents tiny std from amplifying noise
                self._engine_std = max(float(arr.std()), 1.0)
        else:
            # Engine-standardized z: deviation from THIS engine's baseline
            # measured in THIS engine's units of variability
            engine_z = (raw_z - self._engine_mean) / self._engine_std
            increment = engine_z - self.drift_k
            self.cusum = max(0.0, self.decay * self.cusum + increment)

        alarm = self.cusum > self.threshold

        if alarm and not self.detected:
            self.detected = True
            self.detection_idx = idx

        self.z_history.append(raw_z)
        self.cusum_history.append(self.cusum)

        return {
            "z_score": float(raw_z),
            "cusum": float(self.cusum),
            "alarm": bool(alarm),
            "detected": bool(self.detected),
            "detection_idx": self.detection_idx,
        }

    @staticmethod
    def calibrate_threshold(healthy_errors, baseline_mean, baseline_std,
                            drift_k=0.5, target_fpr=0.10, warmup=30,
                            decay=0.95):
        baseline_std = max(baseline_std, 1e-8)
        n = len(healthy_errors)
        all_z = (healthy_errors - baseline_mean) / baseline_std
        chunk_size = max(warmup + 20, 50)
        n_chunks = max(n // chunk_size, 1)
        candidates = np.linspace(0.5, 20.0, 200)

        for h in candidates:
            total_alarms = 0
            total_post_warmup = 0
            for c in range(n_chunks):
                start = c * chunk_size
                end = min(start + chunk_size, n)
                chunk_z = all_z[start:end]
                if len(chunk_z) <= warmup:
                    continue
                wu_z = chunk_z[:warmup]
                eng_mean = float(wu_z.mean())
                eng_std = max(float(wu_z.std()), 1.0)
                cusum = 0.0
                for z in chunk_z[warmup:]:
                    engine_z = (z - eng_mean) / eng_std
                    cusum = max(0.0, decay * cusum + (engine_z - drift_k))
                    if cusum > h:
                        total_alarms += 1
                        cusum = 0.0
                    total_post_warmup += 1
            fpr = total_alarms / max(total_post_warmup, 1)
            if fpr <= target_fpr:
                # *** THE FIX: floor at 3.0 to prevent over-sensitivity ***
                return max(float(h), 3.0)
        return float(candidates[-1])

# def classify_severity(z_score):
#     if z_score < 1.0:
#         return {"level": "HEALTHY", "action": "No action required"}
#     elif z_score < 2.0:
#         return {"level": "MILD", "action": "Monitor closely"}
#     elif z_score < 3.0:
#         return {"level": "MODERATE", "action": "Schedule inspection"}
#     else:
#         return {"level": "SEVERE", "action": "Immediate intervention required"}
def classify_severity(z_score, cusum, threshold_h, streak, confirm_cycles):
    if streak >= confirm_cycles:
        if z_score >= 3.0:
            return {"level": "WARNING", "action": "Persistent severe anomaly"}
        return {"level": "EARLY_WARNING", "action": "Persistent anomaly detected"}

    if cusum > 0.5 * threshold_h:
        return {"level": "MODERATE", "action": "Watch closely"}

    if z_score < 1.0:
        return {"level": "HEALTHY", "action": "No action required"}

    return {"level": "MILD", "action": "Monitor closely"}

# class EarlyWarningSystem:
#     def __init__(self, vae, baseline_mean, baseline_std,
#                  drift_k=0.5, threshold_h=4.0, warmup=20,
#                  decay=0.95, device="cpu"):
#         self.vae = vae.to(device)
#         self.vae.eval()
#         self.device = device
#         self.baseline_mean = float(baseline_mean)
#         self.baseline_std = float(baseline_std)
#         self.cpd = ChangePointDetector(
#             baseline_mean=self.baseline_mean,
#             baseline_std=self.baseline_std,
#             drift_k=drift_k, threshold_h=threshold_h,
#             warmup=warmup, decay=decay,
#         )
#         self.reset()

#     def reset(self):
#         self.cpd.reset()
#         self.cycle_history = []
#         self.error_history = []
#         self.z_history = []
#         self.cusum_history = []
#         self.onset_detected = False
#         self.onset_cycle = None

#     def monitor(self, x, cycle):
#         if x.dim() == 2:
#             x = x.unsqueeze(0)
#         x = x.to(self.device)
#         error = float(self.vae.get_reconstruction_error(x)[0])
#         det = self.cpd.update(error)

#         self.cycle_history.append(cycle)
#         self.error_history.append(error)
#         self.z_history.append(det["z_score"])
#         self.cusum_history.append(det["cusum"])

#         if det["alarm"] and not self.onset_detected:
#             self.onset_detected = True
#             self.onset_cycle = cycle

#         sev = classify_severity(det["z_score"])
#         return {
#             "cycle": cycle, "error": error,
#             "z_score": det["z_score"], "cusum": det["cusum"],
#             "alarm": det["alarm"],
#             "severity": sev["level"], "action": sev["action"],
#         }

#     def get_detection_cycle(self):
#         return self.onset_cycle

#     def get_detection_latency(self, true_onset):
#         if self.onset_cycle is None:
#             return float("nan")
#         return float(true_onset - self.onset_cycle)


class EarlyWarningSystem:
    def __init__(self, vae, baseline_mean, baseline_std,
                 drift_k=0.5, threshold_h=4.0, warmup=20,
                 decay=0.95, device="cpu",
                 confirm_cycles=3,
                 severe_z=3.0,
                 moderate_z=2.0):
        self.vae = vae.to(device)
        self.vae.eval()
        self.device = device
        self.baseline_mean = float(baseline_mean)
        self.baseline_std = float(baseline_std)

        self.cpd = ChangePointDetector(
            baseline_mean=self.baseline_mean,
            baseline_std=self.baseline_std,
            drift_k=drift_k,
            threshold_h=threshold_h,
            warmup=warmup,
            decay=decay,
        )

        self.confirm_cycles = int(confirm_cycles)
        self.severe_z = float(severe_z)
        self.moderate_z = float(moderate_z)

        self.reset()

    def reset(self):
        self.cpd.reset()
        self.cycle_history = []
        self.error_history = []
        self.z_history = []
        self.cusum_history = []

        self.onset_detected = False
        self.onset_cycle = None

        self.alarm_streak = 0
        self.warning_streak = 0

    def _classify_stage(self, z_score, cusum, alarm):
        if not alarm:
            if z_score < 1.0:
                return "HEALTHY", "No action required"
            elif z_score < self.moderate_z:
                return "MILD", "Monitor closely"
            else:
                return "MODERATE", "Inspect trend"

        if self.alarm_streak < self.confirm_cycles:
            return "EARLY_WARNING", "Early anomaly detected — continue monitoring"

        if z_score >= self.severe_z:
            return "WARNING", "Persistent anomaly — schedule inspection"

        return "WARNING", "Persistent anomaly — monitor and inspect"

    def monitor(self, x, cycle):
        if x.dim() == 2:
            x = x.unsqueeze(0)

        x = x.to(self.device)
        error = float(self.vae.get_reconstruction_error(x)[0])
        det = self.cpd.update(error)

        self.cycle_history.append(cycle)
        self.error_history.append(error)
        self.z_history.append(det["z_score"])
        self.cusum_history.append(det["cusum"])

        if det["alarm"]:
            self.alarm_streak += 1
        else:
            self.alarm_streak = 0

        severity, action = self._classify_stage(
            z_score=det["z_score"],
            cusum=det["cusum"],
            alarm=det["alarm"]
        )

        if (self.alarm_streak >= self.confirm_cycles) and (not self.onset_detected):
            self.onset_detected = True
            self.onset_cycle = cycle

        return {
            "cycle": cycle,
            "error": error,
            "z_score": det["z_score"],
            "cusum": det["cusum"],
            "alarm": det["alarm"],
            "alarm_streak": self.alarm_streak,
            "severity": severity,
            "action": action,
        }

    def get_detection_cycle(self):
        return self.onset_cycle

    def get_detection_latency(self, true_onset):
        if self.onset_cycle is None:
            return float("nan")
        return float(true_onset - self.onset_cycle)

# ──────────────────────────────────────────────────────────────────────────────
# SMOKE TEST
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 80)
    print("EARLY WARNING SYSTEM -- SMOKE TEST")
    print("  Engine-adaptive decayed CUSUM")
    print("=" * 80)

    torch.manual_seed(42)
    np.random.seed(42)

    BATCH, SEQ_LEN, N_FEAT = 64, 30, N_SENSORS
    healthy = torch.randn(BATCH, SEQ_LEN, N_FEAT) * 0.5

    vae = TimeSeriesVAE(input_size=N_FEAT, seq_len=SEQ_LEN)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3, weight_decay=1e-5)
    print(f"\nParameters: {sum(p.numel() for p in vae.parameters()):,}")

    vae.train()
    for epoch in range(1, 101):
        optimizer.zero_grad()
        x_r, mu, lv, _ = vae(healthy)
        loss, _, _ = vae_loss(x_r, healthy, mu, lv, beta=min(1.0, epoch / 50.0))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
        optimizer.step()
        if epoch % 25 == 0:
            print(f"  Epoch {epoch:3d}  Loss={loss.item():.6f}")

    errors = vae.get_reconstruction_error(healthy)
    baseline_mean = float(errors.mean())
    baseline_std = float(errors.std())
    print(f"\nBaseline: mu={baseline_mean:.6f}, sigma={baseline_std:.6f}")

    val_errors = vae.get_reconstruction_error(
        torch.randn(BATCH, SEQ_LEN, N_FEAT) * 0.5)
    threshold_h = ChangePointDetector.calibrate_threshold(
        val_errors, baseline_mean, baseline_std,
        drift_k=0.5, target_fpr=0.10, warmup=20, decay=0.95)
    print(f"Calibrated h={threshold_h:.4f}")

    ews = EarlyWarningSystem(
        vae=vae, baseline_mean=baseline_mean, baseline_std=baseline_std,
        drift_k=0.5, threshold_h=threshold_h, warmup=20, decay=0.95)

    TRUE_ONSET = 55
    TOTAL = 80

    print(f"\n{'Cyc':>4} {'Err':>10} {'Z':>7} {'CUSUM':>8} {'Sev':>9} {'Al':>3}")
    print("-" * 55)

    for cycle in range(1, TOTAL + 1):
        if cycle <= 30:
            noise = 0.5; phase = "H"
        elif cycle < TRUE_ONSET:
            p = (cycle - 30) / (TRUE_ONSET - 30)
            noise = 0.5 + 0.20 * p; phase = "D"
        else:
            p = min(1.0, (cycle - TRUE_ONSET) / max(TOTAL - TRUE_ONSET, 1))
            noise = 0.70 + 0.40 * p; phase = "S"

        r = ews.monitor(torch.randn(1, SEQ_LEN, N_FEAT) * noise, cycle)
        al = "**" if r["alarm"] else "  "

        if cycle <= 5 or cycle >= 28 or cycle % 5 == 0 or r["alarm"]:
            print(f"{cycle:4d} {r['error']:10.6f} {r['z_score']:7.2f} "
                  f"{r['cusum']:8.3f} {r['severity']:>9} {al:>3}  {phase}")

    det = ews.get_detection_cycle()
    print("\n" + "=" * 55)
    if det:
        lat = ews.get_detection_latency(TRUE_ONSET)
        print(f"Onset={TRUE_ONSET} Det={det} Lat={lat:+.0f} "
              f"({'EARLY' if lat > 0 else 'LATE'})")
    else:
        print("No detection.")
    print("=" * 55)