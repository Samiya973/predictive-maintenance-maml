# 🔧 Few-Shot Predictive Maintenance with MAML
### Remaining Useful Life Prediction via Model-Agnostic Meta-Learning on NASA C-MAPSS

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10.0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Dataset](https://img.shields.io/badge/Dataset-NASA%20C--MAPSS%20FD001-003087?style=flat-square)](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository)
[![Status](https://img.shields.io/badge/Status-Active%20Development-22c55e?style=flat-square)]()
---

## 🧠 Overview

Industrial equipment failures cost manufacturers billions annually — yet most predictive maintenance models require **hundreds of failure examples per machine type** to generalize. This project challenges that assumption.

We apply **Model-Agnostic Meta-Learning (MAML)** to turbofan engine degradation data, training a model that can predict Remaining Useful Life (RUL) from **as few as 5–10 labeled cycles** of a new, unseen engine. Each engine is treated as a distinct *meta-learning task*, enabling rapid adaptation at deployment time — no retraining required.

> **Core hypothesis:** Inter-engine variability (lifetime range: 128–362 cycles) is not noise — it is signal. MAML exploits this heterogeneity to learn *how to learn* degradation patterns.

---

## 📊 Results at a Glance

| Model | Test RMSE | Test MAE | R² | NASA Score | Status |
|---|---|---|---|---|---|
| **LSTM Baseline** | **14.04 cycles** | 10.19 cycles | 0.8953 | 10,036 | ✅ Complete |
| **Random Forest** | 15.43 cycles | 11.28 cycles | 0.8734 | 14,017 | ✅ Complete |
| **MAML (few-shot)** | — | — | — | — | ⏳ In Progress |

> Both baselines exceed the target of **RMSE < 20 cycles**. MAML aims to match this from ≤10 support samples per unseen engine.

---

## 🎯 Problem Statement

Given multivariate time-series sensor data from a jet engine, predict how many operational cycles remain before failure.

| Challenge | Conventional Approach | Our Approach |
|---|---|---|
| New engine, no history | Retrain from scratch | Few-shot adaptation via MAML |
| Inter-engine variability | Treat as noise | Exploit as task diversity |
| Limited labeled failures | Requires large datasets | Meta-learns from 5–10 support samples |
| Deployment generalization | Engine-specific models | Task-agnostic meta-initializer |

---

## 📦 Dataset

**NASA C-MAPSS FD001** — Commercial Modular Aero-Propulsion System Simulation

| Property | Value |
|---|---|
| Source | NASA Ames Prognostics Data Repository |
| Operating Condition | Single (Sea Level) |
| Fault Mode | High Pressure Compressor Degradation |
| Training Engines | 100 (20,631 cycles) |
| Test Engines | 100 (13,096 cycles) |
| Sensors (raw → after EDA) | 21 → **11 informative** |
| Engine Lifetime Range | 128 – 362 cycles (2.8× variation) |
| Missing Values | **0** (100% complete) |

Each engine = one meta-learning task. The 2.8× lifetime variation across engines provides natural task heterogeneity crucial for MAML.

---

## ⚙️ Preprocessing Pipeline

Raw sensor data → deep-learning-ready tensors in 8 steps:

```
Raw Data  (26 cols, 20,631 rows)
    │
[1] Remove constant sensors (variance < 0.01)   →   17 cols
    │
[2] Min-Max normalization per sensor             →   all in [0, 1]
    │
[3] Rolling mean + std  (windows: 5, 10, 20)    →   80 cols
    │
[4] Velocity + Acceleration (Δ, Δ²)             →   102 cols total
    │
[5] Clip RUL at 130 cycles                       →   RUL ∈ [0, 130]
    │
[6] Sliding window (length=30, stride=1)         →   [17,731, 30, 102]
    │
[7] Engine-level split (no leakage)              →   70 / 15 / 15
    │
[8] Save .npz + scaler .pkl
```

### Feature Breakdown (102 total)

| Feature Type | Count | Description |
|---|---|---|
| Base sensors (normalized) | 11 | 11 informative sensors scaled to [0,1] |
| Operational settings | 3 | `setting_1`, `setting_2`, `setting_3` |
| Rolling mean (w=5,10,20) | 33 | Short/mid/long-term trend per sensor |
| Rolling std (w=5,10,20) | 33 | Local volatility per sensor |
| Velocity (1st derivative) | 11 | Δ sensor per timestep |
| Acceleration (2nd derivative) | 11 | Δ² sensor per timestep |
| **Total** | **102** | |

### Data Split

| Split | Engines | Sequences |
|---|---|---|
| Train | 70 | 12,286 |
| Validation | 15 | 2,735 |
| Test | 15 | 2,710 |

> **Critical:** Split by engine, not by sequence. Random sequence splitting causes data leakage — sequences from the same engine would appear in both train and test sets.

---

## 🏗️ Architecture

### LSTM Baseline (Current Best)

```
Input: [batch=64, timesteps=30, features=102]
        ↓
  LSTM(102 → 128)  +  Dropout(0.3)
        ↓
  LSTM(128 → 64)   +  Dropout(0.3)
        ↓
  Linear(64 → 1)   +  ReLU
        ↓
  RUL prediction ∈ [0, 130]

Total parameters: 168,513
```

## Random Forest Baseline
```
Input: (N, 30, 102)
        ↓
  extract_features()
  mean(axis=1)  →  102 features
  std(axis=1)   →  102 features
  min(axis=1)   →  102 features
  max(axis=1)   →  102 features
        ↓
Flat vector: (N, 408)
        ↓
  RandomForestRegressor(
    n_estimators      = 200
    max_depth         = 20
    max_features      = 0.5
    min_samples_leaf  = 2
    n_jobs            = -1
  )
        ↓
  clip(ŷ, 0, 130)
        ↓
RUL prediction ∈ [0, 130]

Total trees    : 200
Input features : 408  (102 × 4 statistics)
```

### MAML Meta-Training Loop (Upcoming)

```
θ* ← meta-initialization

For each engine (task τᵢ):
  1. Sample support set  →  compute Lᵢ(θ)
  2. Inner update        →  θᵢ' = θ - α∇Lᵢ(θ)    [5 steps]
  3. Evaluate query set with adapted θᵢ'

Meta-update:  θ ← θ - β∇ Σᵢ Lᵢ(θᵢ')
```

---

## 📈 Baseline Training Details

### LSTM — Training Curve Summary

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 5 | 2872.82 | 2720.82 |
| 25 | 463.95 | 315.41 |
| 35 | 200.19 | **167.36** ← best |
| 50 | 132.74 | 195.05 |

Early stopping triggered at epoch 50 (patience=15). Notable plateau at epochs 1–24 before rapid convergence — Adam optimizer escaping a local minimum, common in LSTM time-series regression.

### Random Forest — Key Observations

- 200 trees, 408-dimensional temporal feature vectors (mean + std + min + max over 30 timesteps)
- Trained in **31.1 seconds** vs 50 epochs for LSTM
- Severe overfitting (Train RMSE: 1.17 vs Test RMSE: 15.43) — ensemble averaging partially compensates
- Higher NASA Score than LSTM (14,017 vs 10,037) — makes more dangerous late predictions
- Compensating strengths: native feature importances + free uncertainty quantification via tree variance

### Evaluation Figures

| Figure | Description |
|---|---|
| `lstm_baseline_evaluation.png` | Pred vs actual, error distribution, loss curve, RMSE/MAE by split |
| `rf_01_standard_evaluation.png` | Pred vs actual, top-20 feature importances |
| `rf_02_degradation_trajectories.png` | Actual vs predicted RUL over engine life for 6 test engines |
| `rf_03_residual_analysis.png` | Residuals vs RUL — bias at different degradation stages |
| `rf_04_uncertainty_bands.png` | ±1σ and ±2σ prediction bands from 200 trees |
| `rf_05_shap_explainability.png` | SHAP beeswarm + bar plot for 500 test instances |
| `rf_06_cost_analysis.png` | Early vs late prediction cost-weighted distribution |

---

## 📁 Project Structure

```
predictive-maintenance-maml/
├── data/
│   ├── raw/                        # NASA C-MAPSS source files
│   └── processed/                  # FD001_preprocessed.npz (12.4 MB) + scaler.pkl
├── src/
│   ├── data/
│   │   ├── load_data.py            # Data loading + RUL computation      (243 lines)
│   │   ├── preprocessing.py        # CMAPSSPreprocessor class             (456 lines)
│   │   └── data_loader.py          # PyTorch DataLoader setup             (120 lines)
│   ├── models/
│   │   ├── baselines.py            # LSTMBaseline (168,513 params)
│   │   └── rf_baseline.py          # RFBaseline + temporal feature extractor
│   ├── training/
│   │   ├── train_baseline.py       # LSTM training loop + early stopping
│   │   └── train_rf_baseline.py    # RF training + validation
│   └── evaluation/
│       ├── evaluate_baseline.py    # RMSE, MAE, R², NASA Score
│       └── evaluate_rf_baseline.py # + SHAP, uncertainty bands, cost analysis
├── notebooks/
│   └── 01_Initial_EDA.ipynb        # Sensor variance analysis + lifecycle plots
├── results/
│   ├── figures/                    # 7 evaluation figures
│   └── saved_models/               # lstm_baseline_best.pth + rf_baseline_best.pkl
├── requirements.txt
└── README.md
```

---

## 🗺️ Roadmap

| Phase | Task | Status | Metric |
|---|---|---|---|
| Day 1 | Environment + project scaffold | ✅ Complete | — |
| Day 2 | EDA + sensor analysis | ✅ Complete | 11 sensors retained |
| Day 3 | Preprocessing pipeline | ✅ Complete | X: (17731, 30, 102) |
| Day 4 | LSTM + RF baselines | ✅ Complete | RMSE: 14.04 / 15.43 |
| Days 5–6 | MAML meta-training | 🔄 In Progress | Target: RMSE < 15 |
| Day 7 | Early warning: VAE + CUSUM | ⏳ Upcoming | 30+ cycle earlier detection |
| Day 8 | System integration | ⏳ Upcoming | — |
| Days 9–11 | Evaluation + final report | ⏳ Upcoming | — |

---

## 🔬 Key Technical Decisions

**Why FD001?**
Single operating condition isolates degradation signal from operational variance. Cleaner task structure = fairer evaluation of MAML's few-shot capability.

**Why RUL cap at 130?**
Early-cycle sensor readings are indistinguishable regardless of whether RUL is 200 or 350. Capping focuses loss on the critical degradation window. Standard practice per Zheng et al. (2017) and Li et al. (2018).

**Why window = 30 timesteps?**
Balances temporal context against data efficiency — critical for short-lived engines (128-cycle minimum). Shorter sequences = more training tasks per engine for MAML.

**Why engine-level split?**
Sequence-level splitting causes data leakage: cycles 31–60 of an engine are trivially predictable if cycles 1–30 were seen in training.

**Why `higher` for MAML?**
Enables exact second-order gradients through the inner loop. First-order approximations (FOMAML) lose gradient signal that matters for fast adaptation on small support sets.

**Why SHAP on Random Forest?**
RF's tree structure gives exact (not approximate) SHAP values — a free explainability baseline before applying SHAP to the LSTM/MAML models.

---

## 📦 Dependencies

```
torch==2.10.0+cpu
higher==0.2.1
numpy==1.26.4
pandas==2.2.0
scikit-learn==1.4.0
scipy==1.12.0
matplotlib==3.8.0
seaborn==0.13.0
jupyter==1.0.0
```

---

## 📚 References

1. Finn, C., Abbeel, P., & Levine, S. (2017). **Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks.** *ICML 2017.* [[arXiv]](https://arxiv.org/abs/1703.03400)
2. Saxena, A., et al. (2008). **Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation.** *PHM 2008.*
3. Zheng, S., et al. (2017). **Long Short-Term Memory Network for Remaining Useful Life Estimation.** *IEEE PHM.*
4. Li, X., et al. (2018). **Remaining Useful Life Estimation in Prognostics Using Deep Convolution Neural Networks.** *Reliability Engineering & System Safety.*
5. Lundberg, S. & Lee, S.-I. (2017). **A Unified Approach to Interpreting Model Predictions (SHAP).** *NeurIPS.*


<p align="center">
  <i>Built on NASA's open prognostics data and the meta-learning community's foundations.</i><br>
  <i>If this helps your research, a ⭐ goes a long way.</i>
</p>
