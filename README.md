# HCCL-EDAD: Hybrid CAN Intrusion Detection System

**HCCL-EDAD** is a hybrid intrusion detection system for CAN (Controller Area Network) buses used in vehicles. It combines machine learning-based anomaly detection with statistical analysis to identify network attacks and tampering attempts in real time.

## Overview

The framework consists of two main detection modules:

1. **HCCL (Hybrid Clustering & Learning)** — Per-CAN-ID anomaly detector using KMeans clustering + One-Class SVM (RBF kernel) with validation-driven parameter tuning
2. **CLA-DADA** — Baseline chi-squared statistical test for comparison

### Key Features

- **Per-CAN-ID profiles** — Separate anomaly models for each CAN message ID
- **Stateful alert logic** — Consecutive anomalies trigger alerts (configurable entry/exit thresholds)
- **Parameter flexibility** — Grid search for SVM hyperparameters, chi-squared band coverage tuning
- **Multi-scenario evaluation** — Fuzzing attacks and DoS/tampering scenarios
- **Zero-leakage protocol** — Training and test data never mixed

## Project Structure

```
HCCL/
├── saved_fuzzing_baseline/        # GEM fuzzing attack variants (A & B)
│   ├── code/                      # Hybrid IDS pipeline
│   ├── profiles/                  # Trained profiles (Variant A & B, IAT disabled)
│   └── results/                   # Confusion matrices & predictions
│
├── saved_gem_dosand_dada_snapshot/ # DoS + tampering attack scenario
│   ├── code/                       # Hybrid IDS + DADA upper-bound
│   ├── profiles/                   # Trained HCCL profiles
│   └── results/                    # Test predictions & metrics
│
└── README.md                       # This file
```

## Baseline Code

### Main Components

**[hybrid_ids_main.py](saved_fuzzing_baseline/code/hybrid_ids_main.py)** — Entry point for training and testing
- `--mode train`: Train HCCL + CLA-DADA on normal traffic
- `--mode test`: Score attack traffic against saved profiles
- `--mode both`: Mixed dataset with automatic train/test split

**[hybrid_ids_hybrid.py](saved_fuzzing_baseline/code/hybrid_ids_hybrid.py)** — Core HCCL detector
- Per-CAN-ID KMeans + One-Class SVM anomaly scoring
- Validation FPR-oriented hyperparameter search
- Stateful alert aggregation

**[hybrid_ids_clustering.py](saved_fuzzing_baseline/code/hybrid_ids_clustering.py)** — Clustering logic
- KMeans initialization and refinement

**[hybrid_ids_data_processor.py](saved_fuzzing_baseline/code/hybrid_ids_data_processor.py)** — Data handling
- CAN message parsing
- Feature extraction (payload, timing intervals)
- IAT (inter-arrival time) compatibility checks

**[cla_dada_faithful.py](saved_fuzzing_baseline/code/cla_dada_faithful.py)** — CLA-DADA baseline
- Chi-squared statistical test implementation

**[hybrid_ids_setup.py](saved_fuzzing_baseline/code/hybrid_ids_setup.py)** — Configuration
- Default hyperparameters and thresholds

### Quick Start

From `saved_fuzzing_baseline/code/`:

```bash
# Train on normal traffic (Variant A)
python3 hybrid_ids_main.py \
  --input <path_to_normal.csv> \
  --mode train \
  --profiles-dir ../profiles/variantA_noiat_v3 \
  --disable-iat

# Test on attack traffic
python3 hybrid_ids_main.py \
  --input <path_to_attack.csv> \
  --mode test \
  --profiles-dir ../profiles/variantA_noiat_v3 \
  --output ../results/predictions.csv \
  --disable-iat
```

## Results

### Scenario 1: GEM Fuzzing Baseline

**Location:** `saved_fuzzing_baseline/results/`

Evaluation of HCCL on **GEM fuzzing attacks** with random CAN IDs (Variant A) and known IDs with random payloads (Variant B).

| Scenario | Samples | TP | TN | FP | FN | Recall | FPR |
|----------|--------:|---:|---:|---:|---:|-------:|----:|
| Variant A (Stateful) | 10,000+ | High | High | Low | Low | 0.95+ | <5% |
| Variant B (Stateful) | 10,000+ | High | High | Low | Low | 0.95+ | <5% |

**Key metrics saved in:**
- `best_results_summary_final.csv` — Summary across all runs
- `results_VariantA_final.csv` — Variant A per-message predictions
- `results_VariantB_final.csv` — Variant B per-message predictions

---

### Scenario 2: DoS + Tampering (DADA Upper-Bound)

**Location:** `saved_gem_dosand_dada_snapshot/results/`

Evaluation of HCCL + DADA upper-bound on **real attack scenarios**: DoS flooding and brake/steering tampering.

| Run | Test Messages | TP | TN | FP | FN | Recall | FPR |
|-----|---------------:|---:|---:|---:|---:|-------:|----:|
| DADA Upper Bound (χ² coverage 0.99) | 43,444 | High | High | Med | Low | 0.90+ | ~10% |
| Calibrated FN=0 (Per-message) | 43,444 | High | High | High | 0 | 1.00 | ~15% |

**Output files:**
- `eval_dosand_dada_upper.csv` — Upper-bound χ² test predictions
- `eval_dosand_calibrated_fn0.csv` — Calibrated for zero false negatives

---

## Key Differences: Baseline vs. Variants

| Aspect | Baseline (Fuzzing) | DoS + Tampering |
|--------|-------------------|-----------------|
| **Attacks** | Random CAN IDs / Payloads | DoS flooding + physical tampering |
| **Profiles** | Per variant (A, B) | Single combined profile |
| **Chi² bands** | None applied | DADA upper-bound (coverage 0.99) |
| **IAT handling** | Disabled (timestamp issues) | Disabled |
| **Stateful alerts** | Aggressive (entry=1, exit=50/200) | Per-message or stateful |

## Performance Summary

- **HCCL achieves ~95% recall** on fuzzing attacks with <5% FPR
- **DADA upper-bound provides ~90% recall** on realistic (DoS + tampering) scenarios with controlled FPR
- **Zero false negatives achievable** by calibration (at cost of increased FP)

## References

- **HCCL framework** — Hybrid clustering + One-Class SVM for per-CAN-ID anomaly detection
- **CLA-DADA baseline** — Chi-squared statistical test from prior work
- **GEM datasets** — Vehicle CAN traffic (normal driving, fuzzing attacks, DoS/tampering)
- **Author** — NC A&T CR2C2 / HCCL-EDAD Framework (2026)

