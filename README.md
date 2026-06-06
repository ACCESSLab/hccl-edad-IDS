# HCCL-EDAD

**A Learning-Based Intrusion Detection System for CAN Bus DoS Attacks in Connected Vehicles**


---

## About

HCCL-EDAD is a lightweight on-device intrusion detection system for CAN bus security in connected and automated vehicles (CAVs). The framework establishes per-identifier behavioral baselines through adaptive K-means clustering, refines decision boundaries via a One-Class SVM, and evaluates incoming messages through a hierarchical six-layer detection pipeline — trained exclusively on benign traffic to preserve zero-day detection capability.

**Key properties:**
- No labeled attack data required during training
- Sub-millisecond per-frame inference latency
- Formal statistical guarantee on false positive rate: E[FPR] ≤ α + O(1/√n)
- Deployable on embedded ECUs without cloud connectivity

---

## Repository Contents

| File | Description |
|---|---|
| `HCCL_edad.py` | Proposed HCCL-EDAD detection framework |
| `cla_dada.py` | Re-implementation of CLA-DADA baseline (D'Angelo et al., 2020) |
| `requirements.txt` | Python dependencies |


---

## Dataset

The **GEM-CAN** dataset is publicly available on Zenodo:
**https://doi.org/10.5281/zenodo.18283067**

---

## Results

Performance on the GEM-CAN dataset under DoS flooding and data tampering:

| Method | Recall | TNR | FPR | MCC |
|---|---|---|---|---|
| **HCCL-EDAD** | **100%** | **70.64%** | **29.36%** | **0.8375** |
| CLA-DADA | 100% | 44.37% | 55.63% | 0.662 |

---

## Citation

If you use this code, please cite:

```bibtex
@article{tavasoli2026hccl,
  author  = {Tavasoli, Mahsa and Sarrafzadeh, Abdolhossein and
             Karimoddini, Ali and Khaleghi, Milad and
             Phuapaiboon, Tienake and Pasandi, Hannaneh B.},
  title   = {A Learning-Based Intrusion Detection System for CAN Bus DoS Attacks in Connected Vehicles},
  journal = {IEEE Open Journal of Intelligent Transportation Systems},
  year    = {2026}
}
```

---

