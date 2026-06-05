#!/usr/bin/env python3
"""
CLA-DADA: implementation of D'Angelo et al. (2021)
"A Cluster-Based Multidimensional Approach for Detecting Attacks on Connected Vehicles"
IEEE Internet of Things Journal, Vol. 8, No. 16, 2021
DOI: 10.1109/JIOT.2020.3032935

Algorithms 1 (CLA) and 2 (DADA) from the paper. K-means per CAN ID on 8-byte
payloads; LB/UB are min/max distances to each cluster centroid (Eq. 5-6).
Default K=300 and nPts=30 follow the paper.
"""

import os
import sys
import time
import pickle
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

DATA_COLS = [f"data{i}" for i in range(8)]

def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    for alt in (" CAN ID", "CAN_ID", "can_id", "ArbID", "CANID"):
        if alt in df.columns:
            df = df.rename(columns={alt: "arbitration_id"})
    for i, alt in enumerate(["DATA[0]","DATA[1]","DATA[2]","DATA[3]",
                              "DATA[4]","DATA[5]","DATA[6]","DATA[7]"]):
        if alt in df.columns:
            df = df.rename(columns={alt: f"data{i}"})
    return df

def _extract_normals(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("label", "Flag"):
        if col in df.columns:
            vals = df[col].astype(str).str.strip()
            if vals.isin({"R", "T"}).any():
                return df[vals == "R"].copy()
            try:
                return df[vals.astype(int) == 0].copy()
            except ValueError:
                pass
    if "Attack_Type" in df.columns:
        return df[df["Attack_Type"].astype(str).str.strip() == "Normal"].copy()
    return df.copy()

def _get_bytes(df: pd.DataFrame) -> np.ndarray:
    present = [c for c in DATA_COLS if c in df.columns]
    if len(present) < 8:
        raise ValueError(f"Missing byte columns. Found: {present}")
    def _to_int(s):
        s = str(s).strip()
        try:
            return int(s, 16) if s.startswith("0x") or s.startswith("0X") else int(float(s))
        except (ValueError, TypeError):
            try:
                return int(s, 16)
            except (ValueError, TypeError):
                return 0
    vals = df[present].applymap(_to_int).values
    return vals.astype(float)

def _true_labels(df: pd.DataFrame) -> np.ndarray:
    for col in ("label", "Flag"):
        if col in df.columns:
            vals = df[col].astype(str).str.strip()
            if vals.isin({"R", "T"}).any():
                return (vals == "T").astype(int).values
            try:
                return (vals.astype(int) != 0).astype(int).values
            except ValueError:
                pass
    if "Attack_Type" in df.columns:
        return (df["Attack_Type"].astype(str).str.strip() != "Normal").astype(int).values
    return np.zeros(len(df), dtype=int)


# ══════════════════════════════════════════════════════════════════════
# Algorithm 1 — CLA (Cluster-Based Learning Algorithm)
# ══════════════════════════════════════════════════════════════════════

class CLA:
    """
    Faithful CLA implementation. K-means on raw 8-D byte vectors.
    Output Ψ (sigma_map): { can_id : np.ndarray shape (K, 10) }
        columns: [cx0..cx7, LB, UB]
    """

    def __init__(self, K: int = 300):
        self.K = K
        self.sigma_map: dict = {}
        self.known_ids: set = set()

    def fit(self, df_normal: pd.DataFrame) -> "CLA":
        print(f"\n[CLA] Training on {len(df_normal):,} normal messages  (K={self.K})")

        self.known_ids = set(df_normal["arbitration_id"].unique())
        print(f"      Found {len(self.known_ids)} unique CAN IDs")

        for can_id, grp in df_normal.groupby("arbitration_id"):
            data = _get_bytes(grp)
            n = len(data)

            # Paper requires 1 < K < N^ID; for n<2 we fall back to a single
            # centroid with LB=UB=0 (pragmatic extension, not in D'Angelo 2021).
            k = min(self.K, n)
            if k < 2:
                centroid = data[0]
                row = np.append(centroid, [0.0, 0.0])
                self.sigma_map[can_id] = row.reshape(1, -1)
                continue

            km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
            km.fit(data)
            labels = km.labels_
            centroids = km.cluster_centers_

            # Vectorised LB/UB computation per cluster
            rows = []
            for ci in range(k):
                mask = labels == ci
                pts = data[mask]
                if len(pts) == 0:
                    rows.append(np.append(centroids[ci], [0.0, 0.0]))
                    continue
                dists = np.linalg.norm(pts - centroids[ci], axis=1)
                lb = float(dists.min())
                ub = float(dists.max())
                rows.append(np.append(centroids[ci], [lb, ub]))

            self.sigma_map[can_id] = np.array(rows)
            print(f"      CAN ID {can_id:5}: {n:6,} msgs → {k} clusters  "
                  f"(UB range [{min(r[-1] for r in rows):.1f}, "
                  f"{max(r[-1] for r in rows):.1f}])")

        print(f"[CLA] Done — {sum(len(v) for v in self.sigma_map.values()):,} "
              f"total centroids stored in Ψ")
        return self

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"sigma_map": self.sigma_map,
                         "known_ids": self.known_ids,
                         "K": self.K}, f)
        print(f"[CLA] Saved Ψ → {path}")

    def load(self, path: str) -> "CLA":
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.sigma_map = d["sigma_map"]
        self.known_ids = d["known_ids"]
        self.K         = d["K"]
        print(f"[CLA] Loaded Ψ ← {path}  "
              f"({len(self.known_ids)} IDs, "
              f"{sum(len(v) for v in self.sigma_map.values()):,} centroids)")
        return self


# ══════════════════════════════════════════════════════════════════════
# Algorithm 2 — DADA (fully vectorised)
# ══════════════════════════════════════════════════════════════════════

class DADA:
    """
    Faithful DADA with FULLY VECTORISED detection.

    All three conditions are checked via numpy broadcasting:
      (c) Unknown ID → ATTACK
      (d)(A) All nPts nearest centroids have dist outside [LB, UB] → ATTACK
      (d)(B) None of the nPts nearest centroids belongs to message's ID → ATTACK
      else → NORMAL
    """

    def __init__(self, sigma_map: dict, nPts: int = 30):
        self.sigma_map  = sigma_map
        self.nPts       = nPts
        self.known_ids  = set(sigma_map.keys())
        self._global_centroids: np.ndarray = None
        self._global_ids:       np.ndarray = None
        self._global_lb:        np.ndarray = None
        self._global_ub:        np.ndarray = None
        self._knn_index: NearestNeighbors  = None
        self._build_global_index()

    def predict_df(self, df_test):
        result = self.detect(df_test)
        return result['dada_anomaly'].astype(int).values

    def _build_global_index(self) -> None:
        rows_c, rows_id, rows_lb, rows_ub = [], [], [], []
        for can_id, arr in self.sigma_map.items():
            rows_c.append(arr[:, :8])
            rows_id.extend([can_id] * len(arr))
            rows_lb.append(arr[:, 8])
            rows_ub.append(arr[:, 9])

        self._global_centroids = np.vstack(rows_c)
        self._global_ids       = np.array(rows_id)
        self._global_lb        = np.concatenate(rows_lb)
        self._global_ub        = np.concatenate(rows_ub)

        M = len(self._global_centroids)
        k_search = min(self.nPts, M)
        self._knn_index = NearestNeighbors(
            n_neighbors = k_search,
            algorithm   = "ball_tree",
            metric      = "euclidean",
            n_jobs      = -1,
        )
        self._knn_index.fit(self._global_centroids)
        print(f"[DADA] Global centroid index built: {M:,} centroids, "
              f"nPts={self.nPts}")

    def detect(self, df_test: pd.DataFrame) -> pd.DataFrame:
        """
        Run DADA on df_test — FULLY VECTORISED (no Python for-loop).

        Returns df_test with added columns:
            dada_anomaly  (bool)
            dada_reason   (str)
        """
        print(f"\n[DADA] Detecting anomalies in {len(df_test):,} messages...")

        byte_matrix = _get_bytes(df_test)           # (N, 8)
        can_ids     = df_test["arbitration_id"].values
        N           = len(df_test)

        is_attack = np.zeros(N, dtype=bool)
        reasons   = np.empty(N, dtype=object)
        reasons[:] = "Normal"

        # ── Condition (c): Unknown CAN ID ─────────────────────────────────
        known_set = self.known_ids
        unknown_mask = np.array([cid not in known_set for cid in can_ids])
        is_attack[unknown_mask] = True
        reasons[unknown_mask]   = "Unknown CAN ID"

        known_idx = np.where(~unknown_mask)[0]
        if len(known_idx) == 0:
            print("[DADA] All messages had unknown IDs.")
            df_out = df_test.copy()
            df_out["dada_anomaly"] = is_attack
            df_out["dada_reason"]  = reasons
            return df_out

        # ── Batch KNN search ──────────────────────────────────────────────
        X_known = byte_matrix[known_idx]                     # (n_known, 8)
        distances, indices = self._knn_index.kneighbors(X_known)
        # distances: (n_known, nPts)   indices: (n_known, nPts)

        # ── Condition (d)(A): Crown check — FULLY VECTORISED ─────────────
        # For each known message, check if ANY of its nPts neighbours
        # has distance within [LB, UB]
        nbr_lb = self._global_lb[indices]                    # (n_known, nPts)
        nbr_ub = self._global_ub[indices]                    # (n_known, nPts)
        in_crown = (distances >= nbr_lb) & (distances <= nbr_ub)  # (n_known, nPts)
        any_in_crown = in_crown.any(axis=1)                  # (n_known,)

        # Messages with NO centroid in crown → ATTACK (condition A)
        out_of_range_mask = ~any_in_crown                    # (n_known,)
        oor_global = known_idx[out_of_range_mask]
        is_attack[oor_global] = True
        # Build reason strings only for flagged messages (cheaper than per-row)
        if out_of_range_mask.any():
            min_dists = distances[out_of_range_mask].min(axis=1)
            nearest_ubs = nbr_ub[out_of_range_mask, 0]
            for local_i, global_i in enumerate(oor_global):
                reasons[global_i] = (
                    f"Out of range: min_dist={min_dists[local_i]:.2f}, "
                    f"nearest_UB={nearest_ubs[local_i]:.2f}"
                )

        # ── Condition (d)(B): ID mismatch — VECTORISED ────────────────────
        # For messages that ARE in some crown, check if any crown-matching
        # centroid belongs to the message's own CAN ID
        crown_ok_local = np.where(any_in_crown)[0]          # local indices in known_idx
        if len(crown_ok_local) > 0:
            nbr_ids = self._global_ids[indices]              # (n_known, nPts)
            msg_ids = can_ids[known_idx]                     # (n_known,)

            # For each message in crown_ok_local, check:
            #   does any centroid that is in_crown AND belongs to msg's ID exist?
            sub_in_crown = in_crown[crown_ok_local]          # (n_sub, nPts)
            sub_nbr_ids  = nbr_ids[crown_ok_local]           # (n_sub, nPts)
            sub_msg_ids  = msg_ids[crown_ok_local]            # (n_sub,)

            # Broadcasting: match message ID against all neighbour IDs
            id_match = sub_nbr_ids == sub_msg_ids[:, None]   # (n_sub, nPts)
            # Must be in crown AND have matching ID
            has_own_id_in_crown = (sub_in_crown & id_match).any(axis=1)  # (n_sub,)

            # ID mismatch: in some crown but none with own ID
            id_mismatch = ~has_own_id_in_crown
            if id_mismatch.any():
                mismatch_positions = np.where(id_mismatch)[0]
                mismatch_global = known_idx[crown_ok_local[mismatch_positions]]
                is_attack[mismatch_global] = True

                for j, pos in enumerate(mismatch_positions):
                    g_idx = known_idx[crown_ok_local[pos]]
                    crown_ids_set = set(sub_nbr_ids[pos][sub_in_crown[pos]])
                    reasons[g_idx] = (
                        f"ID mismatch: msg_id={can_ids[g_idx]}, "
                        f"crown_ids={crown_ids_set}"
                    )

        n_attack = int(is_attack.sum())
        print(f"[DADA] Done — flagged {n_attack:,} / {N:,} as attack")

        df_out = df_test.copy()
        df_out["dada_anomaly"] = is_attack
        df_out["dada_reason"]  = reasons
        return df_out


# ══════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ══════════════════════════════════════════════════════════════════════

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str = "CLA-DADA") -> dict:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    acc  = (tp + tn) / (tp + tn + fp + fn) if (tp+tn+fp+fn) > 0 else 0
    prec = tp / (tp + fp)  if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn)  if (tp + fn) > 0 else 0
    f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr  = fn / (fn + tp) if (fn + tp) > 0 else 0

    w = 30
    sep = "─" * (w + 22)
    print(f"\n{'═'*(w+22)}")
    print(f"  {label} — Performance Metrics")
    print(sep)
    print(f"  {'TP':.<{w}} {tp:>10,}")
    print(f"  {'TN':.<{w}} {tn:>10,}")
    print(f"  {'FP':.<{w}} {fp:>10,}")
    print(f"  {'FN':.<{w}} {fn:>10,}")
    print(sep)
    print(f"  {'Accuracy':.<{w}} {acc:>10.4f}")
    print(f"  {'Precision (PPV)':.<{w}} {prec:>10.4f}")
    print(f"  {'Recall (TPR)':.<{w}} {rec:>10.4f}")
    print(f"  {'F1-Score':.<{w}} {f1:>10.4f}")
    print(f"  {'FPR (1-TNR)':.<{w}} {fpr:>10.4f}")
    print(f"  {'FNR (Miss rate)':.<{w}} {fnr:>10.4f}")
    print(f"{'═'*(w+22)}")

    return dict(tp=tp, tn=tn, fp=fp, fn=fn,
                accuracy=acc, precision=prec, recall=rec,
                f1=f1, fpr=fpr, fnr=fnr)


def _save_metrics(out_dir: str, metrics: dict, meta: dict) -> None:
    import json
    os.makedirs(out_dir, exist_ok=True)
    payload = {**metrics, **meta}
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(payload, fh, indent=2)
    with open(os.path.join(out_dir, "metrics.txt"), "w") as fh:
        for k, v in payload.items():
            fh.write(f"{k}: {v}\n")


def _plot_cla_dada(out_dir: str, dataset: str, y_test, y_pred, df_result, metrics: dict):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"CLA-DADA — {dataset}",
                 fontsize=14, fontweight="bold")

    ax = axes[0]
    cm = np.array([[metrics["tp"], metrics["fn"]],
                   [metrics["fp"], metrics["tn"]]])
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Attack", "Pred Normal"])
    ax.set_yticklabels(["Act Attack", "Act Normal"])
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=11, fontweight="bold")

    ax = axes[1]
    reason_summary = {"Unknown ID": 0, "Out of range": 0,
                      "ID mismatch": 0, "Normal": 0}
    for r in df_result["dada_reason"]:
        rs = str(r)
        if "Unknown" in rs:
            reason_summary["Unknown ID"] += 1
        elif "Out of range" in rs:
            reason_summary["Out of range"] += 1
        elif "ID mismatch" in rs:
            reason_summary["ID mismatch"] += 1
        else:
            reason_summary["Normal"] += 1
    labels = list(reason_summary.keys())
    sizes = [reason_summary[l] for l in labels]
    ax.pie(sizes, labels=labels,
           colors=["#FF6B6B", "#4ECDC4", "#FFE66D", "#95E77E"],
           autopct="%1.1f%%", startangle=90)
    ax.set_title("Detection Breakdown")

    plt.tight_layout()
    out_png = os.path.join(out_dir, "summary.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"[6] Plot saved → {out_png}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="CLA-DADA — faithful paper implementation (vectorised)"
    )
    ap.add_argument("--train",   default="input/FreeDrivingData_20180323_SONATA.csv")
    ap.add_argument("--test",    default="input/Flooding_dataset_SONATA.csv")
    ap.add_argument("--K",       type=int, default=300)
    ap.add_argument("--nPts",    type=int, default=30)
    ap.add_argument("--output-dir",   default="output")
    ap.add_argument("--profiles-dir", default=None,
                    help="Cached CLA profiles (default: <output-dir>/profiles)")
    ap.add_argument("--full", action="store_true",
                    help="Use entire test CSV (default: 20%% stratified hold-out)")
    ap.add_argument("--no-cache", action="store_true",
                    help="Force CLA retraining even if profiles exist")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    profiles_dir = args.profiles_dir or os.path.join(args.output_dir, "profiles")
    os.makedirs(profiles_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    cla_path = os.path.join(profiles_dir, "cla_dada_profiles.pkl")

    print(f"\n[1] Loading training data: {args.train}")
    df_train_raw = _load_csv(args.train)
    df_normal    = _extract_normals(df_train_raw)
    print(f"    Normal messages for CLA training: {len(df_normal):,}")

    cla = CLA(K=args.K)
    train_s = 0.0
    if os.path.exists(cla_path) and not args.no_cache:
        print(f"[2] Loading cached CLA profiles from {cla_path}")
        cla.load(cla_path)
    else:
        print(f"[2] Training CLA (K={args.K}) ...")
        t0 = time.time()
        cla.fit(df_normal)
        train_s = time.time() - t0
        print(f"    Training time: {train_s:.1f}s")
        cla.save(cla_path)

    print(f"\n[3] Loading test data: {args.test}")
    df_attack = _load_csv(args.test)
    y_all     = _true_labels(df_attack)

    if args.full:
        df_test = df_attack.reset_index(drop=True)
        y_test  = y_all
        split_label = "full dataset"
    else:
        from sklearn.model_selection import train_test_split
        idx = np.arange(len(df_attack))
        _, test_idx = train_test_split(idx, test_size=0.20, random_state=42,
                                       stratify=y_all)
        df_test = df_attack.iloc[test_idx].reset_index(drop=True)
        y_test  = y_all[test_idx]
        split_label = "20% hold-out"

    n0 = int((y_test == 0).sum())
    n1 = int((y_test == 1).sum())
    print(f"    {split_label}: {len(df_test):,} rows  ({n0:,} normal, {n1:,} attack)")

    print(f"\n[4] Running DADA (nPts={args.nPts}) ...")
    dada = DADA(sigma_map=cla.sigma_map, nPts=args.nPts)
    t0   = time.time()
    df_result = dada.detect(df_test)
    elapsed   = time.time() - t0
    print(f"    Detection time: {elapsed:.1f}s  "
          f"({elapsed/len(df_test)*1000:.3f} ms/msg)")

    y_pred   = df_result["dada_anomaly"].astype(int).values
    cladada_metrics = evaluate(y_test, y_pred, "CLA-DADA (D'Angelo 2021)")

    dataset_name = os.path.basename(os.path.normpath(args.output_dir))
    meta = {
        "train": os.path.basename(args.train),
        "test": os.path.basename(args.test),
        "K": args.K,
        "nPts": args.nPts,
        "split": split_label,
        "test_rows": len(df_test),
        "test_normal": n0,
        "test_attack": n1,
        "train_rows": len(df_normal),
        "detection_time_s": round(elapsed, 3),
        "train_time_s": round(train_s, 3),
    }
    _save_metrics(args.output_dir, cladada_metrics, meta)

    out_csv = os.path.join(args.output_dir, "results.csv")
    df_result.to_csv(out_csv, index=False)
    print(f"\n[5] Results saved → {out_csv}")
    print(f"     Metrics     → {args.output_dir}/metrics.json")

    if not args.no_plots:
        _plot_cla_dada(args.output_dir, dataset_name.upper(),
                       y_test, y_pred, df_result, cladada_metrics)


if __name__ == "__main__":
    main()