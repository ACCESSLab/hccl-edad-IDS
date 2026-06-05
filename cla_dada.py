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
import time
import pickle
import warnings
import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

DATA_COLS = [f"data{i}" for i in range(8)]


def _hex_to_int_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    result = pd.to_numeric(s, errors="coerce")
    mask = result.isna() & s.str.len().gt(0) & (s != "nan")
    if mask.any():
        h = s[mask].str.replace("0x", "", case=False, regex=False)
        result[mask] = h.apply(
            lambda v: int(v, 16)
            if v and all(c in "0123456789abcdefABCDEF" for c in v)
            else np.nan
        )
    return result.fillna(0)


def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    for alt in (" CAN ID", "CAN_ID", "can_id", "ArbID", "CANID"):
        if alt in df.columns:
            df = df.rename(columns={alt: "arbitration_id"})
    for i, alt in enumerate(["DATA[0]", "DATA[1]", "DATA[2]", "DATA[3]",
                             "DATA[4]", "DATA[5]", "DATA[6]", "DATA[7]"]):
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
    vals = pd.DataFrame({c: _hex_to_int_series(df[c]) for c in present}).values
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


class CLA:
    """Cluster-Based Learning Algorithm (Algorithm 1)."""

    def __init__(self, K: int = 300):
        self.K = K
        self.sigma_map: dict = {}
        self.known_ids: set = set()

    def fit(self, df_normal: pd.DataFrame) -> "CLA":
        print(f"\n[CLA] Training on {len(df_normal):,} normal messages (K={self.K})")

        self.known_ids = set(df_normal["arbitration_id"].unique())
        print(f"      {len(self.known_ids)} unique CAN IDs")

        for can_id, grp in df_normal.groupby("arbitration_id"):
            data = _get_bytes(grp)
            n = len(data)

            # Paper: 1 < K < N_ID. Single-sample IDs use one centroid, LB=UB=0.
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
            ubs = [r[-1] for r in rows]
            print(f"      ID {can_id:5}: {n:6,} msgs, {k} clusters, "
                  f"UB [{min(ubs):.1f}, {max(ubs):.1f}]")

        total = sum(len(v) for v in self.sigma_map.values())
        print(f"[CLA] Done: {total:,} centroids in Psi")
        return self

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"sigma_map": self.sigma_map,
                         "known_ids": self.known_ids,
                         "K": self.K}, f)
        print(f"[CLA] Saved Psi -> {path}")

    def load(self, path: str) -> "CLA":
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.sigma_map = d["sigma_map"]
        self.known_ids = d["known_ids"]
        self.K = d["K"]
        total = sum(len(v) for v in self.sigma_map.values())
        print(f"[CLA] Loaded Psi <- {path} ({len(self.known_ids)} IDs, {total:,} centroids)")
        return self


class DADA:
    """Data-Centric Anomaly Detection Algorithm (Algorithm 2)."""

    def __init__(self, sigma_map: dict, nPts: int = 30):
        self.sigma_map = sigma_map
        self.nPts = nPts
        self.known_ids = set(sigma_map.keys())
        self._global_centroids: np.ndarray = None
        self._global_ids: np.ndarray = None
        self._global_lb: np.ndarray = None
        self._global_ub: np.ndarray = None
        self._knn_index: NearestNeighbors = None
        self._build_global_index()

    def predict_df(self, df_test):
        return self.detect(df_test)["dada_anomaly"].astype(int).values

    def _build_global_index(self) -> None:
        rows_c, rows_id, rows_lb, rows_ub = [], [], [], []
        for can_id, arr in self.sigma_map.items():
            rows_c.append(arr[:, :8])
            rows_id.extend([can_id] * len(arr))
            rows_lb.append(arr[:, 8])
            rows_ub.append(arr[:, 9])

        self._global_centroids = np.vstack(rows_c)
        self._global_ids = np.array(rows_id)
        self._global_lb = np.concatenate(rows_lb)
        self._global_ub = np.concatenate(rows_ub)

        M = len(self._global_centroids)
        k_search = min(self.nPts, M)
        self._knn_index = NearestNeighbors(
            n_neighbors=k_search,
            algorithm="ball_tree",
            metric="euclidean",
            n_jobs=-1,
        )
        self._knn_index.fit(self._global_centroids)
        print(f"[DADA] Index: {M:,} centroids, nPts={self.nPts}")

    def detect(self, df_test: pd.DataFrame) -> pd.DataFrame:
        print(f"\n[DADA] Detecting in {len(df_test):,} messages")

        byte_matrix = _get_bytes(df_test)
        can_ids = df_test["arbitration_id"].values
        N = len(df_test)

        is_attack = np.zeros(N, dtype=bool)
        reasons = np.empty(N, dtype=object)
        reasons[:] = "Normal"

        # (c) unknown CAN ID
        known_set = self.known_ids
        unknown_mask = np.array([cid not in known_set for cid in can_ids])
        is_attack[unknown_mask] = True
        reasons[unknown_mask] = "Unknown CAN ID"

        known_idx = np.where(~unknown_mask)[0]
        if len(known_idx) == 0:
            df_out = df_test.copy()
            df_out["dada_anomaly"] = is_attack
            df_out["dada_reason"] = reasons
            return df_out

        X_known = byte_matrix[known_idx]
        distances, indices = self._knn_index.kneighbors(X_known)

        # (d.A) outside all crown areas
        nbr_lb = self._global_lb[indices]
        nbr_ub = self._global_ub[indices]
        in_crown = (distances >= nbr_lb) & (distances <= nbr_ub)
        any_in_crown = in_crown.any(axis=1)

        out_of_range_mask = ~any_in_crown
        oor_global = known_idx[out_of_range_mask]
        is_attack[oor_global] = True
        if out_of_range_mask.any():
            min_dists = distances[out_of_range_mask].min(axis=1)
            nearest_ubs = nbr_ub[out_of_range_mask, 0]
            for local_i, global_i in enumerate(oor_global):
                reasons[global_i] = (
                    f"Out of range: min_dist={min_dists[local_i]:.2f}, "
                    f"nearest_UB={nearest_ubs[local_i]:.2f}"
                )

        # (d.B) in crown but no matching CAN ID
        crown_ok_local = np.where(any_in_crown)[0]
        if len(crown_ok_local) > 0:
            nbr_ids = self._global_ids[indices]
            msg_ids = can_ids[known_idx]

            sub_in_crown = in_crown[crown_ok_local]
            sub_nbr_ids = nbr_ids[crown_ok_local]
            sub_msg_ids = msg_ids[crown_ok_local]

            id_match = sub_nbr_ids == sub_msg_ids[:, None]
            has_own_id_in_crown = (sub_in_crown & id_match).any(axis=1)

            id_mismatch = ~has_own_id_in_crown
            if id_mismatch.any():
                mismatch_positions = np.where(id_mismatch)[0]
                mismatch_global = known_idx[crown_ok_local[mismatch_positions]]
                is_attack[mismatch_global] = True

                for pos in mismatch_positions:
                    g_idx = known_idx[crown_ok_local[pos]]
                    crown_ids_set = set(sub_nbr_ids[pos][sub_in_crown[pos]])
                    reasons[g_idx] = (
                        f"ID mismatch: msg_id={can_ids[g_idx]}, "
                        f"crown_ids={crown_ids_set}"
                    )

        print(f"[DADA] Flagged {int(is_attack.sum()):,} / {N:,} as attack")

        df_out = df_test.copy()
        df_out["dada_anomaly"] = is_attack
        df_out["dada_reason"] = reasons
        return df_out


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str = "CLA-DADA") -> dict:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    print(f"\n--- {label} ---")
    print(f"  TP={tp:,}  TN={tn:,}  FP={fp:,}  FN={fn:,}")
    print(f"  Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}")
    print(f"  F1={f1:.4f}  FPR={fpr:.4f}  FNR={fnr:.4f}")

    return dict(tp=tp, tn=tn, fp=fp, fn=fn,
                accuracy=acc, precision=prec, recall=rec,
                f1=f1, fpr=fpr, fnr=fnr)


def main():
    ap = argparse.ArgumentParser(description="CLA-DADA CAN intrusion detection")
    ap.add_argument("--train", required=True, help="Training CSV (normal traffic)")
    ap.add_argument("--test", required=True, help="Test CSV (normal + attack)")
    ap.add_argument("--K", type=int, default=300,
                    help="K-means clusters per CAN ID (default: 300)")
    ap.add_argument("--nPts", type=int, default=30,
                    help="k-NN neighbours for DADA (default: 30)")
    ap.add_argument("--profiles-dir", default="profiles/")
    ap.add_argument("--no-cache", action="store_true",
                    help="Retrain CLA even if cached profiles exist")
    args = ap.parse_args()

    os.makedirs(args.profiles_dir, exist_ok=True)
    cla_path = os.path.join(args.profiles_dir, "cla_dada_profiles.pkl")

    print(f"\n[1] Loading training data: {args.train}")
    df_train_raw = _load_csv(args.train)
    df_normal = _extract_normals(df_train_raw)
    print(f"    {len(df_normal):,} normal messages")

    cla = CLA(K=args.K)
    if os.path.exists(cla_path) and not args.no_cache:
        print(f"[2] Loading cached profiles from {cla_path}")
        cla.load(cla_path)
    else:
        print(f"[2] Training CLA (K={args.K})")
        t0 = time.time()
        cla.fit(df_normal)
        print(f"    Training time: {time.time() - t0:.1f}s")
        cla.save(cla_path)

    print(f"\n[3] Loading test data: {args.test}")
    df_attack = _load_csv(args.test)
    y_all = _true_labels(df_attack)

    idx = np.arange(len(df_attack))
    _, test_idx = train_test_split(idx, test_size=0.20, random_state=42,
                                   stratify=y_all)
    df_test = df_attack.iloc[test_idx].reset_index(drop=True)
    y_test = y_all[test_idx]

    n0 = int((y_test == 0).sum())
    n1 = int((y_test == 1).sum())
    print(f"    Test split: {len(df_test):,} rows ({n0:,} normal, {n1:,} attack)")

    print(f"\n[4] Running DADA (nPts={args.nPts})")
    dada = DADA(sigma_map=cla.sigma_map, nPts=args.nPts)
    t0 = time.time()
    df_result = dada.detect(df_test)
    elapsed = time.time() - t0
    print(f"    Detection time: {elapsed:.1f}s ({elapsed / len(df_test) * 1000:.3f} ms/msg)")

    y_pred = df_result["dada_anomaly"].astype(int).values
    evaluate(y_test, y_pred, "CLA-DADA")

    out_csv = "cla_dada_results.csv"
    df_result.to_csv(out_csv, index=False)
    print(f"\n[5] Results saved -> {out_csv}")


if __name__ == "__main__":
    main()
