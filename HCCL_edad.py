
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HCCL Clustering-Based Detection Module
"""

import os
import time
import pickle
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist as _sp_cdist
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

warnings.filterwarnings("ignore")

_RNG = np.random.RandomState(42)  # Module-level RNG, no global seed


# ============================================================
# Internal data structures
# ============================================================

@dataclass
class PerIDModel:
    """All per-CAN-ID model artefacts produced during training."""
    method: str                          # "kmeans" | "kmeans_svm"
    scaler: Any
    kmeans: Any
    centroids: np.ndarray
    cluster_bounds: List[Dict[str, Any]]
    n_clusters: int
    n_samples: int
    svm: Optional[Any] = None
    svm_params: Optional[Dict[str, Any]] = None


@dataclass
class PerIDThresholds:
    """Calibrated decision thresholds for a single CAN ID."""
    kmeans_dist_thresh: float
    svm_score_thresh: Optional[float]
    cluster_hi: Optional[np.ndarray] = None
    cluster_lo: Optional[np.ndarray] = None
    knn_k: int = 1


# ============================================================
# Helper utilities
# ============================================================

def _mad(x: np.ndarray) -> float:
    """Median Absolute Deviation (robust spread estimator)."""
    med = np.median(x)
    return float(np.median(np.abs(x - med))) + 1e-9


# ============================================================
# Core training
# ============================================================

def _build_per_id_models(
    df_normal: pd.DataFrame,
    id_col: str,
    feature_cols: List[str],
    alpha: float = 0.01,
    n_clusters_cap: int = 20,
    svm_grid_search: bool = True,
    verbose: bool = True,
) -> Dict[int, PerIDModel]:
    """
    Train per-CAN-ID models on NORMAL traffic only.

    SVM selection strategy (when svm_grid_search=True):
        For each CAN ID, a 20% validation split is held out.
        A grid of (nu, gamma) combinations is evaluated.
        The combination that minimises FPR on the validation split
        is selected and re-fit on the full training data.
        This takes ~3-5 minutes on HCRL but gives substantially
        better SVM thresholds, especially for Gear/RPM attacks.

    When svm_grid_search=False:
        Uses fixed nu=alpha, gamma="scale" — fast but less optimal.
    """
    models: Dict[int, PerIDModel] = {}

    grouped = df_normal.groupby(id_col)
    if verbose:
        print(f"  📊 Training per-ID models for {len(grouped)} unique CAN IDs …")

    for can_id, g in grouped:
        X = g.drop_duplicates(subset=feature_cols)
        Z = X[feature_cols].astype(float).values

        # ---- Tiny-ID tolerant model ----------------------------------------
        if len(Z) < 2:
            scaler = StandardScaler()
            Zs = scaler.fit_transform(Z) if len(Z) > 0 else np.zeros((1, len(feature_cols)))
            cents = np.mean(Zs, axis=0, keepdims=True)
            # Keep tiny-ID fallback strict; loose defaults create FN-heavy acceptance.
            cb = [{"centroid": cents[0], "max_dist_p95": 0.05, "n": len(Zs)}]
            mdl = PerIDModel("kmeans", scaler, None, cents, cb, 1, len(Z))
            models[int(can_id)] = mdl
            if verbose:
                print(f"    • ID {can_id}: tiny-model (k=1, tolerant)")
            continue

        # ---- Normal model --------------------------------------------------
        scaler = StandardScaler()
        Zs = scaler.fit_transform(Z)
        n = len(Zs)

        # Adaptive K — denser coverage, closer to CLA-DADA's K=200-300
        if n < 80:
            k = max(2, min(10, n // 4))
        elif n < 400:
            k = min(n_clusters_cap, 20)
        else:
            k = min(n_clusters_cap, int(np.sqrt(n)) + 10)

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Zs)
        cents  = kmeans.cluster_centers_
        lab    = kmeans.labels_

        # Per-cluster distance bounds — include strict LB/UB (like CLA-DADA)
        cluster_bounds = []
        for i in range(k):
            pts = Zs[lab == i]
            if len(pts) == 0:
                continue
            d = np.linalg.norm(pts - cents[i], axis=1)
            cluster_bounds.append({
                "centroid":      cents[i],
                "max_dist_p95":  float(np.percentile(d, 95)),
                "dist_lb":       float(d.min()),
                "dist_ub":       float(d.max()),
                "n":             int(len(pts)),
            })

        mdl = PerIDModel(
            method="kmeans",
            scaler=scaler,
            kmeans=kmeans,
            centroids=cents,
            cluster_bounds=cluster_bounds,
            n_clusters=k,
            n_samples=n,
        )

        # ---- SVM layer -------------------------------------------------------
        if n >= 30:
            if svm_grid_search:
                # ── Principled grid search ────────────────────────────────────
                # Objective: minimise FPR on held-out validation normals.
                # nu   : upper bound on training outlier fraction and support
                #        vectors. Smaller nu → tighter boundary.
                # gamma: RBF kernel bandwidth. "scale" = 1/(n_feat*var).
                #        Smaller float → wider kernel → smoother boundary.
                n_val   = max(10, int(n * 0.20))
                idx_all = _RNG.permutation(n)
                idx_val = idx_all[:n_val]
                idx_fit = idx_all[n_val:]
                Zs_fit  = Zs[idx_fit]
                Zs_val  = Zs[idx_val]

                nu_grid    = [0.001, 0.005, 0.01, 0.02, 0.05]
                gamma_grid = ["scale", 0.01, 0.1, 1.0]

                best_svm       = None
                best_params    = None
                best_fpr       = float("inf")
                best_score_thr = None

                for nu_val in nu_grid:
                    for gam in gamma_grid:
                        try:
                            svm_cand = OneClassSVM(
                                kernel="rbf", nu=nu_val, gamma=gam
                            )
                            svm_cand.fit(Zs_fit)

                            # Score validation normals
                            val_scores = svm_cand.decision_function(
                                Zs_val
                            ).flatten()

                            # Threshold = alpha quantile of validation scores
                            thr_cand = float(np.quantile(val_scores, alpha))

                            # FPR on validation set at this threshold
                            fpr_val = float(
                                (val_scores < thr_cand).sum()
                            ) / len(val_scores)

                            # Prefer lowest FPR; break ties by choosing
                            # tighter nu (more sensitive to attacks)
                            if fpr_val < best_fpr or (
                                fpr_val == best_fpr
                                and nu_val > (best_params or {}).get("nu", 0)
                            ):
                                best_fpr       = fpr_val
                                best_svm       = svm_cand
                                best_params    = {"nu": nu_val, "gamma": gam}
                                best_score_thr = thr_cand

                        except Exception:
                            continue

                if best_svm is not None:
                    # Re-fit on full Zs for better decision boundary coverage
                    try:
                        best_svm_full = OneClassSVM(
                            kernel="rbf",
                            nu=best_params["nu"],
                            gamma=best_params["gamma"],
                        )
                        best_svm_full.fit(Zs)
                        mdl.method     = "kmeans_svm"
                        mdl.svm        = best_svm_full
                        mdl.svm_params = {
                            **best_params,
                            "val_fpr":   round(best_fpr, 4),
                            "score_thr": best_score_thr,
                        }
                    except Exception:
                        mdl.method     = "kmeans_svm"
                        mdl.svm        = best_svm
                        mdl.svm_params = {
                            **best_params,
                            "val_fpr":   round(best_fpr, 4),
                            "score_thr": best_score_thr,
                        }
                else:
                    # Grid search failed — fall back to IsolationForest
                    if verbose:
                        print(f"    ID {can_id}: SVM grid search failed, "
                              f"trying IsolationForest …")
                    try:
                        iso = IsolationForest(
                            n_estimators=200, contamination="auto",
                            random_state=42, n_jobs=-1,
                        ).fit(Zs)
                        mdl.method     = "kmeans_svm"
                        mdl.svm        = iso
                        mdl.svm_params = {"fallback": "IsolationForest"}
                    except Exception:
                        pass

            else:
                # ── Fast fixed SVM (no grid search) ──────────────────────────
                try:
                    nu  = max(0.001, min(0.1, alpha))
                    svm = OneClassSVM(kernel="rbf", nu=nu, gamma="scale")
                    svm.fit(Zs)
                    mdl.method     = "kmeans_svm"
                    mdl.svm        = svm
                    mdl.svm_params = {"nu": nu, "gamma": "scale",
                                      "val_fpr": None}
                except Exception as e_svm:
                    if verbose:
                        print(f"    ID {can_id}: SVM failed ({e_svm}), "
                              f"trying IsolationForest …")
                    try:
                        iso = IsolationForest(
                            n_estimators=200, contamination="auto",
                            random_state=42, n_jobs=-1,
                        ).fit(Zs)
                        mdl.method     = "kmeans_svm"
                        mdl.svm        = iso
                        mdl.svm_params = {"fallback": "IsolationForest"}
                    except Exception:
                        pass

        if verbose:
            if mdl.svm is not None and mdl.svm_params:
                p = mdl.svm_params
                if "val_fpr" in p:
                    extra = (f" + SVM(nu={p['nu']}, gamma={p['gamma']}, "
                             f"val_fpr={p['val_fpr']:.4f})")
                else:
                    extra = f" + {p.get('fallback','SVM')}"
            else:
                extra = ""
            print(f"    • ID {can_id}: KMeans(k={k}){extra}  (n={n})")

        models[int(can_id)] = mdl

    return models


# ============================================================
# Threshold calibration
# ============================================================

def _calibrate_thresholds(
    models: Dict[int, PerIDModel],
    calib_df: pd.DataFrame,
    id_col: str,
    feature_cols: List[str],
    alpha: float = 0.01,
    verbose: bool = True,
) -> Dict[int, PerIDThresholds]:
    """
    Derive per-ID thresholds AND per-cluster [lo, hi] bands
    from held-out calibration normals (fully vectorised per CAN ID).
    """
    q_hi = 1.0 - alpha
    q_lo = alpha

    per_id_dist:    Dict[int, List[float]] = defaultdict(list)
    per_id_scores:  Dict[int, List[float]] = defaultdict(list)
    per_clust_dist: Dict[int, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))

    calib_norm = (
        calib_df[calib_df["label"] == 0]
        if "label" in calib_df.columns
        else calib_df
    )

    # ── Vectorised calibration pass ───────────────────────────────────────
    feat_arr = calib_norm[feature_cols].astype(float).values
    id_arr   = calib_norm[id_col].astype(int).values

    for cid in np.unique(id_arr):
        if cid not in models:
            continue
        m       = models[cid]
        row_idx = np.where(id_arr == cid)[0]
        X       = feat_arr[row_idx]
        Xs      = (X - m.scaler.mean_) / m.scaler.scale_

        # Batch pairwise distances
        dists       = _sp_cdist(Xs, m.centroids)
        nearest_idx = np.argmin(dists, axis=1)
        nearest_d   = dists[np.arange(len(Xs)), nearest_idx]

        # Collect per-cluster distances
        for j in range(m.n_clusters):
            mask = nearest_idx == j
            if mask.any():
                per_clust_dist[cid][j].extend(nearest_d[mask].tolist())
        per_id_dist[cid].extend(nearest_d.tolist())

        # SVM scores — single batch call
        if m.svm is not None:
            try:
                svm_scores = m.svm.decision_function(Xs).flatten()
                per_id_scores[cid].extend(svm_scores.tolist())
            except Exception:
                pass

    thresholds: Dict[int, PerIDThresholds] = {}

    for cid, m in models.items():
        d_arr = np.array(per_id_dist.get(cid, []), dtype=float)
        train_p95 = (
            np.array([b["max_dist_p95"] for b in m.cluster_bounds])
            if m.cluster_bounds else np.array([1.0])
        )
        train_floor = float(np.mean(train_p95))

        if d_arr.size >= 50:
            dist_thr = float(np.quantile(d_arr, q_hi))
        elif d_arr.size >= 10:
            dist_thr = float(np.median(d_arr) + 3.0 * _mad(d_arr))
        elif d_arr.size > 0:
            dist_thr = float(np.median(d_arr) + 2.5 * _mad(d_arr))
        else:
            dist_thr = train_floor if train_floor > 0 else 1.0
        dist_thr = max(dist_thr, 1.02 * train_floor)

        # Per-cluster [lo, hi] bands
        k  = m.n_clusters
        hi = np.zeros(k, dtype=float)
        lo = np.zeros(k, dtype=float)

        for j in range(k):
            arr = np.array(per_clust_dist[cid].get(j, []), dtype=float)
            if arr.size >= 30:
                lo[j] = float(np.quantile(arr, max(0.0, q_lo / 2)))
                hi[j] = float(np.quantile(arr, q_hi))
            elif arr.size >= 8:
                med = np.median(arr)
                mad_val = _mad(arr)
                lo[j] = max(0.0, float(med - 2.0 * mad_val))
                hi[j] = float(med + 3.0 * mad_val)
            elif arr.size > 0:
                med = np.median(arr)
                mad_val = _mad(arr)
                lo[j] = 0.0
                hi[j] = max(float(med + 2.5 * mad_val), float(train_floor * 1.10))
            else:
                lo[j] = 0.0
                hi[j] = float(train_floor * 1.10)

        # SVM score threshold
        s_arr = np.array(per_id_scores.get(cid, []), dtype=float)
        score_thr = float(np.quantile(s_arr, q_lo)) if s_arr.size >= 30 else None

        thresholds[cid] = PerIDThresholds(
            kmeans_dist_thresh = dist_thr,
            svm_score_thresh   = score_thr,
            cluster_hi         = hi,
            cluster_lo         = lo,
            knn_k              = min(20, k),
        )

        if verbose:
            svm_str = f", svm_thr={score_thr:.4f}" if score_thr is not None else ""
            print(f"    ID {cid}: dist_thr={dist_thr:.4f}, kNN(k={min(20,k)}) bands ready{svm_str}")

    return thresholds


# ============================================================
# Extend models with calibration-only IDs
# ============================================================

def _extend_models_with_calibration(
    models: Dict[int, PerIDModel],
    calib_norm: pd.DataFrame,
    id_col: str,
    feature_cols: List[str],
    alpha: float,
    verbose: bool = True,
) -> None:
    known   = set(models.keys())
    new_ids = []

    for cid, g in calib_norm.groupby(id_col):
        if int(cid) in known:
            continue
        mdf   = g.assign(label=0)
        added = _build_per_id_models(
            mdf, id_col, feature_cols,
            alpha=alpha, n_clusters_cap=20,
            svm_grid_search=False, verbose=False,
        )
        if int(cid) in added:
            models[int(cid)] = added[int(cid)]
            new_ids.append(cid)

    if verbose and new_ids:
        preview = new_ids[:8]
        suffix  = "…" if len(new_ids) > 8 else ""
        print(f"  🔁 Added {len(new_ids)} IDs from calibration normals: {preview}{suffix}")


# ============================================================
# Detection (single-sample scoring — kept for backward compat)
# ============================================================

def _score_row(
    x_raw: np.ndarray,
    cid: int,
    models: Dict[int, PerIDModel],
    thrs: Dict[int, PerIDThresholds],
) -> Tuple[int, str, float, float]:
    if cid not in models or cid not in thrs:
        return 1, "Unknown CAN ID", 1.0, 1.0

    m  = models[cid]
    t  = thrs[cid]
    xs = m.scaler.transform([x_raw])[0]
    dists = np.linalg.norm(m.centroids - xs, axis=1)

    # DCLA-style k-NN primary acceptance gate
    k    = t.knn_k
    idxs = np.argsort(dists)[:k]
    accept = False
    for j in idxs:
        dj = dists[j]
        lo = t.cluster_lo[j] if t.cluster_lo is not None else 0.0
        hi = t.cluster_hi[j] if t.cluster_hi is not None else np.inf
        if lo <= dj <= hi:
            accept = True
            break

    if accept:
        conf  = float(np.clip(1.0 - dists[idxs[0]] / (t.kmeans_dist_thresh + 1e-9), 0.0, 1.0))
        score = float(np.clip(dists[idxs[0]] / (t.kmeans_dist_thresh + 1e-9) * 0.4, 0.0, 0.5))
        return 0, "Centroid-band normal", conf, score

    nearest    = float(np.min(dists))
    k_conf     = float(np.clip(nearest / (t.kmeans_dist_thresh + 1e-9), 0.0, 1.0))
    kmeans_anom = nearest > t.kmeans_dist_thresh

    if kmeans_anom:
        excess = (nearest - t.kmeans_dist_thresh) / (t.kmeans_dist_thresh + 1e-9)
        raw_score = 0.5 + float(np.clip(0.5 * excess, 0.0, 0.5))
    else:
        raw_score = float(np.clip(0.4 * nearest / (t.kmeans_dist_thresh + 1e-9), 0.0, 0.5))

    svm_anom, s_conf = False, 0.0
    if m.svm is not None and t.svm_score_thresh is not None:
        try:
            sc = float(m.svm.decision_function([xs])[0])
            if sc < t.svm_score_thresh:
                svm_anom = True
                s_conf   = float(np.clip(
                    (t.svm_score_thresh - sc) / (abs(t.svm_score_thresh) + 1e-6),
                    0.0, 1.0,
                ))
        except Exception:
            pass

    if m.method == "kmeans_svm":
        is_anom = kmeans_anom or svm_anom
        if kmeans_anom and svm_anom:
            reason = "Both KMeans+SVM anomaly"
            conf   = float(np.clip(max(k_conf, s_conf) * 1.2, 0.0, 1.0))
        elif kmeans_anom:
            reason = "KMeans anomaly"
            conf   = k_conf
        elif svm_anom:
            reason = "SVM anomaly"
            conf   = s_conf
        else:
            reason = "Normal"
            conf   = float(np.clip(1.0 - max(k_conf, s_conf), 0.0, 1.0))
    else:
        is_anom = kmeans_anom
        reason  = "KMeans anomaly" if kmeans_anom else "Normal"
        conf    = k_conf if kmeans_anom else float(np.clip(1.0 - k_conf, 0.0, 1.0))

    return int(is_anom), reason, float(np.clip(conf, 0.0, 1.0)), raw_score


# ============================================================
# Public ClusteringDetector
# ============================================================

class ClusteringDetector:
    """
    Enhanced per-CAN-ID clustering detector (HCCL-EDAD Advanced Edition).

    Detection layers (in order of application):
        Layer 0a — Invariant Byte Check:
                   Bytes that NEVER change in normal training (single unique
                   value) are strictly enforced. Any deviation is an anomaly.
                   Catches Gear/RPM/tampering attacks on known IDs.
                   Zero FP risk: only applied to truly invariant bytes.

        Layer 0b — All-Zero Payload Check:
                   Flags messages where ALL 8 payload bytes are zero on a
                   known ID that never produces all-zero payloads in training.
                   Catches DoS flooding (CAN ID=0 injects zero-padded frames).

        Layer 1  — Unknown CAN ID → anomaly (existing)

        Layer 2  — k-NN centroid band acceptance → normal (existing)

        Layer 3  — Distance threshold → anomaly (existing)

        Layer 4  — SVM/IsolationForest refinement (existing)

        Layer 5  — Stationarity / frozen-payload run detection (existing)

    Lightweight design: Layers 0a/0b add <0.1ms per message overhead.
    No retraining needed after enhancement — constraints learned from
    existing normal profiles during build_profiles_from().
    """

    def __init__(
        self,
        data: pd.DataFrame,
        clusters_per_id: int = 20,
        profile_path: Optional[str] = None,
        alpha: float = 0.01,
        calib_frac: float = 0.30,
        id_col: str = "arbitration_id",
        verbose: bool = True,
        distance_multiplier: float = 1.0,
        svm_grid_search: bool = True,
        global_npts: int = 5,
        cross_id_margin: float = 0.10,
    ):
        self.data                = data
        self.n_clusters_cap      = clusters_per_id
        self.alpha               = alpha
        self.calib_frac          = calib_frac
        self.id_col              = id_col
        self.verbose             = verbose
        self.distance_multiplier = distance_multiplier
        self.svm_grid_search     = svm_grid_search
        self._global_npts        = max(0, int(global_npts))
        self._cross_id_margin    = max(0.0, float(cross_id_margin))
        self.feature_cols: List[str]                  = []
        self.models:        Dict[int, PerIDModel]      = {}
        self.thresholds:    Dict[int, PerIDThresholds] = {}
        self._train_time    = 0.0
        self._detect_time   = 0.0
        self._fast_tables:  Dict[int, dict]            = {}

        # Layer 0a: invariant byte constraints
        # { can_id: { byte_idx: invariant_value } }
        self._invariant_bytes: Dict[int, Dict[int, float]] = {}

        # Layer 0b: IDs that never produce all-zero payloads
        self._no_allzero_ids: Set[int] = set()

        # Layer 0c: per-ID payload whitelist (exact normal payload hashes)
        self._seen_payload_hashes: Dict[int, Set[int]] = {}

        # Layer 5: stationarity limits
        self._stationarity_limits: Dict[int, int] = {}

        # Layer 6: global cross-ID consistency index (like DADA)
        self._global_index: Optional[NearestNeighbors] = None
        self._global_cents:  Optional[np.ndarray] = None   # (M, n_feat)
        self._global_ids:    Optional[np.ndarray] = None   # (M,) CAN ID per centroid

        if profile_path and os.path.exists(profile_path):
            self.load_profiles(profile_path)

    # ------------------------------------------------------------------
    def _build_global_index(self) -> None:
        if self._global_npts == 0:  # Layer 6 disabled
            return
        """
        Layer 6: global cross-ID consistency index (mirrors DADA).

        Concatenates all per-ID centroids into a single global pool and
        builds a NearestNeighbors index over them. During scoring, for each
        message we query the globally nearest centroids and check whether any
        of them belongs to the message's own CAN ID. If none do, the payload
        is closer to a different ID's normal space than to its own, which is
        strong evidence of injection or fuzzing.
        """
        if not self.models:
            return

        cents_list, ids_list = [], []
        for cid, m in self.models.items():
            # Compare in raw feature space: global queries use raw feature rows.
            # Per-ID centroids are stored in scaled space, so convert back.
            c = m.scaler.inverse_transform(m.centroids).astype(np.float64)
            cents_list.append(c)
            ids_list.extend([cid] * len(c))

        self._global_cents = np.vstack(cents_list)
        self._global_ids   = np.array(ids_list, dtype=int)

        k_search = min(self._global_npts, len(self._global_cents))
        self._global_index = NearestNeighbors(
            n_neighbors=k_search,
            algorithm="ball_tree",
            metric="euclidean",
            n_jobs=-1,
        ).fit(self._global_cents)

        if self.verbose:
            print(f"  Layer 6 : global index built — "
                  f"{len(self._global_cents):,} centroids across "
                  f"{len(self.models)} CAN IDs  (nPts={k_search})")

    # ------------------------------------------------------------------
    def _precompute_fast_tables(self) -> None:
        self._fast_tables = {}
        for cid, m in self.models.items():
            t = self.thresholds.get(cid)
            if t is None:
                continue
            K          = m.centroids.shape[0]
            scale_safe = np.maximum(
                m.scaler.scale_.astype(np.float64), 1e-3
            )
            hi = (t.cluster_hi * self.distance_multiplier) \
                 if t.cluster_hi is not None \
                 else np.full(K, np.inf, dtype=np.float64)

            self._fast_tables[cid] = {
                "mean":           m.scaler.mean_.astype(np.float64),
                "scale":          scale_safe,
                "cents":          m.centroids.astype(np.float64),
                "lo":             t.cluster_lo if t.cluster_lo is not None
                                  else np.zeros(K, dtype=np.float64),
                "hi":             hi,
                "dist_thr":       float(t.kmeans_dist_thresh)
                                  * self.distance_multiplier,
                "svm":            m.svm,
                "svm_thr":        t.svm_score_thresh,
                "knn_k":          int(t.knn_k),
                "max_frozen_run": self._stationarity_limits.get(cid, 500),
                # Layer 0a: invariant bytes {byte_idx: expected_value}
                "inv_bytes":      self._invariant_bytes.get(cid, {}),
                # Layer 0b: True if this ID never produces all-zero payload
                "no_allzero":     cid in self._no_allzero_ids,
                # Layer 0c: known payload hashes for this ID
                "seen_payload_hashes": np.array(
                    sorted(self._seen_payload_hashes.get(cid, set())),
                    dtype=np.uint64,
                ),
            }

    # ------------------------------------------------------------------
    def _resolve_id_col(self) -> str:
        if self.id_col in self.data.columns:
            return self.id_col
        for cand in ["arbitration_id", " CAN ID", "CAN_ID", "can_id"]:
            if cand in self.data.columns:
                return cand
        raise ValueError(
            f"Cannot find CAN-ID column. Tried '{self.id_col}' and common aliases. "
            f"Columns present: {list(self.data.columns)}"
        )

    # ------------------------------------------------------------------
    def _learn_lightweight_constraints(
        self, normal_df: pd.DataFrame, id_col: str
    ) -> None:
        """
        Learn two zero-cost lightweight constraints from normal traffic:

        Layer 0a — Invariant bytes:
            For each CAN ID, identify payload bytes whose value NEVER
            changes across ALL normal messages. These are strictly
            invariant (e.g. a status byte always = 0xFE, a DLC byte
            always = 0x08). Any message deviating from these values is
            immediately flagged as anomalous.

            Only applied to truly invariant bytes (nunique == 1).
            Zero FP risk by construction: if a byte varies even once
            in normal traffic, it is not constrained.

        Layer 0b — All-zero payload exclusion:
            If a CAN ID NEVER produces an all-zero payload in normal
            traffic, flag any message from that ID with payload=00...00.
            DoS floods CAN ID=0x000 with zero-padded frames — Layer 1
            already catches these (unknown ID), but this layer also
            catches DoS-style attacks that happen to use a known CAN ID
            with zero-padded payloads.
        """
        byte_cols = [f"data{i}" for i in range(8)]
        avail = [c for c in byte_cols if c in normal_df.columns]
        if len(avail) < 8:
            return

        id_arr   = normal_df[id_col].astype(int).values
        byte_arr = normal_df[avail].values.astype(float)

        inv_bytes:     Dict[int, Dict[int, float]] = {}
        no_allzero_ids: Set[int] = set()
        seen_payload_hashes: Dict[int, Set[int]] = {}
        n_inv_total = 0

        for cid in np.unique(id_arr):
            mask = id_arr == cid
            b    = byte_arr[mask]
            if len(b) < 2:
                continue

            # Layer 0a: find invariant bytes
            inv = {}
            for bi in range(8):
                col_vals = b[:, bi]
                if len(np.unique(col_vals)) == 1:
                    inv[bi] = float(col_vals[0])
            if inv:
                inv_bytes[int(cid)] = inv
                n_inv_total += len(inv)

            # Layer 0b: check if any normal message has all-zero payload
            all_zero_rows = (b == 0).all(axis=1)
            if not all_zero_rows.any():
                no_allzero_ids.add(int(cid))

            # Layer 0c: exact payload whitelist for this CAN ID.
            # Using 64-bit packed hash of the 8-byte payload.
            payload_u8 = b.astype(np.uint8)
            packed = (
                (payload_u8[:, 0].astype(np.uint64) << 56)
                | (payload_u8[:, 1].astype(np.uint64) << 48)
                | (payload_u8[:, 2].astype(np.uint64) << 40)
                | (payload_u8[:, 3].astype(np.uint64) << 32)
                | (payload_u8[:, 4].astype(np.uint64) << 24)
                | (payload_u8[:, 5].astype(np.uint64) << 16)
                | (payload_u8[:, 6].astype(np.uint64) << 8)
                | payload_u8[:, 7].astype(np.uint64)
            )
            seen_payload_hashes[int(cid)] = set(np.unique(packed).tolist())

        self._invariant_bytes  = inv_bytes
        self._no_allzero_ids   = no_allzero_ids
        self._seen_payload_hashes = seen_payload_hashes

        if self.verbose:
            print(f"\n--- HCCL: Lightweight Constraints ---")
            print(f"  Layer 0a: {len(inv_bytes)} IDs with invariant bytes "
                  f"({n_inv_total} total constrained byte positions)")
            print(f"  Layer 0b: {len(no_allzero_ids)} IDs that never "
                  f"produce all-zero payloads")
            n_seen = sum(len(v) for v in seen_payload_hashes.values())
            print(f"  Layer 0c: {len(seen_payload_hashes)} IDs with payload whitelists "
                  f"({n_seen} unique payload signatures)")
            # Show examples
            for cid, inv in list(inv_bytes.items())[:3]:
                ex = ", ".join(f"byte{k}={v:.0f}" for k,v in inv.items())
                print(f"    ID {cid}: invariant [{ex}]")

    # ------------------------------------------------------------------
    def _learn_stationarity_limits(
        self, normal_df: pd.DataFrame, id_col: str,
        margin: float = 1.25, min_threshold: int = 50
    ) -> None:
        """
        Layer 5: learn per-ID maximum frozen-payload run length from
        normal traffic. Any run longer than margin * max_normal_run
        is flagged as a stationarity anomaly (frozen ECU / replay attack).
        """
        byte_cols = [f"data{i}" for i in range(8)]
        avail = [c for c in byte_cols if c in normal_df.columns]
        if len(avail) < 8:
            return

        sort_col = next(
            (c for c in ("Timestamp","timestamp","tiTestaTp","time","ts")
             if c in normal_df.columns), None
        )
        df = (normal_df.sort_values(sort_col).reset_index(drop=True)
              if sort_col else normal_df.reset_index(drop=True))

        id_arr   = df[id_col].astype(int).values
        byte_arr = df[avail].values

        for cid in np.unique(id_arr):
            mask     = id_arr == cid
            payloads = byte_arr[mask]
            n        = len(payloads)
            if n < 2:
                self._stationarity_limits[int(cid)] = min_threshold
                continue
            same    = np.all(payloads[1:] == payloads[:-1], axis=1)
            max_run = 1; run = 1
            for s in same:
                if s:
                    run += 1
                else:
                    max_run = max(max_run, run); run = 1
            max_run = max(max_run, run)
            self._stationarity_limits[int(cid)] = max(
                min_threshold, int(max_run * margin)
            )

        if self.verbose:
            n_active = sum(
                1 for v in self._stationarity_limits.values()
                if v < 500
            )
            print(f"  Layer 5 : stationarity limits for "
                  f"{len(self._stationarity_limits)} IDs "
                  f"({n_active} with active thresholds < 500)")

    # ------------------------------------------------------------------
    def build_profiles(
        self,
        feature_cols: List[str],
        save_path: Optional[str] = None,
    ) -> Dict[int, PerIDModel]:
        print("\n--- HCCL: Building Per-ID Profiles ---")
        self.feature_cols = feature_cols
        id_col = self._resolve_id_col()

        if "label" in self.data.columns:
            normal_df = self.data[self.data["label"] == 0].copy()
        elif "is_attack" in self.data.columns:
            normal_df = self.data[self.data["is_attack"] == 0].copy()
        else:
            normal_df = self.data.copy()

        print(f"  Using {len(normal_df)} normal messages for training.")

        calib_df = normal_df.sample(frac=self.calib_frac, random_state=42)
        train_df = normal_df.drop(index=calib_df.index)
        print(f"  Split → KMeans: {len(train_df):,} rows  |  Calibration: {len(calib_df):,} rows")

        t0 = time.time()
        self.models = _build_per_id_models(
            train_df, id_col, feature_cols,
            alpha=self.alpha, n_clusters_cap=self.n_clusters_cap,
            svm_grid_search=self.svm_grid_search,
            verbose=self.verbose,
        )
        _extend_models_with_calibration(
            self.models, calib_df, id_col, feature_cols,
            alpha=self.alpha, verbose=self.verbose,
        )
        self._train_time = time.time() - t0
        print(f"  ⏱ Training time: {self._train_time:.2f}s")

        self.calibrate(calib_df)

        # Learn lightweight constraints from FULL normal data
        self._learn_lightweight_constraints(normal_df, id_col)
        self._learn_stationarity_limits(normal_df, id_col)

        self._precompute_fast_tables()
        self._build_global_index()

        if save_path:
            self.save_profiles(save_path)

        return self.models

    # ------------------------------------------------------------------
    def calibrate(self, calib_df: pd.DataFrame) -> Dict[int, PerIDThresholds]:
        print("\n--- HCCL: Calibrating Per-Cluster Bands ---")
        id_col = self._resolve_id_col()
        self.thresholds = _calibrate_thresholds(
            self.models, calib_df,
            id_col, self.feature_cols,
            alpha=self.alpha, verbose=self.verbose,
        )
        print(f"  Calibrated thresholds for {len(self.thresholds)} IDs.")
        return self.thresholds

    # ------------------------------------------------------------------
    def detect_anomalies(self, threshold: float = 0.5) -> pd.DataFrame:
        if not self.models:
            raise ValueError("No models available. Call build_profiles() or load_profiles() first.")
        if not self.thresholds:
            raise ValueError("Thresholds not calibrated. Call calibrate() or build_profiles() first.")

        print(f"\n--- HCCL: Detecting Anomalies (fallback threshold={threshold}) ---")
        t1 = time.time()
        preds, reasons, confs, scores = self.score_dataframe_fast(self.data)
        self._detect_time = time.time() - t1

        self.data = self.data.copy()
        self.data["cluster_anomaly_score"] = scores
        self.data["cluster_confidence"]    = confs
        self.data["cluster_reason"]        = reasons
        self.data["cluster_anomaly"]       = preds.astype(bool)
        self.data["cluster_strong"]        = scores > (threshold + 0.2)
        self.data["cluster_weak"]          = (scores > (threshold - 0.1)) & (scores <= (threshold + 0.2))

        n_total  = len(self.data)
        n_anom   = int(self.data["cluster_anomaly"].sum())
        ms_per_msg = (self._detect_time / max(1, n_total)) * 1000
        print(f"  ⏱ Detection: {self._detect_time:.2f}s  (~{ms_per_msg:.3f} ms/msg)")
        print(f"  Flagged {n_anom}/{n_total} messages as anomalies.")
        print("\n  Clustering anomaly score distribution:")
        print(self.data["cluster_anomaly_score"].describe().to_string())

        if "is_attack" in self.data.columns or "label" in self.data.columns:
            gt_col = "is_attack" if "is_attack" in self.data.columns else "label"
            y_true = self.data[gt_col].values.astype(int)
            y_pred = self.data["cluster_anomaly"].values.astype(int)

            tp = int(np.sum((y_pred == 1) & (y_true == 1)))
            tn = int(np.sum((y_pred == 0) & (y_true == 0)))
            fp = int(np.sum((y_pred == 1) & (y_true == 0)))
            fn = int(np.sum((y_pred == 0) & (y_true == 1)))
            n  = len(y_true)

            acc  = (tp + tn) / n if n else 0.0
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec  = tp / (tp + fn) if (tp + fn) else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            fpr  = fp / (fp + tn) if (fp + tn) else 0.0
            fnr  = fn / (fn + tp) if (fn + tp) else 0.0

            print(f"\n  📊 Clustering Results:")
            print(f"     Samples={n}  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
            print(f"     Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")
            print(f"     FPR={fpr:.4f}  FNR={fnr:.4f}")
            print("\n  Detailed Classification Report:")
            print(classification_report(y_true, y_pred, target_names=["Normal", "Attack"]))
            print("  Confusion Matrix:")
            print(confusion_matrix(y_true, y_pred))

        return self.data

    # ------------------------------------------------------------------
    def build_profiles_from(
        self,
        train_df: pd.DataFrame,
        feature_cols: List[str],
        calib_frac: float = 0.30,
        save_path: Optional[str] = None,
    ) -> Dict[int, "PerIDModel"]:
        print("\n--- HCCL: Building Per-ID Profiles (external train data) ---")
        self.feature_cols = feature_cols
        id_col = self._resolve_id_col()

        if id_col not in train_df.columns:
            for cand in ["arbitration_id", " CAN ID", "CAN_ID", "can_id"]:
                if cand in train_df.columns:
                    train_df = train_df.rename(columns={cand: id_col})
                    break

        normal_df = (
            train_df[train_df["label"] == 0].copy()
            if "label" in train_df.columns
            else train_df.copy()
        )
        print(f"  Normal rows for training: {len(normal_df):,}")

        calib_df = normal_df.sample(frac=calib_frac, random_state=42)
        fit_df   = normal_df.drop(index=calib_df.index)
        print(f"  KMeans: {len(fit_df):,}  |  Calibration: {len(calib_df):,}")

        t0 = time.time()
        self.models = _build_per_id_models(
            fit_df, id_col, feature_cols,
            alpha=self.alpha, n_clusters_cap=self.n_clusters_cap,
            svm_grid_search=self.svm_grid_search,
            verbose=self.verbose,
        )
        _extend_models_with_calibration(
            self.models, calib_df, id_col, feature_cols,
            alpha=self.alpha, verbose=self.verbose,
        )
        self._train_time = time.time() - t0
        print(f"  ⏱ Training time: {self._train_time:.2f}s")

        self.calibrate(calib_df)

        # Learn lightweight constraints from FULL normal data (not just fit_df)
        self._learn_lightweight_constraints(normal_df, id_col)
        self._learn_stationarity_limits(normal_df, id_col)

        self._precompute_fast_tables()
        self._build_global_index()

        if save_path:
            self.save_profiles(save_path)

        return self.models

    # ------------------------------------------------------------------
    def score_dataframe(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:
        """Row-by-row scoring (backward compat, prefer score_dataframe_fast)."""
        if not self.models or not self.thresholds:
            raise ValueError("Models not ready. Call build_profiles() first.")

        id_col = self._resolve_id_col()
        preds, reasons, confs, scores = [], [], [], []

        for _, row in df.iterrows():
            cid = int(row[id_col])
            x   = row[self.feature_cols].astype(float).values
            p, r, c, s = _score_row(x, cid, self.models, self.thresholds)
            preds.append(p)
            reasons.append(r)
            confs.append(c)
            scores.append(s)

        return (
            np.array(preds,  dtype=int),
            reasons,
            np.array(confs,  dtype=float),
            np.array(scores, dtype=float),
        )

    # ------------------------------------------------------------------
    # FAST vectorized scoring — Enhanced with Layers 0a, 0b, 5
    # ------------------------------------------------------------------
    def score_dataframe_fast(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:
        """
        Vectorized batch scoring with all detection layers.

        Layer 0a — Invariant Byte Check (NEW, lightweight):
            Bytes that never change in normal training are strictly
            enforced. Any deviation is flagged immediately — before
            cluster distance computation. Catches Gear/RPM/tampering
            attacks. Zero FP risk: only truly invariant bytes constrained.

        Layer 0b — All-Zero Payload Check (NEW, lightweight):
            Flags messages with all-zero payload on IDs that never
            produce zero payloads in normal traffic. Reduces DoS-induced
            FP on known IDs and catches zero-padded injection attacks.

        Layer 1 — Unknown CAN ID → anomaly (unchanged)

        Layer 2 — k-NN centroid band acceptance → normal (unchanged)

        Layer 3 — Distance threshold → anomaly (unchanged)

        Layer 4 — SVM/IsolationForest refinement (unchanged)

        Layer 5 — Stationarity / frozen-payload run detection (NEW):
            Flags abnormally long runs of identical payloads per CAN ID.
            Catches replay attacks and ECU freeze-out.
        """
        if not self.models or not self.thresholds:
            raise ValueError("Models not ready.")

        if not self._fast_tables:
            self._precompute_fast_tables()
        if self._global_index is None:
            self._build_global_index()

        id_col  = self._resolve_id_col()
        n       = len(df)
        preds   = np.ones(n,  dtype=int)
        scores  = np.ones(n,  dtype=float)
        confs   = np.ones(n,  dtype=float)
        reasons = ["Unknown CAN ID"] * n

        feat_arr = df[self.feature_cols].astype(float).values
        id_arr   = df[id_col].astype(int).values

        # Raw byte arrays for Layer 0a/0b
        byte_cols   = [f"data{i}" for i in range(8)]
        avail_bytes = [c for c in byte_cols if c in df.columns]
        has_bytes   = len(avail_bytes) == 8
        if has_bytes:
            raw_byte_arr = df[avail_bytes].values.astype(float)  # (N, 8)

        for cid in np.unique(id_arr):
            row_idx = np.where(id_arr == cid)[0]

            ft = self._fast_tables.get(cid)
            if ft is None:
                continue   # Layer 1: unknown ID, already set to anomaly

            X = feat_arr[row_idx]

            # ── Layer 0a: Invariant Byte Check ────────────────────────────
            # Only bytes with a single unique value in normal training are
            # checked. Any deviation is an immediate anomaly flag.
            layer0_flagged = np.zeros(len(row_idx), dtype=bool)

            if has_bytes:
                inv_bytes = ft.get("inv_bytes", {})
                if inv_bytes:
                    raw = raw_byte_arr[row_idx]        # (Nid, 8)
                    for byte_idx, expected_val in inv_bytes.items():
                        col_vals = raw[:, byte_idx]
                        violations = col_vals != expected_val
                        if violations.any():
                            viol_local = np.where(violations)[0]
                            layer0_flagged[viol_local] = True
                            for li in viol_local:
                                gi = row_idx[li]
                                actual = raw[li, byte_idx]
                                reasons[gi] = (
                                    f"Invariant byte violation "
                                    f"(byte{byte_idx}={actual:.0f}, "
                                    f"expected={expected_val:.0f})"
                                )
                                preds[gi]  = 1
                                scores[gi] = 0.95
                                confs[gi]  = 0.97

                # ── Layer 0b: All-Zero Payload Check ──────────────────────
                # Flag messages with all-zero payload on IDs that never
                # produce all-zero payloads in normal traffic
                if ft.get("no_allzero", False):
                    raw = raw_byte_arr[row_idx]
                    all_zero = (raw == 0).all(axis=1)
                    new_zero = all_zero & ~layer0_flagged
                    if new_zero.any():
                        layer0_flagged |= new_zero
                        for li in np.where(new_zero)[0]:
                            gi = row_idx[li]
                            reasons[gi] = "All-zero payload (DoS pattern)"
                            preds[gi]   = 1
                            scores[gi]  = 0.90
                            confs[gi]   = 0.92

                # ── Layer 0c: Payload Whitelist Check ─────────────────────
                # If payload signature was never seen in normal training
                # for this ID, mark anomaly (strong fuzzy/injection signal).
                seen_hashes = ft.get("seen_payload_hashes")
                if seen_hashes is not None and len(seen_hashes) > 0:
                    raw = raw_byte_arr[row_idx].astype(np.uint8)
                    packed = (
                        (raw[:, 0].astype(np.uint64) << 56)
                        | (raw[:, 1].astype(np.uint64) << 48)
                        | (raw[:, 2].astype(np.uint64) << 40)
                        | (raw[:, 3].astype(np.uint64) << 32)
                        | (raw[:, 4].astype(np.uint64) << 24)
                        | (raw[:, 5].astype(np.uint64) << 16)
                        | (raw[:, 6].astype(np.uint64) << 8)
                        | raw[:, 7].astype(np.uint64)
                    )
                    unseen = ~np.isin(packed, seen_hashes)
                    new_unseen = unseen & ~layer0_flagged
                    if new_unseen.any():
                        layer0_flagged |= new_unseen
                        for li in np.where(new_unseen)[0]:
                            gi = row_idx[li]
                            reasons[gi] = "Unseen payload signature"
                            preds[gi]   = 1
                            scores[gi]  = 0.96
                            confs[gi]   = 0.98

            # Skip cluster scoring for Layer-0 flagged messages
            remaining = ~layer0_flagged
            if not remaining.any():
                continue

            row_idx_r = row_idx[remaining]
            X_r       = X[remaining]

            # ── Layers 2-4: Cluster distance scoring ──────────────────────
            Xs = (X_r - ft["mean"]) / ft["scale"]

            cents  = ft["cents"]
            dists  = _sp_cdist(Xs, cents)
            k      = ft["knn_k"]
            lo_arr = ft["lo"]
            hi_arr = ft["hi"]
            thr    = ft["dist_thr"] + 1e-9

            nn_idx  = np.argmin(dists, axis=1)
            nearest = dists[np.arange(len(X_r)), nn_idx]

            # Band acceptance — fully vectorized
            k_eff    = min(k, dists.shape[1] - 1)
            top_k    = np.argpartition(dists, kth=k_eff, axis=1)[:, :k]
            g_dist   = dists[np.arange(len(X_r))[:, None], top_k]
            g_lo     = lo_arr[top_k]
            g_hi     = hi_arr[top_k]
            accepted = ((g_dist >= g_lo) & (g_dist <= g_hi)).any(axis=1)

            # Normal path
            if accepted.any():
                d_n  = nearest[accepted]
                gi_n = row_idx_r[accepted]
                preds[gi_n]  = 0
                scores[gi_n] = np.clip(d_n / thr * 0.4, 0.0, 0.5)
                confs[gi_n]  = np.clip(1.0 - d_n / thr, 0.0, 1.0)
                for gi in gi_n:
                    reasons[gi] = "Centroid-band normal"

            # Anomaly path
            anom_mask = ~accepted
            if anom_mask.any():
                d_a     = nearest[anom_mask]
                excess  = (d_a - thr) / thr
                sc_a    = np.where(
                    d_a > thr,
                    0.5 + np.clip(0.5 * excess, 0.0, 0.5),
                    np.clip(0.4 * d_a / thr, 0.0, 0.5),
                )
                kmeans_anom = d_a > thr
                cf_a        = np.clip(d_a / thr, 0.0, 1.0)

                # Layer 4: SVM with deduplication
                svm_anom = np.zeros(anom_mask.sum(), dtype=bool)
                svm, svm_thr = ft["svm"], ft["svm_thr"]
                if svm is not None and svm_thr is not None:
                    try:
                        Xs_a      = Xs[anom_mask]
                        uniq, inv = np.unique(Xs_a, axis=0,
                                              return_inverse=True)
                        sv_full   = svm.decision_function(
                            uniq).flatten()[inv]
                        svm_anom  = sv_full < svm_thr
                    except Exception:
                        pass

                final_anom = kmeans_anom | svm_anom
                gi_a       = row_idx_r[anom_mask]
                preds[gi_a]  = final_anom.astype(int)
                scores[gi_a] = sc_a
                confs[gi_a]  = cf_a
                for i, gi in enumerate(gi_a):
                    reasons[gi] = (
                        "KMeans anomaly" if final_anom[i] else "Normal"
                    )

        # ── Layer 5: Stationarity post-filter ─────────────────────────────
        # Flag abnormally long runs of identical payloads per CAN ID.
        # Catches replay attacks and ECU freeze caused by bus saturation.
        if has_bytes and self._stationarity_limits:
            byte_feat = raw_byte_arr   # reuse (N, 8) array
            for cid in np.unique(id_arr):
                ft = self._fast_tables.get(cid)
                if ft is None:
                    continue
                max_run  = ft.get("max_frozen_run", 500)
                cid_rows = np.where(id_arr == cid)[0]
                if len(cid_rows) < max_run:
                    continue
                cid_bytes = byte_feat[cid_rows]
                run_start = 0
                for j in range(1, len(cid_rows)):
                    if np.array_equal(cid_bytes[j], cid_bytes[j - 1]):
                        run_len = j - run_start + 1
                        if run_len > max_run:
                            gi = cid_rows[j]
                            if preds[gi] == 0:
                                preds[gi]   = 1
                                scores[gi]  = 0.75
                                confs[gi]   = 0.85
                                reasons[gi] = (
                                    f"Frozen payload "
                                    f"(run={run_len}, thr={max_run})"
                                )
                    else:
                        run_start = j

        # ── Layer 6: Global cross-ID consistency check (DADA-style) ──────────
        # For each message that passed all prior layers (pred=0), query the
        # nPts globally nearest centroids.  If NONE belong to the message's
        # own CAN ID, the payload is closer to a foreign ID's normal space —
        # strong evidence of injection or fuzzing.
        if self._global_index is not None and self._global_cents is not None:
            normal_mask = np.where(preds == 0)[0]
            if len(normal_mask) > 0:
                X_normal = feat_arr[normal_mask]
                ids_normal = id_arr[normal_mask]

                dist_g, idx_g = self._global_index.kneighbors(X_normal)
                neighbour_ids  = self._global_ids[idx_g]  # (N_normal, nPts)

                # Conservative margin: flag only when foreign-ID proximity is
                # clearly better than the sample's own-ID centroid proximity.
                nearest_foreign = dist_g[:, 0]
                nearest_own = np.full(len(ids_normal), np.inf, dtype=np.float64)
                for cid in np.unique(ids_normal):
                    local_mask = ids_normal == cid
                    m = self.models.get(int(cid))
                    if m is None:
                        continue
                    own_raw_centroids = m.scaler.inverse_transform(m.centroids).astype(np.float64)
                    own_d = _sp_cdist(X_normal[local_mask], own_raw_centroids).min(axis=1)
                    nearest_own[local_mask] = own_d

                # Require finite own reference and a clear distance gap.
                cross_id_anom = np.isfinite(nearest_own) & (
                    nearest_foreign + self._cross_id_margin < nearest_own
                )

                flagged = normal_mask[cross_id_anom]
                preds[flagged]  = 1
                scores[flagged] = 0.80
                confs[flagged]  = 0.85
                for gi in flagged:
                    reasons[gi] = "Cross-ID anomaly (payload closer to foreign CAN ID)"

        return (preds, reasons, confs, scores)

    # ------------------------------------------------------------------
    def summary_stats(self) -> Dict[str, Any]:
        return {
            "n_can_ids_trained":    len(self.models),
            "n_can_ids_calibrated": len(self.thresholds),
            "n_clusters_cap":       self.n_clusters_cap,
            "alpha":                self.alpha,
            "calib_frac":           self.calib_frac,
            "train_time_sec":       self._train_time,
            "detect_time_sec":      self._detect_time,
            "feature_cols":         self.feature_cols,
            "n_invariant_byte_ids": len(self._invariant_bytes),
            "n_no_allzero_ids":     len(self._no_allzero_ids),
            "n_payload_whitelist_ids": len(self._seen_payload_hashes),
            "methods": {cid: m.method for cid, m in self.models.items()},
            "global_npts": self._global_npts,
            "cross_id_margin": self._cross_id_margin,
        }

    # ------------------------------------------------------------------
    def save_profiles(self, save_path: str) -> None:
        if not self.models:
            raise ValueError("Nothing to save. Call build_profiles() first.")

        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        bundle = {
            "can_profiles":       {cid: {"total_messages": m.n_samples}
                                   for cid, m in self.models.items()},
            "centroids":          {cid: m.centroids
                                   for cid, m in self.models.items()},
            "cluster_scalers":    {cid: m.scaler
                                   for cid, m in self.models.items()},
            "feature_cols":       self.feature_cols,
            "hccl_models":        self.models,
            "hccl_thresholds":    self.thresholds,
            "alpha":              self.alpha,
            "id_col":             self.id_col,
            "global_npts":        self._global_npts,
            "cross_id_margin":    self._cross_id_margin,
            # Enhanced constraint bundles
            "invariant_bytes":    self._invariant_bytes,
            "no_allzero_ids":     list(self._no_allzero_ids),
            "seen_payload_hashes": {
                cid: sorted(list(v)) for cid, v in self._seen_payload_hashes.items()
            },
            "stationarity_limits": self._stationarity_limits,
        }

        with open(save_path, "wb") as fh:
            pickle.dump(bundle, fh, protocol=4)

        size_mb = os.path.getsize(save_path) / 1024 / 1024
        n_inv   = len(self._invariant_bytes)
        n_naz   = len(self._no_allzero_ids)
        print(f"  💾 Saved profile bundle to {save_path}  ({size_mb:.2f} MB) "
              f"[{n_inv} inv-byte IDs, {n_naz} no-allzero IDs]")

    # ------------------------------------------------------------------
    def load_profiles(self, load_path: str) -> Dict[int, PerIDModel]:
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Profile file not found: {load_path}")

        with open(load_path, "rb") as fh:
            bundle = pickle.load(fh)

        self.feature_cols = bundle.get("feature_cols", [])

        if "hccl_models" in bundle:
            self.models     = bundle["hccl_models"]
            self.thresholds = bundle.get("hccl_thresholds", {})
            self.alpha      = bundle.get("alpha", self.alpha)
            self.id_col     = bundle.get("id_col", self.id_col)
            self._global_npts = int(bundle.get("global_npts", self._global_npts))
            self._cross_id_margin = float(bundle.get("cross_id_margin", self._cross_id_margin))

            # Load enhanced constraints (graceful fallback for old bundles)
            self._invariant_bytes      = bundle.get("invariant_bytes", {})
            self._no_allzero_ids       = set(bundle.get("no_allzero_ids", []))
            self._seen_payload_hashes  = {
                int(cid): set(vals)
                for cid, vals in bundle.get("seen_payload_hashes", {}).items()
            }
            self._stationarity_limits  = bundle.get("stationarity_limits", {})

            n_inv = len(self._invariant_bytes)
            n_naz = len(self._no_allzero_ids)
            n_pwh = len(self._seen_payload_hashes)
            n_sta = len(self._stationarity_limits)
            print(
                f"  📂 Loaded HCCL bundle from {load_path} "
                f"({len(self.models)} IDs, {len(self.thresholds)} thresholds, "
                f"{n_inv} inv-byte IDs, {n_naz} no-allzero IDs, {n_pwh} payload-whitelist IDs, "
                f"{n_sta} stationarity limits)"
            )
            self._precompute_fast_tables()
            self._build_global_index()
        else:
            print(
                f"  ⚠ Legacy profile format detected. "
                f"Reconstructing models (no SVM / calibrated bands available)."
            )
            centroids = bundle.get("centroids", {})
            scalers   = bundle.get("cluster_scalers", {})
            for cid in centroids:
                cents = centroids[cid]
                k     = len(cents)
                cb    = [{"centroid": c, "max_dist_p95": 1.0, "n": 0}
                         for c in cents]
                mdl   = PerIDModel(
                    method="kmeans",
                    scaler=scalers.get(cid, StandardScaler()),
                    kmeans=None,
                    centroids=cents,
                    cluster_bounds=cb,
                    n_clusters=k,
                    n_samples=0,
                )
                self.models[int(cid)] = mdl
            print(f"  📂 Loaded {len(self.models)} legacy profiles "
                  f"from {load_path}")

        return self.models
