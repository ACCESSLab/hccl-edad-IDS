#!/usr/bin/env python3
"""Generate synthetic CAN traffic for testing CLA-DADA."""

import argparse
import numpy as np
import pandas as pd

CAN_IDS = [0x100, 0x200, 0x300, 0x400]
BYTE_COLS = [f"data{i}" for i in range(8)]


def _make_row(can_id: int, payload: np.ndarray, label: int) -> dict:
    row = {"arbitration_id": can_id, "label": label}
    for i, b in enumerate(payload):
        row[f"data{i}"] = int(b) % 256
    return row


def generate_dataset(
    n_normal_per_id: int = 500,
    n_attack: int = 200,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    centroids = {
        cid: rng.integers(20, 220, size=8).astype(float)
        for cid in CAN_IDS
    }

    train_rows = []
    for cid in CAN_IDS:
        center = centroids[cid]
        for _ in range(n_normal_per_id):
            payload = center + rng.normal(0, 3, size=8)
            payload = np.clip(payload, 0, 255)
            train_rows.append(_make_row(cid, payload, 0))

    test_rows = []
    for cid in CAN_IDS:
        center = centroids[cid]
        for _ in range(n_normal_per_id // 2):
            payload = center + rng.normal(0, 3, size=8)
            payload = np.clip(payload, 0, 255)
            test_rows.append(_make_row(cid, payload, 0))

    n_each = max(1, n_attack // 4)

    # Unknown CAN ID
    for _ in range(n_each):
        payload = rng.integers(0, 256, size=8)
        test_rows.append(_make_row(0x999, payload, 1))

    # Out of range
    for _ in range(n_each):
        cid = rng.choice(CAN_IDS)
        payload = centroids[cid] + rng.normal(0, 80, size=8)
        payload = np.clip(payload, 0, 255)
        test_rows.append(_make_row(cid, payload, 1))

    # ID mismatch: payload near another ID's centroid
    for _ in range(n_each):
        src, tgt = rng.choice(CAN_IDS, size=2, replace=False)
        payload = centroids[tgt] + rng.normal(0, 2, size=8)
        payload = np.clip(payload, 0, 255)
        test_rows.append(_make_row(src, payload, 1))

    # Mixed replay-style offset
    for _ in range(n_attack - 3 * n_each):
        cid = rng.choice(CAN_IDS)
        payload = centroids[cid] + rng.integers(-40, 40, size=8)
        payload = np.clip(payload, 0, 255)
        test_rows.append(_make_row(cid, payload, 1))

    df_train = pd.DataFrame(train_rows)
    df_test = pd.DataFrame(test_rows).sample(frac=1, random_state=seed).reset_index(drop=True)

    return df_train, df_test


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic CAN CSVs")
    ap.add_argument("--out-dir", default="data/synthetic")
    ap.add_argument("--normal-per-id", type=int, default=500)
    ap.add_argument("--attacks", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    df_train, df_test = generate_dataset(
        n_normal_per_id=args.normal_per_id,
        n_attack=args.attacks,
        seed=args.seed,
    )

    train_path = os.path.join(args.out_dir, "train_normal.csv")
    test_path = os.path.join(args.out_dir, "test_mixed.csv")
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)

    n_attack = int((df_test["label"] == 1).sum())
    print(f"Wrote {train_path} ({len(df_train):,} normal rows)")
    print(f"Wrote {test_path} ({len(df_test):,} rows, {n_attack:,} attacks)")


if __name__ == "__main__":
    main()
