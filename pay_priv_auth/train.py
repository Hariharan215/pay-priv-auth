"""Training script for the logistic baseline."""

from __future__ import annotations

import argparse
import pathlib
from typing import Tuple

import joblib
import numpy as np
import pandas as pd

from .evaluation import compute_auc, compute_eer, compute_far_frr
from .models.baseline import LogisticBaseline


def generate_synthetic_data(n_users: int, n_features: int, samples_per_user: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = []
    uids = []
    for u in range(n_users):
        mean = rng.normal(loc=0.0, scale=1.0, size=n_features)
        samples = rng.normal(mean, 1.0, size=(samples_per_user, n_features))
        X.append(samples)
        uids.extend([u] * samples_per_user)
    return np.vstack(X), np.array(uids, dtype=int)


def train_model(X: np.ndarray, uids: np.ndarray) -> LogisticBaseline:
    model = LogisticBaseline(max_iter=200)
    model.fit(X, uids)
    return model


def evaluate(model: LogisticBaseline, X: np.ndarray, uids: np.ndarray) -> Tuple[float, float, float, float]:
    scores = []
    labels = []
    for uid in np.unique(uids):
        mask_g = uids == uid
        s_g = model.score_user(int(uid), X[mask_g])
        s_i = model.score_user(int(uid), X[~mask_g])
        scores.append(np.concatenate([s_g, s_i]))
        labels.append(np.concatenate([np.ones_like(s_g), np.zeros_like(s_i)]))
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    auc = compute_auc(scores, labels)
    eer, th, _ = compute_eer(scores, labels)
    far, frr = compute_far_frr(scores, labels, th)
    return auc, eer, far, frr


def main() -> None:
    ap = argparse.ArgumentParser(description="Train logistic baseline")
    ap.add_argument("--dataset_parquet", type=pathlib.Path, default=None)
    ap.add_argument("--n_users", type=int, default=5)
    ap.add_argument("--n_features", type=int, default=20)
    ap.add_argument("--samples_per_user", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.dataset_parquet and args.dataset_parquet.exists():
        df = pd.read_parquet(args.dataset_parquet)
        feature_cols = [c for c in df.columns if c.startswith("f")]
        X = df[feature_cols].to_numpy()
        uids = df["user_id"].to_numpy().astype(int)
    else:
        X, uids = generate_synthetic_data(args.n_users, args.n_features, args.samples_per_user, args.seed)

    # Split into train/test per user
    rng = np.random.default_rng(args.seed)
    train_idx = []
    test_idx = []
    for uid in np.unique(uids):
        idx = np.where(uids == uid)[0]
        rng.shuffle(idx)
        split = int(0.7 * len(idx))
        train_idx.extend(idx[:split])
        test_idx.extend(idx[split:])
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)

    model = train_model(X[train_idx], uids[train_idx])
    auc, eer, far, frr = evaluate(model, X[test_idx], uids[test_idx])
    print(f"AUC={auc:.3f} EER={eer:.3f} FAR={far:.3f} FRR={frr:.3f}")

    # Save per-user models
    out_dir = pathlib.Path("artifacts")
    out_dir.mkdir(exist_ok=True)
    for uid, um in model.models.items():
        joblib.dump(um.model, out_dir / f"user_{uid}_logreg.joblib")


if __name__ == "__main__":  # pragma: no cover
    main()
