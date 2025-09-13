"""Training script for the logistic baseline on real datasets."""

from __future__ import annotations

import argparse
import pathlib
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from .evaluation import auc_score, eer, far_frr_at, frr_far_at
from .models.baseline import LogisticBaseline


def split_by_session(df: pd.DataFrame, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    users = df["user_id"].unique()
    train_mask = np.zeros(len(df), dtype=bool)
    for u in users:
        sess = df[df["user_id"] == u]["session_id"].unique()
        rng.shuffle(sess)
        split = int(0.7 * len(sess))
        train_sessions = set(sess[:split])
        mask = (df["user_id"] == u) & (df["session_id"].isin(train_sessions))
        train_mask |= mask
    test_mask = ~train_mask
    return train_mask, test_mask


def evaluate(model: LogisticBaseline, X: np.ndarray, uids: np.ndarray) -> Dict[int, Tuple[float, float, float, float]]:
    metrics: Dict[int, Tuple[float, float, float, float]] = {}
    for uid in np.unique(uids):
        mask_g = uids == uid
        scores_g = model.predict_proba_user(int(uid), X[mask_g])
        scores_i = model.predict_proba_user(int(uid), X[~mask_g])
        y_true = np.concatenate([np.ones_like(scores_g), np.zeros_like(scores_i)])
        y_score = np.concatenate([scores_g, scores_i])
        auc = auc_score(y_true, y_score)
        eer_v, _ = eer(y_true, y_score)
        far, _ = far_frr_at(y_true, y_score, 0.05)
        frr, _ = frr_far_at(y_true, y_score, 0.005)
        metrics[int(uid)] = (auc, eer_v, far, frr)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Train logistic baseline")
    ap.add_argument("--dataset_parquet", type=pathlib.Path, required=True)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    df = pd.read_parquet(args.dataset_parquet)
    feature_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feature_cols].to_numpy()
    uid_codes, uniques = pd.factorize(df["user_id"].astype(str))

    train_mask, test_mask = split_by_session(df, args.seed)
    model = LogisticBaseline(max_iter=200)
    model.fit(X[train_mask], uid_codes[train_mask])

    metrics = evaluate(model, X[test_mask], uid_codes[test_mask])
    auc = np.mean([m[0] for m in metrics.values()])
    eer_v = np.mean([m[1] for m in metrics.values()])
    far = np.mean([m[2] for m in metrics.values()])
    frr = np.mean([m[3] for m in metrics.values()])
    print(f"AUC={auc:.3f} EER={eer_v:.3f} FAR@5%FRR={far:.3f} FRR@0.5%FAR={frr:.3f}")

    dataset_name = args.dataset_parquet.stem.replace("_features", "")
    out_dir = pathlib.Path("artifacts") / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, uid in enumerate(uniques):
        if idx in model.models:
            joblib.dump(model.models[idx].model, out_dir / f"user_{uid}_logreg.joblib")


if __name__ == "__main__":  # pragma: no cover
    main()
