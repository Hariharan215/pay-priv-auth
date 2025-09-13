"""End-to-end experiment runner on real datasets."""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

from .evaluation import auc_score, eer, far_frr_at, frr_far_at
from .models.aggregator import fedavg_logistic
from .models.baseline import LogisticBaseline
from .train import split_by_session


def compute_metrics(model, X, uids):
    out = {}
    for uid in np.unique(uids):
        mask_g = uids == uid
        scores_g = model.predict_proba_user(int(uid), X[mask_g]) if hasattr(model, "predict_proba_user") else model.predict_proba(X[mask_g])[:,1]
        scores_i = model.predict_proba_user(int(uid), X[~mask_g]) if hasattr(model, "predict_proba_user") else model.predict_proba(X[~mask_g])[:,1]
        y_true = np.concatenate([np.ones_like(scores_g), np.zeros_like(scores_i)])
        y_score = np.concatenate([scores_g, scores_i])
        auc = auc_score(y_true, y_score)
        e, _ = eer(y_true, y_score)
        far, _ = far_frr_at(y_true, y_score, 0.05)
        frr, _ = frr_far_at(y_true, y_score, 0.005)
        out[int(uid)] = (auc, e, far, frr)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Run experiment")
    ap.add_argument("--dataset_parquet", type=pathlib.Path, required=True)
    ap.add_argument("--use_fedavg", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    df = pd.read_parquet(args.dataset_parquet)
    feature_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feature_cols].to_numpy()
    uid_codes, uniques = pd.factorize(df["user_id"].astype(str))
    train_mask, test_mask = split_by_session(df, args.seed)

    baseline = LogisticBaseline(max_iter=200)
    baseline.fit(X[train_mask], uid_codes[train_mask])

    metrics_per_user = compute_metrics(baseline, X[test_mask], uid_codes[test_mask])
    print("user,AUC,EER,FAR@5%FRR,FRR@0.5%FAR")
    for idx, vals in metrics_per_user.items():
        print(f"{uniques[idx]},{vals[0]:.3f},{vals[1]:.3f},{vals[2]:.3f},{vals[3]:.3f}")

    if args.use_fedavg:
        global_model = fedavg_logistic({uid: um.model for uid, um in baseline.models.items()})
        # wrap global_model to reuse compute_metrics
        class Wrapper:
            def __init__(self, model):
                self.model = model
            def predict_proba_user(self, user_id, X):
                return self.model.predict_proba(X)[:,1]
        global_metrics = compute_metrics(Wrapper(global_model), X[test_mask], uid_codes[test_mask])
        print("global,", end="")
        gm_vals = np.mean(list(global_metrics.values()), axis=0)
        print(f"{gm_vals[0]:.3f},{gm_vals[1]:.3f},{gm_vals[2]:.3f},{gm_vals[3]:.3f}")


if __name__ == "__main__":  # pragma: no cover
    main()
