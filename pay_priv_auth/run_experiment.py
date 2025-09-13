"""End-to-end experiment runner."""

from __future__ import annotations

import argparse
from typing import Dict

import numpy as np

from .evaluation import compute_auc, compute_eer
from .models.aggregator import FedAvgAggregator
from .models.baseline import LogisticBaseline
from .train import generate_synthetic_data


def main() -> None:
    ap = argparse.ArgumentParser(description="Run synthetic experiment")
    ap.add_argument("--n_users", type=int, default=5)
    ap.add_argument("--n_features", type=int, default=20)
    ap.add_argument("--samples_per_user", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_fedavg", action="store_true")
    ap.add_argument("--dp_noise", type=float, default=0.0, help="Std dev of Gaussian noise added to scores")
    args = ap.parse_args()

    X, uids = generate_synthetic_data(args.n_users, args.n_features, args.samples_per_user, args.seed)
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(uids))
    rng.shuffle(idx)
    split = int(0.8 * len(idx))
    train_idx, test_idx = idx[:split], idx[split:]
    X_train, X_test = X[train_idx], X[test_idx]
    uids_train, uids_test = uids[train_idx], uids[test_idx]

    baseline = LogisticBaseline(max_iter=200)
    baseline.fit(X_train, uids_train)

    if args.use_fedavg:
        agg = FedAvgAggregator()
        global_model = agg.average([um.model for um in baseline.models.values()])
        scores = global_model.predict_proba(X_test)[:, 1]
        labels = (uids_test == 0).astype(int)
    else:
        scores = baseline.score_user(0, X_test)
        labels = (uids_test == 0).astype(int)

    if args.dp_noise > 0:
        scores = scores + rng.normal(0.0, args.dp_noise, size=scores.shape)

    auc = compute_auc(scores, labels)
    eer, th, _ = compute_eer(scores, labels)
    print(f"AUC={auc:.3f} EER={eer:.3f} threshold={th:.3f}")


if __name__ == "__main__":  # pragma: no cover
    main()
