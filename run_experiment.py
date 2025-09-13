"""
End‑to‑end experiment runner for Pay‑PrivAuth.

This script orchestrates synthetic data generation, local training on
individual clients, federated aggregation and evaluation.  It uses the
``FedAvgAggregator`` to combine logistic regression models trained on
each client (user) and computes performance metrics of the aggregated
model.  Differential privacy clipping and noise addition are illustrated
using the stubs in ``models.dp``.

Usage:

```bash
python run_experiment.py --n_users 5 --n_features 20 --samples_per_user 100
```

The output summarises per‑client metrics as well as the aggregated
model's performance.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import numpy as np

from models.baseline import LogisticBaseline
from models.aggregator import FedAvgAggregator
from models.dp import dp_clip, DPSanitiser
from evaluation import compute_auc, compute_eer


def generate_synthetic_data(n_users: int, n_features: int, samples_per_user: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Wrapper around ``train.generate_synthetic_data`` to avoid dependency."""
    from train import generate_synthetic_data as gen
    return gen(n_users, n_features, samples_per_user, seed)


def train_client_models(X: np.ndarray, uids: np.ndarray, dp_noise: float = 0.0) -> Dict[int, LogisticBaseline]:
    """
    Train a local logistic regression model for each client (user).

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    uids : ndarray of shape (n_samples,)
        User IDs for each sample.
    dp_noise : float, default=0.0
        Standard deviation of Gaussian noise to add to model parameters (DP).

    Returns
    -------
    client_models : dict mapping user_id → LogisticRegression
        Trained models for each user.
    """
    client_models = {}
    unique_users = np.unique(uids)
    sanitiser = DPSanitiser(noise_std=dp_noise) if dp_noise > 0.0 else None
    for uid in unique_users:
        # Prepare binary labels for this user
        y_binary = (uids == uid).astype(int)
        clf = LogisticBaseline(max_iter=200)
        clf.fit(X, uids)  # logistic baseline handles per‑user logic internally
        # After fitting, we extract the underlying LogisticRegression object for this uid
        lr_model = clf.models[int(uid)].model
        # Optionally add DP noise to parameters
        if sanitiser is not None:
            lr_model.coef_ = lr_model.coef_ + np.random.normal(loc=0.0, scale=dp_noise, size=lr_model.coef_.shape)
            lr_model.intercept_ = lr_model.intercept_ + np.random.normal(loc=0.0, scale=dp_noise, size=lr_model.intercept_.shape)
        client_models[int(uid)] = lr_model
    return client_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a federated synthetic experiment for Pay‑PrivAuth.")
    parser.add_argument("--n_users", type=int, default=5, help="Number of synthetic users")
    parser.add_argument("--n_features", type=int, default=20, help="Feature dimensionality")
    parser.add_argument("--samples_per_user", type=int, default=100, help="Samples per user")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Fraction of samples used for testing")
    parser.add_argument("--dp_noise", type=float, default=0.0, help="Std of Gaussian noise for DP (0 disables noise)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Generate synthetic data
    X, uids = generate_synthetic_data(args.n_users, args.n_features, args.samples_per_user, args.seed)

    # Split into train/test indices
    rng = np.random.default_rng(args.seed)
    indices = np.arange(len(uids))
    rng.shuffle(indices)
    split = int((1.0 - args.test_ratio) * len(indices))
    train_idx, test_idx = indices[:split], indices[split:]

    X_train, uids_train = X[train_idx], uids[train_idx]
    X_test, uids_test = X[test_idx], uids[test_idx]

    # Train local models for each client
    client_models = train_client_models(X_train, uids_train, dp_noise=args.dp_noise)

    # Aggregate local models via federated averaging
    aggregator = FedAvgAggregator()
    aggregated_model = aggregator.average(list(client_models.values()))

    # Evaluate aggregated model on test data
    scores = aggregated_model.predict_proba(X_test)[:, 1]
    # For evaluation we treat the positive class as belonging to user 0
    labels = (uids_test == 0).astype(int)
    auc = compute_auc(scores, labels)
    eer, thresh, _ = compute_eer(scores, labels)
    print("Aggregated model performance (treating user 0 as positive class):")
    print(f"AUC={auc:.3f}, EER={eer:.3f}, threshold={thresh:.3f}")

    # Optional: evaluate per‑client model performance on their own data
    print("\nPer‑client model performance:")
    baseline = LogisticBaseline(max_iter=200)
    baseline.fit(X_train, uids_train)
    for uid in np.unique(uids_train):
        uid_mask = uids_test == uid
        if not np.any(uid_mask):
            continue
        user_scores = baseline.score_user(uid, X_test[uid_mask])
        user_labels = np.ones_like(user_scores, dtype=int)
        impostor_scores = baseline.score_user(uid, X_test[~uid_mask])
        impostor_labels = np.zeros_like(impostor_scores, dtype=int)
        combined_scores = np.concatenate([user_scores, impostor_scores])
        combined_labels = np.concatenate([user_labels, impostor_labels])
        auc_u = compute_auc(combined_scores, combined_labels)
        eer_u, thresh_u, _ = compute_eer(combined_scores, combined_labels)
        print(f"User {uid:>3}: AUC={auc_u:.3f}, EER={eer_u:.3f}, threshold={thresh_u:.3f}")


if __name__ == "__main__":
    main()