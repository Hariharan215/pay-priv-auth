"""
Example training script for the Pay‑PrivAuth baseline.

This script demonstrates how to generate a synthetic behavioural dataset,
train a per‑user logistic regression baseline and evaluate its performance.
It is intended as a starting point for experimentation – you can replace
the synthetic data generator with real preprocessed features and customise
the number of users, feature dimensionality and sample counts via
command‑line arguments.

Usage:

```bash
python train.py --n_users 5 --n_features 20 --samples_per_user 100
```

The script prints the overall AUC and EER across all users as well as
per‑user metrics.
"""

from __future__ import annotations

import argparse
import numpy as np

from models.baseline import LogisticBaseline
from evaluation import compute_auc, compute_eer


def generate_synthetic_data(n_users: int, n_features: int, samples_per_user: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic behavioural dataset.

    For each user, samples are drawn from a multivariate normal distribution with
    a distinct mean vector.  Impostor samples are implicitly included because
    the baseline treats all other users' data as negative examples.  This
    function returns a feature matrix and corresponding user IDs.

    Parameters
    ----------
    n_users : int
        Number of distinct users to simulate.
    n_features : int
        Dimensionality of the feature vectors.
    samples_per_user : int
        Number of samples to generate for each user.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_users * samples_per_user, n_features)
        Synthetic feature matrix.
    user_ids : ndarray of shape (n_users * samples_per_user,)
        Integer user IDs corresponding to each row of ``X``.
    """
    rng = np.random.default_rng(seed)
    X_list = []
    uid_list = []
    for uid in range(n_users):
        # Each user has a mean vector drawn from a standard normal
        mean = rng.normal(loc=uid * 2.0, scale=0.5, size=n_features)
        cov = np.eye(n_features)  # identity covariance for simplicity
        user_samples = rng.multivariate_normal(mean, cov, size=samples_per_user)
        X_list.append(user_samples)
        uid_list.append(np.full(samples_per_user, uid, dtype=int))
    X = np.vstack(X_list)
    user_ids = np.concatenate(uid_list)
    return X, user_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a simple logistic baseline on synthetic data.")
    parser.add_argument("--n_users", type=int, default=5, help="Number of synthetic users")
    parser.add_argument("--n_features", type=int, default=20, help="Dimensionality of feature vectors")
    parser.add_argument("--samples_per_user", type=int, default=100, help="Samples per user")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Generate synthetic dataset
    X, uids = generate_synthetic_data(args.n_users, args.n_features, args.samples_per_user, args.seed)

    # Fit the per‑user logistic regression baseline
    baseline = LogisticBaseline(max_iter=200)
    baseline.fit(X, uids)

    # Evaluate the model
    all_scores = []
    all_labels = []
    for uid in np.unique(uids):
        scores = baseline.score_user(uid, X)
        labels = (uids == uid).astype(int)
        all_scores.append(scores)
        all_labels.append(labels)
        auc = compute_auc(scores, labels)
        eer, thresh, _ = compute_eer(scores, labels)
        print(f"User {uid:>3}: AUC={auc:.3f}, EER={eer:.3f}, threshold={thresh:.3f}")

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    overall_auc = compute_auc(all_scores, all_labels)
    overall_eer, eer_thresh, far_frr = compute_eer(all_scores, all_labels)
    print("-- Overall performance --")
    print(f"AUC={overall_auc:.3f}, EER={overall_eer:.3f}, threshold={eer_thresh:.3f}")


if __name__ == "__main__":
    main()