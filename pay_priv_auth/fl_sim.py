"""Minimal Flower federated learning simulation."""

from __future__ import annotations

import argparse
from typing import Dict, Tuple

import numpy as np
from sklearn.linear_model import SGDClassifier

import flwr as fl

from .train import generate_synthetic_data


def make_user_data(n_users: int, n_features: int, samples_per_user: int, seed: int = 42) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    X, uids = generate_synthetic_data(n_users, n_features, samples_per_user, seed)
    data: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for uid in range(n_users):
        pos = X[uids == uid]
        neg = X[uids != uid][: len(pos)]
        X_u = np.vstack([pos, neg])
        y_u = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        data[uid] = (X_u, y_u)
    return data


class Client(fl.client.NumPyClient):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X, self.y = X, y
        self.model = SGDClassifier(loss="log_loss", max_iter=5)

    def get_parameters(self, config):
        if hasattr(self.model, "coef_"):
            return [self.model.coef_, self.model.intercept_]
        n_features = self.X.shape[1]
        return [np.zeros((1, n_features)), np.zeros(1)]

    def fit(self, parameters, config):
        coef, intercept = parameters
        self.model.coef_ = coef
        self.model.intercept_ = intercept
        self.model.partial_fit(self.X, self.y, classes=np.array([0, 1]))
        return [self.model.coef_, self.model.intercept_], len(self.X), {}

    def evaluate(self, parameters, config):
        coef, intercept = parameters
        self.model.coef_ = coef
        self.model.intercept_ = intercept
        loss = 0.0
        return float(loss), len(self.X), {}


def start_server(num_rounds: int = 3) -> None:
    fl.server.start_server(server_address="0.0.0.0:8080", config=fl.server.ServerConfig(num_rounds=num_rounds))


def start_clients(n_clients: int, n_features: int = 20, samples_per_user: int = 50) -> None:
    data = make_user_data(n_clients, n_features, samples_per_user)
    for uid in range(n_clients):
        X, y = data[uid]
        fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=Client(X, y))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["server", "client"])
    ap.add_argument("--num_rounds", type=int, default=3)
    ap.add_argument("--n_clients", type=int, default=2)
    args = ap.parse_args()
    if args.mode == "server":
        start_server(args.num_rounds)
    else:
        start_clients(args.n_clients)


if __name__ == "__main__":  # pragma: no cover
    main()
