"""Minimal Flower federated learning simulation."""

from __future__ import annotations

import argparse
import pathlib
from typing import Dict, Tuple

import numpy as np
from sklearn.linear_model import SGDClassifier

import flwr as fl

import pandas as pd


def load_user_data(parquet_path: pathlib.Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    df = pd.read_parquet(parquet_path)
    features = [c for c in df.columns if c.startswith("f")]
    data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for uid in df["user_id"].unique():
        pos = df[df["user_id"] == uid][features].to_numpy()
        neg = df[df["user_id"] != uid][features].to_numpy()
        neg = neg[: len(pos)]
        X_u = np.vstack([pos, neg])
        y_u = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        data[str(uid)] = (X_u, y_u)
    return data


class Client(fl.client.NumPyClient):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X, self.y = X, y
        self.model = SGDClassifier(loss="log_loss", max_iter=5)

    def get_parameters(self, config):  # pragma: no cover - simple forwarding
        if hasattr(self.model, "coef_"):
            return [self.model.coef_, self.model.intercept_]
        n_features = self.X.shape[1]
        return [np.zeros((1, n_features)), np.zeros(1)]

    def fit(self, parameters, config):  # pragma: no cover - network IO
        coef, intercept = parameters
        self.model.coef_ = coef
        self.model.intercept_ = intercept
        self.model.partial_fit(self.X, self.y, classes=np.array([0, 1]))
        return [self.model.coef_, self.model.intercept_], len(self.X), {}

    def evaluate(self, parameters, config):  # pragma: no cover
        coef, intercept = parameters
        self.model.coef_ = coef
        self.model.intercept_ = intercept
        loss = 0.0
        return float(loss), len(self.X), {}


def start_server(num_rounds: int = 3) -> None:
    fl.server.start_server(server_address="0.0.0.0:8080", config=fl.server.ServerConfig(num_rounds=num_rounds))


def start_clients(parquet: pathlib.Path, n_clients: int) -> None:
    data = load_user_data(parquet)
    users = list(data.keys())[:n_clients]
    for uid in users:
        X, y = data[uid]
        fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=Client(X, y))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["server", "client"])
    ap.add_argument("--dataset_parquet", type=pathlib.Path, required=True)
    ap.add_argument("--num_rounds", type=int, default=3)
    ap.add_argument("--n_clients", type=int, default=2)
    args = ap.parse_args()
    if args.mode == "server":
        start_server(args.num_rounds)
    else:
        start_clients(args.dataset_parquet, args.n_clients)


if __name__ == "__main__":  # pragma: no cover
    main()
