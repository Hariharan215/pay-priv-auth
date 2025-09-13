"""Federated averaging for linear models."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression


class FedAvgAggregator:
    @staticmethod
    def average(models: Iterable[LogisticRegression], weights: Optional[Iterable[float]] = None) -> LogisticRegression:
        models = list(models)
        if not models:
            raise ValueError("No models provided")
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        else:
            weights = list(weights)
            total = float(sum(weights))
            weights = [w / total for w in weights]
        coef = np.zeros_like(models[0].coef_)
        intercept = np.zeros_like(models[0].intercept_)
        for w, m in zip(weights, models):
            coef += w * m.coef_
            intercept += w * m.intercept_
        agg = LogisticRegression()
        agg.classes_ = models[0].classes_.copy()
        agg.coef_ = coef
        agg.intercept_ = intercept
        return agg
