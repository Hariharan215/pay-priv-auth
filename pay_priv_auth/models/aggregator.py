"""Model aggregation helpers."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.linear_model import LogisticRegression


def fedavg_logistic(models_by_user: Dict[int, LogisticRegression]) -> LogisticRegression:
    """Average logistic regression models via FedAvg."""

    models = [m for m in models_by_user.values() if hasattr(m, "coef_")]
    if not models:
        raise ValueError("No models to average")
    base_shape = models[0].coef_.shape
    models = [m for m in models if m.coef_.shape == base_shape]
    coef = np.mean([m.coef_ for m in models], axis=0)
    intercept = np.mean([m.intercept_ for m in models], axis=0)
    agg = LogisticRegression()
    agg.classes_ = models[0].classes_.copy()
    agg.coef_ = coef
    agg.intercept_ = intercept
    return agg
