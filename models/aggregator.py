"""
Federated averaging utilities.

This module provides simple functionality to perform federated averaging of
model parameters in a privacy‑preserving behavioural authentication setting.  In
the baseline code we focus on logistic regression models from scikit‑learn.
The aggregator can be extended to handle other model types by adding
appropriate hooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Union, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class FedAvgAggregator:
    """
    A simple federated averaging helper for scikit‑learn logistic regression models.

    Federated averaging (FedAvg) combines model parameters from multiple clients
    by computing their (possibly weighted) average.  This implementation
    aggregates the ``coef_`` and ``intercept_`` attributes of scikit‑learn
    ``LogisticRegression`` estimators.  All models must share the same
    architecture (i.e. same feature dimensionality and classes).
    """

    @staticmethod
    def average(models: Iterable[LogisticRegression], weights: Optional[Iterable[float]] = None) -> LogisticRegression:
        """
        Aggregate a collection of ``LogisticRegression`` models by averaging
        their parameters.

        Parameters
        ----------
        models : iterable of LogisticRegression
            Trained estimators from different clients.
        weights : iterable of float, optional
            Relative weights to assign to each model when averaging.  If
            provided, the length must equal the number of models.  If
            omitted, all models are equally weighted.

        Returns
        -------
        aggregated_model : LogisticRegression
            A new estimator whose parameters are the weighted average of the
            input models.

        Note
        ----
        The returned model is a copy of the first model in the iterable with
        averaged coefficients and intercepts.  Training metadata such as
        convergence information is not aggregated.
        """
        models = list(models)
        if not models:
            raise ValueError("At least one model is required for aggregation.")
        n_models = len(models)
        if weights is not None:
            weights = list(weights)
            if len(weights) != n_models:
                raise ValueError("Number of weights must equal number of models.")
            # Normalise weights to sum to 1
            weight_sum = float(sum(weights))
            if weight_sum == 0.0:
                raise ValueError("Sum of weights must be non‑zero.")
            weights = [w / weight_sum for w in weights]
        else:
            weights = [1.0 / n_models] * n_models

        # Check that all models have the same shape
        coef_shapes = [m.coef_.shape for m in models]
        if len(set(coef_shapes)) != 1:
            raise ValueError("All models must have the same coefficient shape for aggregation.")

        # Initialize accumulators
        avg_coef = np.zeros_like(models[0].coef_)
        avg_intercept = np.zeros_like(models[0].intercept_)

        for weight, model in zip(weights, models):
            avg_coef += weight * model.coef_
            avg_intercept += weight * model.intercept_

        # Create a copy of the first model and assign averaged parameters
        aggregated_model = LogisticRegression()
        aggregated_model.classes_ = models[0].classes_.copy()
        aggregated_model.coef_ = avg_coef
        aggregated_model.intercept_ = avg_intercept
        return aggregated_model