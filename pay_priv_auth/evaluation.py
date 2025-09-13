"""Evaluation utilities."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn import metrics


def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute the area under the ROC curve."""

    return float(metrics.roc_auc_score(y_true, y_score))


def eer(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """Return the equal error rate and corresponding threshold."""

    fpr, tpr, thresh = metrics.roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer_val = (fpr[idx] + fnr[idx]) / 2
    return float(eer_val), float(thresh[idx])


def far_frr_at(
    y_true: np.ndarray, y_score: np.ndarray, target_frr: float
) -> Tuple[float, float]:
    """FAR at a desired FRR."""

    fpr, tpr, thresh = metrics.roc_curve(y_true, y_score)
    frr = 1 - tpr
    idx = int(np.nanargmin(np.abs(frr - target_frr)))
    return float(fpr[idx]), float(thresh[idx])


def frr_far_at(
    y_true: np.ndarray, y_score: np.ndarray, target_far: float
) -> Tuple[float, float]:
    """FRR at a desired FAR."""

    fpr, tpr, thresh = metrics.roc_curve(y_true, y_score)
    frr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fpr - target_far)))
    return float(frr[idx]), float(thresh[idx])
