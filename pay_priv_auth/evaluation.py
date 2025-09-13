"""Evaluation utilities."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn import metrics


def compute_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    fpr, tpr, _ = metrics.roc_curve(labels, scores)
    return metrics.auc(fpr, tpr)


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    fpr, tpr, thresh = metrics.roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = (fpr[idx] + fnr[idx]) / 2
    return float(eer), float(thresh[idx]), float(fpr[idx])


def compute_far_frr(scores: np.ndarray, labels: np.ndarray, threshold: float) -> Tuple[float, float]:
    preds = scores >= threshold
    genuine = labels == 1
    impostor = labels == 0
    far = float(np.sum(preds[impostor])) / max(1, np.sum(impostor))
    frr = float(np.sum(~preds[genuine])) / max(1, np.sum(genuine))
    return far, frr
