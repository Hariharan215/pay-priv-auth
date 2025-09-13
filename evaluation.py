"""
Evaluation utilities for behavioural authentication.

This module provides helper functions to compute common metrics used in
authentication tasks, including the area under the ROC curve (AUC), the
equal error rate (EER) and false acceptance/rejection rates at a given
threshold.  We use scikit‑learn to compute ROC curves and derive the
metrics.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn import metrics


def compute_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute the area under the receiver operating characteristic (ROC) curve.

    Parameters
    ----------
    scores : ndarray of shape (n_samples,)
        Predicted probabilities or confidence scores for the positive class.
    labels : ndarray of shape (n_samples,)
        Ground truth binary labels (1 for genuine, 0 for impostor).

    Returns
    -------
    auc : float
        Area under the ROC curve (AUC).  A perfect classifier has AUC=1.
    """
    fpr, tpr, _ = metrics.roc_curve(labels, scores)
    return metrics.auc(fpr, tpr)


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute the equal error rate (EER).

    The EER is the point on the ROC curve where the false acceptance rate
    (FAR) equals the false rejection rate (FRR).  This function returns
    the EER, the corresponding threshold and the FAR/FRR at the EER point.

    Parameters
    ----------
    scores : ndarray of shape (n_samples,)
        Predicted probabilities or confidence scores for the positive class.
    labels : ndarray of shape (n_samples,)
        Ground truth binary labels (1 for genuine, 0 for impostor).

    Returns
    -------
    eer : float
        Equal error rate (in [0, 1]).
    eer_threshold : float
        Score threshold at which FAR equals FRR.
    far_frr : float
        False acceptance/rejection rate at the EER threshold.
    """
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
    fnr = 1.0 - tpr
    # Find the point where |FPR - FNR| is minimal
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    eer_threshold = thresholds[idx]
    return eer, eer_threshold, fpr[idx]


def compute_far_frr(scores: np.ndarray, labels: np.ndarray, threshold: float) -> Tuple[float, float]:
    """
    Compute false acceptance rate (FAR) and false rejection rate (FRR) at a given threshold.

    Parameters
    ----------
    scores : ndarray of shape (n_samples,)
        Predicted probabilities or confidence scores for the positive class.
    labels : ndarray of shape (n_samples,)
        Ground truth binary labels (1 for genuine, 0 for impostor).
    threshold : float
        Decision threshold; samples with scores ≥ threshold are considered genuine.

    Returns
    -------
    far : float
        False acceptance rate: proportion of impostor samples incorrectly accepted.
    frr : float
        False rejection rate: proportion of genuine samples incorrectly rejected.
    """
    preds = scores >= threshold
    genuine_mask = labels == 1
    impostor_mask = labels == 0
    # Avoid division by zero
    n_impostor = np.sum(impostor_mask)
    n_genuine = np.sum(genuine_mask)
    far = np.sum(preds[impostor_mask]) / n_impostor if n_impostor > 0 else 0.0
    frr = np.sum(~preds[genuine_mask]) / n_genuine if n_genuine > 0 else 0.0
    return far, frr