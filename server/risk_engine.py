"""
Simplified risk engine for behavioural authentication.

This module defines the ``RiskEngine`` class which simulates how
behavioural scores can be integrated into a payment flow.  In EMV 3‑D Secure
there are typically two phases: a frictionless flow where the ACS makes a
decision based on risk analysis alone, and a step‑up (challenge) flow where
the user is asked to perform an additional task such as entering an OTP.

The ``RiskEngine`` uses thresholds to determine whether the behavioural
score from a classifier is high enough to approve a transaction without a
challenge, or low enough to decline it outright.  Transactions falling
between these thresholds are sent to a step‑up challenge, where the
behavioural score during OTP entry is evaluated using a lower threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..models.baseline import LogisticBaseline


@dataclass
class RiskEngine:
    """
    Simplified risk engine for deciding payment outcomes based on behavioural scores.

    Parameters
    ----------
    frictionless_threshold : float, default=0.85
        Score above which a transaction is approved without a step‑up challenge.
    stepup_threshold : float, default=0.75
        Score above which a transaction passes the step‑up challenge (OTP).
        Scores below this threshold cause the transaction to be denied.
    """

    frictionless_threshold: float = 0.85
    stepup_threshold: float = 0.75

    def evaluate(self, user_id: int, X_continuous: np.ndarray, X_stepup: np.ndarray, model: LogisticBaseline) -> str:
        """
        Evaluate a transaction using behavioural scores.

        Parameters
        ----------
        user_id : int
            ID of the purported user.
        X_continuous : ndarray of shape (n_windows, n_features)
            Feature windows collected during the frictionless phase (e.g. while
            the user fills in forms).
        X_stepup : ndarray of shape (n_otp_samples, n_features)
            Feature vectors collected during the step‑up challenge (e.g. OTP
            typing).  May be an empty array if no challenge is performed.
        model : LogisticBaseline
            Trained baseline model capable of scoring samples for the given user.

        Returns
        -------
        decision : str
            One of ``"approve"``, ``"challenge"``, ``"deny"`` indicating the
            outcome.  ``"challenge"`` means that a step‑up challenge is
            required; it is only returned if ``X_stepup`` is empty.  If
            ``X_stepup`` is provided, the decision will be ``"approve"`` or
            ``"deny"``.
        """
        # Compute the mean score across continuous windows
        scores_cont = model.score_user(user_id, X_continuous)
        mean_score = float(np.mean(scores_cont)) if scores_cont.size > 0 else 0.0

        # Frictionless decision
        if mean_score >= self.frictionless_threshold:
            return "approve"
        # Score is low enough to justify an immediate denial if no challenge is possible
        if mean_score < self.stepup_threshold and X_stepup.size == 0:
            return "deny"
        # Otherwise we require a challenge
        if X_stepup.size == 0:
            return "challenge"

        # Evaluate step‑up scores
        step_scores = model.score_user(user_id, X_stepup)
        step_mean = float(np.mean(step_scores)) if step_scores.size > 0 else 0.0
        return "approve" if step_mean >= self.stepup_threshold else "deny"