"""
Simple logistic regression baseline for behavioural authentication.

This module defines the ``LogisticBaseline`` class which trains a separate logistic
regression model for each user.  Each model distinguishes the legitimate
behaviour of its corresponding user from the behaviour of all other users
(impostors).  This approach is straightforward and serves as a strong
baseline for behavioural biometrics tasks where the feature representation
already captures sufficient discriminative information.

The implementation uses scikit‑learn's ``LogisticRegression`` under the hood.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted


@dataclass
class UserModel:
    """Container for a per‑user logistic regression model."""

    model: LogisticRegression
    positive_class: int  # the user ID treated as positive class

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the probability that each row in ``X`` is genuine behaviour for
        this user.  Returns a one‑dimensional array of scores in [0, 1].
        """
        check_is_fitted(self.model)
        # ``predict_proba`` returns probabilities for class order [0, 1]; we want
        # probability of class 1 (positive/genuine).
        proba = self.model.predict_proba(X)[:, 1]
        return proba


@dataclass
class LogisticBaseline:
    """
    Per‑user logistic regression baseline.

    When calling ``fit``, the baseline learns a separate binary classifier for
    each user ID in ``user_ids``.  For each user ``u``, samples belonging to
    ``u`` are labelled as genuine (1) while samples from other users are
    labelled as impostor (0).  During scoring, the appropriate model can be
    invoked via ``score_user``.
    """

    penalty: str = "l2"
    C: float = 1.0
    max_iter: int = 100
    models: Dict[int, UserModel] = field(default_factory=dict, init=False)

    def fit(self, X: np.ndarray, user_ids: Iterable[int]) -> "LogisticBaseline":
        """
        Fit a logistic regression model per user.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix.
        user_ids : iterable of length n_samples
            Sequence of integer user IDs indicating which sample belongs to
            which user.  Must contain at least two distinct users.

        Returns
        -------
        self : LogisticBaseline
            Fitted estimator.
        """
        # Convert user IDs to a NumPy array for indexing
        uids = np.asarray(list(user_ids), dtype=int)
        # Determine the unique set of users
        unique_users: np.ndarray = np.unique(uids)
        if unique_users.size < 2:
            raise ValueError("At least two users are required to train the baseline.")

        self.models = {}
        for uid in unique_users:
            # Create binary labels: 1 for samples of uid, 0 for all others
            y_binary = (uids == uid).astype(int)
            # Instantiate a new logistic regression model for this user
            clf = LogisticRegression(
                penalty=self.penalty,
                C=self.C,
                max_iter=self.max_iter,
                solver="liblinear",
                class_weight="balanced",
            )
            clf.fit(X, y_binary)
            self.models[int(uid)] = UserModel(model=clf, positive_class=int(uid))
        return self

    def score_user(self, user_id: int, X: np.ndarray) -> np.ndarray:
        """
        Score a batch of samples for a specific user.

        Parameters
        ----------
        user_id : int
            ID of the user whose model should be used for scoring.
        X : ndarray of shape (n_samples, n_features)
            Feature matrix to be scored.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Scores in [0, 1] representing the probability that each sample
            originates from ``user_id``.
        """
        if user_id not in self.models:
            raise ValueError(f"No model found for user {user_id}. Did you call fit?")
        return self.models[user_id].score(X)

    def predict_user(self, user_id: int, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict whether samples belong to a user given a threshold.

        Parameters
        ----------
        user_id : int
            ID of the user.
        X : ndarray of shape (n_samples, n_features)
            Feature matrix to classify.
        threshold : float, default=0.5
            Threshold above which a sample is considered genuine.

        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Boolean predictions (True for genuine, False for impostor).
        """
        scores = self.score_user(user_id, X)
        return scores >= threshold