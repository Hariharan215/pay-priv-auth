"""Per-user logistic regression baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class UserModel:
    model: LogisticRegression
    positive_class: int

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


def _fit_logreg(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=200)
    clf.fit(X, y)
    return clf


@dataclass
class LogisticBaseline:
    penalty: str = "l2"
    C: float = 1.0
    max_iter: int = 100
    models: Dict[int, UserModel] = field(default_factory=dict, init=False)

    def fit(self, X: np.ndarray, user_ids: Iterable[int]) -> "LogisticBaseline":
        uids = np.asarray(list(user_ids), dtype=int)
        unique = np.unique(uids)
        self.models = {}
        for uid in unique:
            y = (uids == uid).astype(int)
            clf = _fit_logreg(X, y)
            self.models[int(uid)] = UserModel(model=clf, positive_class=int(uid))
        return self

    def predict_proba_user(self, user_id: int, X: np.ndarray) -> np.ndarray:
        """Return positive-class probabilities for ``user_id``."""

        if user_id not in self.models:
            raise ValueError(f"No model for user {user_id}")
        return self.models[user_id].score(X)

    score_user = predict_proba_user  # backwards compatibility

    def save(self, dirpath: Path) -> None:
        dir = Path(dirpath)
        dir.mkdir(parents=True, exist_ok=True)
        for uid, um in self.models.items():
            joblib.dump(um.model, dir / f"user_{uid}_logreg.joblib")

    @classmethod
    def load(cls, dirpath: Path) -> "LogisticBaseline":
        dir = Path(dirpath)
        inst = cls()
        for f in dir.glob("user_*_logreg.joblib"):
            uid = int(f.stem.split("_")[1])
            model = joblib.load(f)
            inst.models[uid] = UserModel(model=model, positive_class=uid)
        return inst
