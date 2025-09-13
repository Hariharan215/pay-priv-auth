"""Embedding privacy layer using random projections."""

from __future__ import annotations

import numpy as np


class RandProjEmbedder:
    """Random sign projection with quantisation and noise."""

    def __init__(self, in_dim: int, out_dim: int = 64, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.R = rng.choice([-1.0, 1.0], size=(out_dim, in_dim)).astype(np.float32)

    def __call__(self, x: np.ndarray, q_step: float = 0.05, noise_std: float = 0.01) -> np.ndarray:
        y = self.R @ x
        y = np.round(y / q_step) * q_step
        y = y + np.random.normal(0.0, noise_std, size=y.shape)
        return y.astype(np.float32)
