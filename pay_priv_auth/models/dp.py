"""Opacus differential privacy helpers."""

from __future__ import annotations

from typing import Tuple

from opacus import PrivacyEngine


def make_private_with_target_epsilon(
    model,
    optimizer,
    dataloader,
    target_epsilon: float,
    target_delta: float,
    epochs: int,
    max_grad_norm: float,
) -> Tuple:
    engine = PrivacyEngine()
    model, optimizer, dataloader = engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        epochs=epochs,
        max_grad_norm=max_grad_norm,
    )
    return model, optimizer, dataloader, engine


def get_epsilon(engine: PrivacyEngine, delta: float) -> float:
    return float(engine.accountant.get_epsilon(delta))
