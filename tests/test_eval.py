import numpy as np
import pytest

from pay_priv_auth.evaluation import compute_eer, compute_far_frr


def test_eer():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1, 1, 0, 0])
    eer, th, far = compute_eer(scores, labels)
    assert pytest.approx(eer, abs=1e-6) == far


def test_far_frr():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1, 1, 0, 0])
    far, frr = compute_far_frr(scores, labels, 0.5)
    assert far == 0.0
    assert frr == 0.0
