import numpy as np
import pytest

from pay_priv_auth.evaluation import eer, far_frr_at


def test_eer():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1, 1, 0, 0])
    e, th = eer(labels, scores)
    far, _ = far_frr_at(labels, scores, e)
    assert pytest.approx(e, abs=1e-6) == far


def test_far_frr():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1, 1, 0, 0])
    far, _ = far_frr_at(labels, scores, 0.0)
    assert far == 0.0
