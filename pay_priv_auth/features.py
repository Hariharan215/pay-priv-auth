"""Feature extraction utilities.

This module aggregates touch, key and IMU events into fixed time windows
and computes simple statistical features for each modality.  Only timing
and coordinate information is used; any textual content is ignored.
"""

from __future__ import annotations

from typing import Iterable, Iterator, List, Tuple

import numpy as np
import pandas as pd


def _window_iter(df: pd.DataFrame, win_ms: int, hop_ms: int) -> Iterator[Tuple[pd.DataFrame, float]]:
    """Yield windows of ``df`` based on the ``t`` column (seconds)."""
    if df.empty:
        return iter([])
    start = float(df["t"].min())
    end = float(df["t"].max())
    win_s = win_ms / 1000.0
    hop_s = hop_ms / 1000.0
    t = start
    while t <= end:
        yield df[(df["t"] >= t) & (df["t"] < t + win_s)], t
        t += hop_s


def _keystroke_feats(df: pd.DataFrame) -> np.ndarray:
    if df.empty:
        return np.zeros(8, dtype=float)
    times = df["t"].values
    intervals = np.diff(times) if len(times) > 1 else np.array([0.0])
    backspace = np.sum(df["event_type"] == "backspace")
    count = len(df)
    span = times.max() - times.min() if count > 1 else 0.0
    bias = df.get("key_code", pd.Series([0]*count)).mean() if count > 0 else 0.0
    feats = [
        intervals.mean(),
        intervals.std(ddof=0),
        np.percentile(intervals, 10),
        np.percentile(intervals, 90),
        backspace / count,
        count,
        span,
        bias,
    ]
    return np.array(feats, dtype=float)


def _gesture_feats(df: pd.DataFrame) -> np.ndarray:
    if df.empty:
        return np.zeros(10, dtype=float)
    times = df["t"].values
    x = df["x"].values
    y = df["y"].values
    dt = np.diff(times) if len(times) > 1 else np.array([1.0])
    dx = np.diff(x) if len(x) > 1 else np.array([0.0])
    dy = np.diff(y) if len(y) > 1 else np.array([0.0])
    speed = np.sqrt(dx**2 + dy**2) / dt
    pressure_mean = df.get("pressure", pd.Series([0]*len(df))).mean()
    size_mean = df.get("size", pd.Series([0]*len(df))).mean()
    span = np.sqrt((x.max() - x.min())**2 + (y.max() - y.min())**2) if len(df) > 1 else 0.0
    bias = x.mean() - y.mean()
    feats = [
        speed.mean(),
        speed.std(ddof=0),
        np.percentile(speed, 90),
        pressure_mean,
        size_mean,
        dx.mean() if len(dx) > 0 else 0.0,
        dy.mean() if len(dy) > 0 else 0.0,
        span,
        len(df),
        bias,
    ]
    return np.array(feats, dtype=float)


def _imu_feats(df: pd.DataFrame) -> np.ndarray:
    if df.empty:
        return np.zeros(12, dtype=float)
    axes = ["ax", "ay", "az", "gx", "gy", "gz"]
    feats: List[float] = []
    for a in axes:
        vals = df.get(a, pd.Series([0.0]))
        feats.append(float(vals.mean()))
        feats.append(float(vals.std(ddof=0)))
    return np.array(feats, dtype=float)


def extract_features(
    touch: pd.DataFrame,
    keys: pd.DataFrame,
    imu: pd.DataFrame,
    win_ms: int = 2000,
    hop_ms: int = 500,
) -> List[Tuple[float, np.ndarray]]:
    """Extract concatenated feature vectors for a session.

    Returns a list of ``(start_t, feature_vector)`` tuples.
    """
    # Determine global time range
    dfs = [df for df in [touch, keys, imu] if not df.empty]
    if not dfs:
        return []
    start = min(df["t"].min() for df in dfs)
    end = max(df["t"].max() for df in dfs)
    win_s = win_ms / 1000.0
    hop_s = hop_ms / 1000.0
    t = float(start)
    out: List[Tuple[float, np.ndarray]] = []
    while t <= end:
        w_touch = touch[(touch["t"] >= t) & (touch["t"] < t + win_s)]
        w_keys = keys[(keys["t"] >= t) & (keys["t"] < t + win_s)]
        w_imu = imu[(imu["t"] >= t) & (imu["t"] < t + win_s)]
        feats = np.concatenate([
            _keystroke_feats(w_keys),
            _gesture_feats(w_touch),
            _imu_feats(w_imu),
        ])
        out.append((t, feats))
        t += hop_s
    return out
