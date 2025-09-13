"""Loader for the HMOG dataset.

This module exposes :func:`iter_sessions` which yields normalised
representations for each session in the raw HMOG directory.  The raw
layout is expected to be ``<root>/<user>/<session>/`` with files
containing touch, key and sensor events.  Only timing and positional
information is retained; any textual content is ignored.
"""

from __future__ import annotations

import json
import pathlib
from typing import Dict, Iterator

import pandas as pd

TOUCH_COLS = ["t", "x", "y", "pressure", "size", "event_type"]
KEYS_COLS = ["t", "event_type", "key_code"]
IMU_COLS = ["t", "ax", "ay", "az", "gx", "gy", "gz"]


def _empty_df(cols) -> pd.DataFrame:
    return pd.DataFrame(columns=cols)


def _load_csv(path: pathlib.Path, cols) -> pd.DataFrame:
    if not path.exists():
        return _empty_df(cols)
    df = pd.read_csv(path)
    df = df.rename(columns={c: c.lower() for c in df.columns})
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
    df = df[cols]
    # Convert timestamps to seconds if they appear to be in milliseconds
    if not df.empty:
        t = df[cols[0]].astype(float)
        if t.max() > 1e6:
            t = t / 1000.0
        df[cols[0]] = t
    return df


def _load_imu(sess: pathlib.Path) -> pd.DataFrame:
    acc = _load_csv(sess / "accelerometer.csv", ["t", "ax", "ay", "az"])
    gyr = _load_csv(sess / "gyroscope.csv", ["t", "gx", "gy", "gz"])
    if acc.empty and gyr.empty:
        return _load_csv(sess / "imu.csv", IMU_COLS)
    if acc.empty:
        acc = _empty_df(["t", "ax", "ay", "az"])
    if gyr.empty:
        gyr = _empty_df(["t", "gx", "gy", "gz"])
    acc = acc.sort_values("t")
    gyr = gyr.sort_values("t")
    imu = pd.merge_asof(acc, gyr, on="t", direction="nearest")
    imu = imu.reindex(columns=IMU_COLS, fill_value=0.0)
    return imu


def iter_sessions(root: pathlib.Path) -> Iterator[Dict]:
    """Iterate over sessions under ``root``.

    Yields dictionaries with keys ``user_id``, ``session_id``, ``phase``,
    ``condition``, ``touch``, ``keys`` and ``imu``.  Missing modalities are
    returned as empty data frames with the expected columns.
    """

    root = pathlib.Path(root)
    if not root.exists():
        return
    for user_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        user_id = user_dir.name
        for sess_dir in sorted(p for p in user_dir.iterdir() if p.is_dir()):
            session_id = sess_dir.name
            phase = "unknown"
            condition = "unknown"
            meta_file = sess_dir / "meta.json"
            if meta_file.exists():
                meta = json.loads(meta_file.read_text())
                phase = meta.get("phase", "unknown")
                condition = meta.get("condition", "unknown")
            touch = _load_csv(sess_dir / "touch.csv", TOUCH_COLS)
            keys = _load_csv(sess_dir / "keys.csv", KEYS_COLS)
            imu = _load_imu(sess_dir)
            yield {
                "user_id": user_id,
                "session_id": session_id,
                "phase": phase,
                "condition": condition,
                "touch": touch,
                "keys": keys,
                "imu": imu,
            }
