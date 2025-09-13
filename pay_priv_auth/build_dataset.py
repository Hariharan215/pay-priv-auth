"""Dataset builder.

Scans a raw data directory and produces a parquet file containing
windowed feature vectors.  The expected layout is
``data/raw/<dataset>/<user>/<session>/`` with files:
``touch.csv``, ``keys.csv``, ``imu.csv`` and ``meta.json``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import List

import pandas as pd

from .features import extract_features


def process_session(sess_path: pathlib.Path, win_ms: int, hop_ms: int) -> pd.DataFrame:
    touch = pd.read_csv(sess_path / "touch.csv") if (sess_path / "touch.csv").exists() else pd.DataFrame()
    keys = pd.read_csv(sess_path / "keys.csv") if (sess_path / "keys.csv").exists() else pd.DataFrame()
    imu = pd.read_csv(sess_path / "imu.csv") if (sess_path / "imu.csv").exists() else pd.DataFrame()
    meta = {}
    if (sess_path / "meta.json").exists():
        meta = json.loads((sess_path / "meta.json").read_text())
    feats = extract_features(touch, keys, imu, win_ms=win_ms, hop_ms=hop_ms)
    rows = []
    for start_t, vec in feats:
        row = {
            "user_id": meta.get("user_id"),
            "session_id": meta.get("session_id"),
            "phase": meta.get("phase"),
            "start_t": start_t,
        }
        for i, v in enumerate(vec):
            row[f"f{i}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def build_dataset(root: pathlib.Path, out: pathlib.Path, win_ms: int, hop_ms: int) -> None:
    if not root.exists():
        print(f"Raw data not found at {root}. Please place dataset in data/raw/<DATASET>.")
        return
    all_rows: List[pd.DataFrame] = []
    for user_dir in sorted(root.glob("*/")):
        for sess_dir in sorted(user_dir.glob("*/")):
            df = process_session(sess_dir, win_ms, hop_ms)
            if not df.empty:
                all_rows.append(df)
    if not all_rows:
        print("No sessions found. Ensure the directory layout matches the specification.")
        return
    final = pd.concat(all_rows, ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    final.to_parquet(out)
    print(f"Wrote {len(final)} windows to {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build feature dataset")
    ap.add_argument("--root", type=pathlib.Path, required=True, help="Root of raw dataset")
    ap.add_argument("--out", type=pathlib.Path, required=True, help="Output parquet path")
    ap.add_argument("--win_ms", type=int, default=2000)
    ap.add_argument("--hop_ms", type=int, default=500)
    args = ap.parse_args()
    build_dataset(args.root, args.out, args.win_ms, args.hop_ms)


if __name__ == "__main__":  # pragma: no cover
    main()
