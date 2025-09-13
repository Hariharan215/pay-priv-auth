"""Dataset builder CLI."""

from __future__ import annotations

import argparse
import pathlib
from typing import List

import pandas as pd

from .datasets import behavepass, hmog
from .features import extract_features


def build_dataset(dataset: str, root: pathlib.Path, out: pathlib.Path, win_ms: int, hop_ms: int) -> None:
    """Build a feature parquet for ``dataset`` under ``root``."""

    root = pathlib.Path(root)
    if not root.exists():
        print(f"Raw data not found at {root}. Please place dataset under data/raw/{dataset}.")
        return
    loader = hmog.iter_sessions if dataset == "hmog" else behavepass.iter_sessions
    rows: List[pd.DataFrame] = []
    for sess in loader(root):
        feats = extract_features(sess["touch"], sess["keys"], sess["imu"], win_ms, hop_ms)
        for start_t, vec in feats:
            row = {
                "user_id": sess["user_id"],
                "session_id": sess["session_id"],
                "phase": sess.get("phase", "unknown"),
                "start_t": start_t,
            }
            for i, v in enumerate(vec):
                row[f"f{i}"] = v
            rows.append(row)
    if not rows:
        print("No sessions found. Ensure the directory layout matches the specification.")
        return
    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"Wrote {len(df)} windows to {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build feature dataset")
    ap.add_argument("--dataset", choices=["hmog", "behavepass"], required=True)
    ap.add_argument("--root", type=pathlib.Path, required=True, help="Root of raw dataset")
    ap.add_argument("--out", type=pathlib.Path, required=True, help="Output parquet path")
    ap.add_argument("--win_ms", type=int, default=2000)
    ap.add_argument("--hop_ms", type=int, default=500)
    args = ap.parse_args()
    build_dataset(args.dataset, args.root, args.out, args.win_ms, args.hop_ms)


if __name__ == "__main__":  # pragma: no cover
    main()
