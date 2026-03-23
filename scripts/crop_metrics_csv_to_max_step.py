#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _crop_csv(path: Path, max_step: int) -> bool:
    df = pd.read_csv(path)
    if "step" not in df.columns:
        return False

    step = pd.to_numeric(df["step"], errors="coerce")
    keep = step.isna() | (step <= int(max_step))
    cropped = df.loc[keep].copy()
    if len(cropped) == len(df):
        return False

    cropped.to_csv(path, index=False)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Crop metrics CSV files in-place to a max global step.")
    ap.add_argument("root", help="Directory to scan")
    ap.add_argument("--pattern", action="append", default=[], help="Glob pattern(s) under root")
    ap.add_argument("--max-step", type=int, required=True, help="Maximum step to keep")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"root not found: {root}")

    patterns = args.pattern or ["**/*.csv"]
    seen: set[Path] = set()
    changed = 0

    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            if _crop_csv(path, args.max_step):
                changed += 1
                print(f"[OK] cropped -> {path}")

    print(f"[DONE] changed={changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
