#!/usr/bin/env python3
"""
Utility to enlarge CSV-based test datasets by repeating each data row block.

Example:
    python expand_test_data_csvs.py --factor 200 --base tests/test_data
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable


def expand_csv(path: Path, factor: int) -> None:
    """Rewrite `path` so its body rows repeat `factor` times."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    if not lines:
        tmp_path.write_text("", encoding="utf-8")
        tmp_path.replace(path)
        return

    header, body = lines[0], lines[1:]
    with tmp_path.open("w", encoding="utf-8") as dst:
        dst.write(header)
        if body:
            dst.writelines(_repeat(body, factor))
    tmp_path.replace(path)


def _repeat(body: Iterable[str], factor: int) -> Iterable[str]:
    for _ in range(factor):
        for line in body:
            yield line


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeat the data rows of each CSV under a directory."
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=Path(""),
        help="Directory to scan for CSV files (default: tests/test_data)",
    )
    parser.add_argument(
        "--factor",
        type=int,
        default=200,
        help="Number of times to repeat CSV body rows (default: 200)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.factor < 0:
        raise ValueError("factor must be non-negative")
    base = args.base
    if not base.exists():
        raise FileNotFoundError(f"Base directory {base} does not exist")
    for path in sorted(base.rglob("*.csv")):
        expand_csv(path, args.factor)
        print(f"Expanded {path}")


if __name__ == "__main__":
    main()
