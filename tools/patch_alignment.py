#!/usr/bin/env python3
"""Patch the one-digit c0 alignment threshold in a generated tree."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tree", type=Path)
    parser.add_argument("visits", type=int, choices=range(1, 9))
    args = parser.parse_args()

    with args.tree.open("r+b") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        window = min(size, 8192)
        stream.seek(size - window)
        tail = stream.read(window)
        anchor = b"          if y > 0\n            - Set + 0\n            if x > "
        start = tail.rfind(anchor)
        if start < 0:
            raise SystemExit("alignment subtree not found at EOF")
        digit_offset = start + len(anchor)
        old = tail[digit_offset:digit_offset + 1]
        if old not in b"01234567" or tail[digit_offset + 1:digit_offset + 2] != b"\n":
            raise SystemExit(f"unexpected threshold bytes: {old!r}")
        new = str(args.visits - 1).encode("ascii")
        stream.seek(size - window + digit_offset)
        stream.write(new)
    print(f"alignment visits: {int(old) + 1} -> {args.visits}")


if __name__ == "__main__":
    main()
