"""Minimal resident helpers required by the hashquine compiler.

The original module's report-only SHA-map inventory is not needed while
building the final controller.  ``constant_runs`` is the sole construction
primitive used by the compiler.
"""

from __future__ import annotations

from typing import Iterable, Sequence


def constant_runs(values: Sequence[int]) -> list[tuple[int, int, int]]:
    """Return half-open maximal constant runs as (start, end, value)."""
    if not values:
        return []
    result: list[tuple[int, int, int]] = []
    start = 0
    value = values[0]
    for index, candidate in enumerate(values[1:], 1):
        if candidate != value:
            result.append((start, index, value))
            start = index
            value = candidate
    result.append((start, len(values), value))
    return result


def unique_sha_maps() -> Iterable[object]:
    """Report-only compatibility shim; construction does not call this."""
    return ()
