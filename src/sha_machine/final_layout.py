#!/usr/bin/env python3
"""Exact 1024-group raster layout and row schedule for the SHA machine.

This module contains no hand-wavy placement.  It assigns every stored
halfword, every schedule READY/reducer pair, the round scratch pixels, and
every logical machine row to an integer coordinate.  ``verify_layout`` checks
all west-to-east causality constraints used by the compiler.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Literal, Mapping, Sequence

from final_unit_design import minimum_schedule_scratch_boundaries


WIDTH = 1024
HEIGHT = 1024
PACK_ROWS = 40
ROUND_ROWS = 12
ROUNDS = 64
FF_ROWS = 32
DISPLAY_ROWS = 32


@dataclass(frozen=True)
class Half:
    m: int
    q: int


@dataclass(frozen=True)
class Word:
    lo: Half
    hi: Half


@dataclass(frozen=True)
class Gap:
    ready: int
    reducer: int


@dataclass(frozen=True)
class StateSlot:
    a: Word
    e: Word
    scratch: tuple[int, int, int]


@dataclass(frozen=True)
class Layout:
    w: Mapping[int, Word]
    schedule_gap: Mapping[int, Gap]
    bootstrap: tuple[int, int, int, int]
    state: Sequence[StateSlot]
    used_width: int


def _word(x: int) -> Word:
    return Word(Half(x, x + 1), Half(x + 2, x + 3))


def build_layout() -> Layout:
    x = 16
    w: Dict[int, Word] = {}
    gaps: Dict[int, Gap] = {}
    points = set(minimum_schedule_scratch_boundaries())

    for index in range(8):
        w[index] = _word(x)
        x += 4
    # W8 is asserted zero.  Boundary 8 is nevertheless a real schedule gap.
    assert 8 in points
    gaps[8] = Gap(x, x + 1)
    x += 2

    # W9 is the low 32 bits of the file length and W15 aliases it.  W10 is
    # 0x80000000 and W11..14 are zero, so none needs another memory word.
    length = _word(x)
    w[9] = length
    w[15] = length
    x += 4
    assert 15 in points
    gaps[15] = Gap(x, x + 1)
    x += 2

    for index in range(16, 64):
        w[index] = _word(x)
        x += 4
        if index in points:
            gaps[index] = Gap(x, x + 1)
            x += 2

    assert set(gaps) == points

    bootstrap = tuple(range(x, x + 4))
    x += 4
    state = []
    for _round in range(ROUNDS):
        a = _word(x)
        e = _word(x + 4)
        scratch = (x + 8, x + 9, x + 10)
        state.append(StateSlot(a, e, scratch))
        x += 11
    assert x == 1002
    return Layout(w, gaps, bootstrap, tuple(state), x)


def p1_row(t: int, byte: int) -> int:
    return PACK_ROWS + ROUND_ROWS * t + byte


def p2_row(t: int, p: int) -> int:
    return PACK_ROWS + ROUND_ROWS * t + 4 + p


def ff_row(word: int, byte: int) -> int:
    return PACK_ROWS + ROUND_ROWS * ROUNDS + 4 * word + byte


DISPLAY_Y0 = PACK_ROWS + ROUND_ROWS * ROUNDS + FF_ROWS
assert DISPLAY_Y0 == 840


def const_w(index: int) -> int | None:
    if index == 8 or 11 <= index <= 14:
        return 0
    if index == 10:
        return 0x80000000
    return None


def a_source(t: int, lag: int) -> tuple[Literal["memory", "initial"], int]:
    """Register a,b,c,d (lag 0..3) at the start of round t."""
    if t > lag:
        return "memory", t - 1 - lag
    # The pre-round state is the variable SHA midstate carried by W0..W7.
    return "initial", lag - t


def e_source(t: int, lag: int) -> tuple[Literal["memory", "initial"], int]:
    """Register e,f,g,h (lag 0..3) at the start of round t."""
    if t > lag:
        return "memory", t - 1 - lag
    return "initial", 4 + lag - t


def choose_gap(start: int, end: int, *, prefer: int | None = None) -> int:
    candidates = [p for p in minimum_schedule_scratch_boundaries()
                  if start <= p <= end]
    if prefer is not None and prefer in candidates:
        return prefer
    if not candidates:
        raise AssertionError((start, end))
    return candidates[-1]


def schedule_reducer_boundaries(t: int) -> tuple[int, int]:
    if not 16 <= t < 64:
        raise ValueError(t)
    # Prefer boundary 15 for the only rounds in which the first phase has no
    # stored source (W10..14 constants); all gaps have a READY then reducer.
    r0 = choose_gap(t - 15, t - 8,
                    prefer=15 if 23 <= t <= 30 else None)
    r1 = choose_gap(t - 2, t - 1)
    return r0, r1


def digest_words(layout: Layout) -> tuple[Word, ...]:
    # Final working state is A63,A62,A61,A60,E63,E62,E61,E60.
    return tuple(layout.state[t].a for t in (63, 62, 61, 60)) + tuple(
        layout.state[t].e for t in (63, 62, 61, 60))


def pack_target(layout: Layout, suffix_byte: int) -> tuple[Half, int] | None:
    """Return (halfword, shift) for an aperture byte, or None if asserted 0."""
    if not 0 <= suffix_byte < 40:
        raise ValueError(suffix_byte)
    if suffix_byte < 32:
        word = layout.w[suffix_byte // 4]
        within = suffix_byte & 3
    elif suffix_byte < 36:
        return None  # high 32 file-length bits, asserted zero
    else:
        word = layout.w[9]
        within = suffix_byte - 36
    # Input bytes are big endian; rows arrive in file order.
    return ((word.hi, 8) if within == 0 else
            (word.hi, 0) if within == 1 else
            (word.lo, 8) if within == 2 else
            (word.lo, 0))


def source_x(word: Word, half: str) -> int:
    return word.lo.m if half == "lo" else word.hi.m


def source_q(word: Word, half: str) -> int:
    return word.lo.q if half == "lo" else word.hi.q


def verify_layout(layout: Layout | None = None) -> dict[str, object]:
    layout = layout or build_layout()
    assert layout.used_width <= WIDTH
    assert DISPLAY_Y0 + DISPLAY_ROWS <= HEIGHT

    schedule_rows = []
    for t in range(16, 64):
        r0_boundary, r1_boundary = schedule_reducer_boundaries(t)
        r0 = layout.schedule_gap[r0_boundary]
        r1 = layout.schedule_gap[r1_boundary]
        target = layout.w[t]

        # Every stored phase-0 source is west of READY/R0.  Constants are
        # injected by a one-leaf control and have no memory coordinate.
        for index in (t - 16, t - 15):
            if index in layout.w:
                assert layout.w[index].hi.q < r0.ready < r0.reducer
        # R0 is before every stored phase-1 source and R1.
        for index in (t - 7, t - 2):
            if index in layout.w:
                assert r0.reducer < layout.w[index].lo.m
                assert layout.w[index].hi.q < r1.ready
        assert r1.ready < r1.reducer < target.lo.m
        schedule_rows.append({
            "round": t,
            "R0_boundary": r0_boundary,
            "R0_xy": [r0.ready, r0.reducer],
            "R1_boundary": r1_boundary,
            "R1_xy": [r1.ready, r1.reducer],
            "target_lo_m": target.lo.m,
        })

    round_rows = []
    for t in range(64):
        if t == 0:
            before = layout.bootstrap
            assert before[-1] + 1 == layout.state[0].a.lo.m
        else:
            prior = layout.state[t - 1]
            before = prior.scratch
            assert prior.e.hi.q < before[0]
            assert before[-1] + 1 == layout.state[t].a.lo.m

        # Generated P1 A/B/C and P2 D/H/G/F/E sources are monotonically west
        # of the reducer scratch belonging to the preceding state.
        for lag in (2, 1, 0):
            kind, index = a_source(t, lag)
            if kind == "memory":
                assert layout.state[index].a.hi.q < before[-1]
        for lag in (3, 2, 1, 0):
            kind, index = e_source(t, lag)
            if kind == "memory":
                assert layout.state[index].e.hi.q < before[-1]
        round_rows.append({
            "round": t,
            "P1_rows": [p1_row(t, b) for b in range(4)],
            "P2_rows": [p2_row(t, p) for p in range(8)],
            "pre_target_scratch": list(before),
            "A_target": asdict(layout.state[t].a),
            "E_target": asdict(layout.state[t].e),
        })

    digests = digest_words(layout)
    for i, destination in enumerate(digests):
        assert layout.w[i].hi.q < destination.lo.m

    return {
        "width": WIDTH,
        "height": HEIGHT,
        "used_width": layout.used_width,
        "right_margin": WIDTH - layout.used_width,
        "pack_rows": [0, PACK_ROWS - 1],
        "round_rows": [PACK_ROWS, PACK_ROWS + ROUND_ROWS * ROUNDS - 1],
        "feedforward_rows": [ff_row(0, 0), ff_row(7, 3)],
        "display_rows": [DISPLAY_Y0, DISPLAY_Y0 + DISPLAY_ROWS - 1],
        "schedule_boundaries": list(minimum_schedule_scratch_boundaries()),
        "schedule_examples": schedule_rows[:8],
        "early_rounds": round_rows[:8],
        "digest_halfword_M_columns_physical_order": [
            half.m for slot in layout.state[60:64]
            for word in (slot.a, slot.e) for half in (word.lo, word.hi)
        ],
        "digest_halfword_M_columns_hash_order": [
            half.m for word in digests for half in (word.hi, word.lo)
        ],
    }


def main() -> None:
    print(json.dumps(verify_layout(), indent=2))


if __name__ == "__main__":
    main()
