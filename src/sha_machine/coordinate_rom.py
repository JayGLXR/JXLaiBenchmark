#!/usr/bin/env python3
"""Generate the sparse (x,y) action ROM for ``full_tree_compiler``.

The ROM is kept separate from the large arithmetic tree so collisions and
causal ordering can be audited cheaply.  It records source retags, READY
setters, reducer/target sites, and feed-forward operations for all 840
compute rows.  A later lowering turns each event into a small MA leaf/subtree
under an x-first/y-second coordinate tree.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Mapping, Sequence

from final_layout import (
    DISPLAY_Y0,
    a_source,
    build_layout,
    const_w,
    digest_words,
    e_source,
    ff_row,
    p1_row,
    p2_row,
    pack_target,
    schedule_reducer_boundaries,
)
from sha_machine import K, small_sigma_fragment
from controller_budget import spread_bits, spread_byte


@dataclass(frozen=True)
class Event:
    x: int
    y: int
    kind: str
    arg: str = ""


class Rom:
    def __init__(self) -> None:
        self.events: Dict[tuple[int, int], Event] = {}

    def add(self, x: int, y: int, kind: str, arg: object = "") -> None:
        event = Event(x, y, kind, str(arg))
        key = x, y
        if key in self.events:
            raise AssertionError(("ROM collision", self.events[key], event))
        self.events[key] = event


def active_half(byte_or_nibble: int, *, byte: bool) -> str:
    if byte:
        return "lo" if byte_or_nibble < 2 else "hi"
    return "lo" if byte_or_nibble < 4 else "hi"


def source_m(word, half: str) -> int:
    return word.lo.m if half == "lo" else word.hi.m


def p1_seed(t: int, byte: int) -> int:
    # The pre-round state is the variable aperture midstate W0..W7, never IV.
    return 0


def p2_seed(t: int, p: int) -> int:
    total = 16 * ((K[t] >> (4 * p)) & 15)
    value = const_w(t)
    if value is not None:
        total += 16 * ((value >> (4 * p)) & 15)
    return total


def _schedule_const(word: int, role: str, byte: int) -> int:
    if role == "direct0" or role == "direct1":
        return (word >> (8 * byte)) & 255
    which, scale = ((0, 256) if role == "sigma0" else (1, 1024))
    p0 = 2 * byte
    lo = (small_sigma_fragment(which, word & 0xFFFF, "lo", p0) |
          small_sigma_fragment(which, word & 0xFFFF, "lo", p0 + 1) << 4)
    hi = (small_sigma_fragment(which, word >> 16, "hi", p0) |
          small_sigma_fragment(which, word >> 16, "hi", p0 + 1) << 4)
    return scale * (spread_byte(lo, 3) + spread_byte(hi, 3))


def schedule_phase_constants(t: int, byte: int) -> tuple[int, int]:
    phase0 = phase1 = 0
    for index, role, phase in ((t - 16, "direct0", 0),
                               (t - 15, "sigma0", 0),
                               (t - 7, "direct1", 1),
                               (t - 2, "sigma1", 1)):
        value = const_w(index)
        if value is not None:
            if phase == 0:
                phase0 += _schedule_const(value, role, byte)
            else:
                phase1 += _schedule_const(value, role, byte)
    return phase0, phase1


def _add_word_source(rom: Rom, y: int, word, aliases: Sequence[str]) -> None:
    for half, alias in zip(("lo", "hi"), aliases):
        if alias:
            rom.add(source_m(word, half), y, "source", alias)


def build_rom() -> Rom:
    layout = build_layout()
    rom = Rom()

    # Forty raw bytes; prefix x0..15 handles the eight individual bits and
    # leaves a READY-tagged byte immediately west of the target M.
    for byte in range(40):
        target = pack_target(layout, byte)
        if target is not None:
            half, shift = target
            rom.add(half.m, byte, "input_byte_target", shift)

    for t in range(64):
        for byte in range(4):
            y = p1_row(t, byte)
            if t >= 16:
                phase0, phase1 = schedule_phase_constants(t, byte)
                rom.add(0, y, "row_seed", phase0)
                r0b, r1b = schedule_reducer_boundaries(t)
                r0, r1 = layout.schedule_gap[r0b], layout.schedule_gap[r1b]

                for index, role in ((t - 16, "direct"), (t - 15, "s0")):
                    if index not in layout.w:
                        continue
                    if role == "direct":
                        half = active_half(byte, byte=True)
                        rom.add(source_m(layout.w[index], half), y, "source",
                                f"schedule.direct.byte{byte}.{half}")
                    else:
                        for half in ("lo", "hi"):
                            rom.add(source_m(layout.w[index], half), y, "source",
                                    f"schedule.byteSigma0.{byte}.{half}")
                rom.add(r0.ready, y, "ready", "SCHED_R0")
                rom.add(r0.reducer, y, "reducer", "SCHED_R0")

                for index, role in ((t - 7, "direct"), (t - 2, "s1")):
                    if index not in layout.w:
                        continue
                    if role == "direct":
                        half = active_half(byte, byte=True)
                        rom.add(source_m(layout.w[index], half), y, "source",
                                f"schedule.direct.byte{byte}.{half}")
                    else:
                        for half in ("lo", "hi"):
                            rom.add(source_m(layout.w[index], half), y, "source",
                                    f"schedule.byteSigma1.{byte}.{half}")
                rom.add(r1.ready, y, "ready", f"SCHED_R1+{phase1}")
                rom.add(r1.reducer, y, "reducer", "SCHED_R1")
                target_half = layout.w[t].lo if byte < 2 else layout.w[t].hi
                rom.add(target_half.m, y, "schedule_byte_target", byte & 1)
            else:
                rom.add(0, y, "row_seed", 0)

            # Never reset at bootstrap: for early rounds it would erase the
            # variable W0..W7 initial-state contributions already scanned.
            seed = p1_seed(t, byte)
            if t == 0:
                rom.add(layout.bootstrap[0], y, "ready", f"P1_R1+{seed}")
                reducers = layout.bootstrap[1:]
            else:
                reducers = layout.state[t - 1].scratch

                # E[t-1].hi is not a P1 source; its Q is the free READY setter.
                rom.add(layout.state[t - 1].e.hi.m, y, "memory_ready", "P1_R1")

            for role, lag in (("c", 2), ("b", 1), ("a", 0)):
                kind, index = a_source(t, lag)
                word = (layout.state[index].a if kind == "memory"
                        else layout.w[index])
                if role == "a":
                    for half in ("lo", "hi"):
                        rom.add(source_m(word, half), y, "source",
                                f"packedP1.a.byte{byte}.{half}")
                else:
                    half = active_half(byte, byte=True)
                    rom.add(source_m(word, half), y, "source",
                            f"packedP1.bc.byte{byte}.{half}")
            for x, name in zip(reducers, ("P1_R1", "P1_R2", "P1_R3")):
                rom.add(x, y, "reducer", name)
            target = layout.state[t].a.lo if byte < 2 else layout.state[t].a.hi
            rom.add(target.m, y, "p1_target", byte & 1)

        for p in range(8):
            y = p2_row(t, p)
            rom.add(0, y, "row_seed", p2_seed(t, p))
            half = active_half(p, byte=False)

            if t in layout.w:
                rom.add(source_m(layout.w[t], half), y, "source",
                        f"pass2.HW.{p}.{half}")
            kind, index = a_source(t, 3)
            word = layout.state[index].a if kind == "memory" else layout.w[index]
            rom.add(source_m(word, half), y, "source",
                    f"pass2.D.{p}.{half}")
            for role, lag, alias in (("h", 3, "HW"), ("g", 2, "Ch.g"),
                                     ("f", 1, "Ch.f")):
                kind, index = e_source(t, lag)
                word = layout.state[index].e if kind == "memory" else layout.w[index]
                rom.add(source_m(word, half), y, "source",
                        f"pass2.{alias}.{p}.{half}")
            kind, index = e_source(t, 0)
            word = layout.state[index].e if kind == "memory" else layout.w[index]
            for source_half in ("lo", "hi"):
                rom.add(source_m(word, source_half), y,
                        "source", f"pass2.e_plus_Sigma1.{p}.{source_half}")

            scratch = layout.bootstrap[1:] if t == 0 else layout.state[t - 1].scratch
            rom.add(scratch[0], y, "ready", "P2_R1")
            rom.add(scratch[1], y, "reducer", "P2_R1")
            rom.add(scratch[2], y, "reducer", "P2_R2")
            target_a = layout.state[t].a.lo if p < 4 else layout.state[t].a.hi
            target_e = layout.state[t].e.lo if p < 4 else layout.state[t].e.hi
            rom.add(target_a.m, y, "p2_a_target", p & 3)
            rom.add(target_e.m, y, "p2_e_target", p & 3)

    # In-place feed-forward.  W[i] supplies the initial-state byte; target N
    # is A63..A60,E63..E60 and therefore already supplies the final-state word.
    for word_index, target in enumerate(digest_words(layout)):
        for byte in range(4):
            y = ff_row(word_index, byte)
            rom.add(0, y, "row_seed", 0)
            half = active_half(byte, byte=True)
            rom.add(source_m(layout.w[word_index], half), y, "source",
                    f"schedule.direct.byte{byte}.{half}")
            target_half = target.lo if byte < 2 else target.hi
            rom.add(target_half.m, y, "feedforward_target", byte & 1)

    return rom


def report() -> dict[str, object]:
    rom = build_rom()
    by_kind: Dict[str, int] = {}
    by_x: Dict[int, int] = {}
    for event in rom.events.values():
        by_kind[event.kind] = by_kind.get(event.kind, 0) + 1
        by_x[event.x] = by_x.get(event.x, 0) + 1
    return {
        "events": len(rom.events),
        "kinds": dict(sorted(by_kind.items())),
        "active_x_columns": len(by_x),
        "max_events_one_x": max(by_x.values()),
        "coordinate_tree_strategy": "x first, sparse y intervals second",
        "rough_coordinate_splits_upper_bound": 2 * len(rom.events) + 2 * len(by_x),
        "display_y0": DISPLAY_Y0,
        "first_events": [asdict(e) for e in sorted(
            rom.events.values(), key=lambda e: (e.y, e.x))[:32]],
    }


def main() -> None:
    print(json.dumps(report(), indent=2))


if __name__ == "__main__":
    main()
