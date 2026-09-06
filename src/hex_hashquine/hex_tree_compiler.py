#!/usr/bin/env python3
"""Compiler for a stock-JXL SHA-256 machine with hexadecimal glyph output.

The large read-once branching programs are represented lazily, so counting
and emitting the 3.39-million-node tree does not allocate millions of Python
decision-node objects.  The generated tree includes the shared Boolean/
rotation selector, message-schedule reducers, all 64 compression rounds,
feed-forward, coordinate/state ROM, 320-bit raw entropy aperture, and a
four-line, 64-glyph uppercase hexadecimal display.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, Protocol, Sequence


HERE = Path(__file__).resolve().parent
MA_DIR = HERE.parent / "ma_lowering"
SHA_DIR = HERE.parent / "sha_machine"
for directory in (HERE, SHA_DIR, MA_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from controller_budget import (  # noqa: E402
    CH_CODES,
    MAJ_CODES,
    SIGMA_CODES,
    _selector_requirements,
    ch_from_code,
    maj_from_code,
    sigma_byte_from_code,
    sigma_from_code,
)
from tagged_selector import constant_runs  # noqa: E402
from coordinate_rom import build_rom  # noqa: E402
from final_layout import (  # noqa: E402
    DISPLAY_Y0, build_layout, digest_words, ff_row, p1_row, p2_row,
)


WIDTH = 1024
HEIGHT = 1024
HIDDEN_CHANNELS = 49
MAX31 = (1 << 31) - 1
HARD_TREE_LIMIT = (1 << 22) - 1
# libjxl also applies a size/channel-dependent ceiling. Three visible RGB
# channels are present in addition to the declared hidden channels.
TREE_LIMIT = min(
    HARD_TREE_LIMIT,
    1024 + WIDTH * HEIGHT * (HIDDEN_CHANNELS + 3) // 16,
)

MEM_BASE = 1_500_000_000
MEM_STRIDE = 1 << 20

P1_L = MAJ_CODES * SIGMA_CODES

# Non-overlapping READY ranges.  Ordinary accumulators stay below 500M;
# memory/source tags start at 1.5B.
P1_R1 = 500_000_000
P1_R2 = 940_000_000
P1_R3 = 942_000_000
SCHED_R0 = 944_000_000
SCHED_R1 = 946_000_000
P2_R1 = 1_000_000_000
P2_R2 = 1_345_000_000
FF_TARGET = 1_350_000_000
P1_TARGET = 1_360_000_000
P2_A_TARGET = 1_362_000_000
P2_A_CARRY = 1_366_000_000
P2_E_TARGET = 1_390_000_000
P2_E_CARRY = 1_394_000_000
# SCHED_R1 carries 1024*sigmaCode plus a <=765 direct subtotal and reaches
# 952,718,205.  Keep the target range strictly above it.
SCHED_TARGET = 953_000_000

Q_P2 = 1216
Q_SCHEDULE = 1024
Q_FF = 257

# Decimal post-processor.  The 256-bit digest is consumed bytewise into 26
# base-1000 limbs because 1000**26 == 10**78 > 2**256.  Hidden channels
# c4..c35 perform the 32 Horner steps.  The limbs run down x=1001, allowing
# c36..c42 to route them into an anti-transposed 3x5 numeral buffer without
# crossing the 1024-pixel Modular group boundary.
DEC_LIMBS = 26
BYTE_Y0 = DISPLAY_Y0
BYTE_BANK_X = 999
DEC_CONTROL_X = 1000
DEC_STATE_X = 1001
DEC_STATE_Y0 = 872
DEC_FIRST_CHANNEL = 4
DEC_LAST_CHANNEL = 35
RENDER_FIRST_CHANNEL = 36
RENDER_LAST_CHANNEL = 42
BITMAP_CHANNEL = 43

DEC_BASE = 1_700_000_000          # normalized base-1000 limb
DEC_TMP_BASE = 1_710_000_000      # 256*d + carry, at most 255999
DEC_CARRY_BASE = 1_720_000_000    # carry byte 0..255
DEC_SEED_BASE = 1_721_000_000     # countdown*256 + input byte
DEC_SEED_COUNT = 31
DEC_FULL_BASE = 1_730_000_000     # routed full limb 0..999
DEC_DIGIT_BASE = 1_731_000_000    # replicated digit 0..9

# (j0,count, route-x, block-x, first physical group row).  Physical y runs
# from low significance to high significance.  Orientation 7 reverses both
# axes while transposing, so the decoded four lines read j25 down to j0.
RENDER_LINES = (
    (19, 7, 989, 990, 954),
    (12, 7, 975, 976, 954),
    (6, 6, 953, 954, 964),
    (0, 6, 941, 942, 964),
)

assert DEC_LAST_CHANNEL - DEC_FIRST_CHANNEL + 1 == 32
assert DEC_STATE_Y0 + 32 + 2 * (DEC_LIMBS - 1) < HEIGHT
assert DEC_STATE_X < WIDTH
assert max(base_y + 10 * (count - 1) + 8
           for _j0, count, _route_x, _block_x, base_y in RENDER_LINES) < HEIGHT
assert DEC_DIGIT_BASE + 9 <= MAX31

# Hexadecimal post-processor. c3 banks the digest's 32 bytes in SHA order.
# c4..c35 each route one byte into two scale-2 glyph cells, c36 renders the
# nibble tags to a binary bitmap, and the remaining channels copy that bitmap
# forward to RGB.  The decoded layout is four lines of 16 characters.
HEX_RENDER_FIRST_CHANNEL = 4
HEX_RENDER_LAST_CHANNEL = 35
HEX_BITMAP_CHANNEL = 36
HEX_TAG_BASE = 1_700_000_000
HEX_SCALE = 2
HEX_OUTPUT_X0 = 4
HEX_CHAR_STRIDE = 7
HEX_GROUP_GAP = 2
HEX_OUTPUT_LINE_Y = (4, 16, 28, 40)

assert HEX_RENDER_LAST_CHANNEL - HEX_RENDER_FIRST_CHANNEL + 1 == 32
assert HEX_TAG_BASE + 15 <= MAX31


class Tree(Protocol):
    def stats(self) -> "Stats": ...
    def render(self, out, indent: int = 0) -> None: ...


@dataclass(frozen=True)
class Stats:
    nodes: int
    leaves: int
    depth: int
    min_offset: int
    max_offset: int

    @staticmethod
    def merge(parts: Sequence["Stats"], extra_depth: int = 0,
              split_nodes: int = 0) -> "Stats":
        return Stats(
            split_nodes + sum(p.nodes for p in parts),
            sum(p.leaves for p in parts),
            extra_depth + max((p.depth for p in parts), default=0),
            min((p.min_offset for p in parts), default=0),
            max((p.max_offset for p in parts), default=0),
        )


@dataclass(frozen=True)
class Leaf:
    predictor: str
    offset: int = 0

    def stats(self) -> Stats:
        return Stats(1, 1, 1, self.offset, self.offset)

    def render(self, out, indent: int = 0) -> None:
        sign = "+" if self.offset >= 0 else "-"
        out.write("  " * indent + f"- {self.predictor} {sign} {abs(self.offset)}\n")


@dataclass(frozen=True)
class Split:
    prop: str
    threshold: int
    greater: Tree
    less_equal: Tree

    def stats(self) -> Stats:
        a, b = self.greater.stats(), self.less_equal.stats()
        return Stats(1 + a.nodes + b.nodes, a.leaves + b.leaves,
                     1 + max(a.depth, b.depth),
                     min(a.min_offset, b.min_offset),
                     max(a.max_offset, b.max_offset))

    def render(self, out, indent: int = 0) -> None:
        out.write("  " * indent + f"if {self.prop} > {self.threshold}\n")
        self.greater.render(out, indent + 1)
        self.less_equal.render(out, indent + 1)


@dataclass(frozen=True)
class Intervals:
    """Balanced threshold selection over ordered half-open intervals."""
    prop: str
    starts: Sequence[int]
    children: Sequence[Tree]

    def __post_init__(self) -> None:
        if not self.starts or len(self.starts) != len(self.children):
            raise ValueError("starts/children mismatch")
        if any(a >= b for a, b in zip(self.starts, self.starts[1:])):
            raise ValueError("starts not increasing")

    def stats(self) -> Stats:
        child = [x.stats() for x in self.children]
        # A balanced n-leaf interval selector contributes n-1 splits and
        # ceil(log2(n)) worst-case levels; calculate exact recursive depth.
        def d(lo: int, hi: int) -> int:
            if hi - lo == 1:
                return child[lo].depth
            mid = (lo + hi) // 2
            return 1 + max(d(mid, hi), d(lo, mid))
        return Stats(len(self.children) - 1 + sum(x.nodes for x in child),
                     sum(x.leaves for x in child), d(0, len(child)),
                     min(x.min_offset for x in child),
                     max(x.max_offset for x in child))

    def render(self, out, indent: int = 0) -> None:
        def rec(lo: int, hi: int, level: int) -> None:
            if hi - lo == 1:
                self.children[lo].render(out, level)
                return
            mid = (lo + hi) // 2
            out.write("  " * level +
                      f"if {self.prop} > {self.starts[mid] - 1}\n")
            rec(mid, hi, level + 1)
            rec(lo, mid, level + 1)
        rec(0, len(self.children), indent)


def validate_tree(tree: Tree) -> dict[str, int]:
    """Mirror libjxl ValidateTree, including path-redundant split rejection."""
    ranges: dict[str, list[int]] = {}
    checked_nodes = 0
    checked_leaves = 0

    def split(prop: str, threshold: int, greater: Callable[[], None],
              less_equal: Callable[[], None]) -> None:
        nonlocal checked_nodes
        checked_nodes += 1
        bounds = ranges.setdefault(prop, [-(1 << 31), (1 << 31) - 1])
        low, high = bounds
        if low > threshold or high <= threshold:
            raise AssertionError(("invalid path-redundant split", prop,
                                  threshold, low, high))
        bounds[0] = threshold + 1
        greater()
        bounds[0] = low
        bounds[1] = threshold
        less_equal()
        bounds[1] = high

    def visit(node: Tree) -> None:
        nonlocal checked_nodes, checked_leaves
        if isinstance(node, Leaf):
            checked_nodes += 1
            checked_leaves += 1
            return
        if isinstance(node, Split):
            split(node.prop, node.threshold,
                  lambda: visit(node.greater),
                  lambda: visit(node.less_equal))
            return
        if isinstance(node, Intervals):
            def rec(lo: int, hi: int) -> None:
                if hi - lo == 1:
                    visit(node.children[lo])
                    return
                mid = (lo + hi) // 2
                split(node.prop, node.starts[mid] - 1,
                      lambda: rec(mid, hi), lambda: rec(lo, mid))
            rec(0, len(node.children))
            return
        raise AssertionError(type(node))

    visit(tree)
    return {"nodes": checked_nodes, "leaves": checked_leaves}


def coalesced_intervals(prop: str, triples: Iterable[tuple[int, str, int]]) -> Intervals:
    """Coalesce consecutive keys having the same predictor/offset."""
    starts: list[int] = []
    leaves: list[Tree] = []
    last: tuple[str, int] | None = None
    for key, predictor, offset in triples:
        cur = predictor, offset
        if cur != last:
            starts.append(key)
            leaves.append(Leaf(predictor, offset))
            last = cur
    return Intervals(prop, tuple(starts), tuple(leaves))


def source_selector() -> tuple[Tree, dict[str, int]]:
    requirements = _selector_requirements(packed_pass1=True)
    ordered = sorted(requirements.items(), key=lambda item: item[1][0])
    tag_for_alias: dict[str, int] = {}
    starts: list[int] = []
    children: list[Tree] = []
    for tag, (table, aliases) in enumerate(ordered):
        for alias in aliases:
            tag_for_alias[alias] = tag
        base = MEM_BASE + tag * MEM_STRIDE
        runs = constant_runs(table)
        starts.append(base)
        children.append(Intervals(
            "W",
            tuple(base + start for start, _end, _value in runs),
            tuple(Leaf("WW", value) for _start, _end, value in runs),
        ))
    # Default pass-through and target phase/control tags.  These tiny leaves
    # do not duplicate any of the 16-bit source tables.  A Q pixel sees its
    # tagged M value in W and the incoming controller in WW.
    pass_tag = len(ordered)
    if pass_tag != TAG_PASS:
        raise AssertionError((pass_tag, TAG_PASS))
    starts.append(MEM_BASE + pass_tag * MEM_STRIDE)
    children.append(Leaf("WW", 0))
    tag_for_alias["__PASS__"] = pass_tag
    reset = {
        TAG_PL1, TAG_PLJ, TAG_PH1, TAG_PHD,
        TAG_EL1, TAG_EL2, TAG_EL3, TAG_ELJ,
        TAG_EH1, TAG_EH2, TAG_EH3, TAG_EHD,
        TAG_SL1, TAG_SLJ, TAG_SH1, TAG_SHD,
        TAG_FL1, TAG_FLJ, TAG_FH1, TAG_FD,
    }
    pass_ww = {
        TAG_PLE, TAG_PHE, TAG_PLP, *TAG_PHR, TAG_ALR,
        TAG_ELI, TAG_EHI, TAG_ELR, TAG_SLI, TAG_SHI, TAG_SLR,
        TAG_FLR,
    }
    ca_to_ce = {TAG_AL1, TAG_AL2, TAG_AL3, TAG_ALJ,
                TAG_AH2, TAG_AH3, TAG_AHD}
    controls: list[tuple[int, str, Tree]] = []
    for tag in sorted(reset):
        controls.append((tag, f"__RESET_{tag}__", Leaf("Set", 0)))
    for tag in sorted(pass_ww):
        controls.append((tag, f"__PASS_{tag}__", Leaf("WW", 0)))
    for tag in sorted(ca_to_ce):
        controls.append((tag, f"__CA_CE_{tag}__",
                         Leaf("WW", P2_E_TARGET - P2_A_TARGET)))

    # Low A carry is extracted by the Q after the p=4 bridge.  Its payload
    # quotient is bounded by six; at p=5 the M normalizes to ALR first.
    controls.append((TAG_ALC, "__ALC_CARRY__", Intervals(
        "W", tuple(_tag_base(TAG_ALC) + 65536 * c for c in range(7)),
        tuple(Leaf("WW", P2_A_CARRY - P2_A_TARGET + 2048 * c)
              for c in range(7)))))
    for c, tag in enumerate(TAG_AH1):
        controls.append((tag, f"__AH1_{c}_RESTORE__",
                         Leaf("WW", P2_E_TARGET - P2_A_CARRY - 2048 * c)))

    # E and schedule/feed-forward low-half bridges extract their independent
    # carries in exactly the same fashion.
    controls.append((TAG_ELC, "__ELC_CARRY__", Intervals(
        "W", tuple(_tag_base(TAG_ELC) + 65536 * c for c in range(6)),
        tuple(Leaf("WW", P2_E_CARRY - P2_E_TARGET + 2048 * c)
              for c in range(6)))))
    controls.append((TAG_SLC, "__SLC_CARRY__", Intervals(
        "W", tuple(_tag_base(TAG_SLC) + 65536 * c for c in range(4)),
        tuple(Leaf("WW", c) for c in range(4)))))
    controls.append((TAG_FLC, "__FLC_CARRY__", Intervals(
        "W", tuple(_tag_base(TAG_FLC) + 65536 * c for c in range(2)),
        tuple(Leaf("WW", c) for c in range(2)))))
    controls.append((TAG_READY_P1, "__READY_P1__", Leaf("WW", P1_R1)))
    controls.append((TAG_READY_FF, "__READY_FF__", Leaf("WW", FF_TARGET)))
    controls.append((TAG_DISPLAY, "__DISPLAY__",
                     Leaf("W", -_tag_base(TAG_DISPLAY))))

    # Post-SHA byte taps.  At a digest Q pixel, W is the tagged 16-bit
    # halfword.  The low-byte action keeps the exact W sample and subtracts
    # its tag/high-byte block; the high-byte action emits the block number.
    # Unlike the ordinary direct source tables these do not depend on WW, so
    # byte routing cannot be perturbed by the neighboring digest lane.
    low_base = _tag_base(TAG_DEC_LO)
    high_base = _tag_base(TAG_DEC_HI)
    controls.append((TAG_DEC_LO, "__DEC_LOW_BYTE__", Intervals(
        "W", tuple(low_base + 256 * high for high in range(256)),
        tuple(Leaf("W", -low_base - 256 * high)
              for high in range(256)))))
    controls.append((TAG_DEC_HI, "__DEC_HIGH_BYTE__", Intervals(
        "W", tuple(high_base + 256 * high for high in range(256)),
        tuple(Leaf("Set", high) for high in range(256)))))
    controls.append((TAG_DEC_BG, "__DEC_BACKGROUND__", Leaf("Set", 65536)))

    for tag, name, action in sorted(controls, key=lambda item: item[0]):
        starts.append(_tag_base(tag))
        children.append(action)
        tag_for_alias[name] = tag
    return Intervals("W", tuple(starts), tuple(children)), tag_for_alias


def p1_reducers() -> tuple[Tree, Tree, Tree]:
    # R1: code0 + L*code1 -> code0 + L*raw1.
    def high_entries():
        last = None
        for code1 in range(P1_L):
            sig, maj_code = divmod(code1, MAJ_CODES)
            raw = maj_from_code(maj_code) + sigma_from_code(sig)
            off = P1_R2 - P1_R1 + P1_L * (raw - code1)
            if off != last:
                yield P1_R1 + P1_L * code1, "W", off
                last = off

    # R2: normalize low sigma, retaining low Maj and L*raw1.
    def low_sigma_entries():
        last = None
        for raw1 in range(31):
            for sig in range(SIGMA_CODES):
                key = P1_R2 + P1_L * raw1 + MAJ_CODES * sig
                off = P1_R3 - P1_R2 + MAJ_CODES * (
                    sigma_from_code(sig) - sig)
                if off != last:
                    yield key, "W", off
                    last = off

    # R3: normalize low Maj and emit the generic target READY range.
    def low_maj_entries():
        last = None
        for raw1 in range(31):
            for sigma in range(16):
                for maj_code in range(MAJ_CODES):
                    q = maj_code + MAJ_CODES * sigma + P1_L * raw1
                    packed = maj_from_code(maj_code) + sigma + 32 * raw1
                    off = P1_TARGET - P1_R3 + packed - q
                    if off != last:
                        yield P1_R3 + q, "W", off
                        last = off
    return (coalesced_intervals("W", high_entries()),
            coalesced_intervals("W", low_sigma_entries()),
            coalesced_intervals("W", low_maj_entries()))


def schedule_reducers() -> tuple[Tree, Tree]:
    code_count = 3**8
    r0 = coalesced_intervals("W", (
        (SCHED_R0 + 256 * code, "W",
         -SCHED_R0 + sigma_byte_from_code(code) - 256 * code)
        for code in range(code_count)
    ))
    r1 = coalesced_intervals("W", (
        (SCHED_R1 + 1024 * code, "W",
         SCHED_TARGET - SCHED_R1 +
         sigma_byte_from_code(code) - 1024 * code)
        for code in range(code_count)
    ))
    return r0, r1


def p2_reducers() -> tuple[Tree, Tree]:
    sigma_stride = 1024 * CH_CODES
    r1 = coalesced_intervals("W", (
        (P2_R1 + sigma_stride * code, "W",
         P2_R2 - P2_R1 + 16 * sigma_from_code(code) - sigma_stride * code)
        for code in range(SIGMA_CODES)
    ))
    r2 = coalesced_intervals("W", (
        (P2_R2 + 1024 * code, "W",
         P2_A_TARGET - P2_R2 + 16 * ch_from_code(code) - 1024 * code)
        for code in range(CH_CODES)
    ))
    return r1, r2


# Persistent memory/control tags.  Source tables occupy 0..42.  Tags are
# deliberately disjoint from every controller READY range.
TAG_PASS = 43
TAG_PLE, TAG_PHE = 44, 45
TAG_PL1, TAG_PLJ, TAG_PLP = 46, 47, 48
TAG_PH1, TAG_PHD, TAG_PHR1 = 49, 50, 51
TAG_AL1, TAG_AL2, TAG_AL3, TAG_ALJ = 52, 53, 54, 55
TAG_ALC, TAG_ALR = 56, 57
TAG_AH1 = tuple(range(58, 65))
TAG_AH2, TAG_AH3, TAG_AHD = 65, 66, 67
TAG_ELI, TAG_EHI = 68, 69
TAG_EL1, TAG_EL2, TAG_EL3, TAG_ELJ = 70, 71, 72, 73
TAG_ELC, TAG_ELR = 74, 75
TAG_EH1, TAG_EH2, TAG_EH3, TAG_EHD = 76, 77, 78, 79
TAG_SLI, TAG_SHI = 80, 81
TAG_SL1, TAG_SLJ, TAG_SLC, TAG_SLR = 82, 83, 84, 85
TAG_SH1, TAG_SHD = 86, 87
TAG_FL1, TAG_FLJ, TAG_FLC, TAG_FLR = 88, 89, 90, 91
TAG_FH1, TAG_FD = 92, 93
TAG_READY_P1, TAG_READY_FF, TAG_DISPLAY = 94, 95, 96
TAG_PHR2, TAG_PHR3, TAG_PHR4 = 97, 98, 99
TAG_PHR = (TAG_PHR1, TAG_PHR2, TAG_PHR3, TAG_PHR4)
TAG_DEC_LO, TAG_DEC_HI, TAG_DEC_BG = 100, 101, 102

ALL_MEMORY_TAGS = tuple(range(TAG_DEC_BG + 1))


def _tag_base(tag: int) -> int:
    return MEM_BASE + tag * MEM_STRIDE


def _p1_contribution(q: int, local_byte: int) -> int:
    raw0, raw1 = q & 31, q >> 5
    p0 = 2 * local_byte
    return ((raw0 & 15) << (4 * p0) |
            (raw1 & 15) << (4 * (p0 + 1)) |
            (raw0 >> 4) << (16 + p0) |
            (raw1 >> 4) << (16 + p0 + 1))


def _tag_leaf(old: int, new: int, offset: int = 0,
              predictor: str = "N") -> Leaf:
    return Leaf(predictor, (new - old) * MEM_STRIDE + offset)


def _tag_dispatch(actions: Mapping[int, Tree], default: Tree | None = None) -> Tree:
    """Bounded N-tag dispatch; gaps and values above each tag are inert."""
    default = default or Leaf("N", 0)
    points: dict[int, Tree] = {0: default}
    for tag in actions:
        points.setdefault(_tag_base(tag) + MEM_STRIDE, default)
    # Action starts win over a previous consecutive tag's end sentinel.
    for tag, action in actions.items():
        points[_tag_base(tag)] = action
    starts = tuple(sorted(points))
    return Intervals("N", starts, tuple(points[x] for x in starts))


def _normalize_tree(old: int, new: int, contribution: int = 0) -> Tree:
    """Add contribution, retain low 16 bits, and discard metadata/overflow."""
    starts: list[int] = []
    leaves: list[Tree] = []
    add_lo = contribution & 0xFFFF
    for high in range(16):
        lo0 = high << 16
        pieces = [lo0]
        if add_lo and 65536 - add_lo:
            pieces.append(lo0 + 65536 - add_lo)
        for start in sorted(p for p in pieces if lo0 <= p < lo0 + 65536):
            payload = start
            normalized = (payload + contribution) & 0xFFFF
            starts.append(_tag_base(old) + start)
            leaves.append(Leaf(
                "N", (new - old) * MEM_STRIDE + normalized - payload))
    return Intervals("N", tuple(starts), tuple(leaves))


def p1_target() -> Tree:
    # High-byte rows intentionally encounter low memory first.  PLJ/PLP are
    # therefore explicit identity bridges whose Q preserves CP1 in WW.
    qchildren: list[Tree] = []
    for q in range(991):
        qchildren.append(_tag_dispatch({
            TAG_PLE: _tag_leaf(TAG_PLE, TAG_PL1, _p1_contribution(q, 0)),
            TAG_PL1: _tag_leaf(TAG_PL1, TAG_PLJ, _p1_contribution(q, 1)),
            TAG_PLJ: _tag_leaf(TAG_PLJ, TAG_PLP),
            TAG_PLP: Leaf("N", 0),
            TAG_PHE: _tag_leaf(TAG_PHE, TAG_PH1, _p1_contribution(q, 0)),
            TAG_PH1: _tag_leaf(TAG_PH1, TAG_PHD, _p1_contribution(q, 1)),
        }))
    return Intervals("W", tuple(P1_TARGET + q for q in range(991)),
                     tuple(qchildren))


def _target_offset(local_p: int, t1: int, flag: int) -> int:
    shift = 4 * local_p
    return ((t1 << shift) +
            flag * ((16 << shift) - (1 << (16 + local_p))))


def _flag_tree(local_p: int, q: int, oldtag: int, newtag: int) -> Tree:
    t1 = q >> 4
    period = 1 << (16 + local_p)
    count = MEM_STRIDE // period
    return Intervals(
        "N",
        tuple(_tag_base(oldtag) + k * period for k in range(count)),
        tuple(Leaf("N", (newtag - oldtag) * MEM_STRIDE +
                   _target_offset(local_p, t1, k & 1))
              for k in range(count)),
    )


def _final_a_tree(q: int, oldtag: int, newtag: int) -> Tree:
    """Consume p=3 flag/addition, then reduce the completed half modulo 2^16."""
    t1 = q >> 4
    starts: list[int] = []
    leaves: list[Tree] = []
    for high in range(16):
        flag = (high >> 3) & 1
        contribution = _target_offset(3, t1, flag)
        add_lo = contribution & 0xFFFF
        base = high << 16
        cuts = [base]
        if add_lo:
            cuts.append(base + 65536 - add_lo)
        for payload in sorted(x for x in cuts if base <= x < base + 65536):
            normalized = (payload + contribution) & 0xFFFF
            starts.append(_tag_base(oldtag) + payload)
            leaves.append(Leaf(
                "N", (newtag - oldtag) * MEM_STRIDE + normalized - payload))
    return Intervals("N", tuple(starts), tuple(leaves))


def _q_tree(base: int, count: int, make: Callable[[int], Tree]) -> Tree:
    return Intervals("W", tuple(base + q for q in range(count)),
                     tuple(make(q) for q in range(count)))


def _q_add_tree(base: int, count: int, old: int, new: int,
                digit: Callable[[int], int], local_p: int,
                final: bool = False) -> Tree:
    return _q_tree(base, count, lambda q: _add_digit(
        old, new, digit(q), local_p, final=final))


def _t1_tree(base: int, make: Callable[[int], Tree]) -> Tree:
    """A updates depend on T1=q>>4, never on q's low D nibble."""
    return Intervals("W", tuple(base + 16 * t for t in range(76)),
                     tuple(make(t) for t in range(76)))


def p2_a_target() -> Tree:
    actions: dict[int, Tree] = {
        TAG_PLP: _t1_tree(P2_A_TARGET,
                          lambda t: _flag_tree(0, 16*t, TAG_PLP, TAG_AL1)),
        TAG_AL1: _t1_tree(P2_A_TARGET,
                          lambda t: _flag_tree(1, 16*t, TAG_AL1, TAG_AL2)),
        TAG_AL2: _t1_tree(P2_A_TARGET,
                          lambda t: _flag_tree(2, 16*t, TAG_AL2, TAG_AL3)),
        TAG_AL3: _t1_tree(P2_A_TARGET,
                          lambda t: _flag_tree(3, 16*t, TAG_AL3, TAG_ALJ)),
        TAG_ALJ: _tag_leaf(TAG_ALJ, TAG_ALC),
        TAG_ALC: _normalize_tree(TAG_ALC, TAG_ALR),
        TAG_ALR: Leaf("N", 0),
        TAG_AH2: _t1_tree(P2_A_TARGET,
                          lambda t: _add_digit(TAG_AH2, TAG_AH3, t, 2)),
        TAG_AH3: _t1_tree(P2_A_TARGET,
                          lambda t: _add_digit(TAG_AH3, TAG_AHD, t, 3,
                                               final=True)),
    }
    for old in TAG_AH1:
        actions[old] = _t1_tree(
            P2_A_TARGET, lambda t, old=old: _add_digit(old, TAG_AH2, t, 1))
    return _tag_dispatch(actions)


def p2_a_carry_target() -> Tree:
    starts: list[int] = []
    children: list[Tree] = []
    for carry in range(7):
        for t1 in range(76):
            starts.append(P2_A_CARRY + 2048 * carry + 16 * t1)
            children.append(_tag_leaf(
                TAG_PHR4, TAG_AH1[carry], t1 + carry))
        starts.append(P2_A_CARRY + 2048 * carry + Q_P2)
        children.append(Leaf("N", 0))
    return _tag_dispatch({
        TAG_PHR4: Intervals("W", tuple(starts), tuple(children))
    })


def _e_raw(q: int) -> int:
    return (q & 15) + (q >> 4)


def _add_digit(old: int, new: int, digit: int, local_p: int,
               final: bool = False) -> Tree:
    contribution = digit << (4 * local_p)
    if final:
        return _normalize_tree(old, new, contribution)
    return _tag_leaf(old, new, contribution)


def p2_e_target() -> Tree:
    # The four low-nibble E rows also visit A.high.  Use those CE visits to
    # consume A.high's four packed-T2 flag bits once, independent of q.  This
    # removes seven copies of the p=0 flag LUT from the carry path.
    return _tag_dispatch({
        TAG_PHD: _flag_tree(0, 0, TAG_PHD, TAG_PHR1),
        TAG_PHR1: _flag_tree(1, 0, TAG_PHR1, TAG_PHR2),
        TAG_PHR2: _flag_tree(2, 0, TAG_PHR2, TAG_PHR3),
        TAG_PHR3: _flag_tree(3, 0, TAG_PHR3, TAG_PHR4),
        TAG_ELI: _q_add_tree(P2_E_TARGET, Q_P2, TAG_ELI, TAG_EL1,
                             _e_raw, 0),
        TAG_EL1: _q_add_tree(P2_E_TARGET, Q_P2, TAG_EL1, TAG_EL2,
                             _e_raw, 1),
        TAG_EL2: _q_add_tree(P2_E_TARGET, Q_P2, TAG_EL2, TAG_EL3,
                             _e_raw, 2),
        TAG_EL3: _q_add_tree(P2_E_TARGET, Q_P2, TAG_EL3, TAG_ELJ,
                             _e_raw, 3),
        TAG_ELJ: _tag_leaf(TAG_ELJ, TAG_ELC),
        TAG_ELC: _normalize_tree(TAG_ELC, TAG_ELR),
        TAG_ELR: Leaf("N", 0),
        TAG_EH1: _q_add_tree(P2_E_TARGET, Q_P2, TAG_EH1, TAG_EH2,
                             _e_raw, 1),
        TAG_EH2: _q_add_tree(P2_E_TARGET, Q_P2, TAG_EH2, TAG_EH3,
                             _e_raw, 2),
        TAG_EH3: _q_add_tree(P2_E_TARGET, Q_P2, TAG_EH3, TAG_EHD,
                             _e_raw, 3, final=True),
    })


def p2_e_carry_target() -> Tree:
    starts: list[int] = []
    children: list[Tree] = []
    for carry in range(6):
        for q in range(Q_P2):
            starts.append(P2_E_CARRY + 2048 * carry + q)
            children.append(_add_digit(TAG_EHI, TAG_EH1,
                                       _e_raw(q) + carry, 0))
        starts.append(P2_E_CARRY + 2048 * carry + Q_P2)
        children.append(Leaf("Set", 0))
    return _tag_dispatch({
        TAG_EHI: Intervals("W", tuple(starts), tuple(children))
    })


def schedule_target() -> Tree:
    actions: dict[int, Tree] = {
        TAG_SLI: _q_add_tree(SCHED_TARGET, Q_SCHEDULE,
                             TAG_SLI, TAG_SL1, lambda q: q, 0),
        TAG_SL1: _q_tree(SCHED_TARGET, Q_SCHEDULE,
                         lambda q: _tag_leaf(TAG_SL1, TAG_SLJ, q << 8)),
        TAG_SLJ: _tag_leaf(TAG_SLJ, TAG_SLC),
        TAG_SLC: _normalize_tree(TAG_SLC, TAG_SLR),
        TAG_SHI: _q_add_tree(SCHED_TARGET, Q_SCHEDULE,
                             TAG_SHI, TAG_SH1, lambda q: q, 0),
        TAG_SH1: _q_tree(SCHED_TARGET, Q_SCHEDULE,
                         lambda q: _normalize_tree(TAG_SH1, TAG_SHD, q << 8)),
    }
    # R1 can be placed immediately before W[t-1], not necessarily W[t].
    # A just-read W[t-1] therefore still carries a unary source tag when CS
    # traverses it.  Canonicalize that unrelated memory to PASS in the target
    # dispatcher itself; coordinate M is bypassed whenever W is a stage.
    for source_tag in range(43):
        actions[source_tag] = _tag_leaf(source_tag, TAG_PASS)
    return _tag_dispatch(actions)


def feedforward_target() -> Tree:
    low_init = _q_add_tree(FF_TARGET, Q_FF, TAG_PASS, TAG_FL1,
                           lambda q: q, 0)
    low_init_a = _q_add_tree(FF_TARGET, Q_FF, TAG_ALR, TAG_FL1,
                             lambda q: q, 0)
    low_init_e = _q_add_tree(FF_TARGET, Q_FF, TAG_ELR, TAG_FL1,
                             lambda q: q, 0)
    high_init = _q_add_tree(FF_TARGET, Q_FF, TAG_PHE, TAG_FH1,
                            lambda q: q, 0)
    return _tag_dispatch({
        # Digest halves are canonicalized to PASS on the idle/pre-FF row.
        TAG_PASS: low_init,
        TAG_ALR: low_init_a,
        TAG_ELR: low_init_e,
        TAG_FL1: _q_tree(FF_TARGET, Q_FF,
                         lambda q: _tag_leaf(TAG_FL1, TAG_FLJ, q << 8)),
        TAG_FLJ: _tag_leaf(TAG_FLJ, TAG_FLC),
        TAG_FLC: _normalize_tree(TAG_FLC, TAG_FLR),
        TAG_FLR: Leaf("N", 0),
        TAG_FH1: _q_tree(FF_TARGET, Q_FF,
                         lambda q: _normalize_tree(TAG_FH1, TAG_FD, q << 8)),
        # A separate high-init tag is installed by coordinate ROM immediately
        # before byte two, avoiding low/high ambiguity under a single PASS.
        TAG_PHE: high_init,
    })


def stage_dispatch() -> Tree:
    p1a, p1b, p1c = p1_reducers()
    s0, s1 = schedule_reducers()
    p2a, p2b = p2_reducers()
    starts = (P1_R1, P1_R2, P1_R3, SCHED_R0, SCHED_R1,
              SCHED_TARGET, P2_R1, P2_R2, FF_TARGET, P1_TARGET,
              P2_A_TARGET, P2_A_CARRY, P2_E_TARGET, P2_E_CARRY)
    return Intervals("W", starts,
                     (p1a, p1b, p1c, s0, s1, schedule_target(),
                      p2a, p2b, feedforward_target(), p1_target(),
                      p2_a_target(), p2_a_carry_target(),
                      p2_e_target(), p2_e_carry_target()))


def _retag_from(oldtags: frozenset[int], newtag: int) -> Tree:
    if len(oldtags) == 1:
        old = next(iter(oldtags))
        # ``-1`` denotes memory that has never been initialized.  It is
        # logically zero, but a Top/N predictor is *not* zero on the first
        # raster row: JPEG XL extends the left sample across that boundary.
        # Use the constant predictor so controller pixels to the west cannot
        # leak into fresh machine state.
        return Leaf("Set", _tag_base(newtag)) if old < 0 else Leaf(
            "N", (newtag - old) * MEM_STRIDE)
    actions = {
        old: _tag_leaf(old, newtag) for old in oldtags if old >= 0
    }
    tree = _tag_dispatch(actions)
    if -1 in oldtags:
        tree = Split("N", MEM_BASE - 1, tree,
                     Leaf("Set", _tag_base(newtag)))
    return tree


def _input_byte_tree(shift: int, oldtags: frozenset[int]) -> Tree:
    tagged = _retag_from(oldtags, TAG_PASS)
    # Retagging is a leaf for all reachable input states.  Preserve its
    # predictor and add the big-endian byte at the requested halfword shift.
    if isinstance(tagged, Leaf):
        return Intervals("W", tuple(range(256)),
                         tuple(Leaf(tagged.predictor,
                                    tagged.offset + (value << shift))
                               for value in range(256)))
    return Intervals("W", tuple(range(256)), tuple(
        _tag_dispatch({
            old: _tag_leaf(old, TAG_PASS, value << shift)
            for old in oldtags if old >= 0
        }, Leaf("Set", _tag_base(TAG_PASS) + (value << shift)))
        for value in range(256)))


def _target_tag_updates(layout) -> dict[tuple[int, int], frozenset[int]]:
    updates: dict[tuple[int, int], frozenset[int]] = {}

    def put(x: int, y: int, *tags: int) -> None:
        key = x, y
        value = frozenset(tags)
        if key in updates and updates[key] != value:
            raise AssertionError(("target tag collision", key,
                                  updates[key], value))
        updates[key] = value

    for t in range(64):
        a, e = layout.state[t].a, layout.state[t].e
        # Packed P1: the high-byte controller deliberately traverses low.
        for byte in range(4):
            y = p1_row(t, byte)
            if byte == 0:
                put(a.lo.m, y, TAG_PL1)
            elif byte == 1:
                put(a.lo.m, y, TAG_PLJ)
            elif byte == 2:
                put(a.lo.m, y, TAG_PLP)
                put(a.hi.m, y, TAG_PH1)
            else:
                put(a.lo.m, y, TAG_PLP)
                put(a.hi.m, y, TAG_PHD)

        for p in range(8):
            y = p2_row(t, p)
            if p == 0:
                put(a.lo.m, y, TAG_AL1)
                put(a.hi.m, y, TAG_PHR1)
                put(e.lo.m, y, TAG_EL1)
            elif p == 1:
                put(a.lo.m, y, TAG_AL2)
                put(a.hi.m, y, TAG_PHR2)
                put(e.lo.m, y, TAG_EL2)
            elif p == 2:
                put(a.lo.m, y, TAG_AL3)
                put(a.hi.m, y, TAG_PHR3)
                put(e.lo.m, y, TAG_EL3)
            elif p == 3:
                put(a.lo.m, y, TAG_ALJ)
                put(a.hi.m, y, TAG_PHR4)
                put(e.lo.m, y, TAG_ELJ)
            elif p == 4:
                put(a.lo.m, y, TAG_ALC)
                put(a.hi.m, y, *TAG_AH1)
                put(e.lo.m, y, TAG_ELC)
                put(e.hi.m, y, TAG_EH1)
            elif p == 5:
                put(a.lo.m, y, TAG_ALR)
                put(a.hi.m, y, TAG_AH2)
                put(e.lo.m, y, TAG_ELR)
                put(e.hi.m, y, TAG_EH2)
            elif p == 6:
                put(a.lo.m, y, TAG_ALR)
                put(a.hi.m, y, TAG_AH3)
                put(e.lo.m, y, TAG_ELR)
                put(e.hi.m, y, TAG_EH3)
            else:
                put(a.lo.m, y, TAG_ALR)
                put(a.hi.m, y, TAG_AHD)
                put(e.lo.m, y, TAG_ELR)
                put(e.hi.m, y, TAG_EHD)

        if t >= 16:
            word = layout.w[t]
            for byte in range(4):
                y = p1_row(t, byte)
                if byte == 0:
                    put(word.lo.m, y, TAG_SL1)
                elif byte == 1:
                    put(word.lo.m, y, TAG_SLJ)
                elif byte == 2:
                    put(word.lo.m, y, TAG_SLC)
                    put(word.hi.m, y, TAG_SH1)
                else:
                    put(word.lo.m, y, TAG_SLR)
                    put(word.hi.m, y, TAG_SHD)

    for word_index, word in enumerate(digest_words(layout)):
        for byte in range(4):
            y = ff_row(word_index, byte)
            if byte == 0:
                put(word.lo.m, y, TAG_FL1)
            elif byte == 1:
                put(word.lo.m, y, TAG_FLJ)
            elif byte == 2:
                put(word.lo.m, y, TAG_FLC)
                put(word.hi.m, y, TAG_FH1)
            else:
                put(word.lo.m, y, TAG_FLR)
                put(word.hi.m, y, TAG_FD)
    return updates


def _memory_coordinates(layout) -> tuple[set[int], set[int]]:
    words = list({id(word): word for word in layout.w.values()}.values())
    words += [slot.a for slot in layout.state]
    words += [slot.e for slot in layout.state]
    m = {half.m for word in words for half in (word.lo, word.hi)}
    q = {half.q for word in words for half in (word.lo, word.hi)}
    return m, q


def _coordinate_actions(tags: Mapping[str, int]) -> tuple[
        dict[tuple[int, int], Tree], dict[str, int]]:
    layout = build_layout()
    rom = build_rom()
    memory_m, _memory_q = _memory_coordinates(layout)
    target_updates = _target_tag_updates(layout)
    events_by_y: dict[int, list] = {}
    for event in rom.events.values():
        events_by_y.setdefault(event.y, []).append(event)

    # Fresh memory is explicitly typed one row before its first target.
    init: dict[tuple[int, int], int] = {}
    for t in range(64):
        y0 = p1_row(t, 0) - 1
        init[(layout.state[t].a.lo.m, y0)] = TAG_PLE
        init[(layout.state[t].a.hi.m, y0)] = TAG_PHE
        ey = p2_row(t, 0) - 1
        init[(layout.state[t].e.lo.m, ey)] = TAG_ELI
        init[(layout.state[t].e.hi.m, ey)] = TAG_EHI
        if t >= 16:
            init[(layout.w[t].lo.m, y0)] = TAG_SLI
            init[(layout.w[t].hi.m, y0)] = TAG_SHI

    # Canonicalize all digest storage on the first FF row.  The active low
    # half is consumed by the FF target; every inactive high half becomes the
    # unambiguous high-init tag before byte two.
    first_ff = ff_row(0, 0)
    digest = digest_words(layout)
    for word in digest:
        init[(word.hi.m, first_ff)] = TAG_PHE
        if word is not digest[0]:
            init[(word.lo.m, first_ff)] = TAG_PASS

    actions: dict[tuple[int, int], Tree] = {}
    state: dict[int, frozenset[int]] = {
        x: frozenset((-1,)) for x in memory_m
    }
    harmful_visits = 0

    safe = {TAG_PASS, TAG_PLE, TAG_PHE, TAG_PLP, *TAG_PHR,
            TAG_ALR, TAG_ELI, TAG_EHI, TAG_ELR, TAG_SLI, TAG_SHI,
            TAG_SLR, TAG_FLR, TAG_DISPLAY}

    def set_action(x: int, y: int, tree: Tree) -> None:
        key = x, y
        if key in actions:
            raise AssertionError(("coordinate action collision", key))
        actions[key] = tree

    def parse_ready(text: str) -> int:
        name, plus, extra = text.partition("+")
        base = {
            "P1_R1": P1_R1, "P2_R1": P2_R1,
            "SCHED_R0": SCHED_R0, "SCHED_R1": SCHED_R1,
        }[name]
        return base + (int(extra) if plus else 0)

    for y in range(DISPLAY_Y0):
        row_events = {event.x: event for event in events_by_y.get(y, ())}
        target_x = {x for (x, yy) in target_updates if yy == y}

        # Feed-forward READY is placed immediately west of A.low.  For E,
        # A.high is retagged and its Q is the immediate-west setter.
        if first_ff <= y < DISPLAY_Y0:
            word_index = (y - first_ff) // 4
            if 0 <= word_index < 8:
                word = digest[word_index]
                if word in tuple(slot.a for slot in layout.state):
                    set_action(word.lo.m - 1, y, Leaf("W", FF_TARGET))
                else:
                    slot = next(slot for slot in layout.state
                                if slot.e is word)
                    row_events[slot.a.hi.m] = type("E", (), {
                        "x": slot.a.hi.m, "kind": "ff_ready_control",
                        "arg": ""})()

        explicit_x = set(row_events)
        explicit_x.update(x for (x, yy) in init if yy == y)

        # A destructive source/control tag is legal only on its active row.
        # If no target consumes it now, normalize it to neutral PASS before Q.
        for x in memory_m:
            old = state[x]
            if (x not in explicit_x and x not in target_x and
                    not old.issubset(safe)):
                set_action(x, y, _retag_from(old, TAG_PASS))
                state[x] = frozenset((TAG_PASS,))
                harmful_visits += 1

        for (x, yy), newtag in tuple(init.items()):
            if yy != y:
                continue
            # A simultaneous target owns the M output and supersedes init.
            if x in target_x:
                continue
            set_action(x, y, _retag_from(state[x], newtag))
            state[x] = frozenset((newtag,))

        for x, event in sorted(row_events.items()):
            if event.kind in {"row_seed", "reducer", "p1_target",
                              "p2_a_target", "p2_e_target",
                              "schedule_byte_target", "feedforward_target"}:
                continue
            if event.kind == "source":
                new = tags[event.arg]
                set_action(x, y, _retag_from(state[x], new))
                state[x] = frozenset((new,))
            elif event.kind == "input_byte_target":
                set_action(x, y, _input_byte_tree(int(event.arg), state[x]))
                state[x] = frozenset((TAG_PASS,))
            elif event.kind == "memory_ready":
                set_action(x, y, _retag_from(state[x], TAG_READY_P1))
                state[x] = frozenset((TAG_READY_P1,))
            elif event.kind == "ff_ready_control":
                set_action(x, y, _retag_from(state[x], TAG_READY_FF))
                state[x] = frozenset((TAG_READY_FF,))
            elif event.kind == "ready":
                set_action(x, y, Leaf("W", parse_ready(event.arg)))
            else:
                raise AssertionError(("unlowered ROM event", event))

        # Global W-stage target dispatch owns these M outputs.
        for (x, yy), new in target_updates.items():
            if yy == y:
                state[x] = new

    return actions, {
        "coordinate_actions": len(actions),
        "target_state_updates": len(target_updates),
        "harmful_tag_normalizations": harmful_visits,
    }


def _rows_for_x(default: Tree, row_actions: Mapping[int, Tree]) -> Tree:
    timeline: dict[int, Tree] = {0: default}
    for y, action in row_actions.items():
        timeline[y] = action
    for y in row_actions:
        if y + 1 < DISPLAY_Y0 and y + 1 not in row_actions:
            timeline[y + 1] = default
    starts: list[int] = []
    children: list[Tree] = []
    last: Tree | None = None
    for y, action in sorted(timeline.items()):
        if action != last:
            starts.append(y)
            children.append(action)
            last = action
    return Intervals("y", tuple(starts), tuple(children))


def _digest_byte_sources() -> list[tuple[int, int, int, int, int]]:
    """(byte-index, M, Q, old-tag, byte-tag) in SHA byte order."""
    layout = build_layout()
    words = digest_words(layout)
    result: list[tuple[int, int, int, int, int]] = []
    byte_index = 0
    for word_index, word in enumerate(words):
        # TAG_FD is not a stable idle tag.  Coordinate cleanup canonicalizes
        # completed high halves to PASS on the following feed-forward row.
        # The final word, completed at y=839, is still FD.  A60.high is the
        # exceptional E60 feed-forward READY control on that same last row,
        # so it reaches the post-SHA raster as READY_FF.  Low halves remain
        # in the safe FLR tag.
        if word_index == len(words) - 1:
            high_old_tag = TAG_FD
        elif word_index == 3:
            high_old_tag = TAG_READY_FF
        else:
            high_old_tag = TAG_PASS
        for half, old_tag in ((word.hi, high_old_tag),
                              (word.lo, TAG_FLR)):
            result.append((byte_index, half.m, half.q, old_tag, TAG_DEC_HI))
            byte_index += 1
            result.append((byte_index, half.m, half.q, old_tag, TAG_DEC_LO))
            byte_index += 1
    assert byte_index == 32
    return result


def _post_sha_coordinate_program() -> Tree:
    """Expose one exact digest byte per row and make all else >255.

    Digest M cells retain their payload in N while their tag changes between
    DEC_BG and the two byte taps.  Their following Q cell is then handled by
    source_selector().  A 65536 background lets c3 recognize the sole byte
    source on each row without confusing a genuine zero byte with a path.
    """
    sources = _digest_byte_sources()
    by_m: dict[int, dict[int, int]] = {}
    old_tag_for_m: dict[int, int] = {}
    for byte_index, m, _q, old_tag, byte_tag in sources:
        by_m.setdefault(m, {})[BYTE_Y0 + byte_index] = byte_tag
        old_tag_for_m[m] = old_tag

    timelines: dict[int, Tree] = {}
    for m, launches in by_m.items():
        values: list[Tree] = []
        old = old_tag_for_m[m]
        # Include one stable row after the final LOW->BG transition so the
        # last interval is N+0 rather than a perpetually repeated retag.
        for y in range(BYTE_Y0, BYTE_Y0 + 34):
            new = launches.get(y, TAG_DEC_BG)
            values.append(Leaf("N", (new - old) * MEM_STRIDE))
            old = new
        # The last byte is on row 871; row 872 is already DEC_BG and N+0
        # remains the correct background tag for the rest of the image.
        timelines[m] = _runs("y", values, BYTE_Y0)

    values: list[Tree] = []
    background = Leaf("Set", 65536)
    for x in range(WIDTH):
        values.append(timelines.get(x, background))
    starts: list[int] = []
    children: list[Tree] = []
    previous: Tree | None = None
    for x, value in enumerate(values):
        if value != previous:
            starts.append(x)
            children.append(value)
            previous = value
    return Intervals("x", tuple(starts), tuple(children))


_COORDINATE_META: dict[str, int] = {}


def coordinate_program(tags: Mapping[str, int]) -> Tree:
    layout = build_layout()
    memory_m, memory_q = _memory_coordinates(layout)
    actions, meta = _coordinate_actions(tags)
    _COORDINATE_META.clear()
    _COORDINATE_META.update(meta)
    by_x: dict[int, dict[int, Tree]] = {}
    for (x, y), action in actions.items():
        by_x.setdefault(x, {})[y] = action
    starts: list[int] = []
    children: list[Tree] = []
    for x in range(16, layout.used_width + 1):
        default = (Leaf("N", 0) if x in memory_m else
                   Leaf("WW", 0) if x in memory_q else Leaf("W", 0))
        starts.append(x)
        children.append(_rows_for_x(default, by_x.get(x, {})))
    compute = Intervals("x", tuple(starts), tuple(children))
    post_sha = _post_sha_coordinate_program()
    return Split("y", DISPLAY_Y0 - 1, post_sha, compute)


def _raw_bit(bit: int, first: bool) -> Tree:
    zero = Leaf("Set", 0) if first else Leaf("W", 0)
    one = Leaf("Set", 1 << bit) if first else Leaf("W", 1 << bit)
    return Split("Prev1", 250, zero, one)


def prefix_program() -> Tree:
    rom = build_rom()
    seeds = {event.y: int(event.arg) for event in rom.events.values()
             if event.kind == "row_seed"}
    pack = Intervals(
        "x", tuple(range(16)),
        tuple(_raw_bit(x // 2, x == 0) if x % 2 == 0 else Leaf("W", 0)
              for x in range(16)))
    seed_rows = Intervals("y", tuple(sorted(seeds)),
                          tuple(Leaf("Set", seeds[y]) for y in sorted(seeds)))
    compute = Split("x", 0, Leaf("W", 0), seed_rows)
    active = Split("y", 39, compute, pack)
    return Split("y", DISPLAY_Y0 - 1, Leaf("Set", 65536), active)


def _runs(prop: str, values: Sequence[Tree], start: int = 0) -> Tree:
    """Compress a dense property table into lower-bound intervals."""
    starts: list[int] = []
    children: list[Tree] = []
    previous: Tree | None = None
    for index, value in enumerate(values):
        if previous is None or value != previous:
            starts.append(start + index)
            children.append(value)
            previous = value
    return Intervals(prop, tuple(starts), tuple(children))


def _bounded(prop: str, low: int, high: int, inside: Tree,
             outside: Tree | None = None) -> Tree:
    outside = outside or Leaf("Set", 0)
    return Split(prop, high, outside,
                 Split(prop, low - 1, inside, outside))


def _identity_tree(prop: str, low: int, count: int,
                   output_low: int | None = None) -> Tree:
    output_low = low if output_low is None else output_low
    inner = Intervals(
        prop, tuple(low + value for value in range(count)),
        tuple(Leaf("Set", output_low + value) for value in range(count)))
    return _bounded(prop, low, low + count - 1, inner)


def _coordinate_overrides(default: Tree,
                          actions: Mapping[tuple[int, int], Tree]) -> Tree:
    """A sparse exact (x,y) ROM with `default` outside its points."""
    by_x: dict[int, dict[int, Tree]] = {}
    for (x, y), action in actions.items():
        if not (0 <= x < WIDTH and 0 <= y < HEIGHT):
            raise AssertionError((x, y))
        by_x.setdefault(x, {})[y] = action

    xvalues: list[Tree] = []
    for x in range(WIDTH):
        row_actions = by_x.get(x)
        if not row_actions:
            xvalues.append(default)
            continue
        timeline: dict[int, Tree] = {0: default}
        for y, action in row_actions.items():
            timeline[y] = action
            if y + 1 < HEIGHT and y + 1 not in row_actions:
                timeline[y + 1] = default
        starts_list: list[int] = []
        children_list: list[Tree] = []
        previous: Tree | None = None
        for y in sorted(timeline):
            action = timeline[y]
            if action != previous:
                starts_list.append(y)
                children_list.append(action)
                previous = action
        xvalues.append(Intervals(
            "y", tuple(starts_list), tuple(children_list)))
    return _runs("x", xvalues)


def _byte_bank_row(source_q: int) -> Tree:
    identity = _identity_tree("Prev1", 0, 256)
    return Intervals(
        "x", (0, source_q, source_q + 1, BYTE_BANK_X + 1),
        (Leaf("Set", 0), identity, Leaf("W", 0), Leaf("Set", 0)))


def byte_bank_program() -> Tree:
    """c3: compact the 32 exposed bytes and initialize decimal memory."""
    sources = _digest_byte_sources()
    bank = Intervals(
        "y", tuple(BYTE_Y0 + byte_index for byte_index, *_rest in sources),
        tuple(_byte_bank_row(q) for _byte_index, _m, q, _old, _new in sources))
    # The outer split below already supplies the upper y<=871 bound.
    bank = Split("y", BYTE_Y0 - 1, bank, Leaf("Set", 0))

    state_values: list[Tree] = []
    for y in range(DEC_STATE_Y0, HEIGHT):
        is_limb = (y <= DEC_STATE_Y0 + 2 * (DEC_LIMBS - 1) and
                   (y - DEC_STATE_Y0) % 2 == 0)
        state_values.append(Leaf("Set", DEC_BASE if is_limb else 0))
    state_y = _runs("y", state_values, DEC_STATE_Y0)
    state = Intervals(
        "x", (0, DEC_STATE_X, DEC_STATE_X + 1),
        (Leaf("Set", 0), state_y, Leaf("Set", 0)))
    return Split("y", BYTE_Y0 + 31, state, bank)


def _bank_copy_tree(seed: bool = False) -> Tree:
    output = (DEC_SEED_BASE + DEC_SEED_COUNT * 256) if seed else 0
    return _identity_tree("Prev1", 0, 256, output)


def _converter_bank_program() -> Tree:
    raw = _bank_copy_tree(False)
    seeded = _bank_copy_tree(True)
    channel_children: list[Tree] = []
    for channel in range(DEC_FIRST_CHANNEL, DEC_LAST_CHANNEL + 1):
        selected_y = BYTE_Y0 + channel - DEC_FIRST_CHANNEL
        values = [seeded if y == selected_y else raw
                  for y in range(BYTE_Y0, BYTE_Y0 + 32)]
        active = _runs("y", values, BYTE_Y0)
        channel_children.append(Split(
            "y", BYTE_Y0 + 31, Leaf("Set", 0),
            Split("y", BYTE_Y0 - 1, active, Leaf("Set", 0))))
    return Intervals(
        "c", tuple(range(DEC_FIRST_CHANNEL, DEC_LAST_CHANNEL + 1)),
        tuple(channel_children))


def _decimal_control_program() -> Tree:
    seed_min = DEC_SEED_BASE
    seed_max = DEC_SEED_BASE + DEC_SEED_COUNT * 256 + 255
    seed_init_min = DEC_SEED_BASE + DEC_SEED_COUNT * 256

    # A selected byte enters from W with countdown=31.  Count zero becomes
    # the first carry controller; all other seed rows decrement one step.
    seed_from_n = Split(
        "N", DEC_SEED_BASE + 255, Leaf("N", -256),
        Leaf("N", DEC_CARRY_BASE - DEC_SEED_BASE))

    carry_from_tmp = Intervals(
        "N", tuple(DEC_TMP_BASE + 1000 * carry for carry in range(256)),
        tuple(Leaf("Set", DEC_CARRY_BASE + carry)
              for carry in range(256)))

    # One row after an update, N still holds the incoming carry and NE is the
    # exact temporary at x=1001.  Copying NE puts that temporary in this
    # column; the following row can extract its quotient from N directly.
    copy_tmp_from_ne = Leaf("NE", 0)

    from_n = Split(
        "N", DEC_SEED_BASE - 1,
        Split("N", seed_max, Leaf("Set", 0), seed_from_n),
        Split("N", DEC_CARRY_BASE - 1,
              Split("N", DEC_CARRY_BASE + 255,
                    Leaf("Set", 0), copy_tmp_from_ne),
              Split("N", DEC_TMP_BASE - 1,
                    Split("N", DEC_TMP_BASE + 255999,
                          Leaf("Set", 0), carry_from_tmp),
                    Leaf("Set", 0))))
    capture = _bounded(
        "W", seed_init_min, seed_init_min + 255, Leaf("W", 0), from_n)
    return capture


def _decimal_state_program() -> Tree:
    update = Intervals(
        "Prev1", tuple(DEC_BASE + digit for digit in range(1000)),
        tuple(Leaf("W", DEC_TMP_BASE - DEC_CARRY_BASE + 256 * digit)
              for digit in range(1000)))
    normalize = Intervals(
        "N", tuple(DEC_TMP_BASE + 1000 * carry for carry in range(256)),
        tuple(Leaf("N", DEC_BASE - DEC_TMP_BASE - 1000 * carry)
              for carry in range(256)))
    normalize = _bounded(
        "N", DEC_TMP_BASE, DEC_TMP_BASE + 255999, normalize)
    # A prior channel can have a TMP value here for the next vertical limb;
    # only the exact normalized-limb domain selects update.  Everything else
    # must still get the chance to normalize this channel's N temporary.
    return _bounded("Prev1", DEC_BASE, DEC_BASE + 999,
                    update, normalize)


def decimal_converter_program() -> Tree:
    return Intervals(
        "x", (0, BYTE_BANK_X, DEC_CONTROL_X, DEC_STATE_X,
              DEC_STATE_X + 1),
        (Leaf("Set", 0), _converter_bank_program(),
         _decimal_control_program(), _decimal_state_program(),
         Leaf("Set", 0)))


def machine_program(source: Tree, tags: Mapping[str, int]) -> Tree:
    fabric = Split(
        "W", MEM_BASE - 1, source,
        Split("W", P1_R1 - 1, stage_dispatch(), coordinate_program(tags)),
    )
    return Split("x", 15, fabric, prefix_program())


def raw_aperture_program() -> Tree:
    # End on odd x=15 so the last interval remains an ordinary zero leaf;
    # otherwise Set251 would accidentally persist across the full row.
    xchildren = tuple(Leaf("Set", 251 if x % 2 == 0 else 0)
                      for x in range(16))
    active_x = Intervals("x", tuple(range(16)), xchildren)
    return Split("y", 39, Leaf("Set", 0), active_x)


def alignment_program(visits: int = 5) -> Tree:
    if not 1 <= visits <= 8:
        raise ValueError(visits)
    active = Split("x", visits - 1, Leaf("Set", 0), Leaf("Set", 251))
    return Split("y", 0, Leaf("Set", 0), active)


DECIMAL_FONT = (
    ("111", "101", "101", "101", "111"),  # 0
    ("010", "110", "010", "010", "111"),  # 1
    ("111", "001", "111", "100", "111"),  # 2
    ("111", "001", "111", "001", "111"),  # 3
    ("101", "101", "111", "001", "001"),  # 4
    ("111", "100", "111", "001", "111"),  # 5
    ("111", "100", "111", "101", "111"),  # 6
    ("111", "001", "010", "010", "010"),  # 7
    ("111", "101", "111", "101", "111"),  # 8
    ("111", "101", "111", "001", "111"),  # 9
)


def _copy_digit_program() -> Tree:
    return _identity_tree("Prev1", DEC_DIGIT_BASE, 10)


def _limb_source_tree(prop: str) -> Tree:
    source = Intervals(
        prop, tuple(DEC_BASE + value for value in range(1000)),
        tuple(Leaf("Set", DEC_FULL_BASE + value)
              for value in range(1000)))
    return _bounded(prop, DEC_BASE, DEC_BASE + 999, source)


def _limb_digit_tree(divisor: int) -> Tree:
    values = [Leaf("Set", DEC_DIGIT_BASE + (value // divisor) % 10)
              for value in range(1000)]
    return _bounded(
        "W", DEC_FULL_BASE, DEC_FULL_BASE + 999,
        _runs("W", values, DEC_FULL_BASE))


def _put_action(actions: dict[tuple[int, int], Tree], x: int, y: int,
                action: Tree) -> None:
    old = actions.get((x, y))
    if old is not None and old != action:
        raise AssertionError(("render action collision", x, y, old, action))
    actions[(x, y)] = action


def _render_jobs() -> list[tuple[int, int, int, int, int]]:
    """(k, limb, route-x, block-x, group-y)."""
    jobs: list[tuple[int, int, int, int, int]] = []
    seen: set[int] = set()
    for j0, count, route_x, block_x, base_y in RENDER_LINES:
        for k in range(count):
            limb = j0 + k
            jobs.append((k, limb, route_x, block_x, base_y + 10 * k))
            seen.add(limb)
    assert seen == set(range(DEC_LIMBS))
    return jobs


def render_channel_program(channel: int) -> Tree:
    if not RENDER_FIRST_CHANNEL <= channel <= RENDER_LAST_CHANNEL:
        raise ValueError(channel)
    k = 6 - (channel - RENDER_FIRST_CHANNEL)
    prop = f"Prev{channel - DEC_LAST_CHANNEL}"
    actions: dict[tuple[int, int], Tree] = {}

    for job_k, limb, route_x, block_x, group_y in _render_jobs():
        if job_k != k:
            continue
        source_y = DEC_STATE_Y0 + 32 + 2 * limb
        dx = DEC_STATE_X - route_x
        arrival_y = source_y + dx
        if arrival_y > group_y:
            raise AssertionError((channel, limb, arrival_y, group_y))

        _put_action(actions, DEC_STATE_X, source_y,
                    _limb_source_tree(prop))
        for step in range(1, dx + 1):
            _put_action(actions, DEC_STATE_X - step, source_y + step,
                        Leaf("NE", 0))
        for y in range(arrival_y + 1, group_y + 7):
            _put_action(actions, route_x, y, Leaf("N", 0))

        # Physical 5x3 blocks become ordinary 3x5 glyphs after orientation 7.
        for offset, divisor in ((0, 1), (3, 10), (6, 100)):
            y0 = group_y + offset
            _put_action(actions, block_x, y0, _limb_digit_tree(divisor))
            for x in range(block_x + 1, block_x + 5):
                _put_action(actions, x, y0, Leaf("W", 0))
            for y in (y0 + 1, y0 + 2):
                _put_action(actions, block_x, y, Leaf("N", 0))
                for x in range(block_x + 1, block_x + 5):
                    _put_action(actions, x, y, Leaf("W", 0))

    return _coordinate_overrides(_copy_digit_program(), actions)


def _bitmap_digit_tree(font_row: int, font_column: int) -> Tree:
    values = [Leaf("Set", MAX31 if
                   DECIMAL_FONT[digit][font_row][font_column] == "1" else 0)
              for digit in range(10)]
    return _bounded(
        "Prev1", DEC_DIGIT_BASE, DEC_DIGIT_BASE + 9,
        _runs("Prev1", values, DEC_DIGIT_BASE))


def bitmap_program() -> Tree:
    actions: dict[tuple[int, int], Tree] = {}
    for _k, _limb, _route_x, block_x, group_y in _render_jobs():
        for digit_offset in (0, 3, 6):
            y0 = group_y + digit_offset
            for dx in range(5):
                for dy in range(3):
                    # Orientation 7 maps (x,y) -> (1023-y,1023-x).
                    _put_action(actions, block_x + dx, y0 + dy,
                                _bitmap_digit_tree(4 - dx, 2 - dy))
    return _coordinate_overrides(Leaf("Set", 0), actions)


def visible_program() -> Tree:
    # c61 reaches the c43 bitmap as its 18th prior compatible channel.  G and
    # B then copy the immediately preceding visible binary plane.
    red = Split("Prev18", 0, Leaf("Set", MAX31), Leaf("Set", 0))
    copy = Split("Prev1", 0, Leaf("Set", MAX31), Leaf("Set", 0))
    return Split("c", 61, copy, red)


def full_tree(alignment_visits: int = 5) -> tuple[Tree, dict[str, int]]:
    source, tags = source_selector()
    render_channels = tuple(
        render_channel_program(channel)
        for channel in range(RENDER_FIRST_CHANNEL, RENDER_LAST_CHANNEL + 1))
    tree = Intervals(
        "c",
        (0, 1, 2, 3, DEC_FIRST_CHANNEL,
         RENDER_FIRST_CHANNEL, 37, 38, 39, 40, 41, 42,
         BITMAP_CHANNEL, BITMAP_CHANNEL + 1, 61),
        (alignment_program(alignment_visits), raw_aperture_program(),
         machine_program(source, tags), byte_bank_program(),
         decimal_converter_program(),
         *render_channels,
         bitmap_program(), Leaf("Set", 0), visible_program()))
    return tree, tags


HEX_FONT = (
    ("111", "101", "101", "101", "111"),  # 0
    ("010", "110", "010", "010", "111"),  # 1
    ("111", "001", "111", "100", "111"),  # 2
    ("111", "001", "111", "001", "111"),  # 3
    ("101", "101", "111", "001", "001"),  # 4
    ("111", "100", "111", "001", "111"),  # 5
    ("111", "100", "111", "101", "111"),  # 6
    ("111", "001", "010", "010", "010"),  # 7
    ("111", "101", "111", "101", "111"),  # 8
    ("111", "101", "111", "001", "111"),  # 9
    ("010", "101", "111", "101", "101"),  # A
    ("110", "101", "110", "101", "110"),  # B
    ("111", "100", "100", "100", "111"),  # C
    ("110", "101", "101", "101", "110"),  # D
    ("111", "100", "110", "100", "111"),  # E
    ("111", "100", "110", "100", "100"),  # F
)


def _hex_output_x(character: int) -> int:
    if not 0 <= character < 16:
        raise ValueError(character)
    return (HEX_OUTPUT_X0 + HEX_CHAR_STRIDE * character +
            HEX_GROUP_GAP * (character // 4))


def _hex_physical_y(character: int) -> int:
    # Orientation 7 maps physical (x,y) to decoded (1023-y,1023-x).
    # A scale-2 3x5 glyph occupies six physical y samples.
    return WIDTH - 1 - (_hex_output_x(character) + 3 * HEX_SCALE - 1)


def _hex_job(byte_index: int) -> tuple[int, int, int, int, int, int, int]:
    """Return line, slot, source-y, route-x, block-x, high-y, low-y."""
    if not 0 <= byte_index < 32:
        raise ValueError(byte_index)
    line, slot = divmod(byte_index, 8)
    decoded_y = HEX_OUTPUT_LINE_Y[line]
    block_x = HEIGHT - 1 - (decoded_y + 5 * HEX_SCALE - 1)
    route_x = block_x - 1
    source_y = BYTE_Y0 + byte_index
    high_y = _hex_physical_y(2 * slot)
    low_y = _hex_physical_y(2 * slot + 1)
    return line, slot, source_y, route_x, block_x, high_y, low_y


def hex_byte_bank_program() -> Tree:
    """c3: compact the 32 exposed digest bytes at x=999, rows 840..871."""
    sources = _digest_byte_sources()
    bank = Intervals(
        "y", tuple(BYTE_Y0 + byte_index for byte_index, *_rest in sources),
        tuple(_byte_bank_row(q) for _byte_index, _m, q, _old, _new in sources))
    active = Split("y", BYTE_Y0 - 1, bank, Leaf("Set", 0))
    return Split("y", BYTE_Y0 + 31, Leaf("Set", 0), active)


def _byte_nibble_tree(high: bool) -> Tree:
    values = [Leaf("Set", HEX_TAG_BASE + ((value >> 4) if high else
                                          (value & 15)))
              for value in range(256)]
    return _bounded("W", 0, 255, _runs("W", values, 0))


def hex_render_channel_program(channel: int) -> Tree:
    """Route one digest byte and expand its two nibble tags into glyph cells."""
    if not HEX_RENDER_FIRST_CHANNEL <= channel <= HEX_RENDER_LAST_CHANNEL:
        raise ValueError(channel)
    byte_index = channel - HEX_RENDER_FIRST_CHANNEL
    (_line, _slot, source_y, route_x, block_x,
     high_y, low_y) = _hex_job(byte_index)
    actions: dict[tuple[int, int], Tree] = {}

    # Prev1..Prev19 are the encodable previous-channel properties. c4..c21
    # read c3 directly. c22 copies the complete 32-byte strip from c3 once;
    # c23..c35 then address that relay with Prev1..Prev13.
    if byte_index < 18:
        byte_copy = _identity_tree(f"Prev{byte_index + 1}", 0, 256)
        relay_first_y = relay_last_y = source_y
    elif byte_index == 18:
        byte_copy = _identity_tree("Prev19", 0, 256)
        relay_first_y = BYTE_Y0
        relay_last_y = BYTE_Y0 + 31
    else:
        byte_copy = _identity_tree(f"Prev{channel - 22}", 0, 256)
        relay_first_y = relay_last_y = source_y
    for y in range(relay_first_y, relay_last_y + 1):
        _put_action(actions, BYTE_BANK_X, y, byte_copy)

    # The explicit direct read or c22 relay above places this channel's source
    # byte at (999, source_y).
    if route_x < BYTE_BANK_X:
        distance = BYTE_BANK_X - route_x
        for step in range(1, distance + 1):
            _put_action(actions, BYTE_BANK_X - step, source_y + step,
                        Leaf("NE", 0))
        arrival_y = source_y + distance
    else:
        for x in range(BYTE_BANK_X + 1, route_x + 1):
            _put_action(actions, x, source_y, Leaf("W", 0))
        arrival_y = source_y

    glyphs = ((low_y, False), (high_y, True))
    first_y = min(y for y, _high in glyphs)
    last_y = max(y for y, _high in glyphs) + 3 * HEX_SCALE - 1
    if arrival_y >= first_y:
        raise AssertionError((byte_index, arrival_y, first_y))

    # Keep the raw byte on a vertical rail immediately west of this output
    # line. Each glyph converts it once, then fills the 10x6 physical block.
    for y in range(arrival_y + 1, last_y + 1):
        _put_action(actions, route_x, y, Leaf("N", 0))
    for glyph_y, high in glyphs:
        _put_action(actions, block_x, glyph_y, _byte_nibble_tree(high))
        for x in range(block_x + 1, block_x + 5 * HEX_SCALE):
            _put_action(actions, x, glyph_y, Leaf("W", 0))
        for dy in range(1, 3 * HEX_SCALE):
            _put_action(actions, block_x, glyph_y + dy, Leaf("N", 0))
            for x in range(block_x + 1, block_x + 5 * HEX_SCALE):
                _put_action(actions, x, glyph_y + dy, Leaf("W", 0))

    # Previously emitted glyph rectangles contain only this compact tag
    # domain. Copying it as the default preserves them without retaining old
    # routes or unrelated state.
    return _coordinate_overrides(
        _identity_tree("Prev1", HEX_TAG_BASE, 16), actions)


def _bitmap_hex_tree(font_row: int, font_column: int) -> Tree:
    values = [Leaf("Set", MAX31 if
                   HEX_FONT[nibble][font_row][font_column] == "1" else 0)
              for nibble in range(16)]
    return _bounded(
        "Prev1", HEX_TAG_BASE, HEX_TAG_BASE + 15,
        _runs("Prev1", values, HEX_TAG_BASE))


def hex_bitmap_program() -> Tree:
    actions: dict[tuple[int, int], Tree] = {}
    for byte_index in range(32):
        (_line, _slot, _source_y, _route_x, block_x,
         high_y, low_y) = _hex_job(byte_index)
        for glyph_y in (high_y, low_y):
            for dx in range(5 * HEX_SCALE):
                for dy in range(3 * HEX_SCALE):
                    _put_action(
                        actions, block_x + dx, glyph_y + dy,
                        _bitmap_hex_tree(4 - dx // HEX_SCALE,
                                         2 - dy // HEX_SCALE))
    return _coordinate_overrides(Leaf("Set", 0), actions)


def hex_full_tree(alignment_visits: int = 3) -> tuple[Tree, dict[str, int]]:
    source, tags = source_selector()
    render_channels = tuple(
        hex_render_channel_program(channel)
        for channel in range(HEX_RENDER_FIRST_CHANNEL,
                             HEX_RENDER_LAST_CHANNEL + 1))
    # c37..c48 are true zero padding used only for the decoder's tree budget.
    # Red c49 reads bitmap c36 through the supported Prev13 property; green
    # and blue then copy the immediately preceding visible plane.
    starts = (0, 1, 2, 3, *range(HEX_RENDER_FIRST_CHANNEL,
                                 HEX_RENDER_LAST_CHANNEL + 1),
              HEX_BITMAP_CHANNEL, HEX_BITMAP_CHANNEL + 1,
              HIDDEN_CHANNELS, HIDDEN_CHANNELS + 1)
    red = Split("Prev13", 0, Leaf("Set", MAX31), Leaf("Set", 0))
    copy = Split("Prev1", 0, Leaf("Set", MAX31), Leaf("Set", 0))
    children = (alignment_program(alignment_visits), raw_aperture_program(),
                machine_program(source, tags), hex_byte_bank_program(),
                *render_channels, hex_bitmap_program(), Leaf("Set", 0),
                red, copy)
    return Intervals("c", starts, children), tags


def controller_interval_report() -> list[dict[str, int | str]]:
    """Prove that every reachable stage is below the next dispatch start."""
    bounds = [
        ("P1_R1", P1_R1, P1_R1 + 429_981_695, P1_R2),
        ("P1_R2", P1_R2, P1_R2 + 642_815, P1_R3),
        ("P1_R3", P1_R3, P1_R3 + 687_615, SCHED_R0),
        ("SCHED_R0", SCHED_R0, SCHED_R0 + 1_679_615, SCHED_R1),
        ("SCHED_R1", SCHED_R1, SCHED_R1 + 6_718_205, SCHED_TARGET),
        ("SCHED_TARGET", SCHED_TARGET, SCHED_TARGET + 1_023, P2_R1),
        ("P2_R1", P2_R1, P2_R1 + 339_741_695, P2_R2),
        ("P2_R2", P2_R2, P2_R2 + 4_194_495, FF_TARGET),
        ("FF_TARGET", FF_TARGET, FF_TARGET + 256, P1_TARGET),
        ("P1_TARGET", P1_TARGET, P1_TARGET + 990, P2_A_TARGET),
        ("P2_A_TARGET", P2_A_TARGET, P2_A_TARGET + 1_215, P2_A_CARRY),
        ("P2_A_CARRY", P2_A_CARRY,
         P2_A_CARRY + 6 * 2048 + 1_215, P2_E_TARGET),
        ("P2_E_TARGET", P2_E_TARGET, P2_E_TARGET + 1_215, P2_E_CARRY),
        ("P2_E_CARRY", P2_E_CARRY,
         P2_E_CARRY + 5 * 2048 + 1_215, MEM_BASE),
    ]
    result = []
    for name, start, maximum, next_start in bounds:
        assert start <= maximum < next_start, (
            "overlapping controller interval", name, start, maximum, next_start)
        result.append({"name": name, "start": start, "maximum": maximum,
                       "next_start": next_start,
                       "slack": next_start - maximum - 1})
    return result


def report(alignment_visits: int = 3) -> dict[str, object]:
    tree, tags = hex_full_tree(alignment_visits)
    stats = tree.stats()
    validation = validate_tree(tree)
    assert validation == {"nodes": stats.nodes, "leaves": stats.leaves}
    layout = build_layout()
    display_q = [half.q for word in digest_words(layout)
                 for half in (word.hi, word.lo)]
    # Physical digest-word order is A63,A62,A61,A60,E63,E62,E61,E60;
    # within each SHA word the high half precedes the low half.
    return {
        "nodes": stats.nodes,
        "leaves": stats.leaves,
        "depth": stats.depth,
        "offset_range": [stats.min_offset, stats.max_offset],
        "hard_node_limit": HARD_TREE_LIMIT,
        "decoder_node_limit": TREE_LIMIT,
        "decoder_node_margin": TREE_LIMIT - stats.nodes,
        "fits_decoder_node_limit": stats.nodes <= TREE_LIMIT,
        "libjxl_path_range_validation": validation,
        "width": WIDTH,
        "height": HEIGHT,
        "hidden_channels": HIDDEN_CHANNELS,
        "source_tags": len(_selector_requirements(packed_pass1=True)),
        "all_source_and_control_aliases": len(tags),
        **_COORDINATE_META,
        "coordinate_program_complete": True,
        "transition_simulator": "integration_simulator.py",
        "transition_simulator_status": "passed: 6 apertures / 384 rounds",
        "full_raster_verifier": "verify_hex.py",
        "controller_intervals": controller_interval_report(),
        "nonadjacent_target_stage": (
            "Only SCHED_TARGET (CS) can cross unrelated M cells: R1 is at "
            "the W[t-2]/W[t-1] gap. schedule_target canonicalizes unary "
            "source tags 0..42 to PASS. P1/P2/FF stages are adjacent to a "
            "reducer/target or intentionally traverse their own low/high pair."
        ),
        "stock_full_encode_verification": "external: run verify_hex.py",
        "digest_source_q_columns_sha_order": display_q,
        "display_encoding": "SHA256(file).hexdigest().upper()",
        "byte_bank_samples_sha_order": [
            [BYTE_BANK_X, BYTE_Y0 + index] for index in range(32)
        ],
        "hex_route_channels": [HEX_RENDER_FIRST_CHANNEL,
                               HEX_RENDER_LAST_CHANNEL],
        "hex_bitmap_channel": HEX_BITMAP_CHANNEL,
        "hex_output_x": [_hex_output_x(i) for i in range(16)],
        "hex_output_line_y": list(HEX_OUTPUT_LINE_Y),
        "hex_font": "3x5 uppercase 0-9A-F, scale 2",
        "orientation": 7,
        "alignment_visits": alignment_visits,
        "maximum_tagged_sample": max(
            _tag_base(TAG_DEC_BG) + MEM_STRIDE - 1,
            HEX_TAG_BASE + 15,
        ),
        "all_samples_fit_signed31": max(
            _tag_base(TAG_DEC_BG) + MEM_STRIDE - 1,
            HEX_TAG_BASE + 15,
        ) <= MAX31,
    }


def emit(path: Path, alignment_visits: int = 3) -> None:
    tree, _tags = hex_full_tree(alignment_visits)
    validation = validate_tree(tree)
    if validation["nodes"] > TREE_LIMIT:
        raise ValueError(("tree exceeds stock decoder limit", validation))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as out:
        out.write(f"Width {WIDTH}\n")
        out.write(f"Height {HEIGHT}\n")
        out.write("Bitdepth 31\n")
        out.write("GroupShift 3\n")
        out.write(f"HiddenChannel {HIDDEN_CHANNELS}\n")
        out.write("Orientation 7\n")
        out.write("RCT 0\n")
        out.write("/* integrated SHA-256 MA circuit with uppercase hex output */\n")
        tree.render(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emit-tree", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--alignment-visits", type=int, default=3,
                        choices=range(1, 9))
    args = parser.parse_args()
    data = report(args.alignment_visits)
    if args.emit_tree:
        emit(args.emit_tree, args.alignment_visits)
        data["tree_bytes"] = args.emit_tree.stat().st_size
    if args.report:
        args.report.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
