#!/usr/bin/env python3
"""A concrete monotone-raster SHA-256 suffix machine.

This module models the part of a JPEG XL hashquine that is easiest to get
wrong: the *geometry* of the computation.  It deliberately does not pretend
that a Modular predictor can read an arbitrary older pixel.

The construction uses three Modular channels:

  input bus     256 raw entropy bits, copied down the channel;
  machine       a left-to-right finite-state scan on every raster row;
  projector     a 16 by 16 bitmap of the final digest.

The machine tape starts at x=256.  Every logical 16-bit tape word occupies
two pixels, Q then M.  Q carries the finite controller.  M carries memory.
At Q_i the decoder recovers the preceding controller from WW and consumes the
*preceding* memory word directly from W.  At M_i it writes the new memory word
from W and N.  Thus each row is a genuine one-way streaming transducer using
only W, WW and N; it does not need to reconstruct NE from N-(N-NE).

The tape is append-only.  Storing only A_t and E_t is sufficient because the
other six SHA working words are delays of those two sequences.  This is what
makes the exact 1024-pixel width bound work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


MASK32 = 0xFFFFFFFF
WIDTH = 1024
INPUT_BITS = 320
INPUT_SIDE = 16
TAPE_X0 = 16

IV = (
    0x6A09E667,
    0xBB67AE85,
    0x3C6EF372,
    0xA54FF53A,
    0x510E527F,
    0x9B05688C,
    0x1F83D9AB,
    0x5BE0CD19,
)

K = (
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5,
    0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
    0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3,
    0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
    0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC,
    0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
    0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7,
    0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
    0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13,
    0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
    0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3,
    0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
    0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5,
    0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
    0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208,
    0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2,
)


def rotr(x: int, n: int) -> int:
    return ((x >> n) | (x << (32 - n))) & MASK32


def sigma0(x: int) -> int:
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3)


def sigma1(x: int) -> int:
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10)


def capsigma0(x: int) -> int:
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22)


def capsigma1(x: int) -> int:
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25)


def ch(x: int, y: int, z: int) -> int:
    return (x & y) ^ ((~x) & z) & MASK32


def maj(x: int, y: int, z: int) -> int:
    return (x & y) ^ (x & z) ^ (y & z)


def compress(state: Sequence[int], block: bytes) -> Tuple[int, ...]:
    """The SHA-256 compression function, independently testable."""
    if len(state) != 8 or len(block) != 64:
        raise ValueError("compress needs eight state words and one 64-byte block")
    w = [int.from_bytes(block[4 * i:4 * i + 4], "big") for i in range(16)]
    for i in range(16, 64):
        w.append((sigma1(w[i - 2]) + w[i - 7] + sigma0(w[i - 15]) + w[i - 16]) & MASK32)
    a, b, c, d, e, f, g, h = state
    for i in range(64):
        t1 = (h + capsigma1(e) + ch(e, f, g) + K[i] + w[i]) & MASK32
        t2 = (capsigma0(a) + maj(a, b, c)) & MASK32
        a, b, c, d, e, f, g, h = (
            (t1 + t2) & MASK32, a, b, c,
            (d + t1) & MASK32, e, f, g,
        )
    return tuple((x + y) & MASK32 for x, y in zip(state, (a, b, c, d, e, f, g, h)))


def midstate_for_block_aligned_prefix(prefix: bytes) -> Tuple[int, ...]:
    if len(prefix) % 64:
        raise ValueError("prefix must end on a SHA-256 block boundary")
    state = IV
    for off in range(0, len(prefix), 64):
        state = compress(state, prefix[off:off + 64])
    return state


def aperture_suffix_block(midstate: Sequence[int], bit_length: int) -> bytes:
    """Final block for ``P || BE32(midstate) || BE64(bit_length)``.

    ``bit_length`` is the length in bits of the *unpadded* complete file,
    i.e. 8 * (len(P) + 40).  The 40-byte decoder aperture therefore occupies
    W0..W9, SHA padding starts at W10, and W15 repeats ``bit_length`` as the
    final block's length field.  The length is aperture data, not a constant
    baked into the MA tree.
    """
    if len(midstate) != 8:
        raise ValueError("midstate must contain eight words")
    if not 0 <= bit_length < 1 << 64:
        raise ValueError("SHA length does not fit uint64")
    raw = b"".join(x.to_bytes(4, "big") for x in midstate)
    raw += bit_length.to_bytes(8, "big")
    return raw + b"\x80" + b"\x00" * 15 + bit_length.to_bytes(8, "big")


def words_from_suffix(midstate: Sequence[int], bit_length: int) -> List[int]:
    block = aperture_suffix_block(midstate, bit_length)
    return [int.from_bytes(block[i:i + 4], "big") for i in range(0, 64, 4)]


@dataclass(frozen=True)
class Slot:
    """One 16-bit logical memory symbol at a fixed Q/M pixel pair."""

    index: int
    name: str
    mx: int
    qx: int


@dataclass(frozen=True)
class ReductionPair:
    name: str
    x0: int
    x1: int


@dataclass(frozen=True)
class RowOp:
    kind: str
    target: Tuple[str, ...]
    round_index: int | None = None
    nibble: int | None = None
    halfword: int | None = None
    sources: Tuple[str, ...] = ()


@dataclass
class Layout:
    bit_length: int
    slots: List[Slot] = field(default_factory=list)
    by_name: Dict[str, Slot] = field(default_factory=dict)
    rows: List[RowOp] = field(default_factory=list)
    w_names: Dict[int, Tuple[str, str]] = field(default_factory=dict)
    a_names: Dict[int, Tuple[str, str]] = field(default_factory=dict)
    e_names: Dict[int, Tuple[str, str]] = field(default_factory=dict)
    digest_names: List[Tuple[str, str]] = field(default_factory=list)
    reductions: List[ReductionPair] = field(default_factory=list)
    cursor: int = TAPE_X0

    def add_slot(self, name: str) -> Slot:
        if name in self.by_name:
            raise ValueError(f"duplicate slot {name}")
        slot = Slot(len(self.slots), name, self.cursor, self.cursor + 1)
        self.cursor += 2
        self.slots.append(slot)
        self.by_name[name] = slot
        return slot

    def add_reduction_pair(self, name: str) -> ReductionPair:
        pair = ReductionPair(name, self.cursor, self.cursor + 1)
        self.cursor += 2
        self.reductions.append(pair)
        return pair

    def add_word(self, prefix: str) -> Tuple[str, str]:
        # Arithmetic targets are physically low halfword then high halfword so
        # nibble carry always travels east.  The returned tuple remains
        # (high, low), i.e. network/logical order.
        hi, lo = f"{prefix}.hi", f"{prefix}.lo"
        self.add_slot(lo)
        self.add_slot(hi)
        return hi, lo

    @property
    def used_width(self) -> int:
        # x=1008..1023 is the feed-forward bit/carry projector tail.
        return WIDTH

    @property
    def allocated_tape_end(self) -> int:
        return self.cursor

    @property
    def machine_rows(self) -> int:
        return len(self.rows)

    @property
    def image_height(self) -> int:
        # Projector reads the stable final tape during sixteen following rows.
        return self.machine_rows + 16

    def validate(self) -> None:
        if len(self.slots) != 370:
            raise AssertionError(f"expected 370 logical slots, got {len(self.slots)}")
        if self.used_width != WIDTH:
            raise AssertionError(f"layout must end exactly at x=1024, got {self.used_width}")
        if self.allocated_tape_end != 1008:
            raise AssertionError(f"tape must leave a 16-cell tail, got {self.allocated_tape_end}")
        if len(self.reductions) != 126:
            raise AssertionError(f"expected 126 reduction pairs, got {len(self.reductions)}")
        if self.machine_rows != 980:
            raise AssertionError(f"expected 980 machine rows, got {self.machine_rows}")
        if self.image_height > 1024:
            raise AssertionError("machine escapes one 1024x1024 Modular group")
        for row in self.rows:
            targets = [self.by_name[n].mx for n in row.target]
            sources = [self.by_name[n].qx for n in row.sources if n in self.by_name]
            if sources and targets and max(sources) >= min(targets):
                raise AssertionError(f"non-monotone row {row}: a source is not west of its target")


def build_layout(bit_length: int) -> Layout:
    """Build the exact 1024 x 992 one-group layout."""
    if not 0 <= bit_length < (1 << 32):
        raise ValueError("one-group construction requires a zero high SHA length word")
    layout = Layout(bit_length)

    # The 320 aperture bits are a 16x20 rectangle in the preceding channel:
    # sixteen midstate halfwords followed by BE64(file_bit_length).  The high
    # length word is asserted zero; rows 18/19 supply W15.
    for j in range(16):
        name = f"M{j}"
        layout.add_slot(name)
        layout.rows.append(RowOp("pack_input_halfword", (name,), halfword=j))

    layout.rows.append(RowOp("idle", ()))       # aperture length high half 0
    layout.rows.append(RowOp("idle", ()))       # aperture length high half 1
    length_hi, length_lo = "L15.hi", "L15.lo"
    layout.add_slot(length_hi)
    layout.rows.append(RowOp("pack_input_halfword", (length_hi,), halfword=18))
    layout.add_slot(length_lo)
    layout.rows.append(RowOp("pack_input_halfword", (length_lo,), halfword=19))

    # Initial SHA words W[0..7] alias the packed midstate halfwords.
    for i in range(8):
        layout.w_names[i] = (f"M{2 * i}", f"M{2 * i + 1}")
    # The low 32-bit aperture length is both message word W9 and the final
    # SHA length word W15.  W8/W14 are the asserted-zero high length words.
    # Aliasing the slots is essential: no file-size-dependent value is baked
    # into the decoder program.
    layout.w_names[9] = (length_hi, length_lo)
    layout.w_names[15] = (length_hi, length_lo)

    # A[-1..-4], E[-1..-4] alias H[0..7].
    for delay in range(1, 5):
        layout.a_names[-delay] = layout.w_names[delay - 1]
        layout.e_names[-delay] = layout.w_names[delay + 3]

    def word_sources(names: Tuple[str, str]) -> Iterable[str]:
        return names

    for t in range(64):
        if t >= 16:
            wnames = layout.add_word(f"W{t}")
            layout.w_names[t] = wnames
            # W[8..15] are compile-time padding/length constants and therefore
            # have no tape slots.
            src_words = tuple(
                layout.w_names.get(i, ())
                for i in (t - 16, t - 15, t - 7, t - 2)
            )
            src = tuple(n for pair in src_words for n in pair)
            for p in range(8):
                layout.rows.append(RowOp("schedule_nibble", wnames, t, p, sources=src))

        # Two W-addressable reducers follow the complete word.  Pair/base
        # encodings keep both halfword contributions below 2^31 until these
        # cells strip the nonlinear and sigma fields.  Round-63 results are
        # never future sources, so their four reducer cells are omitted.
        anames = layout.add_word(f"A{t}")
        if t != 63:
            layout.add_reduction_pair(f"R.A{t}")
        enames = layout.add_word(f"E{t}")
        if t != 63:
            layout.add_reduction_pair(f"R.E{t}")
        layout.a_names[t] = anames
        layout.e_names[t] = enames
        state_words = (
            layout.a_names[t - 4], layout.a_names[t - 3],
            layout.a_names[t - 2], layout.a_names[t - 1],
            layout.e_names[t - 4], layout.e_names[t - 3],
            layout.e_names[t - 2], layout.e_names[t - 1],
        )
        src = tuple(n for pair in state_words for n in pair)
        if t in layout.w_names:
            src += tuple(layout.w_names[t])
        for p in range(8):
            layout.rows.append(RowOp("round_nibble", anames + enames, t, p, sources=src))

    final_state = (
        layout.a_names[63], layout.a_names[62],
        layout.a_names[61], layout.a_names[60],
        layout.e_names[63], layout.e_names[62],
        layout.e_names[61], layout.e_names[60],
    )
    for i, state_names in enumerate(final_state):
        layout.digest_names.append(state_names)
        # In-place feed-forward: q carries the initial nibble; target M reads
        # the final-state halfword from N and adds into it.
        src = tuple(layout.w_names[i])
        for p in range(8):
            layout.rows.append(RowOp("feedforward_nibble", state_names, i, p, sources=src))

    layout.validate()
    return layout


def get_word(memory: Mapping[str, int], names: Tuple[str, str]) -> int:
    return ((memory[names[0]] & 0xFFFF) << 16) | (memory[names[1]] & 0xFFFF)


def set_nibble(memory: MutableMapping[str, int], names: Tuple[str, str], p: int, value: int) -> None:
    """Set nibble p, numbered from the least-significant nibble."""
    if not 0 <= p < 8 or not 0 <= value < 16:
        raise ValueError("bad nibble")
    half = names[1] if p < 4 else names[0]
    shift = 4 * (p if p < 4 else p - 4)
    memory[half] = (memory.get(half, 0) & ~(0xF << shift)) | (value << shift)


def half_nibble(value: int, half: str, p: int) -> int:
    """Contribution of one 16-bit halfword to logical nibble p."""
    value &= 0xFFFF
    if half == "lo" and p < 4:
        return (value >> (4 * p)) & 0xF
    if half == "hi" and p >= 4:
        return (value >> (4 * (p - 4))) & 0xF
    return 0


def bit_fragment(value: int, half: str, p: int,
                 rotates: Sequence[int] = (), shifts: Sequence[int] = ()) -> int:
    """XOR the bits supplied by one halfword to a rotated/shifted nibble.

    This is the local LUT used by a Q cell.  Importantly, it never needs the
    other halfword: fragments XOR together as the scan encounters them.
    """
    value &= 0xFFFF
    base = 0 if half == "lo" else 16
    out = 0
    for b in range(4):
        dst = 4 * p + b
        bit = 0
        for amount in rotates:
            src = (dst + amount) & 31
            if base <= src < base + 16:
                bit ^= (value >> (src - base)) & 1
        for amount in shifts:
            src = dst + amount
            if src < 32 and base <= src < base + 16:
                bit ^= (value >> (src - base)) & 1
        out |= bit << b
    return out


def small_sigma_fragment(which: int, value: int, half: str, p: int) -> int:
    if which == 0:
        return bit_fragment(value, half, p, rotates=(7, 18), shifts=(3,))
    if which == 1:
        return bit_fragment(value, half, p, rotates=(17, 19), shifts=(10,))
    raise ValueError(which)


def cap_sigma_fragment(which: int, value: int, half: str, p: int) -> int:
    if which == 0:
        return bit_fragment(value, half, p, rotates=(2, 13, 22))
    if which == 1:
        return bit_fragment(value, half, p, rotates=(6, 11, 25))
    raise ValueError(which)


def input_bits_from_midstate(midstate: Sequence[int], bit_length: int) -> List[int]:
    raw = (b"".join(x.to_bytes(4, "big") for x in midstate) +
           bit_length.to_bytes(8, "big"))
    # HybridUint raw bits are visited least-significant-bit first in each file
    # byte.  Pack rows compensate with byte-local reversed weights.
    return [(byte >> bit) & 1 for byte in raw for bit in range(8)]


class RasterMachine:
    """Unit-level simulator for the compiled row program.

    ``trace_cells=True`` additionally materializes Q/M pixels and checks the
    local geometry invariant at every pair.  The semantic transition uses a
    Python tuple as q; the netlist emitter lowers that tuple into finite-state
    fields and records the required LUT domains.
    """

    def __init__(self, layout: Layout):
        self.layout = layout

    @staticmethod
    def _add_word_roles(roles: MutableMapping[str, List[Tuple[str, str]]],
                        names: Tuple[str, str], role: str) -> None:
        hi, lo = names
        roles.setdefault(hi, []).append((role, "hi"))
        roles.setdefault(lo, []).append((role, "lo"))

    def _initial_controller(self, op: RowOp) -> Tuple[Dict[str, object], Dict[str, List[Tuple[str, str]]]]:
        """Construct q and the slot-local transition labels for one row."""
        q: Dict[str, object] = {}
        roles: Dict[str, List[Tuple[str, str]]] = {}

        if op.kind == "pack_input_halfword":
            q["packed"] = 0
            return q, roles

        if op.kind == "idle":
            return q, roles

        if op.kind == "schedule_nibble":
            assert op.round_index is not None and op.nibble is not None
            t, p = op.round_index, op.nibble
            q.update(sum=0, fragments={}, seen={})
            terms = (
                ("direct16", t - 16), ("sigma0", t - 15),
                ("direct7", t - 7), ("sigma1", t - 2),
            )
            constants = words_from_suffix((0,) * 8, self.layout.bit_length)
            for role, wi in terms:
                if wi in self.layout.w_names:
                    self._add_word_roles(roles, self.layout.w_names[wi], role)
                else:
                    word = constants[wi]
                    if role.startswith("direct"):
                        q["sum"] = int(q["sum"]) + ((word >> (4 * p)) & 0xF)
                    else:
                        value = sigma0(word) if role == "sigma0" else sigma1(word)
                        q["sum"] = int(q["sum"]) + ((value >> (4 * p)) & 0xF)
            return q, roles

        if op.kind == "round_nibble":
            assert op.round_index is not None and op.nibble is not None
            t = op.round_index
            p = op.nibble
            q.update(
                common=(K[t] >> (4 * p)) & 0xF,
                second=0,
                d=0,
                maj_pending={},
                ch_pending={},
                fragments={"a": 0, "e": 0},
                fragment_seen={"a": 0, "e": 0},
            )
            state = (
                ("d", self.layout.a_names[t - 4]),
                ("c", self.layout.a_names[t - 3]),
                ("b", self.layout.a_names[t - 2]),
                ("a", self.layout.a_names[t - 1]),
                ("h", self.layout.e_names[t - 4]),
                ("g", self.layout.e_names[t - 3]),
                ("f", self.layout.e_names[t - 2]),
                ("e", self.layout.e_names[t - 1]),
            )
            for role, names in state:
                self._add_word_roles(roles, names, role)
            if t in self.layout.w_names:
                self._add_word_roles(roles, self.layout.w_names[t], "w")
            else:
                constants = words_from_suffix((0,) * 8, self.layout.bit_length)
                q["common"] = int(q["common"]) + ((constants[t] >> (4 * p)) & 0xF)
            return q, roles

        if op.kind == "feedforward_nibble":
            assert op.round_index is not None
            i = op.round_index
            q.update(vals={})
            self._add_word_roles(roles, self.layout.w_names[i], "initial")
            return q, roles

        raise AssertionError(op.kind)

    def _consume(self, op: RowOp, q: MutableMapping[str, object],
                 role: str, half: str, symbol: int) -> None:
        """One Q-cell transition from q and NE(old M)."""
        assert op.nibble is not None or op.kind == "pack_input_halfword"
        p = 0 if op.nibble is None else op.nibble
        value = symbol & 0xFFFF

        if op.kind == "schedule_nibble":
            seen = q["seen"]
            fragments = q["fragments"]
            assert isinstance(seen, dict) and isinstance(fragments, dict)
            if role.startswith("direct"):
                q["sum"] = int(q["sum"]) + half_nibble(value, half, p)
                return
            which = 0 if role == "sigma0" else 1
            fragments[role] = int(fragments.get(role, 0)) ^ small_sigma_fragment(which, value, half, p)
            seen[role] = int(seen.get(role, 0)) + 1
            if seen[role] == 2:
                q["sum"] = int(q["sum"]) + int(fragments[role])
            return

        if op.kind == "round_nibble":
            direct_half = (p < 4 and half == "lo") or (p >= 4 and half == "hi")
            if direct_half:
                nib = half_nibble(value, half, p)
                if role == "d":
                    q["d"] = nib
                elif role in ("h", "w"):
                    q["common"] = int(q["common"]) + nib
                elif role in ("a", "b", "c"):
                    pending = q["maj_pending"]
                    assert isinstance(pending, dict)
                    pending[role] = nib
                    if len(pending) == 3:
                        q["second"] = int(q["second"]) + maj(
                            int(pending["a"]), int(pending["b"]), int(pending["c"]))
                        pending.clear()
                elif role in ("e", "f", "g"):
                    pending = q["ch_pending"]
                    assert isinstance(pending, dict)
                    pending[role] = nib
                    if len(pending) == 3:
                        q["common"] = int(q["common"]) + ch(
                            int(pending["e"]), int(pending["f"]), int(pending["g"]))
                        pending.clear()
                else:
                    raise AssertionError(role)
            if role in ("a", "e"):
                fragments = q["fragments"]
                seen = q["fragment_seen"]
                assert isinstance(fragments, dict) and isinstance(seen, dict)
                which = 0 if role == "a" else 1
                fragments[role] = int(fragments[role]) ^ cap_sigma_fragment(which, value, half, p)
                seen[role] = int(seen[role]) + 1
                if seen[role] == 2:
                    if role == "a":
                        q["second"] = int(q["second"]) + int(fragments[role])
                    else:
                        q["common"] = int(q["common"]) + int(fragments[role])
                    del fragments[role]
                    del seen[role]
            return

        if op.kind == "feedforward_nibble":
            vals = q["vals"]
            assert isinstance(vals, dict)
            vals[role] = int(vals.get(role, 0)) | half_nibble(value, half, p)
            return

        raise AssertionError(op.kind)

    @staticmethod
    def _target_digit(old_symbol: int, p: int, raw_sum: int,
                      q: MutableMapping[str, object], key: str,
                      at_half: str) -> int:
        """Nibble-serial write with carry encoded above memory bit 15."""
        payload = old_symbol & 0xFFFF
        if p < 4:
            if at_half != "lo":
                return old_symbol
            carry = 0 if p == 0 else old_symbol >> 16
            total = raw_sum + carry
            payload = (payload & ~(0xF << (4 * p))) | ((total & 0xF) << (4 * p))
            return payload | ((total >> 4) << 16)

        # Low is physically west of high.  At p=4 it hands the carry from the
        # low half to q, then clears its private carry bits.
        if at_half == "lo":
            if p == 4:
                q[f"carry:{key}"] = old_symbol >> 16
                return payload
            return old_symbol
        carry = int(q.pop(f"carry:{key}")) if p == 4 else old_symbol >> 16
        total = raw_sum + carry
        payload = (payload & ~(0xF << (4 * (p - 4)))) | ((total & 0xF) << (4 * (p - 4)))
        # The carry out of bit 31 is discarded modulo 2^32.
        return payload if p == 7 else payload | ((total >> 4) << 16)

    @staticmethod
    def _target_digit32(old_symbol: int, p: int, raw_sum: int) -> int:
        """Write one digest nibble, storing carry in the next free nibble."""
        payload = old_symbol & MASK32
        carry = 0 if p == 0 else (payload >> (4 * p)) & 0xF
        total = raw_sum + carry
        payload = (payload & ~(0xF << (4 * p))) | ((total & 0xF) << (4 * p))
        if p < 7:
            payload = (payload & ~(0xF << (4 * (p + 1)))) | ((total >> 4) << (4 * (p + 1)))
        return payload

    def _target_transition(self, op: RowOp, q: MutableMapping[str, object],
                           slot_name: str, old_symbol: int) -> int:
        """Write target M after its Q cell has seen NE and current q."""
        if op.kind == "pack_input_halfword":
            return int(q["packed"]) if slot_name == op.target[0] else old_symbol

        assert op.nibble is not None
        p = op.nibble

        if op.kind == "schedule_nibble":
            names = (op.target[0], op.target[1])
            half = "hi" if slot_name == names[0] else "lo"
            return self._target_digit(old_symbol, p, int(q["sum"]), q, "W", half)

        if op.kind == "round_nibble":
            if "raw:A" not in q:
                if q["maj_pending"] or q["ch_pending"] or q["fragments"]:
                    raise AssertionError("target reached before all source folds completed")
                common = int(q["common"])
                second = int(q["second"])
                d = int(q["d"])
                q.clear()
                q["raw:A"] = common + second
                q["raw:E"] = common + d
            anames = (op.target[0], op.target[1])
            enames = (op.target[2], op.target[3])
            if slot_name in anames:
                half = "hi" if slot_name == anames[0] else "lo"
                return self._target_digit(old_symbol, p, int(q["raw:A"]), q, "A", half)
            half = "hi" if slot_name == enames[0] else "lo"
            return self._target_digit(old_symbol, p, int(q["raw:E"]), q, "E", half)

        if op.kind == "feedforward_nibble":
            vals = q["vals"]
            assert isinstance(vals, dict)
            names = (op.target[0], op.target[1])
            half = "hi" if slot_name == names[0] else "lo"
            # Unlike fresh W/A/E targets, this slot already contains the
            # compression output.  Include that destination nibble before
            # invoking the same east-going carry mechanism.
            local_p = p if p < 4 else p - 4
            destination_nibble = (old_symbol >> (4 * local_p)) & 0xF
            return self._target_digit(
                old_symbol, p, int(vals["initial"]) + destination_nibble,
                q, "D", half,
            )

        raise AssertionError(op.kind)

    @staticmethod
    def _state_bound_bits(op: RowOp) -> int:
        # Field-product bounds, independent of the sampled input.  These are
        # deliberately conservative and are consumed by the lowering report.
        return {
            "pack_input_halfword": 16,
            "idle": 1,
            "schedule_nibble": 13,
            "round_nibble": 27,
            "feedforward_nibble": 10,
        }[op.kind]

    def run(self, midstate: Sequence[int], trace_cells: bool = False) -> Tuple[bytes, Dict[str, object]]:
        bits = input_bits_from_midstate(midstate, self.layout.bit_length)
        if len(bits) != INPUT_BITS:
            raise AssertionError
        memory: Dict[str, int] = {s.name: 0 for s in self.layout.slots}
        max_q_bits = 0
        cell_checks = 0
        stream_source_reads = 0

        for y, op in enumerate(self.layout.rows):
            q, roles = self._initial_controller(op)
            old = dict(memory)

            # Sixteen-cell controller runway.  On pack rows, Prev1 supplies
            # one complete input-bus row (16 bits).
            runway_pixels: List[Tuple[Tuple[str, str], ...]] = []
            for x in range(INPUT_SIDE):
                if op.kind == "pack_input_halfword":
                    j = op.halfword
                    assert j is not None
                    bit = bits[INPUT_SIDE * j + x]
                    # x0..7 are the low-to-high bits of the first (network
                    # order high) byte; x8..15 are the second byte.
                    weight = 1 << ((8 + x) if x < 8 else (x - 8))
                    q["packed"] = int(q["packed"]) + bit * weight
                runway_pixels.append(tuple(sorted((k, repr(v)) for k, v in q.items())))

            previous_q_pixel = runway_pixels[-1]
            current_row_pixels: List[object] = list(runway_pixels)
            target_set = set(op.target)
            slots_by_m = {s.mx: s for s in self.layout.slots}
            slots_by_q = {s.qx: s for s in self.layout.slots}
            reduction_x = {x for r in self.layout.reductions for x in (r.x0, r.x1)}
            emitted_m: Dict[str, int] = {}
            for x in range(TAPE_X0, WIDTH):
                if x in slots_by_m:
                    slot = slots_by_m[x]
                    old_symbol = old[slot.name]
                    new_symbol = old_symbol
                    if slot.name in target_set:
                        new_symbol = self._target_transition(op, q, slot.name, old_symbol)
                        memory[slot.name] = new_symbol
                    if trace_cells and current_row_pixels[-1] != previous_q_pixel:
                        raise AssertionError(f"M did not receive q through W at row {y}, slot {slot.name}")
                    current_row_pixels.append(new_symbol)
                    emitted_m[slot.name] = new_symbol
                    continue

                if x in slots_by_q:
                    slot = slots_by_q[x]
                    # Q_i consumes its M_i directly through W and recovers the
                    # incoming q through WW.  Source and target sets are
                    # disjoint, so a source M is unchanged in this row.
                    for role, half in roles.get(slot.name, ()):
                        self._consume(op, q, role, half, emitted_m[slot.name])
                        stream_source_reads += 1
                    q_pixel = tuple(sorted((k, repr(v)) for k, v in q.items()))
                    if trace_cells:
                        if current_row_pixels[-1] != emitted_m[slot.name]:
                            raise AssertionError(f"Q did not receive M through W at row {y}, slot {slot.name}")
                        if current_row_pixels[-2] != previous_q_pixel:
                            raise AssertionError(f"Q did not receive q through WW at row {y}, slot {slot.name}")
                        cell_checks += 1
                    current_row_pixels.append(q_pixel)
                    previous_q_pixel = q_pixel
                    continue

                if x in reduction_x:
                    # The semantic simulator has already applied eager folds.
                    # The MA lowering moves those folds to these W-addressable
                    # cells; identity here preserves the SHA oracle.
                    q_pixel = tuple(sorted((k, repr(v)) for k, v in q.items()))
                    if trace_cells and current_row_pixels[-1] != previous_q_pixel:
                        raise AssertionError(f"reducer lost q at row {y}, x={x}")
                    current_row_pixels.append(q_pixel)
                    previous_q_pixel = q_pixel
                    continue

                if x >= self.layout.allocated_tape_end:
                    # Free projector tail.  Computation rows preserve q; the
                    # sixteen later projector rows use it for the packed bit
                    # vector consumed by visible RGB.
                    q_pixel = tuple(sorted((k, repr(v)) for k, v in q.items()))
                    current_row_pixels.append(q_pixel)
                    previous_q_pixel = q_pixel
                    continue

                raise AssertionError(f"unassigned machine cell x={x}")

            if len(current_row_pixels) != WIDTH:
                raise AssertionError("row is not exactly 1024 pixels")
            max_q_bits = max(max_q_bits, self._state_bound_bits(op))

        digest = b"".join(get_word(memory, names).to_bytes(4, "big")
                          for names in self.layout.digest_names)

        # Sixteen destructive projector rows.  A source M doubles its 16-bit
        # payload, exposing the old MSB as a one-bit tag; Q appends that tag
        # with the MA leaf ``2*WW + tag``.  The final sixteen tail cells shift
        # q left modulo 2^16, so visible RGB needs only Prev1>32767.  Source
        # columns are the fixed physical halfword order documented in trace.
        projected = {name: value & 0xFFFF for name, value in memory.items()}
        projector_sources = sorted(
            (name for names in self.layout.digest_names for name in names),
            key=lambda name: self.layout.by_name[name].mx,
        )
        bitmap: List[List[int]] = []
        for _r in range(16):
            qbits = 0
            for name in projector_sources:
                payload = projected[name]
                qbits = ((qbits << 1) | (payload >> 15)) & 0xFFFF
                projected[name] = (payload << 1) & 0xFFFF
            bitmap.append([
                (qbits >> (15 - column)) & 1 for column in range(16)
            ])
        trace = {
            "rows": len(self.layout.rows),
            "width": self.layout.used_width,
            "height_with_projector": self.layout.image_height,
            "slots": len(self.layout.slots),
            "cell_geometry_checks": cell_checks,
            "stream_source_reads": stream_source_reads,
            "naive_controller_bound_bits": max_q_bits,
            "projector_columns": projector_sources,
            "projector_bitmap": bitmap,
            "memory": memory,
        }
        return digest, trace


def expected_digest_from_prefix(prefix: bytes) -> Tuple[Tuple[int, ...], int, bytes]:
    """Return aperture midstate, file bit length and hashlib oracle digest."""
    state = midstate_for_block_aligned_prefix(prefix)
    bit_length = 8 * (len(prefix) + 40)
    aperture = (b"".join(x.to_bytes(4, "big") for x in state) +
                bit_length.to_bytes(8, "big"))
    complete = prefix + aperture
    return state, bit_length, hashlib.sha256(complete).digest()


def compile_netlist(layout: Layout) -> Dict[str, object]:
    """Emit an auditable symbolic netlist for later MA-tree lowering."""
    return {
        "format": "jxl-sha-raster-netlist-v1",
        "geometry": {
            "width": WIDTH,
            "machine_rows": layout.machine_rows,
            "projector_rows": 16,
            "height": layout.image_height,
            "group_size_shift": 3,
            "one_group": True,
        },
        "channels": [
            {
                "name": "budget_padding_c0_c58",
                "hidden": True,
                "count": 59,
                "predictor": "Set 0",
            },
            {
                "name": "input_bus_c59",
                "hidden": True,
                "entropy_cells": {"x0": 0, "width": 16, "height": 20, "count": 320,
                                  "alphabet": {"0": 1, "1": -2}},
            },
            {
                "name": "machine_c60",
                "hidden": True,
                "runway": {"x0": 0, "x1": 15, "input": "Prev1"},
                "tape": {
                    "x0": TAPE_X0,
                    "pairs": len(layout.slots),
                    "M": "16-bit memory; reads q from W and old memory from N; source visits add a unary-op tag",
                    "Q": "consumes tagged M directly from W and incoming q from WW; leaf WW+offset",
                    "reducer_pairs": len(layout.reductions),
                },
            },
            {
                "name": "projector_rgb_c61_c63",
                "hidden": False,
                "mapping": "D0..D7 halfwords become columns; bit 15..0 become sixteen rows via Prev1",
            },
        ],
        "slots": [
            {"index": s.index, "name": s.name, "mx": s.mx, "qx": s.qx}
            for s in layout.slots
        ],
        "rows": [
            {
                "y": y,
                "kind": r.kind,
                "target": list(r.target),
                "round": r.round_index,
                "nibble": r.nibble,
                "halfword": r.halfword,
                "sources": list(r.sources),
                "dependency_check": "all source M.x < all target M.x",
            }
            for y, r in enumerate(layout.rows)
        ],
        "lowering_status": {
            "raster_placement": "complete",
            "semantic_simulator": "complete",
            "ma_tree_integer_state_encoding": "not yet lowered",
            "reason": (
                "round rows have a conservative 27-bit live product state after eager Ch/Maj "
                "folding; a naive scalar threshold LUT can exceed the 2^22 MA-tree limit.  A "
                "field-ordered W+offset encoding or split scratch state is still required"
            ),
        },
    }


def emit_projector_tree(layout: Layout) -> str:
    """Emit a valid jxl_from_tree DSL scaffold for the channel geometry.

    It intentionally emits only the fixed geometry/default-zero tree.  The
    symbolic netlist is the source for the machine subtree; calling this a
    finished JXL would be dishonest because jxl_from_tree itself forces all
    residual tokens to zero and cannot create the 256-bit entropy aperture.
    """
    return "\n".join((
        f"Width {WIDTH}",
        f"Height {layout.image_height}",
        "Bitdepth 31",
        "GroupShift 3",
        "HiddenChannel 61",
        "RCT 0",
        "- Set 0",
        "",
    ))


def emit_text_netlist(layout: Layout) -> str:
    """Compact line-oriented form of the symbolic raster netlist."""
    lines = [
        "JXL_SHA_RASTER_NETLIST 1",
        f"GEOMETRY width={WIDTH} height={layout.image_height} group_shift=3",
        "CHANNEL pad hidden c[0:59] value=0",
        "CHANNEL input_bus hidden c=59 entropy=x[0:16],y[0:20] alphabet=252,249",
        "CHANNEL machine hidden c=60 runway=x[0:16]:Prev1 tape=x[16:1024]",
        "CELLPAIR M=TAG_OR_WRITE(W,N) Q=FST(WW,W)",
        "CHANNEL projector visible c[61:64] source=Prev1 bitmap=16x16",
    ]
    for slot in layout.slots:
        lines.append(f"SLOT {slot.index:03d} {slot.name} Q=({slot.qx}) M=({slot.mx})")
    for y, row in enumerate(layout.rows):
        src = ",".join(row.sources) or "-"
        dst = ",".join(row.target)
        attrs = []
        if row.round_index is not None:
            attrs.append(f"round={row.round_index}")
        if row.nibble is not None:
            attrs.append(f"nibble={row.nibble}")
        if row.halfword is not None:
            attrs.append(f"halfword={row.halfword}")
        lines.append(
            f"ROW {y:03d} {row.kind} {' '.join(attrs)} READ={src} WRITE={dst}"
        )
    lines.append("TAIL rows=16 machine=N")
    lines.append("LOWERING MA_INTEGER_STATE=pending max_live_bits=27 tree_limit_bits=22")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix-blocks", type=int, default=2,
                        help="length of deterministic test prefix in 64-byte blocks")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--trace-cells", action="store_true")
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    prefix = bytes(rng.randrange(256) for _ in range(64 * args.prefix_blocks))
    state, bit_length, oracle = expected_digest_from_prefix(prefix)
    layout = build_layout(bit_length)
    got, trace = RasterMachine(layout).run(state, args.trace_cells)
    if got != oracle:
        raise SystemExit(f"digest mismatch: {got.hex()} != {oracle.hex()}")

    result = {
        "midstate": "".join(f"{x:08x}" for x in state),
        "bit_length": bit_length,
        "digest": got.hex(),
        "trace": {k: v for k, v in trace.items() if k != "memory"},
        "netlist": compile_netlist(layout),
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        args.out.with_suffix(".tree").write_text(emit_projector_tree(layout))
        args.out.with_suffix(".net").write_text(emit_text_netlist(layout))
    print(json.dumps({k: v for k, v in result.items() if k != "netlist"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
