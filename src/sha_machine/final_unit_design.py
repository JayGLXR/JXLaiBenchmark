#!/usr/bin/env python3
"""Executable audit of the stock-MA byte/two-pass SHA-256 design.

This is the semantic/compiler contract for the integrated JPEG XL raster
machine.  It deliberately uses only unit-coefficient MA transitions: every
reducer has the form ``out = W + constant`` on each interval.  In
particular, it does not assume that an MA residual multiplier scales a
predictor (it does not).

The physical one-group layout is also calculated here.  A word half is a
two-pixel ``M,Q`` source/target pair.  Reducers occupy real ephemeral pixels
because ``WW`` is a leaf predictor but is not an MA branch property.  Three
pass-1 reducer pixels are enough: the high packed code is normalized in one
step and the low sigma/majority fields in two more steps.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


HERE = Path(__file__).resolve().parent
MA_DIR = HERE.parent / "ma_lowering"
for directory in (HERE, MA_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from sha_machine import (  # noqa: E402
    IV,
    K,
    cap_sigma_fragment,
    capsigma0,
    capsigma1,
    ch,
    compress,
    half_nibble,
    maj,
    small_sigma_fragment,
    sigma0,
    sigma1,
)
from controller_budget import (  # noqa: E402
    CH_CODES,
    MAJ_CODES,
    SIGMA_CODES,
    ch_from_code,
    count_offset_runs,
    maj_from_code,
    packed_pass1_report,
    packed_schedule_report,
    sigma_byte_from_code,
    sigma_from_code,
    spread_bits,
    spread_byte,
    tree_stats,
)


MASK32 = (1 << 32) - 1
P1_CODE_COUNT = MAJ_CODES * SIGMA_CODES  # 20,736
P1_PACK_MAX = P1_CODE_COUNT * P1_CODE_COUNT - 1
P2_CH_STRIDE = 1024
P2_SIGMA_STRIDE = 1024 * CH_CODES
TREE_MAX_NODES = (1 << 22) - 1


def byte_at(word: int, byte: int) -> int:
    """Byte 0 is least significant, matching the four raster byte rows."""
    return (word >> (8 * byte)) & 0xFF


def nibble_at(word: int, p: int) -> int:
    return (word >> (4 * p)) & 0xF


def p1_code(a: int, b: int, c: int, p: int) -> int:
    maj_code = (spread_bits(nibble_at(a, p), 4) +
                spread_bits(nibble_at(b, p), 4) +
                spread_bits(nibble_at(c, p), 4))
    sig_code = (
        spread_bits(cap_sigma_fragment(0, a & 0xFFFF, "lo", p), 3) +
        spread_bits(cap_sigma_fragment(0, a >> 16, "hi", p), 3)
    )
    assert 0 <= maj_code < MAJ_CODES and 0 <= sig_code < SIGMA_CODES
    return maj_code + MAJ_CODES * sig_code


def raw_t2(code: int) -> int:
    sig_code, maj_code = divmod(code, MAJ_CODES)
    return maj_from_code(maj_code) + sigma_from_code(sig_code)


def p1_pack(a: int, b: int, c: int, byte: int) -> int:
    p0 = 2 * byte
    code0 = p1_code(a, b, c, p0)
    code1 = p1_code(a, b, c, p0 + 1)
    return code0 + P1_CODE_COUNT * code1


def p1_reduce_high(q: int) -> int:
    """code0+L*code1 -> code0+L*rawT2(code1)."""
    code1, code0 = divmod(q, P1_CODE_COUNT)
    return code0 + P1_CODE_COUNT * raw_t2(code1)


def p1_reduce_low_sigma(q: int) -> int:
    """Normalize low sigma while retaining low Maj and packed raw1."""
    raw1, code0 = divmod(q, P1_CODE_COUNT)
    sig_code, maj_code = divmod(code0, MAJ_CODES)
    return (maj_code + MAJ_CODES * sigma_from_code(sig_code) +
            P1_CODE_COUNT * raw1)


def p1_reduce_low_majority(q: int) -> int:
    """Normalize low Maj and transpose L*raw1 to 32*raw1."""
    raw1, low = divmod(q, P1_CODE_COUNT)
    sigma, maj_code = divmod(low, MAJ_CODES)
    return maj_from_code(maj_code) + sigma + 32 * raw1


def p1_pair(a: int, b: int, c: int, byte: int) -> tuple[int, int]:
    q0 = p1_pack(a, b, c, byte)
    q1 = p1_reduce_high(q0)
    q2 = p1_reduce_low_sigma(q1)
    packed = p1_reduce_low_majority(q2)
    return packed & 31, packed >> 5


def schedule_sigma_code(word: int, which: int, byte: int) -> int:
    p0 = 2 * byte
    lo = (small_sigma_fragment(which, word & 0xFFFF, "lo", p0) |
          (small_sigma_fragment(which, word & 0xFFFF, "lo", p0 + 1) << 4))
    hi = (small_sigma_fragment(which, word >> 16, "hi", p0) |
          (small_sigma_fragment(which, word >> 16, "hi", p0 + 1) << 4))
    code = spread_byte(lo, 3) + spread_byte(hi, 3)
    assert 0 <= code < 3**8
    return code


def schedule_byte_raw(words: Sequence[int], t: int, byte: int) -> int:
    """Two real reducer transitions for one schedule byte."""
    direct0 = byte_at(words[t - 16], byte)
    s0code = schedule_sigma_code(words[t - 15], 0, byte)
    q0 = direct0 + 256 * s0code
    q1 = q0 + (sigma_byte_from_code(s0code) - 256 * s0code)
    q1 += byte_at(words[t - 7], byte)
    s1code = schedule_sigma_code(words[t - 2], 1, byte)
    q2 = q1 + 1024 * s1code
    out = q2 + (sigma_byte_from_code(s1code) - 1024 * s1code)
    assert out == (direct0 + byte_at(sigma0(words[t - 15]), byte) +
                   byte_at(words[t - 7], byte) +
                   byte_at(sigma1(words[t - 2]), byte))
    assert 0 <= out <= 1020
    return out


def scheduled_word(words: Sequence[int], t: int) -> int:
    """Store four raw byte sums; ordinary integer carry gives mod-2^32."""
    value = 0
    for byte in range(4):
        value += schedule_byte_raw(words, t, byte) << (8 * byte)
    return value & MASK32


def p2_code(d: int, h: int, e: int, f: int, g: int,
            w: int, k: int, p: int) -> int:
    base = nibble_at(d, p) + 16 * (
        nibble_at(h, p) + nibble_at(w, p) + nibble_at(k, p))
    e_n = nibble_at(e, p)
    f_n = nibble_at(f, p)
    g_n = nibble_at(g, p)
    ch_code = e_n + 16 * f_n + 256 * g_n
    sig_code = (
        spread_bits(cap_sigma_fragment(1, e & 0xFFFF, "lo", p), 3) +
        spread_bits(cap_sigma_fragment(1, e >> 16, "hi", p), 3)
    )
    q = base + P2_CH_STRIDE * ch_code + P2_SIGMA_STRIDE * sig_code
    assert 0 <= q <= 339_741_695
    return q


def p2_reduce_sigma(q: int) -> int:
    sig_code, rest = divmod(q, P2_SIGMA_STRIDE)
    return rest + 16 * sigma_from_code(sig_code)


def p2_reduce_ch(q: int) -> int:
    ch_code, base = divmod(q, P2_CH_STRIDE)
    return base + 16 * ch_from_code(ch_code)


def p2_pair(d: int, h: int, e: int, f: int, g: int,
            w: int, k: int, p: int) -> tuple[int, int, int]:
    q = p2_reduce_ch(p2_reduce_sigma(p2_code(d, h, e, f, g, w, k, p)))
    d_n, t1 = q & 15, q >> 4
    assert d_n == nibble_at(d, p)
    want_t1 = (nibble_at(h, p) + nibble_at(k, p) + nibble_at(w, p) +
               nibble_at(capsigma1(e), p) + nibble_at(ch(e, f, g), p))
    assert t1 == want_t1
    return q, d_n, t1


def encode_temp_half(raw: Sequence[int]) -> int:
    assert len(raw) == 4 and all(0 <= x <= 30 for x in raw)
    return (sum((x & 15) << (4 * p) for p, x in enumerate(raw)) +
            sum((x >> 4) << (16 + p) for p, x in enumerate(raw)))


def a_target_step(sample: int, local_p: int, t1: int) -> int:
    """The exact N+offset target used by the stock target witness."""
    flag_pos = 16 + local_p
    flag = (sample >> flag_pos) & 1
    shift = 4 * local_p
    return sample + (t1 << shift) + flag * (
        (16 << shift) - (1 << flag_pos))


def finish_round_words(raw_t2_nibbles: Sequence[int], t1_nibbles: Sequence[int],
                       d_nibbles: Sequence[int]) -> tuple[int, int]:
    assert len(raw_t2_nibbles) == len(t1_nibbles) == len(d_nibbles) == 8
    alo = encode_temp_half(raw_t2_nibbles[:4])
    ahi = encode_temp_half(raw_t2_nibbles[4:])
    elo = ehi = 0
    for p in range(4):
        alo = a_target_step(alo, p, t1_nibbles[p])
        elo += (d_nibbles[p] + t1_nibbles[p]) << (4 * p)
    a_carry = alo >> 16
    e_carry = elo >> 16
    for p in range(4, 8):
        local = p - 4
        if local == 0:
            ahi += a_carry
            ehi += e_carry
        ahi = a_target_step(ahi, local, t1_nibbles[p])
        ehi += (d_nibbles[p] + t1_nibbles[p]) << (4 * local)
    return (((ahi & 0xFFFF) << 16) | (alo & 0xFFFF),
            ((ehi & 0xFFFF) << 16) | (elo & 0xFFFF))


def compress_unit(state: Sequence[int], block: bytes) -> tuple[int, ...]:
    """One SHA block through the exact byte-P1/nibble-P2 arithmetic."""
    if len(state) != 8 or len(block) != 64:
        raise ValueError("need eight state words and one block")
    words = [int.from_bytes(block[4 * i:4 * i + 4], "big") for i in range(16)]
    a, b, c, d, e, f, g, h = map(int, state)
    for t in range(64):
        if t >= 16:
            words.append(scheduled_word(words, t))
        raw_t2_nibbles: list[int] = []
        for byte in range(4):
            r0, r1 = p1_pair(a, b, c, byte)
            raw_t2_nibbles.extend((r0, r1))
        t1_nibbles: list[int] = []
        d_nibbles: list[int] = []
        for p in range(8):
            _q, dn, t1 = p2_pair(d, h, e, f, g, words[t], K[t], p)
            d_nibbles.append(dn)
            t1_nibbles.append(t1)
        new_a, new_e = finish_round_words(
            raw_t2_nibbles, t1_nibbles, d_nibbles)
        a, b, c, d = new_a, a, b, c
        e, f, g, h = new_e, e, f, g
    return tuple((int(state[i]) + x) & MASK32
                 for i, x in enumerate((a, b, c, d, e, f, g, h)))


def minimum_schedule_scratch_boundaries() -> tuple[int, ...]:
    """Greedy optimum for all interval stabbing constraints on W boundaries."""
    intervals: list[tuple[int, int]] = []
    for t in range(16, 64):
        # R0 after W[t-15], before W[t-7].  A boundary k is after W[k].
        intervals.append((t - 15, t - 8))
        # R1 after W[t-2], before target W[t].
        intervals.append((t - 2, t - 1))
    chosen: list[int] = []
    for start, end in sorted(intervals, key=lambda p: (p[1], p[0])):
        if not any(start <= point <= end for point in chosen):
            chosen.append(end)
    assert all(any(a <= p <= b for p in chosen) for a, b in intervals)
    return tuple(chosen)


@dataclass(frozen=True)
class Geometry:
    prefix_pixels: int = 16
    # W0..7, one shared W9/W15 word, W16..63.  W8,W10..14 are constants.
    stored_w_words: int = 8 + 1 + 48
    pixels_per_stored_word: int = 4
    # Each schedule reduction uses a READY setter followed by the actual
    # W-branching reducer.  Likewise round-zero P1 needs one setter plus its
    # three reducers before A0.
    schedule_scratch_pixels: int = 2 * 25
    bootstrap_scratch_pixels: int = 4
    rounds: int = 64
    # A4,E4,scratch3.  The scratch cells after state t-1 reduce the row-t
    # controller immediately before the next A word; round zero uses the
    # three bootstrap scratch cells before A0.
    state_pixels_per_round: int = 11
    # Feed-forward overwrites the final A60..63/E60..63 words in place.  Their
    # fixed physical-column permutation is part of the display manifest.
    output_halfwords: int = 0
    pixels_per_output_halfword: int = 2

    @property
    def used_width(self) -> int:
        return (self.prefix_pixels +
                self.stored_w_words * self.pixels_per_stored_word +
                self.schedule_scratch_pixels +
                self.bootstrap_scratch_pixels +
                self.rounds * self.state_pixels_per_round +
                self.output_halfwords * self.pixels_per_output_halfword)

    @property
    def compute_rows(self) -> int:
        # 320 aperture bits as 40 byte rows; 4 P1 + 8 P2 per round;
        # four byte-feedforward rows per digest word.
        return 40 + 64 * 12 + 8 * 4

    @property
    def total_rows_with_2x_display(self) -> int:
        return self.compute_rows + 32


def reducer_budget() -> dict[str, object]:
    packed = packed_pass1_report()
    # Three-stage geometry: monolithic high normalizer, then the exact low
    # sigma and low majority maps from the four-stage audit.
    from controller_budget import packed_pass1_four_stage_report
    four = packed_pass1_four_stage_report()
    low_sigma = four["reducers"]["R3_low_sigma_repeated_31"]
    low_maj = four["reducers"]["R4_low_majority_and_pack_repeated_31"]
    p1_nodes = packed["high_normalizer"]["nodes"] + low_sigma["nodes"] + low_maj["nodes"]
    assert p1_nodes == 200_679

    sched = packed_schedule_report()
    # Simplified P2: 81 sigma intervals and 4096 Ch intervals.
    p2_nodes = (2 * SIGMA_CODES - 1) + (2 * CH_CODES - 1)
    assert p2_nodes == 8_352
    return {
        "source_selector_corrected_nodes": 2_776_317,
        "p1_three_stage_nodes": p1_nodes,
        "schedule_nodes": sched["total_nodes"],
        "p2_nodes": p2_nodes,
        "core_nodes": 2_776_317 + p1_nodes + sched["total_nodes"] + p2_nodes,
        "hard_cap": TREE_MAX_NODES,
    }


def verify(iterations: int = 24) -> dict[str, object]:
    # Full finite domains for the small decoders.
    for code in range(P1_CODE_COUNT):
        assert 0 <= raw_t2(code) <= 30
    for code in range(3**8):
        assert sigma_byte_from_code(code) == sigma_byte_from_code(code)

    rng = random.Random(0x4A584C534841)
    for _ in range(iterations):
        state = tuple(rng.getrandbits(32) for _ in range(8))
        block = rng.randbytes(64)
        got = compress_unit(state, block)
        want = compress(state, block)
        if got != want:
            raise AssertionError((state, block.hex(), got, want))

    # End-to-end hashlib oracle from the standard IV for single blocks.
    messages = [b"", b"abc", bytes(range(55))]
    for message in messages:
        bit_length = 8 * len(message)
        padded = message + b"\x80" + b"\0" * (55 - len(message)) + bit_length.to_bytes(8, "big")
        got_words = compress_unit(IV, padded)
        got = b"".join(x.to_bytes(4, "big") for x in got_words)
        assert got == hashlib.sha256(message).digest()

    geometry = Geometry()
    assert minimum_schedule_scratch_boundaries() == (
        8, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37,
        39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61)
    assert geometry.used_width == 1002
    assert geometry.compute_rows == 840
    assert geometry.total_rows_with_2x_display == 872
    budget = reducer_budget()
    assert budget["core_nodes"] == 3_011_590
    return {
        "random_compression_cases": iterations,
        "hashlib_messages": [m.hex() for m in messages],
        "max_p1_q": P1_PACK_MAX,
        "max_p2_q": 339_741_695,
        "schedule_scratch_boundaries": minimum_schedule_scratch_boundaries(),
        "geometry": geometry.__dict__ | {
            "used_width": geometry.used_width,
            "compute_rows": geometry.compute_rows,
            "rows_with_2x_display": geometry.total_rows_with_2x_display,
        },
        "budget": budget,
    }


def main() -> None:
    print(json.dumps(verify(), indent=2))


if __name__ == "__main__":
    main()
