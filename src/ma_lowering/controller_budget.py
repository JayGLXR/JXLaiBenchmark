#!/usr/bin/env python3
"""Executable budget audit for the scalar JPEG XL SHA-256 controller.

This file deliberately models *stock* JPEG XL MA-leaf semantics.  For an MA
leaf with predictor ``W``, offset ``o``, multiplier ``m`` and decoded residual
``r``, the reconstructed sample is::

    W + o + m*r

The multiplier belongs to the entropy residual; it does not multiply ``W``.
In particular all zero-residual leaves are unit-coefficient affine maps
``W + o``.

The audit covers two controller designs:

* ``centered_transpose`` is the compact one-row A-to-E transpose.  Its
  arithmetic works if a scaled W predictor existed.  Stock MA needs an exact
  2,031,616-entry LUT instead, which is counted here.
* ``two_pass`` computes T2 = Maj + Sigma0 on a first round row and the A/E
  correlated pair on a second.  It uses only unit-coefficient reducers and all
  controller values remain signed-31-bit safe.  The script exhaustively checks
  every finite nonlinear code and every aggregate A/E sum.

The selector audit compares retaining the old 39 raw halfword projections
with pruning raw sigma maps that the two-pass design no longer consumes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple


HERE = Path(__file__).resolve().parent
SHA_DIR = HERE.parent / "sha_machine"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(SHA_DIR) not in sys.path:
    sys.path.insert(0, str(SHA_DIR))

from sha_machine import (  # noqa: E402
    cap_sigma_fragment,
    ch,
    half_nibble,
    maj,
    small_sigma_fragment,
)
from tagged_selector import constant_runs, unique_sha_maps  # noqa: E402


INT31_MAX = (1 << 31) - 1
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1
TREE_MAX_NODES = (1 << 22) - 1
HALFWORD_COUNT = 1 << 16

SIGMA_RADIX = 3
SIGMA_CODES = SIGMA_RADIX ** 4       # 81
MAJ_RADIX = 4
MAJ_CODES = MAJ_RADIX ** 4           # 256
CH_CODES = 16 ** 3                    # 4096

SCHEDULE_DIRECT_RADIX = 64
SCHEDULE_SIGMA1_STRIDE = SCHEDULE_DIRECT_RADIX * SIGMA_CODES  # 5184

P1_SIGMA_STRIDE = MAJ_CODES          # 256

P2_LOW_RADIX = 4096
P2_CH_STRIDE = P2_LOW_RADIX
P2_SIGMA_STRIDE = P2_LOW_RADIX * CH_CODES  # 2^24


def spread_bits(nibble: int, radix: int) -> int:
    """Put the four bits of ``nibble`` in independent radix digits."""
    assert 0 <= nibble < 16
    return sum(((nibble >> bit) & 1) * radix ** bit for bit in range(4))


def spread_byte(value: int, radix: int = 3) -> int:
    """Eight independent radix digits, expressed as two four-bit groups."""
    assert 0 <= value < 256
    return spread_bits(value & 0xF, radix) + radix ** 4 * spread_bits(
        value >> 4, radix)


def sigma_byte_from_code(code: int) -> int:
    assert 0 <= code < 3 ** 8
    out = 0
    for bit in range(8):
        code, digit = divmod(code, 3)
        assert 0 <= digit <= 2
        out |= (digit & 1) << bit
    assert code == 0
    return out


def sigma_from_code(code: int) -> int:
    """XOR two bit fragments encoded as four base-3 population counts."""
    assert 0 <= code < SIGMA_CODES
    out = 0
    for bit in range(4):
        code, digit = divmod(code, SIGMA_RADIX)
        assert 0 <= digit <= 2
        out |= (digit & 1) << bit
    assert code == 0
    return out


def maj_from_code(code: int) -> int:
    """Majority of three nibbles encoded as four base-4 bit counts."""
    assert 0 <= code < MAJ_CODES
    out = 0
    for bit in range(4):
        code, digit = divmod(code, MAJ_RADIX)
        assert 0 <= digit <= 3
        out |= int(digit >= 2) << bit
    assert code == 0
    return out


def ch_from_code(code: int) -> int:
    """Ch(e,f,g), with dense code e + 16*f + 256*g."""
    assert 0 <= code < CH_CODES
    g, rem = divmod(code, 256)
    f, e = divmod(rem, 16)
    return ch(e, f, g) & 0xF


def stock_leaf(w: int, offset: int, multiplier: int = 1,
               residual: int = 0) -> int:
    """The reconstruction equation in libjxl's modular MA decoder."""
    return w + offset + multiplier * residual


def count_offset_runs(keys: Sequence[int], outputs: Sequence[int]) -> int:
    """Count exact W+offset intervals over sorted reachable integer keys.

    Invalid gaps may be assigned to either adjacent leaf, so equal offsets on
    adjacent reachable keys coalesce even when the keys are not consecutive.
    """
    assert len(keys) == len(outputs) and keys
    assert all(a < b for a, b in zip(keys, keys[1:]))
    offsets = [out - key for key, out in zip(keys, outputs)]
    return 1 + sum(a != b for a, b in zip(offsets, offsets[1:]))


def tree_stats(runs: int) -> Dict[str, int]:
    return {"runs": runs, "leaves": runs, "nodes": 2 * runs - 1}


def exhaustive_code_checks() -> None:
    # Every base-3 count is reached by a pair of four-bit fragments, and its
    # parity decoder is exactly bitwise XOR.
    seen_sigma = set()
    for left in range(16):
        for right in range(16):
            code = spread_bits(left, 3) + spread_bits(right, 3)
            seen_sigma.add(code)
            assert sigma_from_code(code) == (left ^ right)
    assert seen_sigma == set(range(SIGMA_CODES))

    # Every base-4 count is reached by three nibbles and reduces to Maj.
    seen_maj = set()
    for a in range(16):
        for b in range(16):
            for c in range(16):
                code = (spread_bits(a, 4) + spread_bits(b, 4) +
                        spread_bits(c, 4))
                seen_maj.add(code)
                assert maj_from_code(code) == (maj(a, b, c) & 0xF)
    assert seen_maj == set(range(MAJ_CODES))

    # The dense e,f,g code covers exactly all 4096 Ch inputs.
    for e in range(16):
        for f in range(16):
            for g in range(16):
                code = e + 16 * f + 256 * g
                assert ch_from_code(code) == (ch(e, f, g) & 0xF)


def centered_transpose_report() -> Dict[str, object]:
    """Audit the abandoned single-row centered A-to-E transpose."""
    center = 3968

    # u = majCode + 256*sigCode is dense: 256*81 = 20,736 states.
    u_offsets: List[int] = []
    for sig_code in range(SIGMA_CODES):
        sig = sigma_from_code(sig_code)
        for maj_code in range(MAJ_CODES):
            u = maj_code + MAJ_CODES * sig_code
            s = maj_from_code(maj_code) + sig
            u_offsets.append((1 << 16) * (s - u) - 256 * center)
    r1_runs = 1 + sum(a != b for a, b in zip(u_offsets, u_offsets[1:]))

    q1_min = 256 * (0 - center)
    q1_max = 255 + 256 * (7935 - center)
    assert q1_min == -1_015_808 and q1_max == 1_015_807

    # q1 = chPair + 256*(hds-center), hds = D + 16*H + 256*S.
    # Desired q2 = pairNoK + 2048*chPair.  With a unit W predictor every
    # reachable q1 has a distinct offset; this is the exact stock-MA cost.
    keys: List[int] = []
    outputs: List[int] = []
    fake_scaled_offsets: List[int] = []
    for s in range(31):
        for h in range(16):
            for d in range(16):
                dh = d + 16 * h
                hds = dh + 256 * s
                pair_no_k = 15 + 46 * h + 45 * d + s
                for ch_pair in range(256):
                    q1 = ch_pair + 256 * (hds - center)
                    q2 = pair_no_k + 2048 * ch_pair
                    keys.append(q1)
                    outputs.append(q2)
                fake_scaled_offsets.append(
                    pair_no_k - 2048 * 256 * (hds - center))
    assert keys == list(range(q1_min, q1_max + 1))
    r2_unit_runs = count_offset_runs(keys, outputs)
    assert r2_unit_runs == 2_031_616
    fake_scaled_runs = 1 + sum(
        a != b for a, b in zip(fake_scaled_offsets, fake_scaled_offsets[1:]))

    # Show directly that a leaf multiplier cannot provide the fake affine map.
    witness_w = 123
    witness_offset = 7
    assert stock_leaf(witness_w, witness_offset, 2048, 0) == 130
    assert stock_leaf(witness_w, witness_offset, 2048, 0) != (
        2048 * witness_w + witness_offset)

    r1 = tree_stats(r1_runs)
    r2 = tree_stats(r2_unit_runs)
    return {
        "full_a_q_max": 255 + 256 * (255 + 256 * 20_735),
        "q1_range": [q1_min, q1_max],
        "r1_unit": r1,
        "r2_if_scaled_predictor_existed": tree_stats(fake_scaled_runs),
        "r2_stock_unit_predictor": r2,
        "r1_plus_r2_stock_nodes": r1["nodes"] + r2["nodes"],
        "hard_tree_nodes": TREE_MAX_NODES,
        "nodes_left_after_only_these_two": (
            TREE_MAX_NODES - r1["nodes"] - r2["nodes"]),
        "stock_leaf_multiplier_witness": {
            "W": witness_w,
            "offset": witness_offset,
            "multiplier": 2048,
            "zero_residual_output": stock_leaf(
                witness_w, witness_offset, 2048, 0),
            "incorrect_scaled_W_claim": 2048 * witness_w + witness_offset,
        },
    }


def two_pass_reducer_report() -> Dict[str, object]:
    """Exhaustively verify the unit-coefficient two-pass controller."""
    reducer_runs: Dict[str, int] = {}

    # Schedule:
    # q = direct + 64*s0code + 5184*s1code, direct in 0..30.
    # Strip sigma1, then sigma0.  The low fields remain below their radices.
    sched_r1_keys: List[int] = []
    sched_r1_out: List[int] = []
    for s1code in range(SIGMA_CODES):
        s1 = sigma_from_code(s1code)
        for s0code in range(SIGMA_CODES):
            for direct in range(31):
                q = (direct + SCHEDULE_DIRECT_RADIX * s0code +
                     SCHEDULE_SIGMA1_STRIDE * s1code)
                q1 = direct + s1 + SCHEDULE_DIRECT_RADIX * s0code
                assert q1 < SCHEDULE_SIGMA1_STRIDE
                sched_r1_keys.append(q)
                sched_r1_out.append(q1)
    reducer_runs["schedule_strip_sigma1"] = count_offset_runs(
        sched_r1_keys, sched_r1_out)

    sched_r2_keys: List[int] = []
    sched_r2_out: List[int] = []
    # Use aggregate low=direct+Sigma1.  Every value 0..45 is reachable.
    for s0code in range(SIGMA_CODES):
        for low in range(46):
            q1 = low + SCHEDULE_DIRECT_RADIX * s0code
            out = low + sigma_from_code(s0code)
            assert out <= 60
            sched_r2_keys.append(q1)
            sched_r2_out.append(out)
    reducer_runs["schedule_strip_sigma0"] = count_offset_runs(
        sched_r2_keys, sched_r2_out)

    # Pass 1:
    # q = majCode + 256*sig0Code.  First turn sig0Code into its nibble while
    # retaining its coefficient, then reduce Maj and add the two nibbles.
    p1_r1_keys = list(range(MAJ_CODES * SIGMA_CODES))
    p1_r1_out: List[int] = []
    for q in p1_r1_keys:
        sig_code, maj_code = divmod(q, MAJ_CODES)
        p1_r1_out.append(maj_code + MAJ_CODES * sigma_from_code(sig_code))
    reducer_runs["pass1_sigma0_code_to_nibble"] = count_offset_runs(
        p1_r1_keys, p1_r1_out)

    p1_r2_keys = list(range(MAJ_CODES * 16))
    p1_r2_out: List[int] = []
    for q1 in p1_r2_keys:
        sig, maj_code = divmod(q1, MAJ_CODES)
        t2 = maj_from_code(maj_code) + sig
        assert 0 <= t2 <= 30
        p1_r2_out.append(t2)
    reducer_runs["pass1_maj_code_to_T2"] = count_offset_runs(
        p1_r2_keys, p1_r2_out)

    # Pass 2 reducer offsets depend only on the high nonlinear code, so one
    # interval per code is exact.  Check endpoint separation and reconstruction.
    pair_base_max = 15 + 30 + 45 * 15 + 46 * (15 + 15 + 15)
    assert pair_base_max == 2790
    p2_q_max = (pair_base_max + P2_CH_STRIDE * (CH_CODES - 1) +
                P2_SIGMA_STRIDE * (SIGMA_CODES - 1))
    assert p2_q_max < INT31_MAX

    p2_sig_offsets = [
        46 * sigma_from_code(code) - P2_SIGMA_STRIDE * code
        for code in range(SIGMA_CODES)
    ]
    reducer_runs["pass2_strip_sigma1"] = 1 + sum(
        a != b for a, b in zip(p2_sig_offsets, p2_sig_offsets[1:]))
    p2_ch_offsets = [
        46 * ch_from_code(code) - P2_CH_STRIDE * code
        for code in range(CH_CODES)
    ]
    reducer_runs["pass2_strip_Ch"] = 1 + sum(
        a != b for a, b in zip(p2_ch_offsets, p2_ch_offsets[1:]))

    # Exhaust every correlated final pair.  T1 aggregates H+K+W+Sigma1+Ch.
    # pair = delta + 46*E, with delta=T2-D+15 in 0..45.
    reachable_pairs = set()
    for t2 in range(31):
        for d in range(16):
            for t1 in range(76):
                pair = 15 + t2 + 45 * d + 46 * t1
                delta = pair % 46
                e_out = pair // 46
                a_out = e_out + delta - 15
                assert delta == t2 - d + 15
                assert e_out == d + t1
                assert a_out == t1 + t2
                reachable_pairs.add(pair)

    stats = {name: tree_stats(runs) for name, runs in reducer_runs.items()}
    total_nodes = sum(item["nodes"] for item in stats.values())
    all_offsets: List[int] = []
    all_offsets.extend(out - key for key, out in zip(sched_r1_keys,
                                                     sched_r1_out))
    all_offsets.extend(out - key for key, out in zip(sched_r2_keys,
                                                     sched_r2_out))
    all_offsets.extend(out - key for key, out in zip(p1_r1_keys, p1_r1_out))
    all_offsets.extend(out - key for key, out in zip(p1_r2_keys, p1_r2_out))
    all_offsets.extend(p2_sig_offsets)
    all_offsets.extend(p2_ch_offsets)
    assert min(all_offsets) >= INT32_MIN and max(all_offsets) <= INT32_MAX

    return {
        "schedule_q_max": 30 + 64 * 80 + 5184 * 80,
        "pass1_q_max": 255 + 256 * 80,
        "pass2_pair_base_max": pair_base_max,
        "pass2_q_max": p2_q_max,
        "signed31_margin": INT31_MAX - p2_q_max,
        "final_pair_max": max(reachable_pairs),
        "final_pair_distinct_values": len(reachable_pairs),
        "decoded_A_max": 105,
        "decoded_E_max": 90,
        "offset_range": [min(all_offsets), max(all_offsets)],
        "reducers": stats,
        "total_reducer_nodes": total_nodes,
    }


def packed_pass1_report() -> Dict[str, object]:
    """Audit two T2 nibbles packed into each of four byte-oriented P1 rows.

    ``code0`` and ``code1`` are the dense 20,736-state
    ``majCode + 256*sigCode`` values.  The first reducer normalizes the high
    code; the second repeats the low-code map across 31 high raw-T2 groups.
    The result ``raw0 + 32*raw1`` is carry-free because each raw sum is at
    most 30.  It may be stored as an ordinary integer contribution and split
    when pass 2 consumes the two nibbles.
    """
    code_count = MAJ_CODES * SIGMA_CODES  # 20,736

    def raw_t2(code: int) -> int:
        sig_code, maj_code = divmod(code, MAJ_CODES)
        return maj_from_code(maj_code) + sigma_from_code(sig_code)

    raw = [raw_t2(code) for code in range(code_count)]
    assert min(raw) == 0 and max(raw) == 30

    # q = code0 + 20736*code1 -> code0 + 20736*raw1.
    high_offsets = [code_count * (raw[code] - code)
                    for code in range(code_count)]
    high_runs = 1 + sum(a != b for a, b in
                        zip(high_offsets, high_offsets[1:]))

    # q1 = code0 + 20736*raw1 -> raw0 + 32*raw1.
    low_offsets: List[int] = []
    for raw1 in range(31):
        high_adjust = (32 - code_count) * raw1
        low_offsets.extend(raw0 - code0 + high_adjust
                           for code0, raw0 in enumerate(raw))
    low_runs = 1 + sum(a != b for a, b in
                       zip(low_offsets, low_offsets[1:]))

    # Exhaust the complete 20,736^2 input algebra without materializing all
    # 430 million pairs: both reducer equations are separable in code0/code1.
    for code in range(code_count):
        assert code + high_offsets[code] // code_count == raw[code]
    for raw1 in range(31):
        for code0 in range(code_count):
            q1 = code0 + code_count * raw1
            out = q1 + low_offsets[raw1 * code_count + code0]
            assert out == raw[code0] + 32 * raw1
            assert 0 <= out <= 990

    high = tree_stats(high_runs)
    low = tree_stats(low_runs)
    q_max = code_count * code_count - 1
    q1_max = (code_count - 1) + code_count * 30
    assert q_max == 429_981_695 and q_max < INT31_MAX
    assert q1_max == 642_815
    return {
        "input_equation": "q = code0 + 20736*code1",
        "output_equation": "packed = raw0 + 32*raw1",
        "input_q_max": q_max,
        "after_high_q_max": q1_max,
        "packed_max": 990,
        "high_normalizer": high,
        "low_normalizer_repeated_across_31_groups": low,
        "total_nodes": high["nodes"] + low["nodes"],
        "offset_range": [min(high_offsets + low_offsets),
                         max(high_offsets + low_offsets)],
    }


def packed_pass1_four_stage_report() -> Dict[str, object]:
    """The four-unit-leaf lowering available after the A source word.

    Splitting each 20,736-state normalization into its sigma and majority
    fields is substantially smaller than the two monolithic normalizers.
    The exact interval count is lower than the rectangular upper bounds
    because adjacent base-code values sometimes have the same W offset.
    """
    code_count = MAJ_CODES * SIGMA_CODES

    def raw_t2(code: int) -> int:
        sig_code, maj_code = divmod(code, MAJ_CODES)
        return maj_from_code(maj_code) + sigma_from_code(sig_code)

    # R1: normalize only high sigmaCode, retaining high majCode and all code0.
    r1_offsets = [
        code_count * MAJ_CODES * (sigma_from_code(sig) - sig)
        for sig in range(SIGMA_CODES)
    ]
    r1_runs = 1 + sum(a != b for a, b in zip(r1_offsets, r1_offsets[1:]))

    # R2: high majCode + 256*Sigma -> raw1.  code0 is a complete low block,
    # so the interval behavior is the 4096-entry high field alone.
    r2_offsets: List[int] = []
    for sigma in range(16):
        for maj_code in range(MAJ_CODES):
            r2_offsets.append(code_count * (
                maj_from_code(maj_code) + sigma - maj_code - 256 * sigma))
    r2_runs = 1 + sum(a != b for a, b in zip(r2_offsets, r2_offsets[1:]))

    # R3: normalize low sigmaCode independently in each of 31 raw1 groups.
    r3_offsets: List[int] = []
    for _raw1 in range(31):
        for sig_code in range(SIGMA_CODES):
            r3_offsets.append(256 * (sigma_from_code(sig_code) - sig_code))
    r3_runs = 1 + sum(a != b for a, b in zip(r3_offsets, r3_offsets[1:]))

    # R4: reduce low majCode and transpose L*raw1 to 32*raw1.
    r4_offsets: List[int] = []
    for raw1 in range(31):
        for sigma in range(16):
            for maj_code in range(MAJ_CODES):
                inp = maj_code + 256 * sigma + code_count * raw1
                out = maj_from_code(maj_code) + sigma + 32 * raw1
                r4_offsets.append(out - inp)
    r4_runs = 1 + sum(a != b for a, b in zip(r4_offsets, r4_offsets[1:]))

    # Exhaust the field transitions.  The first two stages are independent of
    # code0, and the last two are independent of the already-normalized raw1.
    for code1 in range(code_count):
        sig_code, maj_code = divmod(code1, MAJ_CODES)
        after_r1 = maj_code + 256 * sigma_from_code(sig_code)
        after_r2 = maj_from_code(maj_code) + sigma_from_code(sig_code)
        assert after_r2 == raw_t2(code1)
        assert 0 <= after_r1 < 4096 and 0 <= after_r2 <= 30
    for raw1 in range(31):
        for code0 in range(code_count):
            sig_code, maj_code = divmod(code0, MAJ_CODES)
            after_r3 = maj_code + 256 * sigma_from_code(sig_code)
            out = maj_from_code(maj_code) + sigma_from_code(sig_code) + 32 * raw1
            assert out == raw_t2(code0) + 32 * raw1
            assert 0 <= after_r3 < 4096 and 0 <= out <= 990

    reducers = {
        "R1_high_sigma": tree_stats(r1_runs),
        "R2_high_majority": tree_stats(r2_runs),
        "R3_low_sigma_repeated_31": tree_stats(r3_runs),
        "R4_low_majority_and_pack_repeated_31": tree_stats(r4_runs),
    }
    total_nodes = sum(item["nodes"] for item in reducers.values())
    all_offsets = r1_offsets + r2_offsets + r3_offsets + r4_offsets
    assert min(all_offsets) >= INT32_MIN and max(all_offsets) <= INT32_MAX
    return {
        "input_q_max": code_count * code_count - 1,
        "packed_max": 990,
        "reducers": reducers,
        "total_nodes": total_nodes,
        "rectangular_leaf_upper_bounds": [81, 4096, 31 * 81,
                                           31 * 16 * 256],
        "offset_range": [min(all_offsets), max(all_offsets)],
    }


def packed_schedule_report() -> Dict[str, object]:
    """Verify the final two-reducer byte schedule.

    The first direct byte is the complete 0..255 low field under a base-3
    Sigma0 code of radix 256.  R0 immediately strips that code to its XOR
    byte.  The second direct byte then raises the partial raw sum only to 765,
    safely below the 1024 coefficient used for Sigma1.  R1 strips Sigma1 and
    returns the complete raw schedule byte sum (at most 1020).
    """
    code_count = 3 ** 8  # 6561 base-3 population-count bytes
    first_direct_max = 255
    partial_max = 3 * 255
    sigma0_stride = 256
    sigma1_stride = 1024

    parity = [sigma_byte_from_code(code) for code in range(code_count)]
    sigma_runs = 1 + sum(
        (parity[a] - a) != (parity[a + 1] - (a + 1))
        for a in range(code_count - 1))
    assert sigma_runs == 4374

    r0_offsets = [parity[code] - sigma0_stride * code
                  for code in range(code_count)]
    r1_offsets = [parity[code] - sigma1_stride * code
                  for code in range(code_count)]
    r0_runs = 1 + sum(a != b for a, b in zip(r0_offsets, r0_offsets[1:]))
    r1_runs = 1 + sum(a != b for a, b in zip(r1_offsets, r1_offsets[1:]))
    # The large negative code coefficient makes both offsets strictly change.
    assert r0_runs == code_count and r1_runs == code_count

    before_r0_max = first_direct_max + sigma0_stride * (code_count - 1)
    before_r1_max = partial_max + sigma1_stride * (code_count - 1)
    assert before_r0_max == 1_679_615
    assert before_r1_max == 6_718_205
    assert before_r1_max < INT31_MAX

    # Exhaust all base-3 codes individually and all normalized byte pairs.
    for code in range(code_count):
        assert 0 <= parity[code] <= 255
    for code in range(code_count):
        for direct in (0, first_direct_max):
            q0 = direct + sigma0_stride * code
            assert q0 + r0_offsets[code] == direct + parity[code]
        for partial in (0, partial_max):
            q1 = partial + sigma1_stride * code
            assert q1 + r1_offsets[code] == partial + parity[code]

    reducers = {
        "R0_strip_sigma0_and_add_byte": tree_stats(r0_runs),
        "R1_strip_sigma1_and_add_byte": tree_stats(r1_runs),
    }
    return {
        "before_R0_q_max": before_r0_max,
        "before_R1_q_max": before_r1_max,
        "raw_schedule_byte_max": 1020,
        "reducers": reducers,
        "total_nodes": sum(item["nodes"] for item in reducers.values()),
        "geometry": (
            "R0 is after W[t-15] and before the second direct source "
            "W[t-7]; R1 is after W[t-2] and before target W[t]."
        ),
    }


def simplified_pass2_report() -> Dict[str, object]:
    """Audit q=D+16*T1 with Ch/Sigma1 held in high code fields."""
    direct_max = 15 + 16 * (15 + 15 + 15)  # D + 16*(H+K+W)
    ch_stride = 1024
    sigma_stride = ch_stride * CH_CODES
    q_max = direct_max + ch_stride * (CH_CODES - 1) + sigma_stride * 80
    assert direct_max == 735 and q_max == 339_738_335
    assert q_max < INT31_MAX

    sigma_offsets = [
        16 * sigma_from_code(code) - sigma_stride * code
        for code in range(SIGMA_CODES)
    ]
    sigma_runs = 1 + sum(a != b for a, b in
                         zip(sigma_offsets, sigma_offsets[1:]))
    assert sigma_runs == SIGMA_CODES

    # After Sigma1, D+16*(H+K+W+Sigma1) <= 975 < 1024.
    after_sigma_max = 975 + ch_stride * (CH_CODES - 1)
    assert after_sigma_max == 4_194_255 < sigma_stride
    ch_offsets = [
        16 * ch_from_code(code) - ch_stride * code
        for code in range(CH_CODES)
    ]
    ch_runs = 1 + sum(a != b for a, b in zip(ch_offsets, ch_offsets[1:]))
    assert ch_runs == CH_CODES

    # Exhaust all aggregate direct states.  The final controller is exactly
    # D+16*T1; target leaves may recover D=q%16 and T1=q//16.
    final_values = set()
    for d in range(16):
        for hkw in range(46):
            for sigma1 in range(16):
                for ch_value in range(16):
                    t1 = hkw + sigma1 + ch_value
                    q = d + 16 * t1
                    assert q % 16 == d and q // 16 == t1
                    final_values.add(q)
    assert min(final_values) == 0 and max(final_values) == 1215

    reducers = {
        "strip_Sigma1": tree_stats(sigma_runs),
        "strip_Ch": tree_stats(ch_runs),
    }
    return {
        "input_q_max": q_max,
        "after_sigma_q_max": after_sigma_max,
        "final_q_max": max(final_values),
        "final_equation": "q = D + 16*T1",
        "reducers": reducers,
        "total_nodes": sum(item["nodes"] for item in reducers.values()),
        "offset_range": [min(sigma_offsets + ch_offsets),
                         max(sigma_offsets + ch_offsets)],
    }


def visible_bit_extractor_report() -> Dict[str, object]:
    """Count a 16-row Prev1 uint16-to-bitmap extractor exactly."""
    runs_by_bit = [1 << (16 - bit) for bit in range(16)]
    assert runs_by_bit[-1] == 2
    total_runs = sum(runs_by_bit)
    subtrees = sum(2 * runs - 1 for runs in runs_by_bit)
    y_dispatch = 15
    return {
        "runs_by_source_bit_0_through_15": runs_by_bit,
        "sum_runs": total_runs,
        "sixteen_subtree_nodes": subtrees,
        "balanced_y_dispatch_nodes": y_dispatch,
        "total_before_channel_and_domain_guards": subtrees + y_dispatch,
        "green_blue_copy": "each uses one Prev1>0 split and two Set leaves",
        "required_machine_invariant": (
            "On display rows, only digest coordinates are raw uint16; every "
            "other machine sample must be outside that domain so the bit "
            "subtree is selected once by Prev1, not duplicated under x."
        ),
    }


def _selector_requirements(
    packed_pass1: bool = True,
) -> Mapping[Tuple[int, ...], List[str]]:
    """All unary source tables in the final byte-schedule/byte-P1 design."""
    if not packed_pass1:
        raise ValueError("the final selector uses packed byte pass 1")
    groups: Dict[Tuple[int, ...], List[str]] = defaultdict(list)

    def add(name: str, fn: Callable[[int], int]) -> None:
        table = tuple(fn(value) for value in range(HALFWORD_COUNT))
        groups[table].append(name)

    # Four byte schedule rows.  R0 folds Sigma0 into the first direct byte,
    # then a second direct byte raises the partial only to 765; Sigma1 can
    # therefore use coefficient 1024.
    for byte in range(4):
        p0, p1 = 2 * byte, 2 * byte + 1
        direct_half = "lo" if byte < 2 else "hi"
        add(
            f"schedule.direct.byte{byte}.{direct_half}",
            lambda v, p0=p0, p1=p1, half=direct_half:
                half_nibble(v, half, p0) +
                16 * half_nibble(v, half, p1),
        )
        for which, scale in ((0, 256), (1, 1024)):
            for source_half in ("lo", "hi"):
                add(
                    f"schedule.byteSigma{which}.{byte}.{source_half}",
                    lambda v, p0=p0, p1=p1, source_half=source_half,
                    which=which, scale=scale:
                        scale * spread_byte(
                            small_sigma_fragment(
                                which, v, source_half, p0) |
                            (small_sigma_fragment(
                                which, v, source_half, p1) << 4),
                            3),
                )

    # Four byte P1 rows.  A combines its base-4 Maj count with its Sigma0
    # fragment; B and C share the Maj-only table for the active half.
    code_count = MAJ_CODES * SIGMA_CODES
    for byte in range(4):
        p0, p1 = 2 * byte, 2 * byte + 1
        direct_half = "lo" if byte < 2 else "hi"
        for source_half in ("lo", "hi"):
            add(
                f"packedP1.a.byte{byte}.{source_half}",
                lambda v, p0=p0, p1=p1, half=source_half:
                    spread_bits(half_nibble(v, half, p0), 4) +
                    code_count * spread_bits(
                        half_nibble(v, half, p1), 4) +
                    P1_SIGMA_STRIDE * (
                        spread_bits(cap_sigma_fragment(0, v, half, p0), 3) +
                        code_count * spread_bits(
                            cap_sigma_fragment(0, v, half, p1), 3)),
            )
        add(
            f"packedP1.bc.byte{byte}.{direct_half}",
            lambda v, p0=p0, p1=p1, half=direct_half:
                spread_bits(half_nibble(v, half, p0), 4) +
                code_count * spread_bits(half_nibble(v, half, p1), 4),
        )

    # Eight nibble P2 rows, using q=D+16*T1.  E again has two logical roles
    # on one visit and must use a single summed Ch.e+Sigma1 table.
    ch_stride = 1024
    sigma_stride = ch_stride * CH_CODES
    for p in range(8):
        direct_half = "lo" if p < 4 else "hi"
        for label, scale in (
            ("D", 1),
            ("HW", 16),
            ("Ch.f", ch_stride * 16),
            ("Ch.g", ch_stride * 256),
        ):
            add(
                f"pass2.{label}.{p}.{direct_half}",
                lambda v, p=p, half=direct_half, scale=scale:
                    scale * half_nibble(v, half, p),
            )
        for source_half in ("lo", "hi"):
            add(
                f"pass2.e_plus_Sigma1.{p}.{source_half}",
                lambda v, p=p, half=source_half:
                    ch_stride * half_nibble(v, half, p) +
                    sigma_stride * spread_bits(
                        cap_sigma_fragment(1, v, half, p), 3),
            )
    return groups


def selector_report() -> Dict[str, object]:
    old_maps = {unary.table: unary for unary in unique_sha_maps()}
    required = _selector_requirements(packed_pass1=True)
    old_tables = set(old_maps)
    required_tables = set(required)
    added = required_tables - old_tables
    retained = required_tables & old_tables
    removed = old_tables - required_tables

    def runs(tables: Iterable[Tuple[int, ...]]) -> int:
        return sum(len(constant_runs(table)) for table in tables)

    old_runs = runs(old_tables)
    added_runs = runs(added)
    required_runs = runs(required_tables)
    union_runs = runs(old_tables | required_tables)

    families: Dict[str, Dict[str, int]] = {}
    family_tables: Dict[str, set[Tuple[int, ...]]] = defaultdict(set)
    for table, aliases in required.items():
        first = aliases[0]
        if first.startswith("schedule.") or first.startswith("pass"):
            family = ".".join(first.split(".")[:2])
        elif first.startswith("Ch."):
            family = ".".join(first.split(".")[:2])
        else:
            family = first.split(".")[0]
        family_tables[family].add(table)
    for family, tables in sorted(family_tables.items()):
        family_runs = runs(tables)
        families[family] = {
            "unique_tags": len(tables),
            "runs": family_runs,
            "standalone_nodes": 2 * family_runs - 1,
        }

    return {
        "design": "final byte schedule + packed P1 + D+16*T1 P2",
        "old_raw_logical_maps": sum(len(m.aliases) for m in old_maps.values()),
        "old_raw_unique_tags": len(old_tables),
        "required_logical_maps": sum(len(v) for v in required.values()),
        "required_unique_tags": len(required_tables),
        "retained_old_tags": len(retained),
        "new_tags_beyond_old_39": len(added),
        "obsolete_old_tags": len(removed),
        "keep_all_old_plus_new": {
            "unique_tags": len(old_tables | required_tables),
            "runs": union_runs,
            "nodes": 2 * union_runs - 1,
            "fits_hard_limit": 2 * union_runs - 1 <= TREE_MAX_NODES,
        },
        "pruned_actual_two_pass_set": {
            "unique_tags": len(required_tables),
            "runs": required_runs,
            "nodes": 2 * required_runs - 1,
            "fits_hard_limit": 2 * required_runs - 1 <= TREE_MAX_NODES,
        },
        "old_runs": old_runs,
        "added_runs": added_runs,
        "families": families,
    }


def geometry_report() -> Dict[str, object]:
    # Current scalar layout: 20 pack + 48*8 schedule + 64*8 round + 8*8
    # feed-forward = 980 machine rows.  A two-row round doubles only the round
    # term.  GroupShift=3 has a 1024x1024 Modular group.
    old_rows = 20 + 48 * 8 + 64 * 8 + 8 * 8
    two_pass_rows = 20 + 48 * 8 + 2 * 64 * 8 + 8 * 8
    packed_p1_rows = 20 + 48 * 8 + 64 * (8 + 4) + 8 * 8
    fused_byte_rows = 20 + 64 * (8 + 4) + 8 * 8
    final_with_display = fused_byte_rows + 16
    assert old_rows == 980 and two_pass_rows == 1492
    return {
        "current_machine_rows": old_rows,
        "two_pass_machine_rows": two_pass_rows,
        "four_byte_P1_machine_rows": packed_p1_rows,
        "fused_byte_schedule_and_P1_rows": fused_byte_rows,
        "fused_design_spare_rows": 1024 - fused_byte_rows,
        "fused_design_rows_with_16_row_display": final_with_display,
        "fused_design_spare_rows_after_display": 1024 - final_with_display,
        "group_height": 1024,
        "overflow_rows_before_display": two_pass_rows - 1024,
        "four_byte_P1_overflow_rows_before_display": packed_p1_rows - 1024,
        "width_reuse": (
            "No new pixels are intrinsically required: nominal A can store "
            "T2, nominal R.A can be the two pass-2 reducers, nominal E can "
            "store final A, and nominal R.E can store final E.  All prior "
            "round sources remain west."
        ),
        "height_blocker": (
            "P1 and P2 are causally separate full west-to-east scans.  The "
            "extra 512 rows cross the 1024-pixel Modular group boundary, "
            "where N/W state is not continuous."
        ),
    }


def build_report(with_selector: bool = True) -> Dict[str, object]:
    exhaustive_code_checks()
    report: Dict[str, object] = {
        "status": "all finite-code and aggregate transition checks passed",
        "stock_semantics": {
            "equation": "pixel = predictor + offset + residual*multiplier",
            "zero_residual_equation": "pixel = predictor + offset",
            "decoder_source": (
                "work/jxlproto-libjxl/lib/jxl/modular/encoding/"
                "encoding.cc:187-192,480"
            ),
            "tree_source": (
                "work/jxlproto-libjxl/lib/jxl/modular/encoding/"
                "dec_ma.cc:132-147"
            ),
        },
        "centered_transpose": centered_transpose_report(),
        "two_pass": two_pass_reducer_report(),
        "packed_pass1": packed_pass1_report(),
        "packed_pass1_four_stage": packed_pass1_four_stage_report(),
        "packed_schedule": packed_schedule_report(),
        "simplified_pass2": simplified_pass2_report(),
        "visible_bit_extractor": visible_bit_extractor_report(),
        "geometry": geometry_report(),
        "hard_tree_nodes": TREE_MAX_NODES,
    }
    if with_selector:
        report["selector"] = selector_report()
        selector_nodes = report["selector"]["pruned_actual_two_pass_set"]["nodes"]
        core_nodes = (
            selector_nodes +
            report["packed_pass1_four_stage"]["total_nodes"] +
            report["packed_schedule"]["total_nodes"] +
            report["simplified_pass2"]["total_nodes"]
        )
        report["final_core_tree_budget"] = {
            "selector_nodes": selector_nodes,
            "packed_P1_reducer_nodes":
                report["packed_pass1_four_stage"]["total_nodes"],
            "schedule_reducer_nodes": report["packed_schedule"]["total_nodes"],
            "pass2_reducer_nodes": report["simplified_pass2"]["total_nodes"],
            "total_before_routing_targets_and_display": core_nodes,
            "remaining_nodes": TREE_MAX_NODES - core_nodes,
        }
        display_nodes = report["visible_bit_extractor"][
            "total_before_channel_and_domain_guards"]
        subtotal = core_nodes + display_nodes
        height = report["geometry"]["fused_design_rows_with_16_row_display"]
        # dec_modular.cc's image-dependent cap is
        # 1024 + width*height*(3+num_extra_channels)/16, clipped by the hard
        # 2^22-1 node ceiling.  Report the first sufficient pad count.
        extras = max(0, math.ceil(
            (subtotal - 1024) * 16 / (1024 * height) - 3))
        report["final_core_tree_budget"].update({
            "plus_visible_extractor_subtotal": subtotal,
            "remaining_after_visible_before_targets_and_routing":
                TREE_MAX_NODES - subtotal,
            "minimum_extra_channels_for_that_subtotal": extras,
        })
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-selector", action="store_true",
                        help="skip the slower 16-bit unary-map enumeration")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = build_report(with_selector=not args.skip_selector)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
