#!/usr/bin/env python3
"""Wrap a naked JXL aperture codestream and install its SHA-256 bootstrap.

The input must end in a 40-byte aperture.  The output is a JPEG XL container
whose bytes before that aperture are block aligned.  The aperture is replaced
with ``SHA256 chaining state after prefix || BE64(file bit length)``.  A decoder
machine can consequently finish the file hash with exactly one compression
block, without collision search or a fixed-point iteration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import tempfile
from pathlib import Path


MASK32 = 0xFFFFFFFF
IV = (
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
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


def rotr(value: int, amount: int) -> int:
    return ((value >> amount) | (value << (32 - amount))) & MASK32


def compress(state: tuple[int, ...], block: bytes) -> tuple[int, ...]:
    if len(block) != 64:
        raise ValueError("compression block must contain 64 bytes")
    words = list(struct.unpack(">16I", block))
    for index in range(16, 64):
        x = words[index - 15]
        y = words[index - 2]
        s0 = rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3)
        s1 = rotr(y, 17) ^ rotr(y, 19) ^ (y >> 10)
        words.append((words[index - 16] + s0 + words[index - 7] + s1) & MASK32)
    a, b, c, d, e, f, g, h = state
    for index, word in enumerate(words):
        big1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)
        choose = (e & f) ^ ((~e) & g)
        t1 = (h + big1 + choose + K[index] + word) & MASK32
        big0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)
        majority = (a & b) ^ (a & c) ^ (b & c)
        t2 = (big0 + majority) & MASK32
        a, b, c, d, e, f, g, h = (
            (t1 + t2) & MASK32, a, b, c,
            (d + t1) & MASK32, e, f, g,
        )
    return tuple((left + right) & MASK32 for left, right in zip(state, (a, b, c, d, e, f, g, h)))


def chaining_state(prefix: bytes) -> tuple[int, ...]:
    if len(prefix) % 64:
        raise ValueError("prefix is not SHA-256 block aligned")
    state = IV
    for offset in range(0, len(prefix), 64):
        state = compress(state, prefix[offset:offset + 64])
    return state


def box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", len(payload) + 8, kind) + payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="naked codestream with a 40-byte EOF aperture")
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--djxl", type=Path, help="optional stock-decoder smoke test")
    args = parser.parse_args()

    source = args.input.read_bytes()
    if len(source) < 40:
        raise SystemExit("input is shorter than the 40-byte aperture")
    codestream_prefix = source[:-40]

    signature = bytes.fromhex("0000000c4a584c200d0a870a")
    file_type = box(b"ftyp", b"jxl " + bytes(4) + b"jxl ")
    jxlc_header = struct.pack(">I4s", 0, b"jxlc")
    fixed = signature + file_type
    # A legal unknown/free box is at least eight bytes.  Its total size is
    # selected so that the final entropy aperture starts on a SHA block edge.
    minimum = 8
    free_size = minimum + ((-(len(fixed) + minimum + len(jxlc_header) + len(codestream_prefix))) % 64)
    free = box(b"free", bytes(free_size - 8))
    prefix = fixed + free + jxlc_header + codestream_prefix
    assert len(prefix) % 64 == 0

    total_size = len(prefix) + 40
    state = chaining_state(prefix)
    midstate = b"".join(struct.pack(">I", value) for value in state)
    suffix = midstate + struct.pack(">Q", total_size * 8)
    result = prefix + suffix
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(result)

    if args.djxl:
        with tempfile.TemporaryDirectory(prefix="jxl-self-hash-") as directory:
            # The machine image uses 31-bit modular samples.  PFM supports
            # the decoder's floating-point output path, whereas PPM requests
            # an integer output depth that stock djxl rejects for this file.
            decoded = Path(directory) / "decoded.pfm"
            completed = subprocess.run(
                [str(args.djxl), "--num_threads=0", "--quiet", str(args.output), str(decoded)],
                capture_output=True,
                text=True,
            )
            if completed.returncode:
                raise SystemExit(completed.stdout + completed.stderr)

    final_block = suffix + b"\x80" + bytes(15) + struct.pack(">Q", total_size * 8)
    machine_digest = b"".join(struct.pack(">I", value) for value in compress(state, final_block))
    actual_digest = hashlib.sha256(result).digest()
    if machine_digest != actual_digest:
        raise AssertionError("one-block bootstrap does not match hashlib")

    manifest = {
        "format": "jxl-sha256-bootstrap-v1",
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "size_bytes": len(result),
        "prefix_bytes": len(prefix),
        "prefix_blocks": len(prefix) // 64,
        "free_box_bytes": free_size,
        "aperture_offset": len(prefix),
        "aperture_bytes": 40,
        "aperture_layout": "BE32(midstate[0..7]) || BE64(unpadded file bit length)",
        "midstate_hex": midstate.hex(),
        "file_bit_length": total_size * 8,
        "sha256": actual_digest.hex(),
        "one_block_machine_digest": machine_digest.hex(),
        "stock_decode": "passed" if args.djxl else "not requested",
    }
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
