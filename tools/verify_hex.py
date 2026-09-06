#!/usr/bin/env python3
"""Independently verify the rendered SHA-256 of the hexadecimal JXL hashquine.

The expected pixels are derived only from the exact input-file SHA-256 and the
documented 3x5 font/layout below.  No compiler modules or manifests are read.
Stock djxl applies the image's encoded orientation before writing the PFM.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import struct
import subprocess
import sys
import zlib
from pathlib import Path


FONT = {
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "010", "010", "010"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
    "A": ("010", "101", "111", "101", "101"),
    "B": ("110", "101", "110", "101", "110"),
    "C": ("111", "100", "100", "100", "111"),
    "D": ("110", "101", "101", "101", "110"),
    "E": ("111", "100", "110", "100", "111"),
    "F": ("111", "100", "110", "100", "100"),
}

WIDTH = 1024
HEIGHT = 1024
SCALE = 2
LINE_Y = (4, 16, 28, 40)
CHARS_PER_LINE = 16
OUTPUT_X0 = 4
CHAR_STRIDE = 7
GROUP_GAP = 2


def sha256_file(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.digest()


def glyph_x(character: int) -> int:
    return OUTPUT_X0 + CHAR_STRIDE * character + GROUP_GAP * (character // 4)


def expected_pixels(digest_hex_upper: str) -> set[tuple[int, int]]:
    if len(digest_hex_upper) != 64 or any(c not in FONT for c in digest_hex_upper):
        raise ValueError("expected exactly 64 uppercase hexadecimal characters")
    lit: set[tuple[int, int]] = set()
    for index, character in enumerate(digest_hex_upper):
        line, slot = divmod(index, CHARS_PER_LINE)
        x0 = glyph_x(slot)
        y0 = LINE_Y[line]
        for font_y, row in enumerate(FONT[character]):
            for font_x, value in enumerate(row):
                if value != "1":
                    continue
                for dy in range(SCALE):
                    for dx in range(SCALE):
                        lit.add((x0 + SCALE * font_x + dx,
                                 y0 + SCALE * font_y + dy))
    return lit


def _read_noncomment_line(source) -> bytes:
    while True:
        line = source.readline()
        if not line:
            raise ValueError("truncated PFM header")
        stripped = line.strip()
        if stripped and not stripped.startswith(b"#"):
            return stripped


def read_pfm(path: Path) -> tuple[int, int, float, array.array]:
    with path.open("rb") as source:
        if _read_noncomment_line(source) != b"PF":
            raise ValueError("expected a three-channel color PFM")
        width, height = map(int, _read_noncomment_line(source).split())
        scale = float(_read_noncomment_line(source))
        if scale == 0:
            raise ValueError("invalid zero PFM scale")
        samples = array.array("f")
        samples.fromfile(source, width * height * 3)
        if len(samples) != width * height * 3:
            raise ValueError("truncated PFM sample data")
        if source.read(1):
            raise ValueError("unexpected trailing PFM data")
    file_is_little_endian = scale < 0
    if file_is_little_endian != (sys.byteorder == "little"):
        samples.byteswap()
    return width, height, abs(scale), samples


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (struct.pack(">I", len(payload)) + kind + payload +
            struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF))


def write_preview(path: Path, lit: set[tuple[int, int]], scale: int = 4) -> None:
    left, top, right, bottom = 0, 0, 124, 53
    width = (right - left + 1) * scale
    height = (bottom - top + 1) * scale
    raw = bytearray()
    for y in range(top, bottom + 1):
        expanded = bytearray()
        for x in range(left, right + 1):
            value = 255 if (x, y) in lit else 0
            expanded.extend(bytes((value,)) * scale)
        for _ in range(scale):
            raw.append(0)  # PNG filter method: None.
            raw.extend(expanded)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    encoded = (b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", ihdr) +
               png_chunk(b"IDAT", zlib.compress(bytes(raw), 9)) +
               png_chunk(b"IEND", b""))
    path.write_bytes(encoded)


def packed_mask_sha256(lit: set[tuple[int, int]]) -> str:
    packed = bytearray((WIDTH * HEIGHT + 7) // 8)
    for x, y in lit:
        bit = y * WIDTH + x
        packed[bit >> 3] |= 1 << (7 - (bit & 7))
    return hashlib.sha256(packed).hexdigest()


def bounds(lit: set[tuple[int, int]]) -> list[int] | None:
    if not lit:
        return None
    xs = [x for x, _y in lit]
    ys = [y for _x, y in lit]
    return [min(xs), min(ys), max(xs), max(ys)]


def recognize(actual: set[tuple[int, int]]) -> tuple[str, list[dict[str, object]]]:
    reverse_font = {pattern: character for character, pattern in FONT.items()}
    if len(reverse_font) != 16:
        raise AssertionError("hex font contains ambiguous glyphs")
    text: list[str] = []
    failures: list[dict[str, object]] = []
    for index in range(64):
        line, slot = divmod(index, CHARS_PER_LINE)
        x0 = glyph_x(slot)
        y0 = LINE_Y[line]
        rows: list[str] = []
        uniform = True
        for font_y in range(5):
            row: list[str] = []
            for font_x in range(3):
                tile = {
                    (x0 + SCALE * font_x + dx,
                     y0 + SCALE * font_y + dy) in actual
                    for dy in range(SCALE) for dx in range(SCALE)
                }
                if len(tile) != 1:
                    uniform = False
                row.append("1" if tile == {True} else "0")
            rows.append("".join(row))
        pattern = tuple(rows)
        character = reverse_font.get(pattern) if uniform else None
        if character is None:
            failures.append({
                "index": index,
                "line": line,
                "slot": slot,
                "uniform_scale_2_tiles": uniform,
                "sampled_pattern": list(pattern),
            })
            text.append("?")
        else:
            text.append(character)
    return "".join(text), failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jxl", type=Path)
    parser.add_argument("--djxl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    jxl = args.jxl.resolve()
    decoder = args.djxl.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pfm = output / "decoded-rgb.pfm"

    decoded = subprocess.run(
        [str(decoder), "--num_threads=0", "--quiet", str(jxl), str(pfm)],
        capture_output=True,
        text=True,
    )
    if decoded.returncode:
        raise SystemExit(decoded.stdout + decoded.stderr)

    digest = sha256_file(jxl)
    expected_hex = digest.hex().upper()
    expected_lit = expected_pixels(expected_hex)
    width, height, pfm_scale, samples = read_pfm(pfm)
    if (width, height) != (WIDTH, HEIGHT):
        raise AssertionError((width, height))

    def rgb(x: int, y: int) -> tuple[float, float, float]:
        # PFM scanlines are stored bottom-to-top.
        file_y = height - 1 - y
        offset = (file_y * width + x) * 3
        return (samples[offset], samples[offset + 1], samples[offset + 2])

    actual_lit: set[tuple[int, int]] = set()
    channels_equal = True
    samples_binary = True
    for y in range(height):
        for x in range(width):
            red, green, blue = rgb(x, y)
            channels_equal &= red == green == blue
            samples_binary &= (red in (0.0, 1.0) and
                               green in (0.0, 1.0) and
                               blue in (0.0, 1.0))
            if red > 0.5:
                actual_lit.add((x, y))

    observed, ocr_failures = recognize(actual_lit)
    unexpected = actual_lit - expected_lit
    missing = expected_lit - actual_lit
    exact_bitmap = not unexpected and not missing
    preview = output / "hex-preview.png"
    write_preview(preview, actual_lit)

    passed = (observed == expected_hex and not ocr_failures and exact_bitmap
              and channels_equal and samples_binary)
    report = {
        "status": "passed" if passed else "failed",
        "jxl": str(jxl),
        "size_bytes": jxl.stat().st_size,
        "decoder": str(decoder),
        "sha256_hex_lower": digest.hex(),
        "expected_hex_upper": expected_hex,
        "expected_lines": [expected_hex[i:i + 16]
                           for i in range(0, 64, 16)],
        "ocr_hex_upper": observed,
        "ocr_lines": [observed[i:i + 16] for i in range(0, 64, 16)],
        "ocr_matches_sha256_case_insensitive": observed == expected_hex,
        "recognized_glyphs": 64 - len(ocr_failures),
        "ocr_failures": ocr_failures,
        "exact_full_bitmap": exact_bitmap,
        "unexpected_white_pixels": len(unexpected),
        "missing_white_pixels": len(missing),
        "rgb_channels_identical": channels_equal,
        "all_rgb_samples_binary": samples_binary,
        "actual_lit_pixels": len(actual_lit),
        "expected_lit_pixels": len(expected_lit),
        "actual_lit_bbox_xyxy": bounds(actual_lit),
        "expected_lit_bbox_xyxy": bounds(expected_lit),
        "actual_packed_mask_sha256": packed_mask_sha256(actual_lit),
        "expected_packed_mask_sha256": packed_mask_sha256(expected_lit),
        "layout": {
            "font": "3x5 uppercase hexadecimal",
            "scale": SCALE,
            "output_x_formula": "4 + 7*slot + 2*floor(slot/4)",
            "line_y": list(LINE_Y),
            "orientation": "already applied by stock djxl output",
        },
        "decoded_pfm": {
            "path": str(pfm),
            "width": width,
            "height": height,
            "scale": pfm_scale,
        },
        "preview_png": str(preview),
    }
    report_path = output / "verification.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
