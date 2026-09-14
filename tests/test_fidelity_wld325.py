"""Audit reader checks using independently specified wire bytes, no game assets."""

import json
import struct

import pytest

from scripts.fidelity_wld325 import CELL, Reader, decode_tiles, read_manifest, read_world


def net_string(value):
    data = value.encode()
    length = len(data)
    prefix = bytearray()
    while length >= 128:
        prefix.append((length & 127) | 128)
        length >>= 7
    return bytes(prefix + bytes([length])) + data


def test_extended_cell_fields_and_long_rle():
    # Four headers; active tile 300, wall 513, framed/painted, shimmer,
    # slope 3, all wires, actuator/inactive and four coatings; 300 cells.
    raw = bytes([0xBF, 0x4F, 0xFF, 0x1E])
    raw += struct.pack("<HhhBBBBBh", 300, 18, 36, 7, 1, 9, 200, 2, 299)
    importance = [False] * 301
    importance[300] = True
    cells, records = decode_tiles(raw, 1, 300, importance)
    assert records == 1
    assert cells == CELL.pack(1, 300, 513, 18, 36, 7, 9, 4, 200, 4, 63, 30) * 300


def test_equivalent_rle_encodings():
    # Water in an empty cell; three repeats encoded separately, byte or short RLE.
    one = bytes([0x08, 50])
    expected = decode_tiles(one * 3, 1, 3, [False])[0]
    assert decode_tiles(bytes([0x48, 50, 2]), 1, 3, [False])[0] == expected
    assert decode_tiles(bytes([0x88, 50, 2, 0]), 1, 3, [False])[0] == expected


@pytest.mark.parametrize("raw", [b"\x40\x03", b"\x80\xff\xff", b"\x02", b"\x00\x00\x00\x00"])
def test_invalid_tile_stream_rejected(raw):
    with pytest.raises(ValueError):
        decode_tiles(raw, 1, 3, [False])


def test_cell_change_survives_decoding():
    dry = decode_tiles(b"\x00", 1, 1, [False])[0]
    wet = decode_tiles(b"\x08\x01", 1, 1, [False])[0]
    assert dry != wet


def test_manifest_normalizes_only_timing():
    original = {
        "GenPassResults": [
            {"Name": "Terrain", "DurationMs": 3, "RandNext": 123, "Hash": 456, "Skipped": False}
        ],
        "Version": "v1.4.5.7",
        "GitSHA": "",
        "FinalHash": 456,
    }
    header = b"opaque" + net_string(json.dumps(original, separators=(",", ":")))
    prefix, manifest, normalized = read_manifest(header)
    assert prefix == 6 and manifest == original
    assert "DurationMs" not in normalized["GenPassResults"][0]
    assert normalized["GenPassResults"][0]["RandNext"] == 123
    assert normalized["FinalHash"] == 456
    with pytest.raises(ValueError):
        read_manifest(header + b"\x00")


def test_bad_string_and_version_rejected(tmp_path):
    with pytest.raises(ValueError):
        Reader(b"\xff\xff\xff\xff\xff").string()
    path = tmp_path / "wrong.wld"
    path.write_bytes(struct.pack("<i", 324))
    with pytest.raises(ValueError, match="version-325"):
        read_world(path)
