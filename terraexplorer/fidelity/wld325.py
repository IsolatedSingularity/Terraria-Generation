"""Shared version-325 decoder for canonical importing and the oracle audit command.

Reads the identifying header, expands serialized cell state and checks the footer.
Undecoded header/sections remain opaque and are compared exactly, never discarded.
Promoted from scripts/fidelity_wld325.py; there is only one live decoding path.
This is not a complete Terraria gameplay loader.
Source: pinned Terraria.IO.WorldFile SaveWorldHeader/Flags/Tiles, LoadWorldTiles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import uuid
from pathlib import Path

SECTIONS = (
    "header",
    "tiles",
    "chests",
    "signs",
    "npcs",
    "tile_entities",
    "pressure_plates",
    "town_manager",
    "bestiary",
    "creative_powers",
    "footer",
)
CELL = struct.Struct("<BHHhhBBBBBBB")
CELL_FIELDS = (
    "active",
    "tile",
    "wall",
    "frame_x",
    "frame_y",
    "tile_paint",
    "wall_paint",
    "liquid_kind",
    "liquid_amount",
    "shape",
    "wires_actuator_inactive",
    "coatings",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


class Reader:
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def take(self, count):
        if count < 0 or self.pos + count > len(self.data):
            raise ValueError(f"Truncated data at {self.pos}, requested {count}")
        result = self.data[self.pos : self.pos + count]
        self.pos += count
        return result

    def value(self, fmt):
        return struct.unpack("<" + fmt, self.take(struct.calcsize("<" + fmt)))[0]

    def string(self):
        length = 0
        for shift in range(0, 35, 7):
            byte = self.value("B")
            if shift == 28 and byte > 7:
                raise ValueError("Invalid .NET string length")
            length |= (byte & 127) << shift
            if byte < 128:
                return self.take(length).decode("utf-8", errors="strict")
        raise ValueError("Unterminated string length")


def decode_tiles(data, width, height, importance):
    reader = Reader(data)
    cells = bytearray()
    records = 0
    for x in range(width):
        y = 0
        while y < height:
            h1 = reader.value("B")
            h2 = reader.value("B") if h1 & 1 else 0
            h3 = reader.value("B") if h2 & 1 else 0
            h4 = reader.value("B") if h3 & 1 else 0
            if h2 & 128 or h4 & ~30:
                raise ValueError("Unknown cell header bits")
            active = bool(h1 & 2)
            tile = wall = tile_paint = wall_paint = amount = 0
            frame_x = frame_y = -1  # sentinel for frames absent from the file
            if active:
                tile = reader.value("H" if h1 & 32 else "B")
                if tile >= len(importance):
                    raise ValueError(f"Tile ID {tile} exceeds importance table")
                if importance[tile]:
                    frame_x, frame_y = reader.value("h"), reader.value("h")
                if h3 & 8:
                    tile_paint = reader.value("B")
            if h1 & 4:
                wall = reader.value("B")
                if h3 & 16:
                    wall_paint = reader.value("B")
            kind = (h1 & 24) >> 3  # 0 absent, 1 water, 2 lava, 3 honey, 4 shimmer
            if kind:
                amount = reader.value("B")
                if h3 & 128:
                    kind = 4
            if h3 & 64:
                wall |= reader.value("B") << 8
            shape = (h2 & 112) >> 4  # 0 full, 1 half, 2..5 slope encoding
            if shape > 5:
                raise ValueError("Invalid saved slope")
            flags = ((h2 & 14) >> 1) | ((h3 & 32) >> 2) | ((h3 & 6) << 3)
            cell = CELL.pack(
                active,
                tile,
                wall,
                frame_x,
                frame_y,
                tile_paint,
                wall_paint,
                kind,
                amount,
                shape,
                flags,
                h4 & 30,
            )
            rle = h1 >> 6
            repeat = 0 if rle == 0 else reader.value("B" if rle == 1 else "h")
            if repeat < 0 or y + repeat >= height:
                raise ValueError(f"RLE crosses column at ({x}, {y})")
            cells.extend(cell * (repeat + 1))
            y += repeat + 1
            records += 1
    if reader.pos != len(data):
        raise ValueError("Tile section boundary mismatch")
    return cells, records


def read_manifest(header):
    """Locate the unique final .NET string, without interpreting the opaque flag tail.

    SaveWorldFlags ends with Manifest.Serialize(). Require its known JSON prefix,
    a valid length prefix ending exactly at the tile pointer, and the pinned version.
    No bytes before this proven boundary are dropped from comparison.
    """
    marker = b'{"GenPassResults":'
    candidates = []
    for position in range(len(header)):
        if not header.startswith(marker, position):
            continue
        for prefix in range(max(0, position - 5), position):
            reader = Reader(header)
            reader.pos = prefix
            try:
                value = reader.string()
                if reader.pos != len(header) or not value.startswith(marker.decode()):
                    continue
                manifest = json.loads(value)
                if manifest["Version"] != "v1.4.5.7":
                    continue
                passes = manifest["GenPassResults"]
                if not isinstance(passes, list) or not passes:
                    continue
                if any(
                    type(item["DurationMs"]) is not int or item["DurationMs"] < 0 for item in passes
                ):
                    continue
            except (ValueError, KeyError, TypeError):
                continue
            candidates.append((prefix, manifest))
    if len(candidates) != 1:
        raise ValueError("Expected exactly one validated terminal generation manifest")
    prefix, manifest = candidates[0]
    normalized = {
        **manifest,
        "GenPassResults": [
            {key: value for key, value in item.items() if key != "DurationMs"}
            for item in manifest["GenPassResults"]
        ],
    }
    return prefix, manifest, normalized


def read_world(path):
    data = path.read_bytes()
    reader = Reader(data)
    version = reader.value("i")
    if version != 325 or reader.take(8) != b"relogic\x02":
        raise ValueError("Expected a version-325 Re-Logic world")
    revision, favorite = reader.value("I"), reader.value("Q")
    if reader.value("h") != 11:
        raise ValueError("Expected 11 section pointers")
    pointers = [reader.value("i") for _ in SECTIONS]
    count = reader.value("H")
    if not count:
        raise ValueError("Empty frame-importance table")
    packed = reader.take((count + 7) // 8)
    importance = [bool(packed[i // 8] & (1 << (i % 8))) for i in range(count)]
    if pointers[0] != reader.pos or pointers != sorted(set(pointers)):
        raise ValueError("Invalid section pointers")
    if pointers[-1] >= len(data):
        raise ValueError("Missing footer")
    metadata = {
        "version": version,
        "revision": revision,
        "favorite_bits": favorite,
        "name": reader.string(),
        "seed": reader.string(),
        "generator_version": reader.value("Q"),
    }
    excluded = {}

    def identity(name, fmt=None, count=None):
        start = reader.pos
        value = reader.take(count) if count else reader.value(fmt)
        excluded[name] = [start, reader.pos]
        metadata[name] = str(uuid.UUID(bytes_le=value)) if count else value

    identity("uuid", count=16)
    metadata["world_id"] = reader.value("i")  # seeded genRand value, not normalized
    for field in ("left", "right", "top", "bottom", "height", "width", "game_mode"):
        metadata[field] = reader.value("i")
    if (metadata["width"], metadata["height"]) != (4200, 1200):
        raise ValueError("This bounded oracle validator supports only Small 4200x1200")
    for field in (
        "drunk",
        "get_good",
        "anniversary",
        "dont_starve",
        "not_the_bees",
        "remix",
        "no_traps",
        "zenith",
        "skyblock",
    ):
        metadata[field] = reader.value("?")
    identity("creation_time_binary", "q")
    identity("last_save_time_binary", "q")
    metadata["moon_type"] = reader.value("B")
    reader.take(17 * 4)  # tree/cave region styles and biome backgrounds
    metadata["spawn_x"], metadata["spawn_y"] = reader.value("i"), reader.value("i")
    for field in ("world_surface", "rock_layer", "time"):
        metadata[field] = reader.value("d")
    metadata["day_time"] = reader.value("?")
    metadata["moon_phase"] = reader.value("i")
    metadata["blood_moon"], metadata["eclipse"] = reader.value("?"), reader.value("?")
    metadata["dungeon_x"], metadata["dungeon_y"] = reader.value("i"), reader.value("i")
    metadata["crimson"] = reader.value("?")
    reader.take(20)  # boss/rescue/invasion/shadow-orb flags, retained in raw header
    metadata["shadow_orb_count"] = reader.value("B")
    metadata["altar_count"] = reader.value("i")
    metadata["hardmode"] = reader.value("?")
    if reader.pos > pointers[1]:
        raise ValueError("Header crossed tile section")
    sections = {
        name: data[start:end]
        for name, start, end in zip(SECTIONS, pointers, pointers[1:] + [len(data)], strict=True)
    }
    normalized_header = bytearray(sections["header"])
    for start, end in excluded.values():
        normalized_header[start - pointers[0] : end - pointers[0]] = b"\0" * (end - start)
    manifest_start, manifest, normalized_manifest = read_manifest(sections["header"])
    normalized_header = normalized_header[:manifest_start] + json.dumps(
        normalized_manifest, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    footer = Reader(sections["footer"])
    if not footer.value("?") or footer.string() != metadata["name"]:
        raise ValueError("Footer name/validity mismatch")
    if footer.value("i") != metadata["world_id"] or footer.pos != len(footer.data):
        raise ValueError("Footer identity/length mismatch")
    cells, records = decode_tiles(
        sections["tiles"], metadata["width"], metadata["height"], importance
    )
    report = {
        "path": str(path.resolve()),
        "size": len(data),
        "sha256": digest(data),
        "metadata": metadata,
        "section_pointers": dict(zip(SECTIONS, pointers, strict=True)),
        "section_sha256": {name: digest(part) for name, part in sections.items()},
        "normalized_header_sha256": digest(normalized_header),
        "normalized_identity_ranges": excluded,
        "importance_count": count,
        "frame_importance": importance,
        "manifest_absolute_offset": pointers[0] + manifest_start,
        "manifest": manifest,
        "cell_count": len(cells) // CELL.size,
        "encoded_cell_records": records,
        "canonical_cells_sha256": digest(cells),
        "footer_valid": True,
    }
    return report, cells, sections, normalized_header, data[4 : pointers[0]]


def compare_worlds(first, second):
    a, ac, asections, ah, af = first
    b, bc, bsections, bh, bf = second
    differences = {}
    first_cells = []
    if ac != bc:
        for offset in range(0, len(ac), CELL.size):
            av, bv = CELL.unpack_from(ac, offset), CELL.unpack_from(bc, offset)
            changed = [name for name, x, y in zip(CELL_FIELDS, av, bv, strict=True) if x != y]
            if changed and len(first_cells) < 10:
                index = offset // CELL.size
                first_cells.append(
                    {"x": index // 1200, "y": index % 1200, "a": av, "b": bv, "fields": changed}
                )
            for name in changed:
                differences[name] = differences.get(name, 0) + 1
    # Pointer offsets encode physical layout, not world state. Compare file metadata
    # and importance tables separately, retaining every other byte of those records.
    preamble_equal = af[:20] == bf[:20] and af[66:] == bf[66:]
    other = {name: asections[name] == bsections[name] for name in SECTIONS[2:]}
    return {
        "raw_equal": a["sha256"] == b["sha256"],
        "metadata_differences": {
            key: [value, b["metadata"][key]]
            for key, value in a["metadata"].items()
            if value != b["metadata"][key]
        },
        "canonical_cells_equal": ac == bc,
        "cell_field_difference_counts": differences,
        "first_different_cells": first_cells,
        "normalized_header_equal": ah == bh,
        "preamble_equal_excluding_offsets": preamble_equal,
        "other_sections_byte_equal": other,
        "manifest_duration_differences": [
            {
                "index": index,
                "name": ap["Name"],
                "milliseconds": [ap["DurationMs"], bp["DurationMs"]],
            }
            for index, (ap, bp) in enumerate(
                zip(a["manifest"]["GenPassResults"], b["manifest"]["GenPassResults"])
            )
            if ap["DurationMs"] != bp["DurationMs"]
        ],
        "semantic_equal_under_declared_normalization": ac == bc
        and ah == bh
        and preamble_equal
        and all(other.values()),
        "normalization": [
            "uuid",
            "creation_time_binary",
            "last_save_time_binary",
            "manifest pass DurationMs",
            "section offsets",
        ],
        "limits": "Saved cell fields only; unsaved frames use -1 sentinel. Opaque sections "
        "are byte-compared; unequal opaque data remains unresolved, never ignored.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("world", type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    first = read_world(args.world)
    result = {"first": first[0]}
    if args.compare:
        second = read_world(args.compare)
        result.update(second=second[0], comparison=compare_worlds(first, second))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))
