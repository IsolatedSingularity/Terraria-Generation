"""Read-only local pass snapshots and exact cell-state comparisons.

Native records retain all 14 bytes of Tile fields, including unsaved state.
Coordinates here are [x, y], explicitly matching the capture traversal.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from terraexplorer.fidelity.world import CELL_DTYPE

RAW_DTYPE = np.dtype(
    [
        ("type", "<u2"),
        ("wall", "<u2"),
        ("liquid", "u1"),
        ("sTileHeader", "<u2"),
        ("bTileHeader", "u1"),
        ("bTileHeader2", "u1"),
        ("bTileHeader3", "u1"),
        ("frameX", "<i2"),
        ("frameY", "<i2"),
    ]
)
assert RAW_DTYPE.itemsize == 14


def load_snapshot(directory: Path, phase: str):
    metadata = json.loads((directory / f"{phase}.json").read_text(encoding="utf-8"))
    raw = gzip.decompress((directory / f"{phase}.raw.gz").read_bytes())
    width, height = metadata["width"], metadata["height"]
    if len(raw) != width * height * RAW_DTYPE.itemsize:
        raise ValueError("Incomplete native Tile snapshot")
    return np.frombuffer(raw, RAW_DTYPE).reshape(width, height), metadata


def canonical_cells(raw, importance):
    """Project native fields to the existing v325 canonical saved-cell schema.

    This does not invoke saving, framing, normalization, or mutate native input.
    The full native stream remains the stronger pass-replay comparison.
    """
    out = np.zeros(raw.shape, dtype=CELL_DTYPE)
    sh, b1, b3 = raw["sTileHeader"], raw["bTileHeader"], raw["bTileHeader3"]
    active = (sh & 32) != 0
    out["active"] = active
    out["tile"] = np.where(active, raw["type"], 0)
    out["wall"] = raw["wall"]
    frame_saved = active & np.asarray(importance, dtype=bool)[raw["type"]]
    out["frame_x"] = np.where(frame_saved, raw["frameX"], -1)
    out["frame_y"] = np.where(frame_saved, raw["frameY"], -1)
    out["tile_paint"] = np.where(active, sh & 31, 0)
    out["wall_paint"] = np.where(raw["wall"] != 0, b1 & 31, 0)
    out["liquid_amount"] = raw["liquid"]
    out["liquid_kind"] = np.where(raw["liquid"] != 0, ((b1 >> 5) & 3) + 1, 0)
    slope = (sh >> 12) & 7
    out["shape"] = np.where(sh & 1024, 1, np.where(slope, slope + 1, 0))
    out["wires_actuator_inactive"] = (
        ((sh >> 7) & 7) | ((b1 >> 4) & 8) | ((sh >> 7) & 16) | ((sh >> 1) & 32)
    )
    out["coatings"] = ((b3 >> 4) & 6) | ((b3 >> 4) & 8) | ((sh >> 11) & 16)
    return out


def array_hash(array):
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def compare_cells(pre, expected, actual, first_limit=10):
    if pre.shape != expected.shape or pre.shape != actual.shape:
        raise ValueError("Cell dimensions differ")
    if pre.dtype != expected.dtype or pre.dtype != actual.dtype:
        raise ValueError("Cell schemas differ")
    wanted, made, differing = expected != pre, actual != pre, expected != actual
    coordinates = np.argwhere(differing)
    names = pre.dtype.names
    return {
        "expected_changed_cells": int(np.count_nonzero(wanted)),
        "actual_changed_cells": int(np.count_nonzero(made)),
        "missing_changed_cells": int(np.count_nonzero(wanted & ~made)),
        "extra_changed_cells": int(np.count_nonzero(made & ~wanted)),
        "different_cells": len(coordinates),
        "per_field_differences": {
            name: int(np.count_nonzero(expected[name] != actual[name])) for name in names
        },
        "missing_field_writes": {
            name: int(np.count_nonzero((expected[name] != pre[name]) & (actual[name] == pre[name])))
            for name in names
        },
        "extra_field_writes": {
            name: int(np.count_nonzero((expected[name] == pre[name]) & (actual[name] != pre[name])))
            for name in names
        },
        "first_differences": [
            {
                "x": int(x),
                "y": int(y),
                "fields": {
                    name: {"expected": int(expected[name][x, y]), "actual": int(actual[name][x, y])}
                    for name in names
                    if expected[name][x, y] != actual[name][x, y]
                },
            }
            for x, y in coordinates[:first_limit]
        ],
        "equal": not len(coordinates),
    }
