"""Immutable native cell view over the established 16-byte audit stream.

Coordinates are cells[y, x]; stored order is x-major, then y (Terraria's save
order). No simulation Tile enum is involved. See docs/fidelity/CANONICAL_SCHEMA.md.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from terraexplorer.fidelity.wld325 import CELL, CELL_FIELDS, SECTIONS, digest, read_world

CELL_DTYPE = np.dtype(
    list(
        zip(
            CELL_FIELDS,
            ("u1", "<u2", "<u2", "<i2", "<i2", "u1", "u1", "u1", "u1", "u1", "u1", "u1"),
            strict=True,
        )
    ),
    align=False,
)
assert CELL_DTYPE.itemsize == CELL.size == 16


@dataclass(frozen=True)
class CanonicalWorld:
    """Native serialized state; views use immutable bytes as their backing store."""

    cell_bytes: bytes
    report_json: str
    preamble: bytes
    section_bytes: tuple[bytes, ...]

    def __post_init__(self):
        if not isinstance(self.cell_bytes, bytes) or not isinstance(self.preamble, bytes):
            raise TypeError("Canonical byte buffers must be immutable bytes")
        if not isinstance(self.section_bytes, tuple) or any(
            not isinstance(section, bytes) for section in self.section_bytes
        ):
            raise TypeError("Canonical saved sections must be immutable bytes")
        meta = self.metadata
        if len(self.cell_bytes) != meta["width"] * meta["height"] * CELL.size:
            raise ValueError("Incomplete canonical cell stream")
        if len(self.section_bytes) != len(SECTIONS):
            raise ValueError("Incomplete saved section set")

    @property
    def report(self):
        return json.loads(self.report_json)

    @property
    def metadata(self):
        return MappingProxyType(self.report["metadata"])

    @property
    def cells(self):
        meta = self.metadata
        return np.frombuffer(self.cell_bytes, CELL_DTYPE).reshape(meta["width"], meta["height"]).T

    @property
    def frame_saved(self):
        cells = self.cells
        importance = np.asarray(self.report["frame_importance"], dtype=bool)
        return (cells["active"] != 0) & importance[cells["tile"]]

    @property
    def half_block(self):
        return self.cells["shape"] == 1

    @property
    def slope(self):
        shape = self.cells["shape"]
        return np.where(shape > 1, shape - 1, 0).astype(np.uint8)

    @property
    def semantic_sha256(self):
        return digest(self.cell_bytes)

    def original_bytes(self):
        """Exact source representation, including opaque sections and original RLE."""
        return struct.pack("<i", 325) + self.preamble + b"".join(self.section_bytes)


def import_world(path: str | Path) -> CanonicalWorld:
    report, cells, sections, _normalized, preamble = read_world(Path(path))
    world = CanonicalWorld(
        bytes(cells), json.dumps(report), preamble, tuple(sections[name] for name in SECTIONS)
    )
    if world.semantic_sha256 != report["canonical_cells_sha256"]:
        raise ValueError("Canonical cell transfer changed state")
    if digest(world.original_bytes()) != report["sha256"]:
        raise ValueError("Source section transfer changed state")
    return world
