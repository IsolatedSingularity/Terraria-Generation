"""User-facing world configuration and stable seed parsing."""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from enum import StrEnum


class WorldScale(StrEnum):
    """Supported render/simulation sizes.

    Preview is intentionally compact for interactive exploration. Small uses
    Terraria's familiar 4200 x 1200 dimensions.
    """

    PREVIEW = "preview"
    SMALL = "small"

    @property
    def dimensions(self) -> tuple[int, int]:
        return (240, 140) if self is WorldScale.PREVIEW else (4200, 1200)


class Evil(StrEnum):
    CORRUPTION = "corruption"
    CRIMSON = "crimson"


class Difficulty(StrEnum):
    CLASSIC = "classic"
    EXPERT = "expert"
    MASTER = "master"


def seed_to_uint32(seed: str | int) -> int:
    """Return a deterministic unsigned 32-bit seed.

    Numeric values retain their low 32 bits. Text uses CRC-32, mirroring the
    broad idea of Terraria's text-seed conversion while deliberately avoiding
    a claim of reproducing its complete RNG stream.
    """

    if isinstance(seed, int):
        return seed & 0xFFFFFFFF
    normalized = seed.strip()
    if normalized and normalized.lstrip("+-").isdigit():
        return int(normalized) & 0xFFFFFFFF
    return zlib.crc32(normalized.encode("utf-8")) & 0xFFFFFFFF


@dataclass(frozen=True, slots=True)
class WorldConfig:
    """Immutable options consumed by the pass pipeline."""

    seed: str | int = "TerraExplorer"
    scale: WorldScale = WorldScale.PREVIEW
    evil: Evil = Evil.CORRUPTION
    difficulty: Difficulty = Difficulty.CLASSIC
    hardmode: bool = False
    enabled_phases: tuple[str, ...] | None = None

    @property
    def width(self) -> int:
        return self.scale.dimensions[0]

    @property
    def height(self) -> int:
        return self.scale.dimensions[1]

    @property
    def seed_value(self) -> int:
        return seed_to_uint32(self.seed)
