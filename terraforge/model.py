"""In-memory world representation used by every TerraForge surface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from terraforge.config import WorldConfig
from terraforge.tiles import Biome, Liquid, Tile, Wall


@dataclass(frozen=True, slots=True)
class LayerDepths:
    world_surface: int
    rock_layer: int
    underworld: int

    @classmethod
    def for_shape(cls, width: int, height: int) -> LayerDepths:
        del width
        underworld_height = 200 if height >= 1200 else max(16, round(height / 6))
        return cls(
            world_surface=round(height * 0.19),
            rock_layer=round(height * 0.46),
            underworld=height - underworld_height,
        )


@dataclass(frozen=True, slots=True)
class StructureMarker:
    kind: str
    x: int
    y: int
    width: int
    height: int
    symbol: str


@dataclass(slots=True)
class PassResult:
    index: int
    name: str
    phase: str
    fidelity: str
    elapsed_ms: float
    changed_tiles: int | None = None
    note: str = ""


@dataclass(slots=True)
class GeneratedWorld:
    config: WorldConfig
    tiles: npt.NDArray[np.uint8]
    walls: npt.NDArray[np.uint8]
    liquid_amount: npt.NDArray[np.uint8]
    liquid_kind: npt.NDArray[np.uint8]
    biomes: npt.NDArray[np.uint8]
    surface: npt.NDArray[np.int16]
    layers: LayerDepths
    metadata: dict[str, Any] = field(default_factory=dict)
    structures: list[StructureMarker] = field(default_factory=list)
    pass_results: list[PassResult] = field(default_factory=list)

    @classmethod
    def empty(cls, config: WorldConfig) -> GeneratedWorld:
        shape = (config.height, config.width)
        layers = LayerDepths.for_shape(config.width, config.height)
        return cls(
            config=config,
            tiles=np.full(shape, Tile.AIR, dtype=np.uint8),
            walls=np.full(shape, Wall.NONE, dtype=np.uint8),
            liquid_amount=np.zeros(shape, dtype=np.uint8),
            liquid_kind=np.full(shape, Liquid.NONE, dtype=np.uint8),
            biomes=np.full(shape, Biome.SKY, dtype=np.uint8),
            surface=np.full(config.width, layers.world_surface, dtype=np.int16),
            layers=layers,
            metadata={"target": "Terraria 1.4.5-era concepts / 1.4.4.9 pass order"},
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self.tiles.shape

    @property
    def memory_bytes(self) -> int:
        return sum(
            array.nbytes
            for array in (
                self.tiles,
                self.walls,
                self.liquid_amount,
                self.liquid_kind,
                self.biomes,
                self.surface,
            )
        )
