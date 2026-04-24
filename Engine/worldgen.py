"""SMALL-world generator producing crop-ready tile grids.

Used by ``Code/`` and ``Advanced/`` figures that follow the redesign-audit
recipe: generate at SMALL (4200x1200) so ``LayerDepths`` ratios stay correct,
then crop ~600x500 windows for display.

Public API:
    generateSmallWorld(seed, evilType="corruption", altarsSmashed=0)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from Engine.algorithms import (
    ADAMANTITE, AIR, ASH, CHLOROPHYTE, COBALT, COPPER, CORRUPT_DIRT, CRIMSON_DIRT,
    CRIMSTONE, DIRT, EBONSTONE, GOLD, GRASS, HARDENED_SAND, HELLSTONE, ICE, IRON,
    LEAD, MUD, MYTHRIL, ORICHALCUM, PALLADIUM, PEARLSAND, PEARLSTONE, PLATINUM,
    SAND, SANDSTONE_BLOCK, SILVER, SNOW, STONE, TIN, TITANIUM, TUNGSTEN,
    cavinator, cellularAutomataSmooth, tileRunner,
)
from Engine.constants import SMALL, LayerDepths


@dataclass
class SmallWorld:
    """Generated SMALL-world payload."""
    grid: npt.NDArray[np.int32]
    layers: LayerDepths
    spawnX: int
    jungleX: int
    desertX: int
    snowX: int
    evilX: int
    evilType: str


def _carveSurface(
    grid: npt.NDArray[np.int32],
    layers: LayerDepths,
    rng: np.random.Generator,
) -> npt.NDArray[np.int32]:
    """Etch a fractal-ish surface line and fill dirt + grass + stone bands."""
    height, width = grid.shape
    base = layers.worldSurface

    # Multi-octave sine surface line (small amplitude so layer markers stay valid).
    xs = np.arange(width)
    surface = np.full(width, base, dtype=np.float64)
    for amp, freq in [(18.0, 0.0035), (9.0, 0.011), (4.0, 0.04)]:
        surface += amp * np.sin(2 * np.pi * freq * xs + rng.uniform(0, 2 * np.pi))
    surfaceInt = surface.astype(np.int32)

    dirtBottom = int(layers.rockLayer)
    for x in range(width):
        sy = int(surfaceInt[x])
        grid[:sy, x] = AIR
        grid[sy, x] = GRASS
        grid[sy + 1: sy + 4, x] = DIRT
        grid[sy + 4: dirtBottom, x] = DIRT
        grid[dirtBottom:, x] = STONE

    # Hellstone shell at hellLayer.
    grid[layers.hellLayer:, :] = ASH
    grid[layers.hellLayer + 4:, :] = HELLSTONE
    return grid


def _placeBiomes(
    grid: npt.NDArray[np.int32],
    layers: LayerDepths,
    rng: np.random.Generator,
    evilType: str,
    compact: bool = False,
) -> dict[str, int]:
    """Convert dirt/stone in surface biome columns; return biome X centers.

    When ``compact`` is True the surface biomes are packed within ~500 tiles
    of the world center so a single 600x500 crop can show Forest -> Jungle ->
    Desert (and the evil biome) simultaneously. The cave/ore passes still run
    over the full 4200 width so the crop reflects realistic underground
    density.
    """
    height, width = grid.shape
    spawnX = width // 2

    if compact:
        # Pack biomes near world center: spawn-Forest | Jungle | Desert | Evil
        # within ~520 tiles total so a 600-wide crop fits all of them.
        snowX = int(spawnX - 360)
        jungleX = int(spawnX - 130)
        desertX = int(spawnX + 110)
        evilX = int(spawnX + 320)
        snowHalf, jungleHalf, desertHalf, evilHalf = 90, 100, 90, 90
    else:
        dungeonSide = rng.choice(["left", "right"])
        if dungeonSide == "left":
            snowX = int(rng.integers(width // 8, width // 4))
            jungleX = int(rng.integers(3 * width // 4, 7 * width // 8))
        else:
            snowX = int(rng.integers(3 * width // 4, 7 * width // 8))
            jungleX = int(rng.integers(width // 8, width // 4))

        desertX = int(rng.integers(width // 3, 2 * width // 3))
        while abs(desertX - spawnX) < 300 or abs(desertX - jungleX) < 350:
            desertX = int(rng.integers(width // 3, 2 * width // 3))
        evilX = int(rng.integers(int(width * 0.18), int(width * 0.82)))
        while abs(evilX - jungleX) < 400 or abs(evilX - spawnX) < 400:
            evilX = int(rng.integers(int(width * 0.18), int(width * 0.82)))
        snowHalf, jungleHalf, desertHalf, evilHalf = 220, 240, 200, 180

    rock = int(layers.rockLayer)
    hell = int(layers.hellLayer)

    # Snow band: convert surface dirt -> snow, stone -> ice up to rockLayer.
    for x in range(max(0, snowX - snowHalf), min(width, snowX + snowHalf)):
        for y in range(rock):
            t = grid[y, x]
            if t == DIRT or t == GRASS:
                grid[y, x] = SNOW
            elif t == STONE:
                grid[y, x] = ICE

    # Jungle band: dirt -> mud, deeper too.
    for x in range(max(0, jungleX - jungleHalf), min(width, jungleX + jungleHalf)):
        for y in range(hell):
            t = grid[y, x]
            if t == DIRT or t == GRASS:
                grid[y, x] = MUD

    # Desert band: dirt/grass -> sand, surface stone -> hardened sand,
    # deeper stone -> sandstone for ant-hive feel.
    for x in range(max(0, desertX - desertHalf), min(width, desertX + desertHalf)):
        for y in range(rock + 80):
            t = grid[y, x]
            if t == DIRT or t == GRASS:
                grid[y, x] = SAND
            elif t == STONE and y < rock + 30:
                grid[y, x] = HARDENED_SAND
            elif t == STONE:
                grid[y, x] = SANDSTONE_BLOCK

    # Evil band: surface dirt -> corrupt/crimson dirt; stone -> ebonstone/crimstone.
    evilDirt = CORRUPT_DIRT if evilType == "corruption" else CRIMSON_DIRT
    evilStone = EBONSTONE if evilType == "corruption" else CRIMSTONE
    for x in range(max(0, evilX - evilHalf), min(width, evilX + evilHalf)):
        for y in range(hell):
            t = grid[y, x]
            if t == DIRT or t == GRASS:
                grid[y, x] = evilDirt
            elif t == STONE:
                grid[y, x] = evilStone

    return {
        "spawnX": spawnX,
        "jungleX": jungleX,
        "desertX": desertX,
        "snowX": snowX,
        "evilX": evilX,
    }


def _carveCaves(
    grid: npt.NDArray[np.int32],
    layers: LayerDepths,
    rng: np.random.Generator,
) -> None:
    """Aggressive cavinator pass + smoothing to hit ~30-40% air underground."""
    height, width = grid.shape
    rockTop = int(layers.worldSurface) + 8
    hellTop = int(layers.hellLayer) - 4

    # Many small surface caves.
    for _ in range(int(width * 0.18)):
        sx = int(rng.integers(20, width - 20))
        sy = int(rng.integers(rockTop, int(layers.rockLayer)))
        tileRunner(grid, sx, sy, float(rng.uniform(5.0, 9.0)),
                   int(rng.integers(40, 90)), tileType=-1,
                   seed=int(rng.integers(0, 1 << 30)))

    # Big underground macro caverns via cavinator.
    for _ in range(int(width * 0.05)):
        sx = int(rng.integers(20, width - 20))
        sy = int(rng.integers(int(layers.rockLayer) + 20, hellTop))
        cavinator(grid, sx, sy, float(rng.uniform(40.0, 80.0)),
                  int(rng.integers(80, 180)),
                  seed=int(rng.integers(0, 1 << 30)))

    # Smooth so cave edges look organic.
    cellularAutomataSmooth(grid, iterations=2,
                           birthThreshold=5, deathThreshold=3)


# Pre-Hardmode ore configs: (oreId, depthMin, depthMax, attempts, strength).
_PRE_HM_ORES = [
    (COPPER, 0.05, 0.40, 280, 4.5),
    (TIN, 0.05, 0.40, 220, 4.5),
    (IRON, 0.20, 0.65, 260, 5.0),
    (LEAD, 0.20, 0.65, 200, 5.0),
    (SILVER, 0.40, 0.85, 200, 5.5),
    (TUNGSTEN, 0.40, 0.85, 160, 5.5),
    (GOLD, 0.55, 0.92, 160, 5.5),
    (PLATINUM, 0.55, 0.92, 130, 5.5),
]

_HM_TIER_1 = [(COBALT, 0.55, 0.85, 110, 5.5), (PALLADIUM, 0.55, 0.85, 110, 5.5)]
_HM_TIER_2 = [(MYTHRIL, 0.65, 0.90, 95, 5.5), (ORICHALCUM, 0.65, 0.90, 95, 5.5)]
_HM_TIER_3 = [(ADAMANTITE, 0.75, 0.95, 80, 5.5), (TITANIUM, 0.75, 0.95, 80, 5.5)]


def _placeOres(
    grid: npt.NDArray[np.int32],
    layers: LayerDepths,
    rng: np.random.Generator,
    altarsSmashed: int,
) -> None:
    height, width = grid.shape
    # Pre-HM (always present).
    for oreId, dMin, dMax, attempts, strength in _PRE_HM_ORES:
        yMin = int(dMin * height)
        yMax = int(dMax * height)
        for _ in range(attempts):
            ox = int(rng.integers(10, width - 10))
            oy = int(rng.integers(yMin, yMax))
            tileRunner(grid, ox, oy, strength, int(rng.integers(2, 7)),
                       tileType=oreId, addTile=False, overRide=False,
                       seed=int(rng.integers(0, 1 << 30)))

    if altarsSmashed >= 3:
        # Pearlstone backbone (light pink hardmode rock seam) at rockLayer.
        for _ in range(int(width * 0.04)):
            ox = int(rng.integers(10, width - 10))
            oy = int(rng.integers(int(layers.rockLayer), int(layers.hellLayer) - 80))
            tileRunner(grid, ox, oy, 8.0, 30, tileType=PEARLSTONE,
                       addTile=False, overRide=False,
                       seed=int(rng.integers(0, 1 << 30)))
        for oreId, dMin, dMax, attempts, strength in _HM_TIER_1:
            yMin = int(dMin * height)
            yMax = int(dMax * height)
            for _ in range(attempts):
                ox = int(rng.integers(10, width - 10))
                oy = int(rng.integers(yMin, yMax))
                tileRunner(grid, ox, oy, strength, int(rng.integers(3, 6)),
                           tileType=oreId, addTile=False, overRide=False,
                           seed=int(rng.integers(0, 1 << 30)))
    if altarsSmashed >= 6:
        for oreId, dMin, dMax, attempts, strength in _HM_TIER_2:
            yMin = int(dMin * height)
            yMax = int(dMax * height)
            for _ in range(attempts):
                ox = int(rng.integers(10, width - 10))
                oy = int(rng.integers(yMin, yMax))
                tileRunner(grid, ox, oy, strength, int(rng.integers(3, 6)),
                           tileType=oreId, addTile=False, overRide=False,
                           seed=int(rng.integers(0, 1 << 30)))
    if altarsSmashed >= 9:
        for oreId, dMin, dMax, attempts, strength in _HM_TIER_3:
            yMin = int(dMin * height)
            yMax = int(dMax * height)
            for _ in range(attempts):
                ox = int(rng.integers(10, width - 10))
                oy = int(rng.integers(yMin, yMax))
                tileRunner(grid, ox, oy, strength, int(rng.integers(3, 6)),
                           tileType=oreId, addTile=False, overRide=False,
                           seed=int(rng.integers(0, 1 << 30)))
        # Chlorophyte in the jungle mud band.
        mudYs, mudXs = np.where(grid == MUD)
        if mudYs.size > 0:
            for _ in range(80):
                idx = int(rng.integers(0, mudYs.size))
                ox = int(mudXs[idx])
                oy = int(mudYs[idx])
                tileRunner(grid, ox, oy, 4.5, 4, tileType=CHLOROPHYTE,
                           addTile=False, overRide=False,
                           seed=int(rng.integers(0, 1 << 30)))


def generateSmallWorld(
    seed: int = 20260423,
    evilType: str = "corruption",
    altarsSmashed: int = 0,
    compactBiomes: bool = False,
) -> SmallWorld:
    """Generate a SMALL-world tile grid with surface, biomes, caves, and ores.

    Args:
        seed: RNG seed for full reproducibility.
        evilType: ``"corruption"`` or ``"crimson"``.
        altarsSmashed: 0 (pre-HM), 3, 6, or 9 (Hardmode tiers unlocked).
        compactBiomes: If True, pack Forest/Jungle/Desert/Evil within ~520
            tiles of world center so a single 600-wide crop captures them.
            If False, use realistic game placement (biomes spread across the
            full 4200-wide world).

    Returns:
        ``SmallWorld`` payload with the grid plus biome center X-coordinates.
    """
    rng = np.random.default_rng(seed)
    layers = LayerDepths.forSmall()
    grid = np.full((SMALL.height, SMALL.width), STONE, dtype=np.int32)

    _carveSurface(grid, layers, rng)
    biomeCenters = _placeBiomes(grid, layers, rng, evilType, compact=compactBiomes)
    _carveCaves(grid, layers, rng)
    _placeOres(grid, layers, rng, altarsSmashed)

    return SmallWorld(
        grid=grid,
        layers=layers,
        spawnX=biomeCenters["spawnX"],
        jungleX=biomeCenters["jungleX"],
        desertX=biomeCenters["desertX"],
        snowX=biomeCenters["snowX"],
        evilX=biomeCenters["evilX"],
        evilType=evilType,
    )


__all__ = ["SmallWorld", "generateSmallWorld"]
