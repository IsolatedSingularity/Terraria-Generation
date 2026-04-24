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
    ADAMANTITE,
    AIR,
    ASH,
    CHLOROPHYTE,
    COBALT,
    COPPER,
    CORRUPT_DIRT,
    CRIMSON_DIRT,
    CRIMSTONE,
    DIRT,
    EBONSTONE,
    GOLD,
    GRASS,
    HARDENED_SAND,
    HELLSTONE,
    ICE,
    IRON,
    LEAD,
    MUD,
    MYTHRIL,
    ORICHALCUM,
    PALLADIUM,
    PEARLSTONE,
    PLATINUM,
    SAND,
    SANDSTONE_BLOCK,
    SILVER,
    SNOW,
    STONE,
    TIN,
    TITANIUM,
    TUNGSTEN,
    cavinator,
    cellularAutomataSmooth,
    tileRunner,
)
from Engine.constants import SMALL, TINY, LayerDepths
from Engine.structures import dirtInRocks, rocksInDirt


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
    """Aggressive cavinator + CA smoothing producing reference-style lacy
    underground (mostly air with organic dirt/stone islands).

    Strategy:
        1. Stone patches in dirt (rocksInDirt) and dirt patches in rock
           (dirtInRocks) so the caverns look mixed, not stratified.
        2. Heavy cavinator chamber pass spanning surface+8 to hellLayer-4.
        3. Thinner cavinator "connector" pass for tunnel variety.
        4. Multi-iteration CA smoothing tuned to GROW caves (death=4 so
           half-buried tiles erode), producing the lacy look.
    """
    height, width = grid.shape
    rockTop = int(layers.worldSurface) + 8
    hellTop = int(layers.hellLayer) - 4
    surfaceY = int(layers.worldSurface)
    rockY = int(layers.rockLayer)

    # Stone-in-dirt and dirt-in-rock to break up the strata.
    rocksInDirt(grid, count=int(width * 0.06),
                worldSurface=surfaceY, rockLayer=rockY,
                seed=int(rng.integers(0, 1 << 30)))
    dirtInRocks(grid, count=int(width * 0.05),
                rockLayer=rockY, hellLayer=hellTop,
                seed=int(rng.integers(0, 1 << 30)))

    # Big macro chambers covering the entire underground (not just rockLayer+).
    # Cavinator with strength 18-32 produces ~9-16 tile radius blobs that
    # bounce and merge into organic chambers.
    chamberCount = int(width * 0.55)
    for _ in range(chamberCount):
        sx = int(rng.integers(20, width - 20))
        sy = int(rng.integers(rockTop, hellTop))
        cavinator(grid, sx, sy,
                  float(rng.uniform(22.0, 38.0)),
                  int(rng.integers(45, 95)),
                  seed=int(rng.integers(0, 1 << 30)))

    # Smaller scattered carve passes for finer texture.
    for _ in range(int(width * 0.45)):
        sx = int(rng.integers(20, width - 20))
        sy = int(rng.integers(rockTop, hellTop))
        cavinator(grid, sx, sy,
                  float(rng.uniform(9.0, 16.0)),
                  int(rng.integers(20, 50)),
                  seed=int(rng.integers(0, 1 << 30)))

    # CA smoothing tuned to grow caves: death=4 means tiles with <=3 solid
    # neighbors erode, so isolated dirt/stone fingers vanish and chambers
    # merge into the lacy reference look.
    cellularAutomataSmooth(grid, iterations=5,
                           birthThreshold=5, deathThreshold=4)


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
    _carveCaves(grid, layers, rng)
    biomeCenters = _placeBiomes(grid, layers, rng, evilType, compact=compactBiomes)
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


__all__ = [
    "SmallWorld",
    "generateSmallWorld",
    "MiniWorld",
    "generateMiniWorld",
    "renderMiniWorld",
]


# ===========================================================================
# TINY world primitive (240 x 140) for native-resolution renders.
# All scripts default to this so individual tiles render at ~6 px/tile in a
# standard 1440 x 840 figure. No cropping anywhere.
# ===========================================================================


@dataclass
class MiniWorld:
    """Generated TINY-world payload."""
    grid: npt.NDArray[np.int32]
    layers: LayerDepths
    spawnX: int
    snowX: int
    jungleX: int
    desertX: int
    evilX: int
    evilType: str


def _miniSurface(
    grid: npt.NDArray[np.int32],
    layers: LayerDepths,
    rng: np.random.Generator,
) -> None:
    height, width = grid.shape
    base = layers.worldSurface
    xs = np.arange(width)
    surface = np.full(width, base, dtype=np.float64)
    for amp, freq in [(3.5, 0.045), (1.8, 0.12), (0.9, 0.30)]:
        surface += amp * np.sin(2 * np.pi * freq * xs + rng.uniform(0, 2 * np.pi))
    surfaceInt = surface.astype(np.int32)

    dirtBottom = int(layers.rockLayer)
    for x in range(width):
        sy = int(surfaceInt[x])
        grid[:sy, x] = AIR
        grid[sy, x] = GRASS
        grid[sy + 1: dirtBottom, x] = DIRT
        grid[dirtBottom:, x] = STONE
    grid[layers.hellLayer:, :] = ASH
    grid[layers.hellLayer + 3:, :] = HELLSTONE


def _miniCaves(
    grid: npt.NDArray[np.int32],
    layers: LayerDepths,
    rng: np.random.Generator,
) -> None:
    """Aggressive cave carving tuned for the 240x140 TINY world.

    Uses absolute brush strengths (not world-fraction scaled) so caves stay
    visible. Target air ratio in the rock layer: 45-55 percent.
    """
    height, width = grid.shape
    rockTop = int(layers.worldSurface) + 2
    hellTop = int(layers.hellLayer) - 2

    # Stone-in-dirt and dirt-in-rock to mottle the strata.
    rocksInDirt(grid, count=int(width * 0.10),
                worldSurface=int(layers.worldSurface),
                rockLayer=int(layers.rockLayer),
                seed=int(rng.integers(0, 1 << 30)))
    dirtInRocks(grid, count=int(width * 0.08),
                rockLayer=int(layers.rockLayer),
                hellLayer=hellTop,
                seed=int(rng.integers(0, 1 << 30)))

    # Big chambers: ~width * 0.55 passes with strength 4-7.
    chamberCount = int(width * 0.55)
    for _ in range(chamberCount):
        sx = int(rng.integers(4, width - 4))
        sy = int(rng.integers(rockTop, hellTop))
        cavinator(grid, sx, sy,
                  float(rng.uniform(4.0, 6.5)),
                  int(rng.integers(20, 42)),
                  seed=int(rng.integers(0, 1 << 30)))

    # Small connectors so chambers chain into tunnels.
    for _ in range(int(width * 0.45)):
        sx = int(rng.integers(4, width - 4))
        sy = int(rng.integers(rockTop, hellTop))
        cavinator(grid, sx, sy,
                  float(rng.uniform(2.0, 3.4)),
                  int(rng.integers(12, 26)),
                  seed=int(rng.integers(0, 1 << 30)))

    # CA smoothing: birth=5, death=3 so chambers merge but solid pockets
    # survive, producing a lacy 45-55 percent air density.
    cellularAutomataSmooth(grid, iterations=3,
                           birthThreshold=5, deathThreshold=3)


def _miniBiomes(
    grid: npt.NDArray[np.int32],
    layers: LayerDepths,
    rng: np.random.Generator,
    evilType: str,
) -> dict[str, int]:
    """Place Snow, Forest (implicit), Jungle, Desert, Evil across 240 tiles.

    Layout (left to right): Snow | Forest (spawn) | Jungle | Desert | Evil.
    Half-widths sized so all four bands fit inside the world. Caves already
    carved, so this only converts SOLID tiles, leaving air pockets intact.
    """
    height, width = grid.shape
    spawnX = width // 2  # 120

    snowX = 28
    jungleX = 95
    desertX = 165
    evilX = 215

    snowHalf = 24
    jungleHalf = 26
    desertHalf = 22
    evilHalf = 22

    rock = int(layers.rockLayer)
    hell = int(layers.hellLayer)

    for x in range(max(0, snowX - snowHalf), min(width, snowX + snowHalf)):
        for y in range(rock):
            t = grid[y, x]
            if t == DIRT or t == GRASS:
                grid[y, x] = SNOW
            elif t == STONE:
                grid[y, x] = ICE

    for x in range(max(0, jungleX - jungleHalf), min(width, jungleX + jungleHalf)):
        for y in range(hell):
            t = grid[y, x]
            if t == DIRT or t == GRASS:
                grid[y, x] = MUD

    for x in range(max(0, desertX - desertHalf), min(width, desertX + desertHalf)):
        for y in range(rock + 12):
            t = grid[y, x]
            if t == DIRT or t == GRASS:
                grid[y, x] = SAND
            elif t == STONE and y < rock + 4:
                grid[y, x] = HARDENED_SAND
            elif t == STONE:
                grid[y, x] = SANDSTONE_BLOCK

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
        "snowX": snowX,
        "jungleX": jungleX,
        "desertX": desertX,
        "evilX": evilX,
    }


# Pre-Hardmode ore configs at TINY scale (oreId, depthMin, depthMax,
# attempts, strength). Attempts are absolute (not width-scaled) so each ore
# family lays down 8-25 visible veins in the 240x140 world.
_MINI_PRE_HM_ORES = [
    (COPPER, 0.20, 0.55, 22, 2.5),
    (TIN, 0.20, 0.55, 18, 2.5),
    (IRON, 0.30, 0.70, 22, 2.7),
    (LEAD, 0.30, 0.70, 18, 2.7),
    (SILVER, 0.45, 0.85, 18, 2.8),
    (TUNGSTEN, 0.45, 0.85, 14, 2.8),
    (GOLD, 0.55, 0.90, 14, 2.8),
    (PLATINUM, 0.55, 0.90, 12, 2.8),
]
_MINI_HM_TIER_1 = [(COBALT, 0.55, 0.85, 14, 2.8), (PALLADIUM, 0.55, 0.85, 14, 2.8)]
_MINI_HM_TIER_2 = [(MYTHRIL, 0.65, 0.90, 12, 2.8), (ORICHALCUM, 0.65, 0.90, 12, 2.8)]
_MINI_HM_TIER_3 = [(ADAMANTITE, 0.75, 0.95, 10, 2.8), (TITANIUM, 0.75, 0.95, 10, 2.8)]


def _miniOres(
    grid: npt.NDArray[np.int32],
    layers: LayerDepths,
    rng: np.random.Generator,
    altarsSmashed: int,
    oreScale: float = 1.0,
) -> None:
    height, width = grid.shape

    def runFamily(family):
        for oreId, dMin, dMax, attempts, strength in family:
            yMin = int(dMin * height)
            yMax = max(yMin + 1, int(dMax * height))
            count = max(1, int(attempts * oreScale))
            for _ in range(count):
                ox = int(rng.integers(2, width - 2))
                oy = int(rng.integers(yMin, yMax))
                tileRunner(grid, ox, oy, strength, int(rng.integers(2, 5)),
                           tileType=oreId, addTile=False, overRide=False,
                           seed=int(rng.integers(0, 1 << 30)))

    runFamily(_MINI_PRE_HM_ORES)
    if altarsSmashed >= 3:
        for _ in range(int(width * 0.06 * oreScale)):
            ox = int(rng.integers(2, width - 2))
            oy = int(rng.integers(int(layers.rockLayer),
                                  max(int(layers.rockLayer) + 1,
                                      int(layers.hellLayer) - 8)))
            tileRunner(grid, ox, oy, 3.5, 12, tileType=PEARLSTONE,
                       addTile=False, overRide=False,
                       seed=int(rng.integers(0, 1 << 30)))
        runFamily(_MINI_HM_TIER_1)
    if altarsSmashed >= 6:
        runFamily(_MINI_HM_TIER_2)
    if altarsSmashed >= 9:
        runFamily(_MINI_HM_TIER_3)
        mudYs, mudXs = np.where(grid == MUD)
        if mudYs.size > 0:
            for _ in range(int(20 * oreScale)):
                idx = int(rng.integers(0, mudYs.size))
                ox = int(mudXs[idx])
                oy = int(mudYs[idx])
                tileRunner(grid, ox, oy, 2.2, 3, tileType=CHLOROPHYTE,
                           addTile=False, overRide=False,
                           seed=int(rng.integers(0, 1 << 30)))


def generateMiniWorld(
    seed: int = 20260423,
    evilType: str = "corruption",
    altarsSmashed: int = 0,
    oreScale: float = 1.0,
    skipBiomes: bool = False,
    skipCaves: bool = False,
    skipOres: bool = False,
) -> MiniWorld:
    """Generate a 240x140 TINY world rendered natively at ~6 px per tile.

    Args:
        seed: RNG seed for full reproducibility.
        evilType: ``"corruption"`` or ``"crimson"``.
        altarsSmashed: 0 (pre-HM), 3, 6, or 9 (Hardmode tiers unlocked).
        oreScale: Multiplier on ore-vein attempt counts (e.g. 10 for the
            ore-distribution figure where density needs to pop visually).
        skipBiomes / skipCaves / skipOres: Stage skips for animations that
            need intermediate snapshots (e.g. show a pre-cave snapshot).

    Returns:
        ``MiniWorld`` payload with the full 240x140 grid plus biome centers.
    """
    rng = np.random.default_rng(seed)
    layers = LayerDepths.forTiny()
    grid = np.full((TINY.height, TINY.width), STONE, dtype=np.int32)

    _miniSurface(grid, layers, rng)
    if not skipCaves:
        _miniCaves(grid, layers, rng)
    if not skipBiomes:
        centers = _miniBiomes(grid, layers, rng, evilType)
    else:
        centers = {"spawnX": 120, "snowX": 28, "jungleX": 95,
                   "desertX": 165, "evilX": 215}
    if not skipOres:
        _miniOres(grid, layers, rng, altarsSmashed, oreScale=oreScale)

    return MiniWorld(
        grid=grid,
        layers=layers,
        spawnX=centers["spawnX"],
        snowX=centers["snowX"],
        jungleX=centers["jungleX"],
        desertX=centers["desertX"],
        evilX=centers["evilX"],
        evilType=evilType,
    )


def renderMiniWorld(
    grid: npt.NDArray[np.int32],
    ax,
    title: str | None = None,
    showLayers: bool = False,
    layers: LayerDepths | None = None,
    highlightTiles: set[int] | None = None,
    dimAlpha: float = 0.30,
):
    """Render a TINY world to the given Matplotlib axes at native resolution.

    Args:
        grid: 240x140 tile grid.
        ax: Matplotlib axes.
        title: Concise plot title (no parens, no tile dims).
        showLayers: If True, draw dashed worldSurface/rockLayer/hellLayer lines.
        layers: Required when ``showLayers=True``.
        highlightTiles: If set, dim every other tile to ``dimAlpha`` so the
            chosen ore family (or biome) pops. Used by the ore-distribution
            figure.
        dimAlpha: Opacity for non-highlighted tiles.

    Returns:
        ``AxesImage`` from matplotlib's ``imshow``.
    """
    from Engine.theme import PALETTE, buildTileColormap  # local import: avoid cycles

    cmap = buildTileColormap()
    if highlightTiles is None:
        im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=200,
                       interpolation="nearest", aspect="equal")
    else:
        # Render a dim base layer + bright overlay for highlighted tiles.
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=200,
                  interpolation="nearest", aspect="equal", alpha=dimAlpha)
        mask = np.isin(grid, list(highlightTiles))
        overlay = np.where(mask, grid, 0)
        im = ax.imshow(overlay, cmap=cmap, vmin=0, vmax=200,
                       interpolation="nearest", aspect="equal", alpha=1.0)

    if showLayers and layers is not None:
        height, width = grid.shape
        ax.axhline(layers.worldSurface, color=PALETTE["cyan"],
                   linestyle="--", linewidth=0.8, alpha=0.55)
        ax.axhline(layers.rockLayer, color=PALETTE["yellow"],
                   linestyle="--", linewidth=0.8, alpha=0.55)
        ax.axhline(layers.hellLayer, color=PALETTE["red"],
                   linestyle="--", linewidth=0.8, alpha=0.55)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, color=PALETTE["fg"], fontsize=12,
                     pad=6, loc="center")
    return im

