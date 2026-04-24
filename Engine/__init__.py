"""
Engine: Core Terraria world generation algorithms.

Provides accurate implementations of TileRunner, digTunnel, Cavinator,
cellular automata smoothing, SettleLiquids, StructureMap, and game constants
derived from decompiled C# source analysis.
"""

from Engine import spriteRenderer
from Engine.algorithms import (
    cavinator,
    cellularAutomataSmooth,
    digTunnel,
    settleLiquids,
    tileRunner,
)
from Engine.constants import (
    ALTAR,  # noqa: F401
    DETAIL_PLOT,
    FEATURE_PLOT,
    HALLOW_GRASS,  # noqa: F401
    LARGE,
    LIFE_CRYSTAL,  # noqa: F401
    MEDIUM,
    SMALL,
    TINY,
    DungeonConfig,
    LayerDepths,
    LivingTreeConfig,
    OreConfig,
    PyramidConfig,
    ShimmerConfig,
    StructureQuotas,
    TempleConfig,
    WorldSize,
)
from Engine.spriteRenderer import applyMapDecorations, cropSmallWorld, drawTileGrid
from Engine.structureMap import StructureMap
from Engine.structures import (
    clentaminatorSpray,
    dirtInRocks,
    dropMeteor,
    generateDungeon,
    generateFloatingIslandHouse,
    generateGemCave,
    generateGraniteCave,
    generateJungleTemple,
    generateLivingTree,
    generateMarbleCave,
    generateMushroomBiome,
    generatePyramid,
    generateShimmerBiome,
    generateSpiderCave,
    generateUndergroundDesert,
    placeClay,
    placeMinecartTracks,
    placePots,
    placeSilt,
    placeSunflowers,
    placeTraps,
    rocksInDirt,
    spreadGrass,
)
from Engine.theme import (
    BIOME_COLORS,
    COLORS,
    DEFAULT_TILE_COLOR,
    ORE_COLORS,
    PALETTE,
    TILE_COLORS,
    applyTokyoNight,
    buildTileColormap,
    divCmap,
    lightCmap,
    saveTinyGif,
    seqCmap,
)
from Engine.worldgen import (
    MiniWorld,
    SmallWorld,
    generateMiniWorld,
    generateSmallWorld,
    renderMiniWorld,
)

__all__ = [
    "WorldSize", "LayerDepths", "StructureQuotas", "OreConfig",
    "SMALL", "MEDIUM", "LARGE", "TINY", "FEATURE_PLOT", "DETAIL_PLOT",
    "DungeonConfig", "TempleConfig", "PyramidConfig", "LivingTreeConfig",
    "ShimmerConfig",
    "tileRunner", "digTunnel", "cavinator", "cellularAutomataSmooth",
    "settleLiquids",
    "StructureMap",
    "applyTokyoNight", "COLORS", "PALETTE", "BIOME_COLORS", "TILE_COLORS",
    "ORE_COLORS", "DEFAULT_TILE_COLOR", "buildTileColormap", "saveTinyGif",
    "seqCmap", "divCmap", "lightCmap",
    "spriteRenderer",
    "cropSmallWorld", "applyMapDecorations", "drawTileGrid",
    "SmallWorld", "generateSmallWorld",
    "MiniWorld", "generateMiniWorld", "renderMiniWorld",
    "generateDungeon", "generateJungleTemple", "generateLivingTree",
    "generatePyramid", "generateSpiderCave", "generateGemCave",
    "generateUndergroundDesert", "generateMarbleCave", "generateGraniteCave",
    "generateShimmerBiome", "generateFloatingIslandHouse",
    "rocksInDirt", "dirtInRocks", "placeClay", "placeSilt",
    "placeSunflowers", "placeTraps", "placePots", "placeMinecartTracks",
    "dropMeteor", "clentaminatorSpray", "generateMushroomBiome", "spreadGrass",
]
