"""
Engine: Core Terraria world generation algorithms.

Provides accurate implementations of TileRunner, digTunnel, Cavinator,
cellular automata smoothing, SettleLiquids, StructureMap, and game constants
derived from decompiled C# source analysis.
"""

from Engine.constants import WorldSize, LayerDepths, StructureQuotas, OreConfig
from Engine.constants import (
    SMALL, MEDIUM, LARGE, FEATURE_PLOT, DETAIL_PLOT,
    DungeonConfig, TempleConfig, PyramidConfig, LivingTreeConfig, ShimmerConfig,
    LIFE_CRYSTAL, ALTAR, HALLOW_GRASS,
)
from Engine.algorithms import tileRunner, digTunnel, cavinator, cellularAutomataSmooth
from Engine.algorithms import settleLiquids
from Engine.structureMap import StructureMap
from Engine.theme import (
    applyTokyoNight, COLORS, PALETTE, BIOME_COLORS, TILE_COLORS, ORE_COLORS,
    DEFAULT_TILE_COLOR, buildTileColormap, seqCmap, divCmap, lightCmap,
)
from Engine import spriteRenderer
from Engine.spriteRenderer import cropSmallWorld, applyMapDecorations, drawTileGrid
from Engine.worldgen import SmallWorld, generateSmallWorld
from Engine.structures import (
    generateDungeon, generateJungleTemple, generateLivingTree,
    generatePyramid, generateSpiderCave, generateGemCave,
    generateUndergroundDesert, generateMarbleCave, generateGraniteCave,
    generateShimmerBiome, generateFloatingIslandHouse,
    rocksInDirt, dirtInRocks, placeClay, placeSilt,
    placeSunflowers, placeTraps, placePots, placeMinecartTracks,
    dropMeteor, clentaminatorSpray, generateMushroomBiome, spreadGrass,
)

__all__ = [
    "WorldSize", "LayerDepths", "StructureQuotas", "OreConfig",
    "SMALL", "MEDIUM", "LARGE", "FEATURE_PLOT", "DETAIL_PLOT",
    "DungeonConfig", "TempleConfig", "PyramidConfig", "LivingTreeConfig",
    "ShimmerConfig",
    "tileRunner", "digTunnel", "cavinator", "cellularAutomataSmooth",
    "settleLiquids",
    "StructureMap",
    "applyTokyoNight", "COLORS", "PALETTE", "BIOME_COLORS", "TILE_COLORS",
    "ORE_COLORS", "DEFAULT_TILE_COLOR", "buildTileColormap",
    "seqCmap", "divCmap", "lightCmap",
    "spriteRenderer",
    "cropSmallWorld", "applyMapDecorations", "drawTileGrid",
    "SmallWorld", "generateSmallWorld",
    "generateDungeon", "generateJungleTemple", "generateLivingTree",
    "generatePyramid", "generateSpiderCave", "generateGemCave",
    "generateUndergroundDesert", "generateMarbleCave", "generateGraniteCave",
    "generateShimmerBiome", "generateFloatingIslandHouse",
    "rocksInDirt", "dirtInRocks", "placeClay", "placeSilt",
    "placeSunflowers", "placeTraps", "placePots", "placeMinecartTracks",
    "dropMeteor", "clentaminatorSpray", "generateMushroomBiome", "spreadGrass",
]
