"""
Engine: Core Terraria world generation algorithms.

Provides accurate implementations of TileRunner, digTunnel, Cavinator,
cellular automata smoothing, SettleLiquids, StructureMap, and game constants
derived from decompiled C# source analysis.
"""

from Engine.constants import WorldSize, LayerDepths, StructureQuotas, OreConfig
from Engine.constants import (
    DungeonConfig, TempleConfig, PyramidConfig, LivingTreeConfig, ShimmerConfig,
)
from Engine.algorithms import tileRunner, digTunnel, cavinator, cellularAutomataSmooth
from Engine.algorithms import settleLiquids
from Engine.structureMap import StructureMap
from Engine.theme import applyDarkTheme, COLORS, BIOME_COLORS, TILE_COLORS, ORE_COLORS
from Engine.theme import seqCmap, divCmap, lightCmap
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
    "DungeonConfig", "TempleConfig", "PyramidConfig", "LivingTreeConfig",
    "ShimmerConfig",
    "tileRunner", "digTunnel", "cavinator", "cellularAutomataSmooth",
    "settleLiquids",
    "StructureMap",
    "applyDarkTheme", "COLORS", "BIOME_COLORS", "TILE_COLORS", "ORE_COLORS",
    "seqCmap", "divCmap", "lightCmap",
    "generateDungeon", "generateJungleTemple", "generateLivingTree",
    "generatePyramid", "generateSpiderCave", "generateGemCave",
    "generateUndergroundDesert", "generateMarbleCave", "generateGraniteCave",
    "generateShimmerBiome", "generateFloatingIslandHouse",
    "rocksInDirt", "dirtInRocks", "placeClay", "placeSilt",
    "placeSunflowers", "placeTraps", "placePots", "placeMinecartTracks",
    "dropMeteor", "clentaminatorSpray", "generateMushroomBiome", "spreadGrass",
]
