"""
Engine: Core Terraria world generation algorithms.

Provides accurate implementations of TileRunner, digTunnel, Cavinator,
cellular automata smoothing, SettleLiquids, StructureMap, and game constants
derived from decompiled C# source analysis.
"""

from Engine.constants import WorldSize, LayerDepths, StructureQuotas, OreConfig
from Engine.algorithms import tileRunner, digTunnel, cavinator, cellularAutomataSmooth
from Engine.algorithms import settleLiquids
from Engine.structureMap import StructureMap

__all__ = [
    "WorldSize",
    "LayerDepths",
    "StructureQuotas",
    "OreConfig",
    "tileRunner",
    "digTunnel",
    "cavinator",
    "cellularAutomataSmooth",
    "settleLiquids",
    "StructureMap",
]
