"""TerraExplorer: an educational, deterministic world-generation laboratory.

The package models the public shape of Terraria's vanilla generation pipeline
without using Re-Logic art or claiming seed-for-seed compatibility.
"""

from terraexplorer.config import Evil, WorldConfig, WorldScale
from terraexplorer.model import GeneratedWorld
from terraexplorer.pipeline import GenerationCancelledError, TerraExplorerPipeline, generate_world

__version__ = "1.0.0"

__all__ = [
    "Evil",
    "GeneratedWorld",
    "GenerationCancelledError",
    "TerraExplorerPipeline",
    "WorldConfig",
    "WorldScale",
    "generate_world",
]
