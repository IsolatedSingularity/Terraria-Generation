"""TerraForge: an educational, deterministic world-generation laboratory.

The package models the public shape of Terraria's vanilla generation pipeline
without using Re-Logic art or claiming seed-for-seed compatibility.
"""

from terraforge.config import Evil, WorldConfig, WorldScale
from terraforge.model import GeneratedWorld
from terraforge.pipeline import GenerationCancelledError, TerraForgePipeline, generate_world

__version__ = "1.0.0"

__all__ = [
    "Evil",
    "GeneratedWorld",
    "GenerationCancelledError",
    "TerraForgePipeline",
    "WorldConfig",
    "WorldScale",
    "generate_world",
]
