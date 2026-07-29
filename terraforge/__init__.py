"""Compatibility aliases for the former TerraForge package name."""

from terraexplorer import (
    Evil,
    GeneratedWorld,
    GenerationCancelledError,
    TerraExplorerPipeline,
    WorldConfig,
    WorldScale,
    generate_world,
)

TerraForgePipeline = TerraExplorerPipeline

__all__ = [
    "Evil",
    "GeneratedWorld",
    "GenerationCancelledError",
    "TerraExplorerPipeline",
    "TerraForgePipeline",
    "WorldConfig",
    "WorldScale",
    "generate_world",
]
