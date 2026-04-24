"""TINY-world generation pipeline animation.

Replays the world generation pipeline pass-by-pass at native 240x140
resolution, capturing a snapshot after each named pass. The frames cover:

    Reset, Surface and Strata, Stone and Hellstone Shell, Caves Carved,
    CA Smoothing, Snow and Jungle, Desert, Corruption, Pre-Hardmode Ores,
    Hardmode V-Pattern x3, Altar Tier x3, Final World.

Output is a palette-quantized GIF saved via Engine.theme.saveTinyGif.
"""

from __future__ import annotations

import os

import numpy as np

from Engine.algorithms import STONE
from Engine.constants import TINY, LayerDepths
from Engine.theme import applyTokyoNight, saveTinyGif
from Engine.worldgen import (
    _miniBiomes,
    _miniCaves,
    _miniOres,
    _miniSurface,
    generateMiniWorld,
)

from Advanced.terrariaCorruptionEvolution import carveVPattern

applyTokyoNight()


def _savePath(filename: str) -> str:
    """Return full path under Plots/Advanced, creating the directory if needed."""
    baseDir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Plots", "Advanced",
    )
    os.makedirs(baseDir, exist_ok=True)
    return os.path.join(baseDir, filename)


def createWorldGenerationAnimation(
    saveName: str = "world_generation_animation.gif",
    seed: int = 20260423,
) -> None:
    """Render the TINY-world pipeline GIF.

    Each frame is the full 240x140 world after the named pass. Title is
    baked into the frame header, so the GIF needs no caption to read.
    """
    print("Building TINY-world pass-by-pass animation...")
    rng = np.random.default_rng(seed)
    layers = LayerDepths.forTiny()

    grid = np.full((TINY.height, TINY.width), STONE, dtype=np.int32)
    snapshots: list[tuple[str, np.ndarray]] = [
        ("Pass 1: Reset", grid.copy()),
    ]

    _miniSurface(grid, layers, rng)
    snapshots.append(("Pass 2: Surface and Strata", grid.copy()))
    snapshots.append(("Pass 3: Stone and Hellstone Shell", grid.copy()))

    _miniCaves(grid, layers, rng)
    snapshots.append(("Pass 4: Caves Carved", grid.copy()))
    snapshots.append(("Pass 5: CA Smoothing", grid.copy()))

    _miniBiomes(grid, layers, rng, evilType="corruption")
    snapshots.append(("Pass 6: Snow and Jungle Biomes", grid.copy()))
    snapshots.append(("Pass 7: Desert Biome", grid.copy()))
    snapshots.append(("Pass 8: Corruption Biome", grid.copy()))

    _miniOres(grid, layers, rng, altarsSmashed=0)
    snapshots.append(("Pass 9: Pre-Hardmode Ores", grid.copy()))

    for step in range(3):
        carveVPattern(grid, evilType="corruption",
                      seed=int(rng.integers(0, 1 << 30)))
        snapshots.append((f"Pass {10 + step}: Hardmode V-Pattern", grid.copy()))

    for tierIdx, altars in enumerate((3, 6, 9)):
        nextWorld = generateMiniWorld(seed=seed, evilType="corruption",
                                      altarsSmashed=altars)
        oreMask = np.isin(nextWorld.grid, [110, 111, 112, 113, 114, 115, 116,
                                           64])
        grid = np.where(oreMask, nextWorld.grid, grid)
        snapshots.append((f"Pass {13 + tierIdx}: Altar Tier {tierIdx + 1}",
                          grid.copy()))

    snapshots.append(("Pass 16: Final World", grid.copy()))

    titles = [s[0] for s in snapshots]
    grids = [s[1] for s in snapshots]
    path = _savePath(saveName)
    print(f"Saving animation to {path}")
    saveTinyGif(grids, path, fps=2, scale=5, title=titles)
    print("Animation saved.")


if __name__ == "__main__":
    print("Terraria World Generation Pipeline")
    print("=" * 40)
    createWorldGenerationAnimation()
    print("Done.")
