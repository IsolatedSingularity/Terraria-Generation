"""Master Evolution: hero GIF spanning the full TINY-world lifecycle.

Frame sequence (~25 frames at 6 fps):
  0-2 : Bare stone shell (pre-surface).
  3-4 : Surface carved + grass band.
  5-6 : Caves carved (lacy chambers appear).
  7-9 : Biomes painted (Snow, Jungle, Desert, Corruption tints).
  10-12: Pre-Hardmode ores added.
  13-16: V-pattern grows.
  17-21: Altar smashing reveals HM ore tiers.
  22-24: Late infection spread.

Every frame is the full 240x140 world rendered at native resolution. No
overview panel, no crop, no inset. Single axes, ~6 px/tile.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from Engine.algorithms import (
    AIR,
    DIRT,
    GRASS,
    HELLSTONE,
    STONE,
)
from Engine.constants import TINY, LayerDepths
from Engine.theme import COLORS, applyTokyoNight, buildTileColormap
from Engine.worldgen import (
    _miniBiomes,
    _miniCaves,
    _miniOres,
    _miniSurface,
    generateMiniWorld,
)

from Advanced.terrariaCorruptionEvolution import (
    carveVPattern,
    spreadInfection,
)

applyTokyoNight()


def _buildLifecycleFrames(seed: int = 20260423) -> list[np.ndarray]:
    """Replay the full generation pipeline, capturing snapshots between passes."""
    rng = np.random.default_rng(seed)
    layers = LayerDepths.forTiny()

    grid = np.full((TINY.height, TINY.width), STONE, dtype=np.int32)
    frames: list[np.ndarray] = []

    # 0: bare stone shell.
    for _ in range(2):
        frames.append(grid.copy())

    # 1: surface + dirt strata + hellstone shell.
    _miniSurface(grid, layers, rng)
    for _ in range(2):
        frames.append(grid.copy())

    # 2: caves carved.
    _miniCaves(grid, layers, rng)
    for _ in range(2):
        frames.append(grid.copy())

    # 3: biomes painted.
    _miniBiomes(grid, layers, rng, evilType="corruption")
    for _ in range(3):
        frames.append(grid.copy())

    # 4: pre-HM ores.
    _miniOres(grid, layers, rng, altarsSmashed=0)
    for _ in range(3):
        frames.append(grid.copy())

    # 5: V-pattern grows in 4 incremental passes.
    for _ in range(4):
        carveVPattern(grid, evilType="corruption",
                      seed=int(rng.integers(0, 1 << 30)))
        frames.append(grid.copy())

    # 6: altar smashing layers in tiers (3 -> 6 -> 9).
    for altars in (3, 6, 9, 9, 9):
        nextWorld = generateMiniWorld(seed=seed, evilType="corruption",
                                      altarsSmashed=altars)
        oreMask = np.isin(nextWorld.grid, [110, 111, 112, 113, 114, 115, 116,
                                           64])
        grid = np.where(oreMask, nextWorld.grid, grid)
        frames.append(grid.copy())

    # 7: late infection spread.
    for _ in range(4):
        spreadInfection(grid, cycles=2, seed=int(rng.integers(0, 1 << 30)))
        frames.append(grid.copy())

    return frames


def renderHeroAnimation(savePath: str | None = None,
                        seed: int = 20260423) -> None:
    """Render the master evolution hero GIF."""
    if savePath is None:
        savePath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "Plots", "Advanced", "terraria_master_evolution.gif",
        )

    print("Building master evolution frames...")
    frames = _buildLifecycleFrames(seed=seed)

    cmap = buildTileColormap()
    fig, ax = plt.subplots(figsize=(14.4, 8.4))
    im = ax.imshow(frames[0], cmap=cmap, vmin=0, vmax=200,
                   interpolation="nearest", aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    titleObj = ax.set_title("World Evolution", color=COLORS["fg"],
                            fontsize=14, fontweight="bold", pad=8)

    def _update(f: int):
        im.set_data(frames[f])
        return [im, titleObj]

    anim = FuncAnimation(fig, _update, frames=len(frames),
                         interval=170, blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(savePath)), exist_ok=True)
    anim.save(savePath, writer="pillow", fps=6)
    plt.close()
    print(f"Saved {savePath}")


if __name__ == "__main__":
    print("Terraria Master Evolution")
    print("=" * 40)
    renderHeroAnimation()
    print("Done.")
