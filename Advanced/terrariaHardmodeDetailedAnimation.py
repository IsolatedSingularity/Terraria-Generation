"""Detailed Hardmode animation rendered as a TINY world per frame.

Sequence:
  1. Pre-Hardmode baseline (a few hold frames so viewers see the start).
  2. V-pattern carving (interpolated reveal across ~6 frames).
  3. Altar smashing: cumulative ore appearance over ~10 frames as altars
     break and Hardmode tiers unlock (Cobalt -> Mythril -> Adamantite ->
     Chlorophyte).
  4. Final hold frames so the end state stays on-screen at the GIF loop point.

Total: ~30 frames at 6 fps. Every frame is the full 240x140 world.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from Engine.theme import COLORS, applyTokyoNight, buildTileColormap
from Engine.worldgen import generateMiniWorld

from Advanced.terrariaCorruptionEvolution import carveVPattern

applyTokyoNight()


def _buildFrames(seed: int = 20260427) -> list[np.ndarray]:
    """Build the cumulative-progression frame stack."""
    base = generateMiniWorld(seed=seed, evilType="corruption", altarsSmashed=0)
    grid = base.grid.copy()

    frames: list[np.ndarray] = []

    # Hold pre-HM baseline.
    for _ in range(3):
        frames.append(grid.copy())

    # V-pattern grows in 6 incremental passes (each adds tiles to the V).
    rng = np.random.default_rng(seed + 1)
    for step in range(6):
        carveVPattern(grid, evilType="corruption",
                      seed=int(rng.integers(0, 1 << 30)))
        frames.append(grid.copy())

    # Altar smashing: progressively layer ore tiers.
    for altars in (3, 3, 6, 6, 9, 9):
        nextWorld = generateMiniWorld(seed=seed, evilType="corruption",
                                      altarsSmashed=altars)
        # Overlay ores onto the V-pattern world by copying ore tiles only.
        oreMask = np.isin(nextWorld.grid, [110, 111, 112, 113, 114, 115, 116,
                                           64])  # Pearlstone + ores
        grid = np.where(oreMask, nextWorld.grid, grid)
        frames.append(grid.copy())

    # Hold final state.
    for _ in range(4):
        frames.append(grid.copy())
    return frames


def renderAnimation(savePath: str | None = None,
                    seed: int = 20260427) -> None:
    """Render the detailed Hardmode animation GIF."""
    if savePath is None:
        savePath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "Plots", "Advanced", "terraria_hardmode_animation.gif",
        )

    print("Building Hardmode animation frames...")
    frames = _buildFrames(seed=seed)

    cmap = buildTileColormap()
    fig, ax = plt.subplots(figsize=(14.4, 8.4))
    im = ax.imshow(frames[0], cmap=cmap, vmin=0, vmax=200,
                   interpolation="nearest", aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    titleObj = ax.set_title("Hardmode Transition", color=COLORS["fg"],
                            fontsize=13, fontweight="bold", pad=8)

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
    print("Terraria Hardmode Detailed Animation")
    print("=" * 40)
    renderAnimation()
    print("Done.")
