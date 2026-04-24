"""Terraria Crimson Evolution -- 4-panel SMALL-world crop figure + spread GIF.

Mirrors terrariaCorruptionEvolution.py but with ``evilType="crimson"``.
Outputs:
  - Plots/Advanced/crimson_evolution.png  (2x2 panel, 600x500 crops)
  - Plots/Advanced/crimson_spread.gif     (animated spread)
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt

from Advanced.terrariaCorruptionEvolution import TerrariaCorruptionEvolution, _gridToRgb
from Engine.theme import COLORS, applyTokyoNight

applyTokyoNight()

_PLOTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "Plots", "Advanced"
)


def createCrimsonEvolutionFigure(savePath: str | None = None) -> plt.Figure:
    """4-panel 600x500 SMALL-crop figure for crimson evolution."""
    from Engine.spriteRenderer import applyMapDecorations, cropSmallWorld, drawTileGrid
    from Engine.worldgen import generateSmallWorld

    if savePath is None:
        savePath = os.path.join(_PLOTS_DIR, "crimson_evolution.png")

    print("Generating SMALL world for crimson evolution (seed=20260424)...")
    world = generateSmallWorld(seed=20260424, evilType="crimson", compactBiomes=True)
    layers = world.layers
    centerX = world.spawnX
    centerY = int((layers.worldSurface + layers.rockLayer) / 2)

    sim = TerrariaCorruptionEvolution(
        worldWidth=world.grid.shape[1], worldHeight=world.grid.shape[0],
        evilType="crimson", seed=20260424,
    )
    sim.grid = world.grid.copy()
    sim.layers = layers
    sim.world = sim.grid

    snapPreHM = sim.grid.copy()

    sim.triggerHardmode()
    snapV = sim.grid.copy()

    sim.simulateSpread(5000.0)
    snapSpread1 = sim.grid.copy()

    for _ in range(9):
        sim.simulateSpread(5000.0)
    snapSpread2 = sim.grid.copy()

    titles = [
        "Phase 1: Pre-Hardmode Crimson Pockets",
        "Phase 2: Hardmode V-Pattern (WoF Defeated)",
        "Phase 3: Spread T+1 (~5 000 s)",
        "Phase 4: Spread T+10 (~50 000 s)",
    ]
    snaps = [snapPreHM, snapV, snapSpread1, snapSpread2]

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    for ax, snap, title in zip(axes.flat, snaps, titles):
        cropped, bounds = cropSmallWorld(snap, centerX=centerX, centerY=centerY,
                                          width=600, height=500)
        drawTileGrid(ax, cropped)
        applyMapDecorations(ax, cropped, layers, cropBounds=bounds)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("X (tiles, crop-local)")
        ax.set_ylabel("Depth (tiles, crop-local)")

    fig.suptitle(
        "Crimson Evolution (600x500 SMALL-World Crop)",
        fontsize=14, fontweight="bold", y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(os.path.dirname(os.path.abspath(savePath)), exist_ok=True)
    fig.savefig(savePath, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    print(f"Saved: {savePath}")
    plt.close(fig)
    return fig


def createCrimsonSpreadAnimation(savePath: str | None = None) -> None:
    """Animated GIF showing crimson spread (SMALL world crop)."""
    from matplotlib.animation import FuncAnimation
    from Engine.worldgen import generateSmallWorld

    if savePath is None:
        savePath = os.path.join(_PLOTS_DIR, "crimson_spread.gif")

    print("Generating SMALL world for crimson spread GIF...")
    world = generateSmallWorld(seed=20260424, evilType="crimson", compactBiomes=True)
    layers = world.layers
    centerX = world.spawnX
    centerY = int((layers.worldSurface + layers.rockLayer) / 2)
    h0, w0 = world.grid.shape

    x0 = max(0, centerX - 300)
    x1 = min(w0, centerX + 300)
    y0 = max(0, centerY - 250)
    y1 = min(h0, centerY + 250)

    def _crop(grid: np.ndarray) -> np.ndarray:
        return grid[y0:y1, x0:x1]

    sim = TerrariaCorruptionEvolution(
        worldWidth=w0, worldHeight=h0,
        evilType="crimson", seed=20260424,
    )
    sim.grid = world.grid.copy()
    sim.layers = layers
    sim.world = sim.grid

    frames: list[np.ndarray] = [_crop(sim.grid)]
    for _ in range(10):
        sim.simulateSpread(1000.0)
        frames.append(_crop(sim.grid))

    sim.triggerHardmode()
    frames.append(_crop(sim.grid))
    hmFrame = len(frames) - 1

    for _ in range(30):
        sim.simulateSpread(3000.0)
        frames.append(_crop(sim.grid))

    fig, ax = plt.subplots(figsize=(13, 10))
    im = ax.imshow(_gridToRgb(frames[0]), aspect="auto", interpolation="nearest",
                   origin="upper")
    ax.set_xlabel("X (tiles, crop-local)")
    ax.set_ylabel("Depth (tiles, crop-local)")
    titleObj = ax.set_title("", fontsize=12, fontweight="bold")

    def _update(f: int):
        im.set_data(_gridToRgb(frames[f]))
        phase = "HARDMODE" if f >= hmFrame else "Pre-Hardmode"
        titleObj.set_text(
            f"Crimson Spread -- Frame {f}/{len(frames) - 1} [{phase}]"
        )
        return [im, titleObj]

    anim = FuncAnimation(fig, _update, frames=len(frames), interval=200, blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(savePath)), exist_ok=True)
    anim.save(savePath, writer="pillow", fps=5)
    print(f"Saved: {savePath}")
    plt.close(fig)


def main() -> None:
    print("Terraria Crimson Evolution")
    print("=" * 40)
    createCrimsonEvolutionFigure()
    createCrimsonSpreadAnimation()
    print("Done.")


if __name__ == "__main__":
    main()
