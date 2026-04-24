"""Terraria Structure Generation -- SMALL-world placement scene.

Single output figure ``terraria_structure_density.png``: a 600x500 crop of
a generated SMALL world with composer-rendered structures (underground
cabins, floating island, pyramid, dungeon corner) overlaid on real cave
and biome topology. Replaces the prior LARGE-world scatter density and the
4-panel detail figure with one tile-scale composition.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from Engine.algorithms import AIR, DIRT, EBONSTONE, SAND, STONE
from Engine.spriteRenderer import (
    Rect as SpriteRect,
)
from Engine.spriteRenderer import (
    cropSmallWorld,
    drawCabin,
    drawDungeon,
    drawFloatingIsland,
    drawPyramid,
    drawTileGrid,
)
from Engine.theme import COLORS, applyTokyoNight
from Engine.worldgen import generateSmallWorld

applyTokyoNight()


# ---------------------------------------------------------------------------
# Sub-grid builders for composer overlays
# ---------------------------------------------------------------------------
def _buildDungeonGrid(w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Stone background + brick room cutouts for dungeon detail."""
    grid = np.full((h, w), STONE, dtype=np.int32)
    walls = np.zeros((h, w), dtype=np.int32)
    walls[2:h - 2, 2:w - 2] = EBONSTONE
    for (x0, y0, rw, rh) in [(4, 6, 18, 12), (24, 8, 14, 10),
                             (6, 22, 18, 12), (26, 22, 16, 10)]:
        grid[y0:y0 + rh, x0:x0 + rw] = AIR
    return grid, walls


def _dungeonRooms(originX: int, originY: int) -> list[SpriteRect]:
    return [
        SpriteRect(originX + 4, originY + 6, 18, 12),
        SpriteRect(originX + 24, originY + 8, 14, 10),
        SpriteRect(originX + 6, originY + 22, 18, 12),
        SpriteRect(originX + 26, originY + 22, 16, 10),
    ]


def _buildIslandGrid(w: int, h: int) -> np.ndarray:
    grid = np.full((h, w), AIR, dtype=np.int32)
    cx, cy = w // 2, h // 2
    for j in range(h):
        for i in range(w):
            r = ((i - cx) / (w * 0.45)) ** 2 + ((j - cy) / (h * 0.4)) ** 2
            if r <= 1.0:
                grid[j, i] = DIRT
    return grid


def _buildPyramidGrid(w: int, h: int) -> np.ndarray:
    grid = np.full((h, w), AIR, dtype=np.int32)
    for j in range(h):
        rowWidth = int((j / h) * w)
        x0 = (w - rowWidth) // 2
        grid[j, x0:x0 + rowWidth] = SAND
    return grid


# ---------------------------------------------------------------------------
# D5: structure placement scene (SMALL crop with composer overlays)
# ---------------------------------------------------------------------------
def _stampSubGrid(target: np.ndarray, sub: np.ndarray,
                   originX: int, originY: int) -> None:
    """Copy non-AIR tiles from ``sub`` into ``target`` at given origin."""
    h, w = sub.shape
    th, tw = target.shape
    y0 = max(0, originY)
    x0 = max(0, originX)
    y1 = min(th, originY + h)
    x1 = min(tw, originX + w)
    if y0 >= y1 or x0 >= x1:
        return
    sy0, sx0 = y0 - originY, x0 - originX
    sy1, sx1 = sy0 + (y1 - y0), sx0 + (x1 - x0)
    region = sub[sy0:sy1, sx0:sx1]
    mask = region != AIR
    target[y0:y1, x0:x1][mask] = region[mask]


def createStructurePlacementScene(savePath: str) -> None:
    """Render four 180x130 detail crops, one per structure archetype.

    Each panel anchors a different structure (floating island, dungeon corner,
    pyramid, surface cabin cluster) on top of a fresh SMALL-world crop, so the
    underlying terrain matches the structure's natural placement context.
    """
    print("Creating structure placement scene (2x2 detail crops)...")
    world = generateSmallWorld(seed=20260423)
    layers = world.layers
    surfaceY = int(layers.worldSurface)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    cropW, cropH = 180, 130

    def _crop(centerX: int, centerY: int):
        return cropSmallWorld(world.grid, centerX=centerX, centerY=centerY,
                              width=cropW, height=cropH)

    # ---- Panel 1: Floating Island in the sky ---------------------------
    ax = axes[0, 0]
    cropped, bounds = _crop(world.spawnX - 200, surfaceY - 60)
    h, w = cropped.shape
    fiW, fiH = 70, 24
    fiGrid = _buildIslandGrid(fiW, fiH)
    fiX, fiY = (w - fiW) // 2, max(2, h // 2 - fiH // 2 - 10)
    _stampSubGrid(cropped, fiGrid, originX=fiX, originY=fiY)
    drawTileGrid(ax, cropped)
    drawFloatingIsland(ax, fiGrid, fiX, fiY, fiW, fiH)
    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    ax.set_title("Floating Island (sky band)", fontsize=10, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    # ---- Panel 2: Dungeon corner ---------------------------------------
    ax = axes[0, 1]
    cropped, bounds = _crop(world.spawnX + 600, surfaceY + 30)
    h, w = cropped.shape
    dgW, dgH = 70, 60
    dgGrid, dgWalls = _buildDungeonGrid(dgW, dgH)
    dgX, dgY = (w - dgW) // 2, max(2, h // 2 - dgH // 2)
    _stampSubGrid(cropped, dgGrid, originX=dgX, originY=dgY)
    drawTileGrid(ax, cropped)
    drawDungeon(ax, dgGrid, dgWalls, _dungeonRooms(dgX, dgY))
    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    ax.set_title("Dungeon entrance", fontsize=10, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    # ---- Panel 3: Surface cabin cluster --------------------------------
    ax = axes[1, 0]
    cropped, bounds = _crop(world.spawnX, surfaceY + 30)
    h, w = cropped.shape
    drawTileGrid(ax, cropped)
    surfaceLocal = surfaceY - bounds[2]
    for cx in (35, 90, 140):
        if surfaceLocal - 8 > 0 and surfaceLocal < h:
            drawCabin(ax, cx, surfaceLocal - 8, w=14, h=9)
    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    ax.set_title("Surface cabin cluster", fontsize=10, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    # ---- Panel 4: Pyramid in desert ------------------------------------
    ax = axes[1, 1]
    cropped, bounds = _crop(world.desertX, surfaceY + 50)
    h, w = cropped.shape
    pyW, pyH = 70, 56
    pyGrid = _buildPyramidGrid(pyW, pyH)
    pyX, pyY = (w - pyW) // 2, max(2, h - pyH - 8)
    _stampSubGrid(cropped, pyGrid, originX=pyX, originY=pyY)
    drawTileGrid(ax, cropped)
    drawPyramid(ax, pyGrid, pyX, pyY, pyW, pyH)
    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    ax.set_title("Pyramid (underground desert)", fontsize=10, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        "Structure Placement (180x130 detail crops, SMALL world)",
        fontsize=13, fontweight="bold", y=0.995,
    )
    plt.tight_layout()
    plt.savefig(savePath, dpi=200, bbox_inches="tight",
                facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"Structure placement scene saved to {savePath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    plotsDir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Plots"
    )
    os.makedirs(plotsDir, exist_ok=True)

    print("Terraria Structure Placement (SMALL crop)")
    print("=" * 42)
    createStructurePlacementScene(
        os.path.join(plotsDir, "terraria_structure_density.png")
    )


if __name__ == "__main__":
    main()
