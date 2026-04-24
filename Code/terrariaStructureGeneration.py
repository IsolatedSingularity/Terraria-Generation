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
    applyMapDecorations,
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
    """Render a 600x500 crop of a SMALL world with structures overlaid."""
    print("Creating structure placement scene (600x500 SMALL crop)...")
    world = generateSmallWorld(seed=20260423)
    layers = world.layers

    # Anchor crop near surface so cabins, floating island, pyramid, and a
    # dungeon corner all fit visually.
    centerX = world.spawnX
    centerY = int(layers.worldSurface) + 180
    cropped, bounds = cropSmallWorld(
        world.grid, centerX=centerX, centerY=centerY,
        width=600, height=500,
    )

    # Stamp structure grids into the cropped tile array.
    h, w = cropped.shape

    # Pyramid (lower-right) - rises from sandstone area.
    pyW, pyH = 60, 50
    pyGrid = _buildPyramidGrid(pyW, pyH)
    _stampSubGrid(cropped, pyGrid, originX=w - pyW - 30, originY=h - pyH - 80)

    # Floating island (upper-left in sky band).
    fiW, fiH = 70, 28
    fiGrid = _buildIslandGrid(fiW, fiH)
    fiOriginY = max(2, int(layers.worldSurface) - bounds[2] - 90)
    _stampSubGrid(cropped, fiGrid, originX=20, originY=fiOriginY)

    # Dungeon corner (upper-right surface).
    dgW, dgH = 50, 38
    dgGrid, dgWalls = _buildDungeonGrid(dgW, dgH)
    dgOriginX = w - dgW - 110
    dgOriginY = max(2, int(layers.worldSurface) - bounds[2] - 8)
    _stampSubGrid(cropped, dgGrid, originX=dgOriginX, originY=dgOriginY)

    # Render base + decorations.
    fig, ax = plt.subplots(figsize=(13, 10))
    drawTileGrid(ax, cropped)
    applyMapDecorations(ax, cropped, layers, cropBounds=bounds)

    # Composer overlays for sub-tile primitives (doors, chests, torches).
    for cabinX, cabinY in [(140, fiOriginY + 90),
                            (260, fiOriginY + 130),
                            (430, fiOriginY + 200)]:
        if 0 <= cabinX < w - 12 and 0 <= cabinY < h - 8:
            drawCabin(ax, cabinX, cabinY, w=12, h=8)

    drawFloatingIsland(ax, fiGrid, 20, fiOriginY, fiW, fiH)
    drawPyramid(ax, pyGrid, w - pyW - 30, h - pyH - 80, pyW, pyH)
    drawDungeon(ax, dgGrid, dgWalls,
                _dungeonRooms(dgOriginX, dgOriginY))

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xlabel("X (tiles, crop-local)", fontweight="bold")
    ax.set_ylabel("Depth (tiles, crop-local)", fontweight="bold")
    ax.set_title(
        "Structure Placement (600x500 crop)",
        fontsize=14, fontweight="bold",
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
