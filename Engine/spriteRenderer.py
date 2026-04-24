"""Procedural pixel-tile renderer for Terraria structures and ore veins.

No external sprite assets (Re-Logic copyright). All visuals derive from
``Engine.theme.TILE_COLORS`` rendered as pixel-perfect bitmaps via
``imshow(..., interpolation='nearest')`` plus matplotlib primitive overlays
for sub-tile features (doors, chests, torches, platforms).

Public API:
    buildTileColormap, drawTileGrid
    drawDoor, drawChest, drawPlatform, drawTorch, drawWallFill
    drawOreVein
    drawDungeon, drawTemple, drawCabin, drawPyramid,
    drawLivingTree, drawFloatingIsland, drawSpiderCave, drawGemCave
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, Rectangle

from Engine.constants import LayerDepths
from Engine.theme import (
    DEFAULT_TILE_COLOR,
    PALETTE,
    TILE_COLORS,
    buildTileColormap as _buildFromTheme,
)


# ---------------------------------------------------------------------------
# Sub-tile primitive colors (decorative overlays only)
# ---------------------------------------------------------------------------
DOOR_COLOR = TILE_COLORS[136]
DOOR_EDGE = "#3A1F0E"
DOOR_KNOB = "#C0A04A"
CHEST_COLOR = TILE_COLORS[135]
CHEST_TRIM = "#8B6914"
PLATFORM_COLOR = TILE_COLORS[133]
TORCH_COLOR = TILE_COLORS[134]
WALL_DARK = "#2D2D2D"


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Rect:
    """Axis-aligned tile-coordinate rectangle (x, y top-left; w, h tiles)."""
    x: int
    y: int
    w: int
    h: int


def buildTileColormap(maxTileId: int = 200) -> ListedColormap:
    """Re-export of `Engine.theme.buildTileColormap` for callers importing
    only `Engine.spriteRenderer`."""
    return _buildFromTheme(maxTileId)


# ---------------------------------------------------------------------------
# Core grid renderer
# ---------------------------------------------------------------------------
def drawTileGrid(
    ax: Axes,
    grid: npt.NDArray[np.integer],
    walls: npt.NDArray[np.integer] | None = None,
    extent: tuple[float, float, float, float] | None = None,
    wallAlpha: float = 0.35,
) -> None:
    """Render a tile grid as a pixel-perfect bitmap.

    Args:
        ax: Matplotlib axis.
        grid: 2D array of tile IDs (foreground).
        walls: Optional 2D array of background wall tile IDs (semi-transparent).
        extent: ``(x0, x1, y0, y1)`` for axis coordinates. Default uses array
            shape so origin is top-left and one cell == one tile unit.
        wallAlpha: Alpha for the wall layer.
    """
    cmap = buildTileColormap()
    if extent is None:
        h, w = grid.shape
        extent = (0.0, float(w), float(h), 0.0)

    if walls is not None:
        ax.imshow(
            walls, cmap=cmap, alpha=wallAlpha,
            interpolation="nearest", extent=extent, vmin=0, vmax=cmap.N - 1,
            aspect="equal", zorder=0,
        )

    ax.imshow(
        grid, cmap=cmap, alpha=1.0,
        interpolation="nearest", extent=extent, vmin=0, vmax=cmap.N - 1,
        aspect="equal", zorder=1,
    )


# ---------------------------------------------------------------------------
# Sub-tile decorative primitives
# ---------------------------------------------------------------------------
def drawDoor(ax: Axes, x: float, y: float, height: int = 3) -> None:
    """1xN brown door with brass knob centered vertically."""
    ax.add_patch(Rectangle(
        (x, y), 1, height,
        facecolor=DOOR_COLOR, edgecolor=DOOR_EDGE, linewidth=0.4, zorder=4,
    ))
    ax.plot(
        x + 0.8, y + height / 2,
        marker="o", markersize=1.8, color=DOOR_KNOB, zorder=5,
    )


def drawChest(ax: Axes, x: float, y: float) -> None:
    """2x2 gold chest with darker lid stripe."""
    ax.add_patch(Rectangle(
        (x, y), 2, 2,
        facecolor=CHEST_COLOR, edgecolor=CHEST_TRIM, linewidth=0.5, zorder=4,
    ))
    ax.add_patch(Rectangle(
        (x, y), 2, 0.6,
        facecolor=CHEST_TRIM, edgecolor="none", zorder=5,
    ))


def drawPlatform(ax: Axes, x: float, y: float, length: int) -> None:
    """Horizontal wood platform, ~1 tile tall."""
    ax.add_patch(Rectangle(
        (x, y + 0.4), length, 0.2,
        facecolor=PLATFORM_COLOR, edgecolor="none", zorder=3,
    ))


def drawTorch(ax: Axes, x: float, y: float) -> None:
    """Single torch with halo."""
    ax.add_patch(Circle(
        (x + 0.5, y + 0.5), 0.7,
        facecolor=TORCH_COLOR, alpha=0.32, edgecolor="none", zorder=3,
    ))
    ax.plot(
        x + 0.5, y + 0.5,
        marker="*", markersize=4.0, color=TORCH_COLOR, zorder=6,
    )


def drawWallFill(
    ax: Axes, x: float, y: float, w: float, h: float,
    color: str = WALL_DARK, alpha: float = 0.5,
) -> None:
    """Background wall fill (semi-transparent darker rectangle)."""
    ax.add_patch(Rectangle(
        (x, y), w, h,
        facecolor=color, alpha=alpha, edgecolor="none", zorder=0,
    ))


# ---------------------------------------------------------------------------
# Ore-vein detail rendering
# ---------------------------------------------------------------------------
def drawOreVein(
    ax: Axes,
    grid: npt.NDArray[np.integer],
    oreId: int,
    lusterColor: str = "#ffffff",
    lusterAlpha: float = 0.35,
) -> None:
    """Overlay a small luster highlight on every tile of ``oreId`` in ``grid``.

    Use after ``drawTileGrid`` to add depth to ore-vein cross-section plots.
    """
    ys, xs = np.where(grid == oreId)
    if ys.size == 0:
        return
    # Offset the highlight to top-left of each tile for a "lit from upper-left" feel.
    ax.scatter(
        xs + 0.3, ys + 0.3,
        s=6, c=lusterColor, alpha=lusterAlpha, marker="o",
        edgecolors="none", zorder=2,
    )


# ---------------------------------------------------------------------------
# Per-structure composers
# ---------------------------------------------------------------------------
def drawDungeon(
    ax: Axes,
    grid: npt.NDArray[np.integer],
    walls: npt.NDArray[np.integer] | None,
    rooms: Iterable[Rect],
) -> None:
    """Dungeon tile grid + per-room doors, chests, torches."""
    drawTileGrid(ax, grid, walls)
    for r in rooms:
        # Door on the floor center.
        drawDoor(ax, r.x + r.w // 2, r.y + r.h - 3, height=3)
        # Chest tucked into one corner.
        drawChest(ax, r.x + 2, r.y + r.h - 3)
        # Torches on opposite upper corners.
        drawTorch(ax, r.x + 2, r.y + 1)
        drawTorch(ax, r.x + r.w - 3, r.y + 1)


def drawTemple(
    ax: Axes,
    grid: npt.NDArray[np.integer],
    walls: npt.NDArray[np.integer] | None,
    rooms: Iterable[Rect],
) -> None:
    """Lihzahrd temple: platforms across each room + altar chest in last."""
    drawTileGrid(ax, grid, walls)
    roomList = list(rooms)
    for r in roomList:
        drawPlatform(ax, r.x + 1, r.y + r.h // 2, r.w - 2)
    if roomList:
        last = roomList[-1]
        drawChest(ax, last.x + last.w // 2, last.y + last.h - 3)


def drawCabin(ax: Axes, x: int, y: int, w: int = 12, h: int = 8) -> None:
    """Underground cabin schematic: wood walls, door, chest, torch.

    Use when no pre-built grid exists; renders directly as overlays.
    """
    drawWallFill(ax, x, y, w, h, color=TILE_COLORS[132], alpha=0.7)
    # Floor and ceiling outline
    ax.add_patch(Rectangle((x, y + h - 1), w, 1,
                           facecolor=TILE_COLORS[132], zorder=2))
    ax.add_patch(Rectangle((x, y), w, 1,
                           facecolor=TILE_COLORS[132], zorder=2))
    drawDoor(ax, x + w // 2, y + h - 4, height=3)
    drawChest(ax, x + 2, y + h - 3)
    drawTorch(ax, x + w - 3, y + 1)


def drawPyramid(
    ax: Axes,
    grid: npt.NDArray[np.integer],
    x: int, y: int, w: int, h: int,
) -> None:
    """Sandstone pyramid: tile grid + chamber + chest."""
    drawTileGrid(ax, grid)
    drawWallFill(ax, x + w // 4, y + h // 2, w // 2, h // 4,
                 color=TILE_COLORS[125], alpha=0.55)
    drawChest(ax, x + w // 2, y + 3 * h // 4)


def drawLivingTree(
    ax: Axes,
    grid: npt.NDArray[np.integer],
    baseX: int, surfaceY: int, height: int = 30,
) -> None:
    """Living tree silhouette + door at base of trunk."""
    drawTileGrid(ax, grid)
    drawDoor(ax, baseX, surfaceY - 3, height=3)


def drawFloatingIsland(
    ax: Axes,
    grid: npt.NDArray[np.integer],
    x: int, y: int, w: int, h: int,
) -> None:
    """Floating island grid + central house with door + chest."""
    drawTileGrid(ax, grid)
    drawWallFill(ax, x + w // 3, y + 1, w // 3, h // 2,
                 color=TILE_COLORS[132], alpha=0.6)
    drawChest(ax, x + w // 2, y + 2)
    drawDoor(ax, x + w // 2 - 2, y + 4, height=3)


def drawSpiderCave(
    ax: Axes,
    grid: npt.NDArray[np.integer],
    x: int, y: int, w: int, h: int,
    seed: int = 42,
) -> None:
    """Spider cave: cobweb tile grid + scattered egg dots."""
    drawTileGrid(ax, grid)
    rng = np.random.default_rng(seed)
    n = max(6, (w * h) // 80)
    ex = x + rng.integers(0, max(1, w), size=n)
    ey = y + rng.integers(0, max(1, h), size=n)
    ax.scatter(ex, ey, s=10, c="#8B0000", marker="o",
               edgecolors=PALETTE["bg"], linewidths=0.4, zorder=3)


def drawGemCave(
    ax: Axes,
    grid: npt.NDArray[np.integer],
    x: int, y: int, w: int, h: int,
    gemColor: str = "#9B59B6",
    seed: int = 42,
) -> None:
    """Gem cave: tile grid + sparkling gem markers."""
    drawTileGrid(ax, grid)
    rng = np.random.default_rng(seed)
    n = max(8, (w * h) // 60)
    ex = x + rng.integers(0, max(1, w), size=n)
    ey = y + rng.integers(0, max(1, h), size=n)
    ax.scatter(ex, ey, s=14, c=gemColor, marker="*",
               edgecolors=PALETTE["fg"], linewidths=0.3, zorder=3)


__all__ = [
    "Rect",
    "buildTileColormap",
    "drawTileGrid",
    "drawDoor", "drawChest", "drawPlatform", "drawTorch", "drawWallFill",
    "drawOreVein",
    "drawDungeon", "drawTemple", "drawCabin", "drawPyramid",
    "drawLivingTree", "drawFloatingIsland", "drawSpiderCave", "drawGemCave",
    "applyMapDecorations", "cropSmallWorld",
]


# ---------------------------------------------------------------------------
# Map decoration + cropping helpers (Phase B/C of redesign audit)
# ---------------------------------------------------------------------------
def applyMapDecorations(
    ax: Axes,
    grid: npt.NDArray[np.integer],
    layers: LayerDepths,
    *,
    grassBand: bool = True,
    hellstoneBand: bool = True,
    layerMarkers: bool = True,
    cropBounds: tuple[int, int, int, int] | None = None,
) -> None:
    """Apply standard decorations to a map ax (after ``drawTileGrid``).

    Args:
        ax: Matplotlib axis already populated with a tile-grid imshow.
        grid: The (possibly cropped) tile grid being rendered, used only for
            bounds inference when ``cropBounds`` is None.
        layers: World-coordinate ``LayerDepths`` for the *full* world from
            which ``grid`` was generated. Depths are remapped into crop space
            when ``cropBounds`` is provided.
        grassBand: If True, draw a thin bright-green strip at ``worldSurface``.
        hellstoneBand: If True, draw an orange strip from ``hellLayer`` to
            the bottom of the visible window.
        layerMarkers: If True, draw cyan/yellow/red dashed horizontal lines at
            ``worldSurface``, ``rockLayer``, ``hellLayer``.
        cropBounds: ``(x0, x1, y0, y1)`` of the crop in world coordinates. If
            None, the grid is assumed to span the whole world (no remap).
    """
    if cropBounds is None:
        h, w = grid.shape
        x0, x1, y0, y1 = 0, w, 0, h
    else:
        x0, x1, y0, y1 = cropBounds

    def toLocalY(worldY: float) -> float:
        return worldY - y0

    surfaceLocal = toLocalY(layers.worldSurface)
    rockLocal = toLocalY(layers.rockLayer)
    hellLocal = toLocalY(float(layers.hellLayer))
    cropW = x1 - x0
    cropH = y1 - y0

    if grassBand and 0 <= surfaceLocal < cropH:
        ax.add_patch(Rectangle(
            (0, surfaceLocal - 1), cropW, 2,
            facecolor=PALETTE.get("green", "#9ece6a"),
            edgecolor="none", alpha=0.55, zorder=2,
        ))

    if hellstoneBand and hellLocal < cropH:
        bandTop = max(hellLocal, 0.0)
        ax.add_patch(Rectangle(
            (0, bandTop), cropW, cropH - bandTop,
            facecolor=PALETTE.get("orange", "#ff9e64"),
            edgecolor="none", alpha=0.28, zorder=2,
        ))

    if layerMarkers:
        markers = (
            (surfaceLocal, PALETTE.get("cyan", "#7dcfff")),
            (rockLocal, PALETTE.get("yellow", "#e0af68")),
            (hellLocal, PALETTE.get("red", "#f7768e")),
        )
        for yLocal, color in markers:
            if 0 <= yLocal <= cropH:
                ax.axhline(
                    y=yLocal, color=color, linestyle="--",
                    linewidth=0.8, alpha=0.55, zorder=6,
                )


def cropSmallWorld(
    fullGrid: npt.NDArray[np.integer],
    centerX: int,
    centerY: int,
    width: int = 600,
    height: int = 500,
) -> tuple[npt.NDArray[np.integer], tuple[int, int, int, int]]:
    """Crop a centered window from a full SMALL-world grid.

    Clamps the requested window to the grid bounds so callers near the edges
    still receive a window of the requested size when possible.

    Args:
        fullGrid: 2D tile-id array shaped ``(H, W)`` where ``H, W`` are the
            full SMALL-world height and width (e.g. 1200 x 4200).
        centerX, centerY: World-coordinate center of the desired crop.
        width, height: Crop dimensions in tiles.

    Returns:
        ``(croppedGrid, (x0, x1, y0, y1))`` where the bounds are in world
        coordinates so callers can re-anchor layer markers into crop space.
    """
    fullH, fullW = fullGrid.shape
    halfW, halfH = width // 2, height // 2

    x0 = max(0, min(fullW - width, centerX - halfW))
    y0 = max(0, min(fullH - height, centerY - halfH))
    x1 = min(fullW, x0 + width)
    y1 = min(fullH, y0 + height)

    return fullGrid[y0:y1, x0:x1], (x0, x1, y0, y1)
