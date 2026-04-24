"""
Terraria Noise Systems Visualization

Visualizes the core terrain and cave generation algorithms used in Terraria:
1. Surface terrain generation via 1D multi-octave wave superposition (numpy)
2. Cave carving via TileRunner diamond-brush random walks (Engine.algorithms)
3. Cellular automata cave smoothing (before/after comparison)
4. Biome tile-type conversion with hard boundaries (no gradient interpolation)

All algorithms reference decompiled WorldGen.cs behavior.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.patches import Patch

from Engine.algorithms import (
    AIR,
    CORRUPT_DIRT,
    CRIMSON_DIRT,
    CRIMSTONE,
    DIRT,
    EBONSTONE,
    GRASS,
    ICE,
    MUD,
    SAND,
    SNOW,
    STONE,
    cellularAutomataSmooth,
    tileRunner,
)
from Engine.constants import DETAIL_PLOT, FEATURE_PLOT, LayerDepths
from Engine.spriteRenderer import drawTileGrid
from Engine.theme import COLORS, PALETTE, applyTokyoNight
from Engine.theme import TILE_COLORS as ENGINE_TILE_COLORS

applyTokyoNight()

# ---------------------------------------------------------------------------
# World constants. FEATURE_PLOT canvas keeps features visible at pixel scale.
# Layer depths scale proportionally from the Large reference world.
# ---------------------------------------------------------------------------
WORLD_WIDTH: int = FEATURE_PLOT.width   # 500
WORLD_HEIGHT: int = FEATURE_PLOT.height  # 300
_REF = LayerDepths.forLarge()
_Y_SCALE = WORLD_HEIGHT / _REF.maxTilesY
LAYERS: LayerDepths = LayerDepths(
    worldSurface=_REF.worldSurface * _Y_SCALE,
    rockLayer=_REF.rockLayer * _Y_SCALE,
    hellLayer=int(_REF.hellLayer * _Y_SCALE),
    maxTilesY=WORLD_HEIGHT,
)


# ===================================================================
# 1. Multi-octave 1D noise for surface terrain
# ===================================================================

def fractalNoise1D(
    length: int,
    octaves: int = 6,
    baseAmplitude: float = 40.0,
    persistence: float = 0.45,
    basePeriod: float = 800.0,
    seed: int = 42,
) -> npt.NDArray[np.float64]:
    """Generate 1D fractal noise via wave superposition (vectorized).

    Each octave doubles frequency and decays amplitude by persistence.
    Uses random-phase sine waves, which is directionally correct for
    Terraria's surface height array computation.

    Returns:
        1D array of length `length` with noise values centered near 0.
    """
    rng = np.random.default_rng(seed)
    x = np.arange(length, dtype=np.float64)
    result = np.zeros(length, dtype=np.float64)
    amplitude = baseAmplitude

    for i in range(octaves):
        period = basePeriod / (2 ** i)
        phase = rng.uniform(0, 2 * np.pi)
        result += amplitude * np.sin(2 * np.pi * x / period + phase)
        amplitude *= persistence

    return result


def createSurfaceTerrainVisualization(savePath: str) -> None:
    """Visualize 1D multi-octave surface terrain for all major biome types."""
    print("Creating surface terrain visualization (multi-octave wave superposition)...")

    worldWidth = WORLD_WIDTH
    x = np.arange(worldWidth)

    # Four representative biomes (was 8); amplitudes/periods rescaled for
    # the FEATURE_PLOT canvas so the wave structure is visible.
    biomeConfigs = [
        ("Forest", dict(octaves=5, baseAmplitude=12, persistence=0.50, basePeriod=120, seed=10), PALETTE["green"],  "#9ece6a"),
        ("Jungle", dict(octaves=6, baseAmplitude=18, persistence=0.48, basePeriod=90,  seed=30), "#73daca",          "#9ece6a"),
        ("Desert", dict(octaves=3, baseAmplitude=6,  persistence=0.40, basePeriod=160, seed=20), PALETTE["yellow"], "#ff9e64"),
        ("Snow",   dict(octaves=5, baseAmplitude=14, persistence=0.45, basePeriod=110, seed=40), PALETTE["cyan"],   "#c0caf5"),
    ]

    xPlot = x  # FEATURE_PLOT is already small; no subsample needed.

    fig, axes = plt.subplots(len(biomeConfigs), 1, figsize=(12, 9))
    fig.suptitle(
        f"Surface Terrain (FEATURE_PLOT {worldWidth} wide)\n"
        "1D Multi-Octave Wave Superposition per Biome",
        fontsize=14, fontweight="bold", y=0.98,
    )

    for i, (name, params, baseColor, surfColor) in enumerate(biomeConfigs):
        noise = fractalNoise1D(worldWidth, **params)
        baseSurface = LAYERS.worldSurface
        terrain = baseSurface + noise
        terrainPlot = terrain

        axes[i].fill_between(xPlot, 0, terrainPlot, color=baseColor, alpha=0.85, label="Base Terrain")
        axes[i].fill_between(xPlot, terrainPlot, terrainPlot + 4, color=surfColor, alpha=0.85, label="Surface Layer")
        axes[i].set_title(f"{name} Biome", fontsize=11, fontweight="bold", pad=6)
        axes[i].set_ylabel("Depth (tiles)", fontsize=9, fontweight="bold")
        axes[i].legend(loc="upper right", frameon=True, fontsize=7)
        axes[i].grid(True, alpha=0.25, linestyle="--")
        axes[i].set_xlim(0, worldWidth)
        axes[i].invert_yaxis()

    axes[-1].set_xlabel("World X Position (tiles)", fontsize=11, fontweight="bold")

    fig.text(
        0.02, 0.02,
        r"$h(x) = \mathrm{worldSurface} + \sum_{i=0}^{N} A \cdot p^i \cdot \sin\left(\frac{2\pi x}{P / 2^i} + \phi_i\right)$",
        fontsize=12, ha="left", va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS["legend_bg"], alpha=0.9, edgecolor=COLORS["edge"]),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(savePath, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"Surface terrain visualization saved to {savePath}")


# ===================================================================
# 2. Cave generation via TileRunner random walks
# ===================================================================

def buildBaseGrid(width: int, height: int) -> npt.NDArray[np.int32]:
    """Create a solid tile grid with dirt above rockLayer and stone below."""
    grid = np.full((height, width), STONE, dtype=np.int32)
    surfaceY = int(LAYERS.worldSurface)

    # Sky is air
    grid[:surfaceY, :] = AIR
    # Dirt layer between surface and rockLayer
    rockY = int(LAYERS.rockLayer)
    grid[surfaceY:rockY, :] = DIRT

    return grid


def createCaveSystemVisualization(savePath: str) -> None:
    """Visualize TileRunner diamond-brush random walks at different depths.

    Shows three depth zones: Surface Caves, Dirt Layer Caves, Rock Layer Caves.
    Also shows before/after cellular automata smoothing.
    """
    print("Creating TileRunner cave system visualization...")

    # Use DETAIL_PLOT (600x400) so vein-scale features are visible.
    sliceWidth = DETAIL_PLOT.width
    sliceHeight = DETAIL_PLOT.height
    rng = np.random.default_rng(seed=777)

    grid = np.full((sliceHeight, sliceWidth), STONE, dtype=np.int32)
    surfaceRow = int(sliceHeight * 0.10)
    dirtBoundary = int(sliceHeight * 0.36)
    grid[:surfaceRow, :] = AIR
    grid[surfaceRow:dirtBoundary, :] = DIRT

    # --- Surface Caves ---
    for _ in range(25):
        sx = rng.integers(20, sliceWidth - 20)
        sy = rng.integers(surfaceRow + 5, dirtBoundary - 10)
        tileRunner(grid, sx, sy, rng.uniform(3.0, 7.0),
                   rng.integers(15, 50), tileType=-1, noYChange=True)

    # --- Dirt Layer Caves ---
    for _ in range(30):
        sx = rng.integers(20, sliceWidth - 20)
        sy = rng.integers(dirtBoundary - 30, dirtBoundary + 60)
        tileRunner(grid, sx, sy, rng.uniform(5.0, 12.0),
                   rng.integers(30, 80), tileType=-1)

    # --- Rock Layer Caves ---
    for _ in range(40):
        sx = rng.integers(20, sliceWidth - 20)
        sy = rng.integers(dirtBoundary + 40, sliceHeight - 40)
        tileRunner(grid, sx, sy, rng.uniform(8.0, 18.0),
                   rng.integers(40, 120), tileType=-1,
                   speedX=rng.uniform(-1.0, 1.0),
                   speedY=rng.uniform(-0.5, 1.5))

    gridBeforeSmooth = grid.copy()
    cellularAutomataSmooth(grid, iterations=3, birthThreshold=5, deathThreshold=3)
    gridAfterSmooth = grid

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        "Cave Carving: TileRunner Diamond-Brush + Cellular Automata Smoothing",
        fontsize=13, fontweight="bold", y=0.98,
    )

    drawTileGrid(axes[0], gridBeforeSmooth)
    axes[0].set_title("Before Smoothing (raw TileRunner)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("X (tiles)", fontweight="bold")
    axes[0].set_ylabel("Depth (tiles)", fontweight="bold")
    axes[0].axhline(surfaceRow, color=PALETTE["cyan"], linewidth=1, linestyle="--", alpha=0.7)
    axes[0].axhline(dirtBoundary, color=PALETTE["orange"], linewidth=1, linestyle="--", alpha=0.7)
    axes[0].text(8, surfaceRow + 8, "Surface", color=PALETTE["cyan"], fontsize=8, fontweight="bold")
    axes[0].text(8, dirtBoundary + 12, "Rock Layer", color=PALETTE["orange"], fontsize=8, fontweight="bold")
    axes[0].set_xlim(0, sliceWidth)
    axes[0].set_ylim(sliceHeight, 0)

    drawTileGrid(axes[1], gridAfterSmooth)
    axes[1].set_title("After Smoothing (3 iterations)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("X (tiles)", fontweight="bold")
    axes[1].axhline(surfaceRow, color=PALETTE["cyan"], linewidth=1, linestyle="--", alpha=0.7)
    axes[1].axhline(dirtBoundary, color=PALETTE["orange"], linewidth=1, linestyle="--", alpha=0.7)
    axes[1].set_xlim(0, sliceWidth)
    axes[1].set_ylim(sliceHeight, 0)

    legendElements = [
        Patch(facecolor=ENGINE_TILE_COLORS[AIR], label="Air (carved)"),
        Patch(facecolor=ENGINE_TILE_COLORS[DIRT], label="Dirt"),
        Patch(facecolor=ENGINE_TILE_COLORS[STONE], label="Stone"),
    ]
    fig.legend(handles=legendElements, loc="lower center", ncol=3,
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    plt.savefig(savePath, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"Cave system visualization saved to {savePath}")


# ===================================================================
# 3. Biome tile-type conversion (hard boundaries, no gradients)
# ===================================================================

def convertBiomeTiles(
    grid: npt.NDArray[np.int32],
    xStart: int,
    xEnd: int,
    conversions: dict[int, int],
) -> None:
    """Apply hard tile-type conversion within a horizontal range.

    This is how Terraria defines biomes: Dirt becomes Mud (Jungle),
    Stone becomes Ebonstone (Corruption), etc. No gradient blending;
    tile types switch at the boundary.
    """
    region = grid[:, xStart:xEnd]
    for srcTile, dstTile in conversions.items():
        mask = region == srcTile
        region[mask] = dstTile


def createBiomeTileConversionVisualization(savePath: str) -> None:
    """Visualize hard-boundary biome tile-type conversion across a world slice."""
    print("Creating biome tile-type conversion visualization...")

    sliceWidth = FEATURE_PLOT.width  # 500
    sliceHeight = FEATURE_PLOT.height  # 300
    rng = np.random.default_rng(seed=999)

    # Build base terrain: air, grass surface, dirt, stone
    grid = np.full((sliceHeight, sliceWidth), STONE, dtype=np.int32)
    surfaceRow = int(sliceHeight * 0.13)
    dirtBottom = int(sliceHeight * 0.40)
    grid[:surfaceRow, :] = AIR
    grid[surfaceRow, :] = GRASS
    grid[surfaceRow + 1:dirtBottom, :] = DIRT

    # Carve some caves so the conversion is visible on varied terrain
    for _ in range(30):
        sx = rng.integers(10, sliceWidth - 10)
        sy = rng.integers(surfaceRow + 10, sliceHeight - 20)
        tileRunner(grid, sx, sy, rng.uniform(4.0, 10.0), rng.integers(20, 60), tileType=-1)

    gridBefore = grid.copy()

    # --- Apply biome conversions at hard boundaries ---
    # Boundaries scaled to FEATURE_PLOT width (500): 5 biome bands.
    boundaries = [int(sliceWidth * f) for f in (0.16, 0.33, 0.49, 0.66, 0.82)]
    convertBiomeTiles(grid, boundaries[0], boundaries[1], {DIRT: MUD, GRASS: MUD})
    convertBiomeTiles(grid, boundaries[1], boundaries[2], {DIRT: SNOW, STONE: ICE, GRASS: SNOW})
    convertBiomeTiles(grid, boundaries[2], boundaries[3], {DIRT: SAND, GRASS: SAND})
    convertBiomeTiles(grid, boundaries[3], boundaries[4], {DIRT: CORRUPT_DIRT, STONE: EBONSTONE, GRASS: CORRUPT_DIRT})
    convertBiomeTiles(grid, boundaries[4], sliceWidth, {DIRT: CRIMSON_DIRT, STONE: CRIMSTONE, GRASS: CRIMSON_DIRT})

    gridAfter = grid

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(
        "Biome Tile-Type Conversion (Hard Boundaries)\n"
        "No gradient interpolation: tile types switch instantly at biome edges",
        fontsize=13, fontweight="bold", y=0.98,
    )

    drawTileGrid(axes[0], gridBefore)
    axes[0].set_title("Before Conversion (Dirt / Stone base)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Depth (tiles)", fontweight="bold")
    axes[0].set_xlim(0, sliceWidth)
    axes[0].set_ylim(sliceHeight, 0)

    drawTileGrid(axes[1], gridAfter)
    axes[1].set_title("After Conversion (hard tile-type replacement)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("X (tiles)", fontweight="bold")
    axes[1].set_ylabel("Depth (tiles)", fontweight="bold")
    axes[1].set_xlim(0, sliceWidth)
    axes[1].set_ylim(sliceHeight, 0)

    biomeLabels = ["Forest", "Jungle", "Snow", "Desert", "Corruption", "Crimson"]
    biomeMidpoints = [
        boundaries[0] // 2,
        (boundaries[0] + boundaries[1]) // 2,
        (boundaries[1] + boundaries[2]) // 2,
        (boundaries[2] + boundaries[3]) // 2,
        (boundaries[3] + boundaries[4]) // 2,
        (boundaries[4] + sliceWidth) // 2,
    ]
    labelColors = [
        PALETTE["green"], "#73daca", PALETTE["cyan"],
        PALETTE["yellow"], PALETTE["purple"], PALETTE["red"],
    ]

    for bx in boundaries:
        for ax in axes:
            ax.axvline(bx, color=PALETTE["fg"], linewidth=1.0, linestyle="--", alpha=0.7)

    for mid, label, lc in zip(biomeMidpoints, biomeLabels, labelColors):
        axes[1].text(
            mid, surfaceRow * 0.4, label, ha="center", va="center",
            fontweight="bold", fontsize=8, color=PALETTE["bg"],
            bbox=dict(boxstyle="round,pad=0.25", facecolor=lc, alpha=0.9),
        )

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    plt.savefig(savePath, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"Biome tile-type conversion visualization saved to {savePath}")


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    print("Starting Terraria noise systems visualization generation")

    outputDir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Plots")
    os.makedirs(outputDir, exist_ok=True)

    try:
        createSurfaceTerrainVisualization(os.path.join(outputDir, "terraria_surface_terrain.png"))
        createCaveSystemVisualization(os.path.join(outputDir, "terraria_cave_systems.png"))
        createBiomeTileConversionVisualization(os.path.join(outputDir, "terraria_biome_tile_conversion.png"))
        print("All noise system visualizations completed successfully")

    except Exception as e:
        print(f"Error in noise systems visualization generation: {e}")
        raise
