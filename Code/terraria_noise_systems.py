"""
Terraria Noise Systems Visualization

Visualizes the core terrain and cave generation algorithms used in Terraria:
1. Surface terrain generation via 1D multi-octave wave superposition (numpy)
2. Cave carving via TileRunner diamond-brush random walks (Engine.algorithms)
3. Cellular automata cave smoothing (before/after comparison)
4. Biome tile-type conversion with hard boundaries (no gradient interpolation)

All algorithms reference decompiled WorldGen.cs behavior.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import seaborn as sns

from Engine.algorithms import tileRunner, cellularAutomataSmooth, AIR, DIRT, STONE, GRASS
from Engine.algorithms import MUD, SNOW, ICE, SAND, EBONSTONE, CORRUPT_DIRT
from Engine.algorithms import CRIMSTONE, CRIMSON_DIRT, PEARLSTONE, HALLOW_DIRT
from Engine.constants import LARGE, LayerDepths

# ---------------------------------------------------------------------------
# Seaborn publication styling
# ---------------------------------------------------------------------------
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"

# ---------------------------------------------------------------------------
# World constants for a Large world
# ---------------------------------------------------------------------------
WORLD_WIDTH: int = LARGE.width   # 8400
WORLD_HEIGHT: int = LARGE.height  # 2400
LAYERS: LayerDepths = LayerDepths.forLarge()


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

    # Each biome has distinct noise parameters (octaves, amplitude, period)
    biomeConfigs = [
        ("Forest",     dict(octaves=5, baseAmplitude=25, persistence=0.50, basePeriod=900,  seed=10), "#2E8B57", "#90EE90"),
        ("Desert",     dict(octaves=3, baseAmplitude=12, persistence=0.40, basePeriod=1200, seed=20), "#DEB887", "#FFFF99"),
        ("Jungle",     dict(octaves=6, baseAmplitude=40, persistence=0.48, basePeriod=700,  seed=30), "#228B22", "#ADFF2F"),
        ("Snow",       dict(octaves=5, baseAmplitude=30, persistence=0.45, basePeriod=850,  seed=40), "#87CEEB", "#F0F8FF"),
        ("Corruption", dict(octaves=6, baseAmplitude=35, persistence=0.52, basePeriod=500,  seed=50), "#9370DB", "#E6E6FA"),
        ("Crimson",    dict(octaves=6, baseAmplitude=33, persistence=0.50, basePeriod=550,  seed=60), "#DC143C", "#FFB6C1"),
        ("Mushroom",   dict(octaves=3, baseAmplitude=18, persistence=0.40, basePeriod=1000, seed=70), "#8A2BE2", "#DDA0DD"),
        ("Hallow",     dict(octaves=5, baseAmplitude=30, persistence=0.47, basePeriod=650,  seed=80), "#FFB6C1", "#FFF0F5"),
    ]

    # Subsample for plotting performance
    step = 4
    xPlot = x[::step]

    fig, axes = plt.subplots(len(biomeConfigs), 1, figsize=(18, 16))
    fig.suptitle(
        "Terraria Surface Terrain Generation, Large World (8400 wide)\n"
        "1D Multi-Octave Wave Superposition per Biome",
        fontsize=18, fontweight="bold", y=0.98,
    )

    for i, (name, params, baseColor, surfColor) in enumerate(biomeConfigs):
        noise = fractalNoise1D(worldWidth, **params)
        baseSurface = LAYERS.worldSurface
        terrain = baseSurface + noise
        terrainPlot = terrain[::step]

        axes[i].fill_between(xPlot, 0, terrainPlot, color=baseColor, alpha=0.8, label="Base Terrain")
        axes[i].fill_between(xPlot, terrainPlot, terrainPlot + 6, color=surfColor, alpha=0.7, label="Surface Layer")
        axes[i].set_title(f"{name} Biome", fontsize=13, fontweight="bold", pad=8)
        axes[i].set_ylabel("Depth (tiles)", fontsize=10, fontweight="bold")
        axes[i].legend(loc="upper right", frameon=True, fancybox=True, shadow=True, fontsize=8)
        axes[i].grid(True, alpha=0.3, linestyle="--")
        axes[i].set_xlim(0, worldWidth)
        axes[i].set_facecolor("#FAFAFA")
        axes[i].invert_yaxis()

    axes[-1].set_xlabel("World X Position (tiles)", fontsize=11, fontweight="bold")

    fig.text(
        0.02, 0.02,
        r"$h(x) = \mathrm{worldSurface} + \sum_{i=0}^{N} A \cdot p^i \cdot \sin\left(\frac{2\pi x}{P / 2^i} + \phi_i\right)$",
        fontsize=12, ha="left", va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="gray"),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(savePath, dpi=300, bbox_inches="tight", facecolor="white")
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

    # Use a smaller slice for visualization clarity
    sliceWidth = 600
    sliceHeight = 500
    rng = np.random.default_rng(seed=777)

    # Build a solid grid (all stone, simulating a vertical cross-section)
    grid = np.full((sliceHeight, sliceWidth), STONE, dtype=np.int32)
    surfaceRow = 50
    dirtBoundary = 180
    grid[:surfaceRow, :] = AIR
    grid[surfaceRow:dirtBoundary, :] = DIRT

    # --- Surface Caves (small strength, shallow) ---
    numSurfaceCaves = 25
    for _ in range(numSurfaceCaves):
        sx = rng.integers(20, sliceWidth - 20)
        sy = rng.integers(surfaceRow + 5, dirtBoundary - 10)
        strength = rng.uniform(3.0, 7.0)
        steps = rng.integers(15, 50)
        tileRunner(grid, sx, sy, strength, steps, tileType=-1, noYChange=True)

    # --- Dirt Layer Caves (medium strength) ---
    numDirtCaves = 30
    for _ in range(numDirtCaves):
        sx = rng.integers(20, sliceWidth - 20)
        sy = rng.integers(dirtBoundary - 30, dirtBoundary + 60)
        strength = rng.uniform(5.0, 12.0)
        steps = rng.integers(30, 80)
        tileRunner(grid, sx, sy, strength, steps, tileType=-1)

    # --- Rock Layer Caves (large strength, deep) ---
    numRockCaves = 40
    for _ in range(numRockCaves):
        sx = rng.integers(20, sliceWidth - 20)
        sy = rng.integers(dirtBoundary + 40, sliceHeight - 40)
        strength = rng.uniform(8.0, 18.0)
        steps = rng.integers(40, 120)
        speedX = rng.uniform(-1.0, 1.0)
        speedY = rng.uniform(-0.5, 1.5)
        tileRunner(grid, sx, sy, strength, steps, tileType=-1,
                   speedX=speedX, speedY=speedY)

    # Snapshot before smoothing
    gridBeforeSmooth = grid.copy()

    # Apply cellular automata smoothing
    cellularAutomataSmooth(grid, iterations=3, birthThreshold=5, deathThreshold=3)
    gridAfterSmooth = grid

    # --- Color maps ---
    tileColors = {
        AIR: np.array([0.05, 0.05, 0.08]),
        DIRT: np.array([0.55, 0.35, 0.17]),
        STONE: np.array([0.50, 0.50, 0.52]),
    }

    def gridToRGB(g: npt.NDArray[np.int32]) -> npt.NDArray[np.float64]:
        rgb = np.zeros((*g.shape, 3), dtype=np.float64)
        for tileID, color in tileColors.items():
            mask = g == tileID
            rgb[mask] = color
        return rgb

    imgBefore = gridToRGB(gridBeforeSmooth)
    imgAfter = gridToRGB(gridAfterSmooth)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle(
        "Terraria Cave Carving: TileRunner Diamond-Brush Random Walks\n"
        "Left: Raw TileRunner output | Right: After Cellular Automata Smoothing",
        fontsize=16, fontweight="bold", y=0.98,
    )

    axes[0].imshow(imgBefore, origin="upper", aspect="auto")
    axes[0].set_title("Before Smoothing (raw TileRunner)", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("X (tiles)", fontweight="bold")
    axes[0].set_ylabel("Depth (tiles)", fontweight="bold")

    # Annotate depth zones
    axes[0].axhline(surfaceRow, color="cyan", linewidth=1, linestyle="--", alpha=0.7)
    axes[0].axhline(dirtBoundary, color="orange", linewidth=1, linestyle="--", alpha=0.7)
    axes[0].text(10, surfaceRow + 8, "Surface Caves", color="cyan", fontsize=9, fontweight="bold")
    axes[0].text(10, dirtBoundary + 12, "Rock Layer Caves", color="orange", fontsize=9, fontweight="bold")
    axes[0].text(10, (surfaceRow + dirtBoundary) // 2, "Dirt Layer Caves", color="white", fontsize=9, fontweight="bold")

    axes[1].imshow(imgAfter, origin="upper", aspect="auto")
    axes[1].set_title("After Cellular Automata Smoothing (3 iterations)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("X (tiles)", fontweight="bold")
    axes[1].axhline(surfaceRow, color="cyan", linewidth=1, linestyle="--", alpha=0.7)
    axes[1].axhline(dirtBoundary, color="orange", linewidth=1, linestyle="--", alpha=0.7)

    # Legend
    legendElements = [
        Patch(facecolor=tileColors[AIR], label="Air (carved)"),
        Patch(facecolor=tileColors[DIRT], label="Dirt"),
        Patch(facecolor=tileColors[STONE], label="Stone"),
    ]
    fig.legend(
        handles=legendElements, loc="lower center", ncol=3,
        fontsize=12, frameon=True, fancybox=True, shadow=True,
        bbox_to_anchor=(0.5, 0.01),
    )

    # Algorithm description
    fig.text(
        0.02, 0.95,
        "TileRunner: diamond brush (manhattan dist), strength decay per step,\n"
        "drunkard's walk drift vectors, clamped speed [-2, 2]",
        fontsize=11, ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="gray"),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    plt.savefig(savePath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Cave system visualization saved to {savePath}")


# ===================================================================
# 3. Biome tile-type conversion (hard boundaries, no gradients)
# ===================================================================

# Display colors per tile type for the biome visualization
TILE_DISPLAY_COLORS: dict[int, tuple[float, float, float]] = {
    AIR:           (0.53, 0.81, 0.92),  # sky blue
    GRASS:         (0.13, 0.55, 0.13),  # dark green
    DIRT:          (0.55, 0.35, 0.17),  # brown
    STONE:         (0.50, 0.50, 0.52),  # gray
    MUD:           (0.30, 0.20, 0.10),  # dark brown
    SNOW:          (0.90, 0.93, 0.96),  # near white
    ICE:           (0.68, 0.85, 0.90),  # light blue
    SAND:          (0.86, 0.80, 0.55),  # tan
    EBONSTONE:     (0.33, 0.17, 0.50),  # dark purple
    CORRUPT_DIRT:  (0.40, 0.25, 0.55),  # purple-brown
    CRIMSTONE:     (0.60, 0.10, 0.10),  # dark red
    CRIMSON_DIRT:  (0.55, 0.15, 0.15),  # red-brown
    PEARLSTONE:    (0.85, 0.75, 0.90),  # light lavender
    HALLOW_DIRT:   (0.80, 0.70, 0.85),  # pastel purple
}


def convertBiomeTiles(
    grid: npt.NDArray[np.int32],
    xStart: int,
    xEnd: int,
    conversions: dict[int, int],
) -> None:
    """Apply hard tile-type conversion within a horizontal range.

    This is how Terraria defines biomes: Dirt becomes Mud (Jungle),
    Dirt becomes Snow Block (Snow), Stone becomes Ebonstone (Corruption), etc.
    There is NO gradient or blending; tile types switch at the boundary.
    """
    region = grid[:, xStart:xEnd]
    for srcTile, dstTile in conversions.items():
        mask = region == srcTile
        region[mask] = dstTile


def createBiomeTileConversionVisualization(savePath: str) -> None:
    """Visualize hard-boundary biome tile-type conversion across a world slice."""
    print("Creating biome tile-type conversion visualization...")

    sliceWidth = 800
    sliceHeight = 300
    rng = np.random.default_rng(seed=999)

    # Build base terrain: air, grass surface, dirt, stone
    grid = np.full((sliceHeight, sliceWidth), STONE, dtype=np.int32)
    surfaceRow = 40
    dirtBottom = 120
    grid[:surfaceRow, :] = AIR
    grid[surfaceRow, :] = GRASS
    grid[surfaceRow + 1:dirtBottom, :] = DIRT

    # Carve some caves so the conversion is visible on varied terrain
    for _ in range(30):
        sx = rng.integers(10, sliceWidth - 10)
        sy = rng.integers(surfaceRow + 10, sliceHeight - 20)
        tileRunner(grid, sx, sy, rng.uniform(4.0, 10.0), rng.integers(20, 60), tileType=-1)

    # Snapshot: before biome conversion
    gridBefore = grid.copy()

    # --- Apply biome conversions at hard boundaries ---
    # Forest: columns 0-130 (unchanged, already dirt/stone)
    # Jungle: columns 130-260
    convertBiomeTiles(grid, 130, 260, {DIRT: MUD, GRASS: MUD})
    # Snow: columns 260-390
    convertBiomeTiles(grid, 260, 390, {DIRT: SNOW, STONE: ICE, GRASS: SNOW})
    # Desert: columns 390-520
    convertBiomeTiles(grid, 390, 520, {DIRT: SAND, GRASS: SAND})
    # Corruption: columns 520-660
    convertBiomeTiles(grid, 520, 660, {DIRT: CORRUPT_DIRT, STONE: EBONSTONE, GRASS: CORRUPT_DIRT})
    # Crimson: columns 660-800
    convertBiomeTiles(grid, 660, 800, {DIRT: CRIMSON_DIRT, STONE: CRIMSTONE, GRASS: CRIMSON_DIRT})

    gridAfter = grid

    # --- Render ---
    def gridToRGB(g: npt.NDArray[np.int32]) -> npt.NDArray[np.float64]:
        rgb = np.zeros((*g.shape, 3), dtype=np.float64)
        for tileID, color in TILE_DISPLAY_COLORS.items():
            mask = g == tileID
            rgb[mask] = color
        # Fallback for unmapped tiles
        unmapped = np.all(rgb == 0, axis=-1) & (g != AIR)
        rgb[unmapped] = (0.4, 0.4, 0.4)
        return rgb

    imgBefore = gridToRGB(gridBefore)
    imgAfter = gridToRGB(gridAfter)

    fig, axes = plt.subplots(2, 1, figsize=(20, 12))
    fig.suptitle(
        "Terraria Biome Tile-Type Conversion (Hard Boundaries)\n"
        "No gradient interpolation: tile types switch instantly at biome edges",
        fontsize=16, fontweight="bold", y=0.98,
    )

    axes[0].imshow(imgBefore, origin="upper", aspect="auto")
    axes[0].set_title("Before Biome Conversion (base terrain: Dirt / Stone)", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Depth (tiles)", fontweight="bold")
    axes[0].set_facecolor("#2F2F2F")

    axes[1].imshow(imgAfter, origin="upper", aspect="auto")
    axes[1].set_title("After Biome Conversion (hard tile-type replacement)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("X (tiles)", fontweight="bold")
    axes[1].set_ylabel("Depth (tiles)", fontweight="bold")
    axes[1].set_facecolor("#2F2F2F")

    # Biome boundary lines and labels
    boundaries = [130, 260, 390, 520, 660]
    biomeLabels = ["Forest", "Jungle", "Snow", "Desert", "Corruption", "Crimson"]
    biomeMidpoints = [65, 195, 325, 455, 590, 730]
    labelColors = ["#2E8B57", "#228B22", "#87CEEB", "#DEB887", "#9370DB", "#DC143C"]

    for bx in boundaries:
        for ax in axes:
            ax.axvline(bx, color="white", linewidth=1.5, linestyle="--", alpha=0.8)

    for mid, label, lc in zip(biomeMidpoints, biomeLabels, labelColors):
        axes[1].text(
            mid, 15, label, ha="center", va="center", fontweight="bold",
            fontsize=10, color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=lc, alpha=0.85),
        )

    # Conversion rules text
    rulesText = (
        "Conversion rules (from WorldGen.cs):\n"
        "  Jungle: Dirt -> Mud, Grass -> Mud\n"
        "  Snow: Dirt -> Snow Block, Stone -> Ice, Grass -> Snow Block\n"
        "  Desert: Dirt -> Sand, Grass -> Sand\n"
        "  Corruption: Dirt -> Corrupt Dirt, Stone -> Ebonstone\n"
        "  Crimson: Dirt -> Crimson Dirt, Stone -> Crimstone"
    )
    fig.text(
        0.02, 0.02, rulesText, fontsize=10, ha="left", va="bottom",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="gray"),
    )

    plt.tight_layout(rect=[0, 0.10, 1, 0.95])
    plt.savefig(savePath, dpi=300, bbox_inches="tight", facecolor="white")
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
