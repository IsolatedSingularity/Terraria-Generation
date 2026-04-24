"""
Terraria Biome Distribution Analysis

Two figures, both rendered as 600x500 crops of a generated SMALL world
(4200x1200) so layer-depth ratios stay game-accurate:

- ``terraria_biome_layouts.png``: single crop spanning Forest -> Jungle ->
  Desert from surface to deep cavern, decorated with grass/hellstone bands
  and dashed worldSurface/rockLayer/hellLayer markers.
- ``terraria_biome_transition_detail.png``: vertical slice through three
  adjacent surface biomes with aggressive cavinator carving and ore clusters.
"""

import os

import matplotlib.pyplot as plt

from Engine.spriteRenderer import (
    applyMapDecorations,
    cropSmallWorld,
    drawTileGrid,
)
from Engine.theme import COLORS, applyTokyoNight
from Engine.worldgen import generateSmallWorld

applyTokyoNight()


# ---------------------------------------------------------------------------
# D1: Biome distribution (single SMALL-world crop)
# ---------------------------------------------------------------------------
def createBiomeLayoutVisualization(savePath: str) -> None:
    """Render a 600x500 crop of a SMALL world spanning multiple biomes.

    Replaces the prior symbolic Rectangle/Ellipse layout with real generated
    terrain so biome converters and cave topology are visible at tile scale.
    """
    print("Creating biome distribution (SMALL crop, sprite render)...")
    world = generateSmallWorld(seed=20260423, evilType="corruption",
                                compactBiomes=True)
    layers = world.layers

    # With compact biomes, world center spans Forest -> Jungle -> Desert.
    centerX = world.spawnX
    centerY = int(layers.worldSurface) + 200

    cropped, bounds = cropSmallWorld(
        world.grid, centerX=centerX, centerY=centerY,
        width=600, height=500,
    )

    fig, ax = plt.subplots(figsize=(12, 9))
    drawTileGrid(ax, cropped)
    applyMapDecorations(ax, cropped, layers, cropBounds=bounds)

    h, w = cropped.shape
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xlabel("X (tiles, crop-local)", fontweight="bold")
    ax.set_ylabel("Depth (tiles, crop-local)", fontweight="bold")
    ax.set_title(
        "Biome Distribution (SMALL world, 600x500 crop)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(savePath, dpi=200, bbox_inches="tight",
                facecolor=COLORS["bg"])
    plt.close()
    print(f"Biome distribution saved to {savePath}")


# ---------------------------------------------------------------------------
# D2: Biome transition detail (3-band vertical slice with caves + ores)
# ---------------------------------------------------------------------------
def createBiomeTransitionDetail(savePath: str) -> None:
    """Render a 600x500 crop showing 3 adjacent biome columns with caves.

    The crop is centered on a Forest -> Jungle -> Desert horizontal sweep
    drawn from the same SMALL-world generator so the depth structure
    (cavinator carving, ore veins, mud band, hellstone shell) is internally
    consistent with the distribution figure.
    """
    print("Creating biome transition detail (sprite render)...")
    world = generateSmallWorld(seed=20260424, evilType="corruption",
                                altarsSmashed=0, compactBiomes=True)
    layers = world.layers

    # Center deeper so the rockLayer and hellstone band are visible.
    centerX = world.spawnX
    centerY = int((layers.rockLayer + layers.hellLayer) / 2)

    cropped, bounds = cropSmallWorld(
        world.grid, centerX=centerX, centerY=centerY,
        width=280, height=210,
    )

    fig, ax = plt.subplots(figsize=(7, 5.3))
    drawTileGrid(ax, cropped)
    applyMapDecorations(ax, cropped, layers, cropBounds=bounds)

    h, w = cropped.shape
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xlabel("X (tiles, crop-local)", fontweight="bold")
    ax.set_ylabel("Depth (tiles, crop-local)", fontweight="bold")
    ax.set_title(
        "Biome Transition Detail (280x210 tight crop)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(savePath, dpi=200, bbox_inches="tight",
                facecolor=COLORS["bg"])
    plt.close()
    print(f"Biome transition detail saved to {savePath}")


if __name__ == "__main__":
    print("Starting Terraria biome distribution analysis")

    outputDir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Plots"
    )
    os.makedirs(outputDir, exist_ok=True)

    try:
        createBiomeLayoutVisualization(
            os.path.join(outputDir, "terraria_biome_layouts.png")
        )
        createBiomeTransitionDetail(
            os.path.join(outputDir, "terraria_biome_transition_detail.png")
        )
        print("All biome analysis visualizations complete.")
    except Exception as e:
        print(f"Error: {e}")
        raise
