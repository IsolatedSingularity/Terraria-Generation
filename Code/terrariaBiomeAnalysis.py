"""Terraria biome distribution figures rendered as native TINY worlds.

Two figures, both 240x140 worlds drawn at native resolution (~6 px/tile)
with no cropping, per the mini-world redesign audit:

- ``terraria_biome_layouts.png``: full TINY world showing the canonical
  Snow, Forest, Jungle, Desert, Corruption layout with caves carved through
  every biome.
- ``terraria_biome_transition_detail.png``: same layout with a different
  seed so cave topology and biome shapes vary while the global structure
  remains comparable.
"""

import os

import matplotlib.pyplot as plt

from Engine.theme import COLORS, applyTokyoNight
from Engine.worldgen import generateMiniWorld, renderMiniWorld

applyTokyoNight()


def createBiomeLayoutVisualization(savePath: str) -> None:
    """Render a full 240x140 TINY world centered on Snow to Corruption."""
    print("Creating biome layout (TINY native render)...")
    world = generateMiniWorld(seed=20260423, evilType="corruption")

    fig, ax = plt.subplots(figsize=(14.4, 8.4))
    renderMiniWorld(world.grid, ax,
                    title="Biome Distribution",
                    showLayers=True, layers=world.layers)
    plt.tight_layout()
    plt.savefig(savePath, dpi=110, bbox_inches="tight",
                facecolor=COLORS["bg"])
    plt.close()
    print(f"Biome layout saved to {savePath}")


def createBiomeTransitionDetail(savePath: str) -> None:
    """Render a second TINY world (different seed) for the transition view."""
    print("Creating biome transitions (TINY native render)...")
    world = generateMiniWorld(seed=20260424, evilType="corruption")

    fig, ax = plt.subplots(figsize=(14.4, 8.4))
    renderMiniWorld(world.grid, ax,
                    title="Biome Transitions",
                    showLayers=True, layers=world.layers)
    plt.tight_layout()
    plt.savefig(savePath, dpi=110, bbox_inches="tight",
                facecolor=COLORS["bg"])
    plt.close()
    print(f"Biome transitions saved to {savePath}")


if __name__ == "__main__":
    print("Starting Terraria biome distribution analysis")

    outputDir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Plots"
    )
    os.makedirs(outputDir, exist_ok=True)

    createBiomeLayoutVisualization(
        os.path.join(outputDir, "terraria_biome_layouts.png")
    )
    createBiomeTransitionDetail(
        os.path.join(outputDir, "terraria_biome_transition_detail.png")
    )
    print("All biome analysis visualizations complete.")
