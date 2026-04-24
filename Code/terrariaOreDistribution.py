"""Terraria ore distribution figures.

Two figures, mini-world redesign edition:

- ``ore_distribution.png``: 3-panel row, each a full TINY world rendered at
  native resolution with one ore family highlighted (other tiles dimmed).
  Panels: Pre-Hardmode, Hardmode Tier 1, Hardmode Tier 3.
- ``ore_density.png``: heatmap with depth bins on the Y axis and ore types
  on the X axis, computed from a SMALL world for honest statistics.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from Engine.algorithms import (
    ADAMANTITE,
    CHLOROPHYTE,
    COBALT,
    COPPER,
    GOLD,
    IRON,
    LEAD,
    MYTHRIL,
    ORICHALCUM,
    PALLADIUM,
    PLATINUM,
    SILVER,
    TIN,
    TITANIUM,
    TUNGSTEN,
)
from Engine.theme import COLORS, PALETTE, applyTokyoNight
from Engine.worldgen import (
    generateMiniWorld,
    generateSmallWorld,
    renderMiniWorld,
)

applyTokyoNight()


_PRE_HM = {COPPER, TIN, IRON, LEAD, SILVER, TUNGSTEN, GOLD, PLATINUM}
_HM_TIER_1 = {COBALT, PALLADIUM}
_HM_TIER_3 = {ADAMANTITE, TITANIUM, CHLOROPHYTE}

_ORE_NAMES = [
    ("Copper", COPPER), ("Tin", TIN),
    ("Iron", IRON), ("Lead", LEAD),
    ("Silver", SILVER), ("Tungsten", TUNGSTEN),
    ("Gold", GOLD), ("Platinum", PLATINUM),
    ("Cobalt", COBALT), ("Palladium", PALLADIUM),
    ("Mythril", MYTHRIL), ("Orichalcum", ORICHALCUM),
    ("Adamantite", ADAMANTITE), ("Titanium", TITANIUM),
    ("Chlorophyte", CHLOROPHYTE),
]


def createOreDistributionFigure(savePath: str) -> None:
    """Three TINY-world panels with one ore family highlighted in each."""
    print("Creating ore distribution (3 TINY worlds)...")

    preHm = generateMiniWorld(seed=20260423, altarsSmashed=0, oreScale=10.0)
    midHm = generateMiniWorld(seed=20260424, altarsSmashed=3, oreScale=10.0)
    lateHm = generateMiniWorld(seed=20260425, altarsSmashed=9, oreScale=10.0)

    fig, axes = plt.subplots(1, 3, figsize=(20, 4.6))

    renderMiniWorld(preHm.grid, axes[0],
                    title="Pre-Hardmode Ores",
                    showLayers=True, layers=preHm.layers,
                    highlightTiles=_PRE_HM)
    renderMiniWorld(midHm.grid, axes[1],
                    title="Hardmode Tier 1",
                    showLayers=True, layers=midHm.layers,
                    highlightTiles=_HM_TIER_1)
    renderMiniWorld(lateHm.grid, axes[2],
                    title="Hardmode Tier 3",
                    showLayers=True, layers=lateHm.layers,
                    highlightTiles=_HM_TIER_3)

    plt.tight_layout()
    plt.savefig(savePath, dpi=110, bbox_inches="tight",
                facecolor=COLORS["bg"])
    plt.close()
    print(f"Ore distribution saved to {savePath}")


def createOreDensityFigure(savePath: str) -> None:
    """Heatmap of ore tile counts by depth bin and ore type.

    Computed from a SMALL world so statistics are honest. Pre-HM, post-3
    altar, and post-9 altar runs are summed so all ore families appear in
    the heatmap.
    """
    print("Creating ore density heatmap (SMALL world stats)...")

    world0 = generateSmallWorld(seed=42, altarsSmashed=0)
    world3 = generateSmallWorld(seed=42, altarsSmashed=3)
    world9 = generateSmallWorld(seed=42, altarsSmashed=9)
    height = world0.layers.maxTilesY

    # 10 depth bins from worldSurface to hellLayer.
    binEdges = np.linspace(int(world0.layers.worldSurface),
                           int(world0.layers.hellLayer),
                           11).astype(int)
    binLabels = [f"{binEdges[i]}-{binEdges[i + 1]}"
                 for i in range(len(binEdges) - 1)]

    # Use the world that contains each ore family.
    def gridForOre(oreId: int) -> np.ndarray:
        if oreId in _PRE_HM:
            return world0.grid
        if oreId in _HM_TIER_1:
            return world3.grid
        return world9.grid

    densityMatrix = np.zeros((len(binLabels), len(_ORE_NAMES)), dtype=np.int64)
    for col, (_, oreId) in enumerate(_ORE_NAMES):
        grid = gridForOre(oreId)
        for row in range(len(binLabels)):
            y0, y1 = binEdges[row], binEdges[row + 1]
            densityMatrix[row, col] = int((grid[y0:y1] == oreId).sum())

    cmap = LinearSegmentedColormap.from_list(
        "tokyoOreDensity",
        ["#1a1b26", "#3b4261", "#7aa2f7", "#bb9af7"],
        N=256,
    )

    fig, ax = plt.subplots(figsize=(13, 7.5))
    # Log1p so low-count cells stay readable next to high-count cells.
    displayed = np.log1p(densityMatrix)
    im = ax.imshow(displayed, cmap=cmap, aspect="auto",
                   interpolation="nearest")

    ax.set_xticks(np.arange(len(_ORE_NAMES)))
    ax.set_xticklabels([n for n, _ in _ORE_NAMES], rotation=35, ha="right",
                       color=PALETTE["fg"])
    ax.set_yticks(np.arange(len(binLabels)))
    ax.set_yticklabels(binLabels, color=PALETTE["fg"])
    ax.set_xlabel("Ore Type", color=PALETTE["fg"], fontweight="bold")
    ax.set_ylabel("Depth Range (tiles)", color=PALETTE["fg"], fontweight="bold")
    ax.set_title("Ore Density by Depth", color=PALETTE["fg"],
                 fontsize=14, fontweight="bold", pad=10)

    # Annotate non-zero cells with raw counts.
    for r in range(densityMatrix.shape[0]):
        for c in range(densityMatrix.shape[1]):
            count = int(densityMatrix[r, c])
            if count > 0:
                ax.text(c, r, str(count), ha="center", va="center",
                        color=PALETTE["fg"], fontsize=8,
                        fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("log(1 + tile count)", color=PALETTE["fg"])
    cbar.ax.tick_params(colors=PALETTE["muted"])

    plt.tight_layout()
    plt.savefig(savePath, dpi=130, bbox_inches="tight",
                facecolor=COLORS["bg"])
    plt.close()
    print(f"Ore density saved to {savePath}")


if __name__ == "__main__":
    print("Starting Terraria ore distribution analysis")

    outputDir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Plots"
    )
    os.makedirs(outputDir, exist_ok=True)

    createOreDistributionFigure(
        os.path.join(outputDir, "ore_distribution.png")
    )
    createOreDensityFigure(
        os.path.join(outputDir, "ore_density.png")
    )
    print("All ore distribution visualizations complete.")
