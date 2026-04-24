"""Terraria Ore Distribution -- SMALL-world crops + altar progression.

Two figures, both rendered as 600x500 crops of generated SMALL worlds:

- ``ore_distribution.png``: 3-panel column showing pre-Hardmode, post-3-altar,
  and post-9-altar states centered on the rock layer so the altar-tier
  ladder (Cobalt/Mythril/Adamantite) is visible.
- ``ore_depth_density.png``: 3-panel column of ore-tile-per-row counts vs
  depth at the same three altar checkpoints, with shared external legend
  and dashed worldSurface/rockLayer/hellLayer markers.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

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
from Engine.spriteRenderer import (
    applyMapDecorations,
    cropSmallWorld,
    drawTileGrid,
)
from Engine.theme import COLORS, ORE_COLORS, PALETTE, applyTokyoNight
from Engine.worldgen import generateSmallWorld

applyTokyoNight()


_PRE_HM_ORES = [COPPER, TIN, IRON, LEAD, SILVER, TUNGSTEN, GOLD, PLATINUM]
_HM_TIER_1 = [COBALT, PALLADIUM]
_HM_TIER_2 = [MYTHRIL, ORICHALCUM]
_HM_TIER_3 = [ADAMANTITE, TITANIUM, CHLOROPHYTE]

_ORE_NAME = {
    COPPER: "Copper", TIN: "Tin", IRON: "Iron", LEAD: "Lead",
    SILVER: "Silver", TUNGSTEN: "Tungsten",
    GOLD: "Gold", PLATINUM: "Platinum",
    COBALT: "Cobalt", PALLADIUM: "Palladium",
    MYTHRIL: "Mythril", ORICHALCUM: "Orichalcum",
    ADAMANTITE: "Adamantite", TITANIUM: "Titanium",
    CHLOROPHYTE: "Chlorophyte",
}
_ORE_COLOR = {
    COPPER: ORE_COLORS["copper"], TIN: ORE_COLORS["tin"],
    IRON: ORE_COLORS["iron"], LEAD: ORE_COLORS["lead"],
    SILVER: ORE_COLORS["silver"], TUNGSTEN: ORE_COLORS["tungsten"],
    GOLD: ORE_COLORS["gold"], PLATINUM: ORE_COLORS["platinum"],
    COBALT: ORE_COLORS["cobalt"], PALLADIUM: ORE_COLORS["palladium"],
    MYTHRIL: ORE_COLORS["mythril"], ORICHALCUM: ORE_COLORS["orichalcum"],
    ADAMANTITE: ORE_COLORS["adamantite"], TITANIUM: ORE_COLORS["titanium"],
    CHLOROPHYTE: ORE_COLORS["chlorophyte"],
}


# ===================================================================
# D3: Ore distribution (3-panel SMALL crop)
# ===================================================================
def createOreDistributionFigure(savePath: str) -> None:
    """Stack three 600x500 crops at altar checkpoints 0/3/9."""
    print("Creating ore distribution (3-panel SMALL crops)...")
    seed = 20260423
    worlds = [
        ("Pre-Hardmode (0 altars)", generateSmallWorld(seed=seed, altarsSmashed=0)),
        ("Hardmode (3 altars smashed)", generateSmallWorld(seed=seed, altarsSmashed=3)),
        ("Late Hardmode (9 altars smashed)", generateSmallWorld(seed=seed, altarsSmashed=9)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    for ax, (title, world) in zip(axes, worlds):
        layers = world.layers
        # Center deep enough to span rock layer to hellstone band.
        centerX = world.spawnX
        centerY = int((layers.rockLayer + layers.hellLayer) / 2)
        cropped, bounds = cropSmallWorld(
            world.grid, centerX=centerX, centerY=centerY,
            width=260, height=200,
        )
        drawTileGrid(ax, cropped)
        applyMapDecorations(ax, cropped, layers, cropBounds=bounds,
                            grassBand=False)
        h, w = cropped.shape
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("X (tiles)")
        ax.set_ylabel("Depth (tiles)")

    fig.suptitle(
        "Ore Distribution (260x200 tight crops, SMALL world)",
        fontsize=13, fontweight="bold", y=1.0,
    )
    plt.tight_layout()
    plt.savefig(savePath, dpi=200, bbox_inches="tight",
                facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"Ore distribution saved to {savePath}")


# ===================================================================
# D4: Ore density (3-panel column with external legend)
# ===================================================================
def _countOreByDepth(
    grid: np.ndarray, oreId: int, binSize: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    h, _ = grid.shape
    edges = np.arange(0, h + binSize, binSize)
    counts = np.array([
        np.sum(grid[edges[i]:min(edges[i + 1], h), :] == oreId)
        for i in range(len(edges) - 1)
    ], dtype=float)
    centers = (edges[:-1] + np.minimum(edges[1:], h)) / 2.0
    return centers, counts


def _drawDepthMarkers(ax: plt.Axes, layers, maxDepth: int) -> None:
    for y, color in [
        (layers.worldSurface, PALETTE["cyan"]),
        (layers.rockLayer, PALETTE["yellow"]),
        (float(layers.hellLayer), PALETTE["red"]),
    ]:
        if 0 <= y <= maxDepth:
            ax.axhline(y=y, color=color, linestyle="--",
                       linewidth=0.9, alpha=0.55)


def createOreDensityFigure(savePath: str) -> None:
    """3-panel column of ore-density-vs-depth at altar checkpoints 0/3/9.

    Each ore is rendered as a smoothed filled-area profile (binSize=24,
    gaussian-smoothed) so the curves read as bell-shaped depth bands rather
    than jagged 8-tile noise. Pre-HM ores fade in post-altar panels.
    """
    print("Creating ore depth-density (3-panel column, smoothed)...")
    seed = 20260423
    worlds = {
        0: generateSmallWorld(seed=seed, altarsSmashed=0),
        3: generateSmallWorld(seed=seed, altarsSmashed=3),
        9: generateSmallWorld(seed=seed, altarsSmashed=9),
    }
    layers = worlds[0].layers
    maxDepth = layers.maxTilesY

    panelOres = {
        0: [(_PRE_HM_ORES, 1.0)],
        3: [(_PRE_HM_ORES, 0.25), (_HM_TIER_1, 1.0), (_HM_TIER_2, 1.0)],
        9: [(_PRE_HM_ORES, 0.20), (_HM_TIER_1, 0.45),
            (_HM_TIER_2, 1.0), (_HM_TIER_3, 1.0)],
    }
    titles = {
        0: "Pre-Hardmode (0 altars)",
        3: "Hardmode (3 altars: tier-1 + tier-2 visible)",
        9: "Late Hardmode (9 altars: full tier ladder)",
    }

    def _smoothProfile(grid: np.ndarray, oreId: int) -> tuple[np.ndarray, np.ndarray]:
        depths, counts = _countOreByDepth(grid, oreId, binSize=24)
        if counts.sum() == 0:
            return depths, counts
        # Gaussian smoothing kernel (sigma=1.5 bins)
        kernel = np.exp(-0.5 * (np.arange(-4, 5) / 1.5) ** 2)
        kernel /= kernel.sum()
        smoothed = np.convolve(counts, kernel, mode="same")
        return depths, smoothed

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    for ax, altars in zip(axes, [0, 3, 9]):
        grid = worlds[altars].grid
        for oreList, alpha in panelOres[altars]:
            for oreId in oreList:
                depths, counts = _smoothProfile(grid, oreId)
                if counts.sum() == 0:
                    continue
                color = _ORE_COLOR[oreId]
                # Filled area on the left of the curve so colors blend cleanly.
                ax.fill_betweenx(depths, 0, counts,
                                 color=color, alpha=alpha * 0.45,
                                 linewidth=0)
                ax.plot(counts, depths, color=color,
                        linewidth=1.6, alpha=alpha,
                        label=_ORE_NAME[oreId] if alpha >= 0.99 else None)
        _drawDepthMarkers(ax, layers, maxDepth)
        ax.invert_yaxis()
        ax.set_title(titles[altars], fontsize=11, fontweight="bold")
        ax.set_ylabel("Depth (tiles)", fontweight="bold")
        ax.grid(True, alpha=0.15, linestyle="--")
        ax.set_xlim(left=0)

    axes[-1].set_xlabel("Smoothed ore tiles per 24-row band", fontweight="bold")

    # External shared legend above the figure.
    seenIds = list(dict.fromkeys(
        _PRE_HM_ORES + _HM_TIER_1 + _HM_TIER_2 + _HM_TIER_3
    ))
    handles = [
        Line2D([0], [0], color=_ORE_COLOR[oid], linewidth=3.0,
               label=_ORE_NAME[oid])
        for oid in seenIds
    ]
    fig.legend(
        handles=handles, loc="upper center",
        bbox_to_anchor=(0.5, 0.995), ncol=5, fontsize=9, frameon=False,
    )

    fig.suptitle(
        "Ore Density vs Depth (3 altar checkpoints)",
        fontsize=14, fontweight="bold", y=1.04,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(savePath, dpi=200, bbox_inches="tight",
                facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"Ore depth-density saved to {savePath}")


# ===================================================================
# Main
# ===================================================================
def main() -> None:
    plotsDir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Plots"
    )
    os.makedirs(plotsDir, exist_ok=True)

    print("=" * 60)
    print("Terraria Ore Distribution (SMALL-world crops + altar progression)")
    print("=" * 60)

    createOreDistributionFigure(
        os.path.join(plotsDir, "ore_distribution.png")
    )
    createOreDensityFigure(
        os.path.join(plotsDir, "ore_depth_density.png")
    )


if __name__ == "__main__":
    main()
