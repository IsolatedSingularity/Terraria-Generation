"""
Terraria Liquid Settling Simulation
====================================

Simulates the game's actual liquid settling mechanics: a bottom-up deterministic
gravity scan. Liquids fall when AIR is below, spread horizontally when blocked.

Liquid types: Water, Lava, Honey (matching the actual game).
Interactions: Water + Lava = Obsidian, Honey + Lava = Crispy Honey Block.

No hydrostatic pressure, no Torricelli, no viscosity, no evaporation.
Uses Engine.algorithms.settleLiquids for the correct bottom-up scan.

Author: Terraria Generation Project
"""

import sys
import os

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Tuple, Dict

from Engine.algorithms import (
    settleLiquids,
    AIR,
    STONE,
    DIRT,
    WATER,
    LAVA,
    HONEY,
    OBSIDIAN,
    CRISPY_HONEY_BLOCK,
)


# ---------------------------------------------------------------------------
# Tile color palette for visualization
# ---------------------------------------------------------------------------
TILE_COLORS: Dict[int, Tuple[float, float, float]] = {
    AIR: (0.86, 0.93, 1.0),
    STONE: (0.50, 0.50, 0.50),
    DIRT: (0.55, 0.37, 0.24),
    WATER: (0.12, 0.36, 0.75),
    LAVA: (0.85, 0.20, 0.05),
    HONEY: (0.90, 0.72, 0.15),
    OBSIDIAN: (0.15, 0.05, 0.20),
    CRISPY_HONEY_BLOCK: (0.75, 0.45, 0.10),
}


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------
class TerrariaLiquidSimulator:
    """Simulates Terraria's discrete, gravity-driven liquid settling."""

    def __init__(self, width: int = 200, height: int = 150) -> None:
        self.width = width
        self.height = height
        self.grid = np.full((height, width), AIR, dtype=np.int32)
        self.interactionSites: List[Tuple[int, int, int]] = []

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------
    def createTestScene(self) -> None:
        """Build a terrain grid with caves, shafts, ledges, and a hell layer."""
        surfaceY = self.height // 3
        hellY = int(self.height * 0.85)

        # Solid ground below surface
        self.grid[surfaceY:, :] = STONE

        # --- Main cave (center, elliptical) ---
        caveCY, caveCX = surfaceY + 20, 100
        ys = np.arange(surfaceY + 5, surfaceY + 35)
        xs = np.arange(40, 160)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        ellipse = (xx - caveCX) ** 2 / 3600.0 + (yy - caveCY) ** 2 / 225.0
        self.grid[surfaceY + 5 : surfaceY + 35, 40:160][ellipse < 1.0] = AIR

        # --- Left pocket ---
        lpCY, lpCX = surfaceY + 17, 30
        ys2 = np.arange(surfaceY + 10, surfaceY + 25)
        xs2 = np.arange(10, 50)
        yy2, xx2 = np.meshgrid(ys2, xs2, indexing="ij")
        ellipse2 = (xx2 - lpCX) ** 2 / 400.0 + (yy2 - lpCY) ** 2 / 49.0
        self.grid[surfaceY + 10 : surfaceY + 25, 10:50][ellipse2 < 1.0] = AIR

        # --- Jungle/honey pocket (right) ---
        jpCY, jpCX = surfaceY + 15, 170
        ys3 = np.arange(surfaceY + 8, surfaceY + 22)
        xs3 = np.arange(150, 190)
        yy3, xx3 = np.meshgrid(ys3, xs3, indexing="ij")
        ellipse3 = (xx3 - jpCX) ** 2 / 400.0 + (yy3 - jpCY) ** 2 / 49.0
        self.grid[surfaceY + 8 : surfaceY + 22, 150:190][ellipse3 < 1.0] = AIR

        # --- Hell layer (open with pillars) ---
        for y in range(hellY, self.height - 2):
            for x in range(3, self.width - 3):
                if x % 25 > 3:
                    self.grid[y, x] = AIR

        # --- Vertical shaft: surface to main cave ---
        self.grid[2 : surfaceY + 6, 98:103] = AIR

        # --- Vertical shaft: main cave to hell ---
        self.grid[surfaceY + 34 : hellY + 1, 79:84] = AIR

        # --- Ledges inside main cave (horizontal spread demo) ---
        self.grid[surfaceY + 28, 60:90] = STONE
        self.grid[surfaceY + 28, 110:140] = STONE

    def placeLiquids(self) -> None:
        """Place water at top, lava in hell, honey in jungle pocket."""
        surfaceY = self.height // 3
        hellY = int(self.height * 0.85)

        # Water pool above the shaft
        region = self.grid[2:10, 93:108]
        region[region == AIR] = WATER

        # Lava in hell layer
        hellRegion = self.grid[hellY : hellY + 4, 10 : self.width - 10]
        hellRegion[hellRegion == AIR] = LAVA

        # Honey in the jungle pocket
        honeyRegion = self.grid[surfaceY + 9 : surfaceY + 14, 155:185]
        honeyRegion[honeyRegion == AIR] = HONEY

        # Small lava pocket adjacent to honey (interaction demo)
        lavaRegion = self.grid[surfaceY + 14 : surfaceY + 18, 168:176]
        lavaRegion[lavaRegion == AIR] = LAVA

    # ------------------------------------------------------------------
    # Settling and interactions
    # ------------------------------------------------------------------
    def runSettling(self, maxPasses: int = 50) -> np.ndarray:
        """Call Engine's bottom-up settleLiquids. Returns pre-settling snapshot."""
        beforeSnapshot = self.grid.copy()
        self.grid = settleLiquids(self.grid, maxPasses=maxPasses)
        return beforeSnapshot

    def detectInteractions(self) -> None:
        """Post-pass: record interaction products created during settling.

        Water + Lava = Obsidian (handled inside settleLiquids).
        Honey + Lava = Crispy Honey Block (handled inside settleLiquids).
        This method also does a final adjacency sweep for any remaining contacts.
        """
        self.interactionSites.clear()

        # Record obsidian formed by settleLiquids
        obsY, obsX = np.where(self.grid == OBSIDIAN)
        for y, x in zip(obsY, obsX):
            self.interactionSites.append((int(y), int(x), OBSIDIAN))

        # Record crispy honey blocks formed by settleLiquids
        chY, chX = np.where(self.grid == CRISPY_HONEY_BLOCK)
        for y, x in zip(chY, chX):
            self.interactionSites.append((int(y), int(x), CRISPY_HONEY_BLOCK))

        # Final adjacency sweep: convert any remaining honey+lava contacts
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                if self.grid[y, x] != HONEY:
                    continue
                for dy, dx in directions:
                    ny, nx = y + dy, x + dx
                    if self.grid[ny, nx] == LAVA:
                        self.grid[ny, nx] = CRISPY_HONEY_BLOCK
                        self.interactionSites.append((ny, nx, CRISPY_HONEY_BLOCK))

        # Final adjacency sweep: water+lava contacts missed by settling
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                if self.grid[y, x] != WATER:
                    continue
                for dy, dx in directions:
                    ny, nx = y + dy, x + dx
                    if self.grid[ny, nx] == LAVA:
                        self.grid[ny, nx] = OBSIDIAN
                        self.interactionSites.append((ny, nx, OBSIDIAN))

    def countLiquids(self) -> Dict[str, int]:
        """Return tile counts for each liquid type and interaction product."""
        return {
            "water": int(np.sum(self.grid == WATER)),
            "lava": int(np.sum(self.grid == LAVA)),
            "honey": int(np.sum(self.grid == HONEY)),
            "obsidian": int(np.sum(self.grid == OBSIDIAN)),
            "crispyHoney": int(np.sum(self.grid == CRISPY_HONEY_BLOCK)),
        }


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------
def buildColorGrid(grid: np.ndarray) -> np.ndarray:
    """Map tile IDs to an RGB array for imshow."""
    h, w = grid.shape
    rgb = np.full((h, w, 3), 0.5, dtype=np.float64)
    for tileId, color in TILE_COLORS.items():
        mask = grid == tileId
        rgb[mask] = color
    return rgb


def visualize(
    beforeGrid: np.ndarray,
    afterGrid: np.ndarray,
    interactionSites: List[Tuple[int, int, int]],
    beforeCounts: Dict[str, int],
    afterCounts: Dict[str, int],
    savePath: str,
) -> None:
    """Create a 2x2 figure: before, after, interaction highlights, tile counts."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # --- Top-left: Before settling ---
    axes[0, 0].imshow(buildColorGrid(beforeGrid), aspect="auto")
    axes[0, 0].set_title("Before Settling (Initial Placement)", fontweight="bold", fontsize=12)
    axes[0, 0].set_xlabel("X (tiles)")
    axes[0, 0].set_ylabel("Y (tiles)")

    # --- Top-right: After settling ---
    axes[0, 1].imshow(buildColorGrid(afterGrid), aspect="auto")
    axes[0, 1].set_title("After Settling (Bottom-Up Gravity Scan)", fontweight="bold", fontsize=12)
    axes[0, 1].set_xlabel("X (tiles)")
    axes[0, 1].set_ylabel("Y (tiles)")

    # --- Bottom-left: Interaction highlights ---
    axes[1, 0].imshow(buildColorGrid(afterGrid), aspect="auto")
    for y, x, product in interactionSites:
        color = "magenta" if product == OBSIDIAN else "yellow"
        axes[1, 0].plot(x, y, "s", color=color, markersize=2, alpha=0.85)
    axes[1, 0].set_title("Interaction Zones Highlighted", fontweight="bold", fontsize=12)
    axes[1, 0].set_xlabel("X (tiles)")
    axes[1, 0].set_ylabel("Y (tiles)")
    obsidianPatch = mpatches.Patch(
        color="magenta", label=f"Obsidian (Water+Lava): {afterCounts['obsidian']}"
    )
    crispyPatch = mpatches.Patch(
        color="yellow", label=f"Crispy Honey (Honey+Lava): {afterCounts['crispyHoney']}"
    )
    axes[1, 0].legend(handles=[obsidianPatch, crispyPatch], loc="upper right", fontsize=9)

    # --- Bottom-right: Bar chart of liquid/product counts ---
    labels = ["Water", "Lava", "Honey", "Obsidian", "Crispy\nHoney"]
    beforeVals = [
        beforeCounts["water"],
        beforeCounts["lava"],
        beforeCounts["honey"],
        beforeCounts["obsidian"],
        beforeCounts["crispyHoney"],
    ]
    afterVals = [
        afterCounts["water"],
        afterCounts["lava"],
        afterCounts["honey"],
        afterCounts["obsidian"],
        afterCounts["crispyHoney"],
    ]

    xPos = np.arange(len(labels))
    barWidth = 0.35
    barColorsBefore = [
        (c[0] * 0.6 + 0.4, c[1] * 0.6 + 0.4, c[2] * 0.6 + 0.4)
        for c in [
            TILE_COLORS[WATER],
            TILE_COLORS[LAVA],
            TILE_COLORS[HONEY],
            TILE_COLORS[OBSIDIAN],
            TILE_COLORS[CRISPY_HONEY_BLOCK],
        ]
    ]
    barColorsAfter = [
        TILE_COLORS[WATER],
        TILE_COLORS[LAVA],
        TILE_COLORS[HONEY],
        TILE_COLORS[OBSIDIAN],
        TILE_COLORS[CRISPY_HONEY_BLOCK],
    ]

    axes[1, 1].bar(
        xPos - barWidth / 2,
        beforeVals,
        barWidth,
        color=barColorsBefore,
        edgecolor="black",
        linewidth=0.5,
        label="Before",
    )
    bars2 = axes[1, 1].bar(
        xPos + barWidth / 2,
        afterVals,
        barWidth,
        color=barColorsAfter,
        edgecolor="black",
        linewidth=0.5,
        label="After",
    )
    axes[1, 1].set_xticks(xPos)
    axes[1, 1].set_xticklabels(labels)
    axes[1, 1].set_title("Tile Counts: Before vs After Settling", fontweight="bold", fontsize=12)
    axes[1, 1].set_ylabel("Tile Count")
    axes[1, 1].legend(fontsize=10)

    maxVal = max(max(beforeVals), max(afterVals)) if max(max(beforeVals), max(afterVals)) > 0 else 1
    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            axes[1, 1].text(
                bar.get_x() + bar.get_width() / 2,
                h + maxVal * 0.01,
                str(int(h)),
                ha="center",
                fontsize=8,
                fontweight="bold",
            )

    # Global legend for tile types
    legendPatches = [
        mpatches.Patch(color=TILE_COLORS[WATER], label="Water"),
        mpatches.Patch(color=TILE_COLORS[LAVA], label="Lava"),
        mpatches.Patch(color=TILE_COLORS[HONEY], label="Honey"),
        mpatches.Patch(color=TILE_COLORS[OBSIDIAN], label="Obsidian"),
        mpatches.Patch(color=TILE_COLORS[CRISPY_HONEY_BLOCK], label="Crispy Honey Block"),
        mpatches.Patch(color=TILE_COLORS[STONE], label="Stone"),
    ]
    fig.legend(
        handles=legendPatches,
        loc="lower center",
        ncol=6,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.suptitle(
        "Terraria Liquid Settling Simulation\n"
        "Bottom-Up Gravity Scan via Engine.algorithms.settleLiquids",
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(savePath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {savePath}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the full liquid settling simulation and produce visualization."""
    print("Terraria Liquid Settling Simulation")
    print("=" * 40)

    sim = TerrariaLiquidSimulator(width=200, height=150)
    sim.createTestScene()
    sim.placeLiquids()

    beforeCounts = sim.countLiquids()
    print(f"Before settling: {beforeCounts}")

    beforeGrid = sim.runSettling(maxPasses=60)
    sim.detectInteractions()

    afterCounts = sim.countLiquids()
    print(f"After settling:  {afterCounts}")

    plotDir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "Plots", "Excess"
    )
    os.makedirs(plotDir, exist_ok=True)
    savePath = os.path.join(plotDir, "liquid_settling_simulation.png")

    visualize(beforeGrid, sim.grid, sim.interactionSites, beforeCounts, afterCounts, savePath)


if __name__ == "__main__":
    main()
