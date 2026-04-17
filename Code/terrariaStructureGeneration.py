"""
Terraria Structure Generation Visualization
============================================

Accurate large-world structure placement using game-derived quotas,
layer depths, and StructureMap exclusion zones. Attempt-loop placement
replicates the actual C# WorldGen logic.

Author: Terraria Generation Project
"""

import sys
import os
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# Engine imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Engine.constants import LARGE, LayerDepths, StructureQuotas
from Engine.structureMap import StructureMap, Rectangle

# Plot styling
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"


class TerrariaStructureGenerator:
    """Generates and visualizes Terraria large-world structures
    using game-accurate quotas, layer depths, and StructureMap
    exclusion zones."""

    def __init__(
        self,
        worldWidth: int = 8400,
        worldHeight: int = 2400,
        seed: int = 12345,
    ) -> None:
        self.worldWidth = worldWidth
        self.worldHeight = worldHeight
        self.seed = seed

        np.random.seed(seed)
        random.seed(seed)

        # Game-accurate layer depths and quotas
        self.layers = LayerDepths.forLarge()
        self.quotas = StructureQuotas.forLarge()
        self.borderBuffer = LARGE.borderBuffer  # 45 tiles

        # Exclusion zone tracker
        self.structureMap = StructureMap()

        # Placed structure storage
        self.structures: Dict[str, List[Tuple[int, int, int, int]]] = {}

        # Dungeon side (True = right, False = left)
        self.dungeonOnRight: bool = random.choice([True, False])

    # ------------------------------------------------------------------
    # Placement helpers
    # ------------------------------------------------------------------

    def _attemptPlace(
        self,
        xMin: int,
        xMax: int,
        yMin: int,
        yMax: int,
        structW: int,
        structH: int,
        padding: int = 5,
        maxAttempts: int = 1000,
    ) -> Optional[Rectangle]:
        """Try up to maxAttempts random positions within bounds.
        Returns a Rectangle if placement succeeds, None otherwise."""
        for _ in range(maxAttempts):
            x = random.randint(xMin, xMax - structW)
            y = random.randint(yMin, yMax - structH)
            rect = Rectangle(x, y, structW, structH)
            if self.structureMap.canPlace(rect, padding):
                self.structureMap.addProtectedStructure(rect, padding)
                return rect
        return None

    # ------------------------------------------------------------------
    # Structure placement methods
    # ------------------------------------------------------------------

    def placeFloatingIslands(self) -> List[Tuple[int, int, int, int]]:
        """Place exactly 6 floating islands for a large world.
        Islands sit above worldSurface with attempt-loop spacing."""
        count = self.quotas.floatingIslands  # 6
        islandW, islandH = 80, 40
        yMin = 80  # well above surface
        yMax = int(self.layers.worldSurface) - islandH - 20
        xMin = self.borderBuffer
        xMax = self.worldWidth - self.borderBuffer

        placed: List[Tuple[int, int, int, int]] = []
        for _ in range(count):
            rect = self._attemptPlace(
                xMin, xMax, yMin, yMax, islandW, islandH,
                padding=60, maxAttempts=2000,
            )
            if rect is not None:
                placed.append((rect.x, rect.y, rect.width, rect.height))

        self.structures["floatingIslands"] = placed
        return placed

    def placeUndergroundCabins(self) -> List[Tuple[int, int, int, int]]:
        """Place 140-160 underground cabins between rockLayer and hellLayer."""
        count = random.randint(
            self.quotas.undergroundCabinsMin,
            self.quotas.undergroundCabinsMax,
        )
        cabinW, cabinH = 24, 18
        yMin = int(self.layers.rockLayer)
        yMax = self.layers.hellLayer - cabinH
        xMin = self.borderBuffer
        xMax = self.worldWidth - self.borderBuffer

        placed: List[Tuple[int, int, int, int]] = []
        for _ in range(count):
            rect = self._attemptPlace(
                xMin, xMax, yMin, yMax, cabinW, cabinH,
                padding=8, maxAttempts=1000,
            )
            if rect is not None:
                placed.append((rect.x, rect.y, rect.width, rect.height))

        self.structures["undergroundCabins"] = placed
        return placed

    def placeLifeCrystals(self) -> List[Tuple[int, int, int, int]]:
        """Place up to 403 Life Crystals (2x2 FrameImportant tiles)
        between worldSurface and hellLayer."""
        maxCount = self.quotas.lifeCrystalsMax  # 403
        crystalW, crystalH = 2, 2
        yMin = int(self.layers.worldSurface) + 20
        yMax = self.layers.hellLayer - crystalH
        xMin = self.borderBuffer
        xMax = self.worldWidth - self.borderBuffer

        placed: List[Tuple[int, int, int, int]] = []
        for _ in range(maxCount):
            rect = self._attemptPlace(
                xMin, xMax, yMin, yMax, crystalW, crystalH,
                padding=2, maxAttempts=200,
            )
            if rect is not None:
                placed.append((rect.x, rect.y, rect.width, rect.height))

        self.structures["lifeCrystals"] = placed
        return placed

    def placeSurfaceChests(self) -> List[Tuple[int, int, int, int]]:
        """Place 42 surface chests near the surface layer."""
        count = self.quotas.surfaceChests  # 42
        chestW, chestH = 2, 2
        yMin = int(self.layers.worldSurface) - 30
        yMax = int(self.layers.worldSurface) + 60
        xMin = self.borderBuffer
        xMax = self.worldWidth - self.borderBuffer

        placed: List[Tuple[int, int, int, int]] = []
        for _ in range(count):
            rect = self._attemptPlace(
                xMin, xMax, yMin, yMax, chestW, chestH,
                padding=20, maxAttempts=1000,
            )
            if rect is not None:
                placed.append((rect.x, rect.y, rect.width, rect.height))

        self.structures["surfaceChests"] = placed
        return placed

    def placeDungeon(self) -> List[Tuple[int, int, int, int]]:
        """Place the dungeon on one side of the world (large rectangle).
        Dungeon side is chosen at init; jungle temple goes opposite."""
        dungeonW, dungeonH = 200, 300
        yMin = int(self.layers.worldSurface) - 20
        yMax = int(self.layers.rockLayer)

        if self.dungeonOnRight:
            xMin = int(self.worldWidth * 0.75)
            xMax = self.worldWidth - self.borderBuffer
        else:
            xMin = self.borderBuffer
            xMax = int(self.worldWidth * 0.25)

        placed: List[Tuple[int, int, int, int]] = []
        rect = self._attemptPlace(
            xMin, xMax, yMin, yMax, dungeonW, dungeonH,
            padding=30, maxAttempts=2000,
        )
        if rect is not None:
            placed.append((rect.x, rect.y, rect.width, rect.height))

        self.structures["dungeon"] = placed
        return placed

    def placeJungleTemple(self) -> List[Tuple[int, int, int, int]]:
        """Place the jungle temple on the opposite side from the dungeon,
        deep in the cavern layer."""
        templeW, templeH = 150, 120
        yMin = int(self.layers.rockLayer) + 50
        yMax = self.layers.hellLayer - templeH - 50

        # Opposite side from dungeon
        if self.dungeonOnRight:
            xMin = self.borderBuffer
            xMax = int(self.worldWidth * 0.35)
        else:
            xMin = int(self.worldWidth * 0.65)
            xMax = self.worldWidth - self.borderBuffer

        placed: List[Tuple[int, int, int, int]] = []
        rect = self._attemptPlace(
            xMin, xMax, yMin, yMax, templeW, templeH,
            padding=40, maxAttempts=2000,
        )
        if rect is not None:
            placed.append((rect.x, rect.y, rect.width, rect.height))

        self.structures["jungleTemple"] = placed
        return placed

    def generateAllStructures(self) -> Dict[str, List[Tuple[int, int, int, int]]]:
        """Run all placement passes in order (large structures first)."""
        self.structureMap.clear()
        self.structures.clear()

        self.placeDungeon()
        self.placeJungleTemple()
        self.placeFloatingIslands()
        self.placeUndergroundCabins()
        self.placeSurfaceChests()
        self.placeLifeCrystals()

        return self.structures

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def visualize(self, savePath: Optional[str] = None) -> None:
        """Two-panel figure: world cross-section with all structures
        and a statistics panel comparing counts to game quotas."""
        if not self.structures:
            self.generateAllStructures()

        fig, (axMap, axStats) = plt.subplots(
            1, 2, figsize=(26, 14),
            gridspec_kw={"width_ratios": [3, 1]},
        )

        # -- Left panel: world cross-section --
        colorMap = {
            "floatingIslands": "#87CEEB",
            "undergroundCabins": "#DEB887",
            "lifeCrystals": "#FF69B4",
            "surfaceChests": "#FFD700",
            "dungeon": "#4A4A4A",
            "jungleTemple": "#8B4513",
        }
        markerMap = {
            "floatingIslands": ("o", 90),
            "undergroundCabins": ("s", 18),
            "lifeCrystals": ("D", 8),
            "surfaceChests": ("^", 40),
            "dungeon": ("D", 250),
            "jungleTemple": ("^", 200),
        }

        # World background
        worldRect = patches.Rectangle(
            (0, 0), self.worldWidth, self.worldHeight,
            linewidth=2, edgecolor="black",
            facecolor="#F0F8FF", alpha=0.3,
        )
        axMap.add_patch(worldRect)

        # Border buffer shading
        for bx in [0, self.worldWidth - self.borderBuffer]:
            buf = patches.Rectangle(
                (bx, 0), self.borderBuffer, self.worldHeight,
                facecolor="gray", alpha=0.15,
            )
            axMap.add_patch(buf)

        # Layer lines
        ws = self.layers.worldSurface
        rl = self.layers.rockLayer
        hl = self.layers.hellLayer
        axMap.axhline(y=ws, color="saddlebrown", lw=2.5, label=f"worldSurface ({int(ws)})")
        axMap.axhline(y=rl, color="slategray", ls="--", lw=2, label=f"rockLayer ({int(rl)})")
        axMap.axhline(y=hl, color="darkred", ls="-.", lw=2.5, label=f"hellLayer ({hl})")

        # Plot structures
        for key, rects in self.structures.items():
            if not rects:
                continue
            color = colorMap.get(key, "#000000")
            marker, size = markerMap.get(key, (".", 20))
            xs = [r[0] + r[2] / 2 for r in rects]
            ys = [r[1] + r[3] / 2 for r in rects]
            label = f"{key} ({len(rects)})"
            axMap.scatter(
                xs, ys, c=color, s=size, marker=marker,
                alpha=0.75, edgecolors="black", linewidth=0.5,
                label=label, zorder=5,
            )

            # Draw rectangles for large structures
            if key in ("dungeon", "jungleTemple"):
                for r in rects:
                    p = patches.Rectangle(
                        (r[0], r[1]), r[2], r[3],
                        linewidth=2, edgecolor=color,
                        facecolor=color, alpha=0.25,
                    )
                    axMap.add_patch(p)

        axMap.set_xlim(0, self.worldWidth)
        axMap.set_ylim(0, self.worldHeight)
        axMap.invert_yaxis()
        axMap.set_xlabel("X (tiles)", fontsize=13, fontweight="bold")
        axMap.set_ylabel("Y (tiles)", fontsize=13, fontweight="bold")
        axMap.set_title(
            "Terraria Large World Structure Placement\n"
            "Game-Accurate Quotas with StructureMap Exclusion",
            fontsize=16, fontweight="bold", pad=15,
        )
        axMap.legend(
            loc="upper left", fontsize=10, frameon=True,
            fancybox=True, shadow=True, ncol=2,
        )
        axMap.grid(True, alpha=0.25, ls="--")

        # -- Right panel: statistics table --
        axStats.axis("off")
        axStats.set_title(
            "Placement vs Game Quotas", fontsize=14, fontweight="bold", pad=15,
        )

        tableData = [
            ["Structure", "Placed", "Quota", "Match"],
            [
                "Floating Islands",
                str(len(self.structures.get("floatingIslands", []))),
                str(self.quotas.floatingIslands),
                "",
            ],
            [
                "Underground Cabins",
                str(len(self.structures.get("undergroundCabins", []))),
                f"{self.quotas.undergroundCabinsMin}-{self.quotas.undergroundCabinsMax}",
                "",
            ],
            [
                "Life Crystals",
                str(len(self.structures.get("lifeCrystals", []))),
                f"<= {self.quotas.lifeCrystalsMax}",
                "",
            ],
            [
                "Surface Chests",
                str(len(self.structures.get("surfaceChests", []))),
                str(self.quotas.surfaceChests),
                "",
            ],
            [
                "Dungeon",
                str(len(self.structures.get("dungeon", []))),
                "1",
                "",
            ],
            [
                "Jungle Temple",
                str(len(self.structures.get("jungleTemple", []))),
                "1",
                "",
            ],
        ]

        # Compute match column
        for row in tableData[1:]:
            placed = int(row[1])
            quotaStr = row[2]
            if quotaStr.startswith("<="):
                ok = placed <= int(quotaStr.split()[-1])
            elif "-" in quotaStr:
                lo, hi = quotaStr.split("-")
                ok = int(lo) <= placed <= int(hi)
            else:
                ok = placed == int(quotaStr)
            row[3] = "YES" if ok else "NO"

        cellColors = []
        for i, row in enumerate(tableData):
            if i == 0:
                cellColors.append(["#4472C4"] * 4)
            else:
                matchColor = "#C6EFCE" if row[3] == "YES" else "#FFC7CE"
                cellColors.append(["#F2F2F2", "#F2F2F2", "#F2F2F2", matchColor])

        table = axStats.table(
            cellText=tableData,
            cellColours=cellColors,
            cellLoc="center",
            loc="upper center",
            bbox=[0.0, 0.30, 1.0, 0.60],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        for (r, c), cell in table.get_celld().items():
            if r == 0:
                cell.set_text_props(color="white", fontweight="bold")
            cell.set_edgecolor("#CCCCCC")

        # Layer depth summary below table
        depthText = (
            f"Layer Depths (large world):\n"
            f"  worldSurface = {int(self.layers.worldSurface)}\n"
            f"  rockLayer    = {int(self.layers.rockLayer)}\n"
            f"  hellLayer    = {self.layers.hellLayer}\n"
            f"  borderBuffer = {self.borderBuffer}\n\n"
            f"StructureMap zones: {len(self.structureMap.protectedZones)}"
        )
        axStats.text(
            0.5, 0.18, depthText, transform=axStats.transAxes,
            fontsize=11, va="top", ha="center", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="gray", alpha=0.9),
        )

        plt.tight_layout()
        if savePath:
            os.makedirs(os.path.dirname(savePath), exist_ok=True)
            plt.savefig(savePath, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close()
            print(f"Saved: {savePath}")
        else:
            plt.show()


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main() -> None:
    """Generate structure placement and save visualization."""
    print("Terraria Large World Structure Generation")
    print("=" * 42)

    gen = TerrariaStructureGenerator(worldWidth=8400, worldHeight=2400, seed=12345)
    gen.generateAllStructures()

    # Print summary
    for key, rects in gen.structures.items():
        print(f"  {key}: {len(rects)}")
    print(f"  StructureMap zones: {len(gen.structureMap.protectedZones)}")

    # Save plot
    plotDir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Plots")
    savePath = os.path.join(plotDir, "terraria_structure_placement_large.png")
    gen.visualize(savePath=savePath)


if __name__ == "__main__":
    main()
