"""
Terraria Ore Distribution -- TileRunner-based Vein Generation
=============================================================

Implements ore vein placement using the game's actual TileRunner algorithm
with area-proportional formula: int(area * 6E-05) invocations per ore type.

Each world picks ONE ore from each alternating pair:
Copper/Tin, Iron/Lead, Silver/Tungsten, Gold/Platinum.

Depth bounds derived from LayerDepths constants (worldSurface, rockLayer, hellLayer).
Hardmode ores unlocked by breaking Demon/Crimson Altars in a 3-tier cycle.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import seaborn as sns
from typing import Dict, List, Tuple

from Engine.algorithms import tileRunner, AIR, STONE, DIRT
from Engine.algorithms import (
    COPPER, TIN, IRON, LEAD, SILVER, TUNGSTEN, GOLD, PLATINUM,
    COBALT, PALLADIUM, MYTHRIL, ORICHALCUM, ADAMANTITE, TITANIUM,
    CHLOROPHYTE, HELLSTONE,
)
from Engine.constants import LARGE, LayerDepths, OreConfig

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
sns.set_style("darkgrid")
plt.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "text.color": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
})

# ---------------------------------------------------------------------------
# Ore metadata: tileID -> (displayName, hexColor, strengthRange, stepsRange)
# ---------------------------------------------------------------------------
ORE_META: Dict[int, Tuple[str, str, Tuple[int, int], Tuple[int, int]]] = {
    COPPER:      ("Copper",      "#B87333", (2, 5), (10, 20)),
    TIN:         ("Tin",         "#A8A8A8", (2, 5), (10, 20)),
    IRON:        ("Iron",        "#C19A6B", (2, 5), (10, 20)),
    LEAD:        ("Lead",        "#6B7B8D", (2, 5), (10, 20)),
    SILVER:      ("Silver",      "#C0C0C0", (2, 4), (8, 15)),
    TUNGSTEN:    ("Tungsten",    "#32CD32", (2, 4), (8, 15)),
    GOLD:        ("Gold",        "#FFD700", (2, 4), (8, 15)),
    PLATINUM:    ("Platinum",    "#E5E4E2", (2, 4), (8, 15)),
    COBALT:      ("Cobalt",      "#0047AB", (2, 5), (10, 25)),
    PALLADIUM:   ("Palladium",   "#FF6347", (2, 5), (10, 25)),
    MYTHRIL:     ("Mythril",     "#00FF7F", (2, 4), (8, 20)),
    ORICHALCUM:  ("Orichalcum",  "#FF1493", (2, 4), (8, 20)),
    ADAMANTITE:  ("Adamantite",  "#FF2400", (2, 4), (8, 18)),
    TITANIUM:    ("Titanium",    "#8B008B", (2, 4), (8, 18)),
    CHLOROPHYTE: ("Chlorophyte", "#7CFC00", (2, 5), (12, 25)),
    HELLSTONE:   ("Hellstone",   "#FF4500", (3, 6), (10, 20)),
}

# Background tile colors for grid rendering
BACKGROUND_COLORS: Dict[int, str] = {
    AIR:   "#87CEEB",
    DIRT:  "#8B4513",
    STONE: "#505050",
}


# ===================================================================
# Core class
# ===================================================================

class TerrariaOreDistribution:
    """Ore distribution using TileRunner with area-proportional formula."""

    def __init__(self, worldWidth: int = 8400, worldHeight: int = 2400,
                 seed: int = 42) -> None:
        self.worldWidth = worldWidth
        self.worldHeight = worldHeight
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.layers = LayerDepths.forLarge()
        self.area = worldWidth * worldHeight
        self.loopCount = OreConfig.loopCount(self.area)

        # Pick one ore from each alternating pair
        self.preHardmodeOres: List[int] = self._pickOrePairs()
        self.hardmodeOres: List[int] = self._pickHardmodeTiers()

        # Visualization grid (world-width section)
        self.sectionWidth: int = min(worldWidth, 1200)
        self.grid: np.ndarray | None = None

    # ---------------------------------------------------------------
    # Pair selection
    # ---------------------------------------------------------------

    def _pickOrePairs(self) -> List[int]:
        """Pick one ore per pre-hardmode alternating pair based on seed."""
        nameToID = {
            "Copper": COPPER, "Tin": TIN,
            "Iron": IRON, "Lead": LEAD,
            "Silver": SILVER, "Tungsten": TUNGSTEN,
            "Gold": GOLD, "Platinum": PLATINUM,
        }
        chosen = []
        for pair in OreConfig.PRE_HARDMODE_PAIRS:
            pick = pair[self.rng.integers(0, 2)]
            chosen.append(nameToID[pick])
        return chosen

    def _pickHardmodeTiers(self) -> List[int]:
        """Pick one ore per hardmode tier based on seed."""
        nameToID = {
            "Cobalt": COBALT, "Palladium": PALLADIUM,
            "Mythril": MYTHRIL, "Orichalcum": ORICHALCUM,
            "Adamantite": ADAMANTITE, "Titanium": TITANIUM,
        }
        chosen = []
        for pair in OreConfig.HARDMODE_TIERS:
            pick = pair[self.rng.integers(0, 2)]
            chosen.append(nameToID[pick])
        return chosen

    # ---------------------------------------------------------------
    # Depth bounds from game layer constants
    # ---------------------------------------------------------------

    def _depthBounds(self, oreID: int) -> Tuple[int, int]:
        """Return (yMin, yMax) placement bounds from game layer constants.

        worldSurface=340, rockLayer=880, hellLayer=2200 for Large world.
        """
        ws = int(self.layers.worldSurface)   # 340
        rl = int(self.layers.rockLayer)      # 880
        hl = self.layers.hellLayer           # 2200
        maxY = self.layers.maxTilesY         # 2400

        # Tier 1 (Copper/Tin): surface to rockLayer
        if oreID in (COPPER, TIN):
            return (0, rl)
        # Tier 2 (Iron/Lead): worldSurface to hellLayer
        if oreID in (IRON, LEAD):
            return (ws, hl)
        # Tier 3 (Silver/Tungsten): mid-cavern to hellLayer
        if oreID in (SILVER, TUNGSTEN):
            return (rl // 2, hl)
        # Tier 4 (Gold/Platinum): rockLayer to hellLayer
        if oreID in (GOLD, PLATINUM):
            return (rl, hl)
        # Hellstone: underworld only
        if oreID == HELLSTONE:
            return (hl, maxY)
        # Hardmode tier 1 (Cobalt/Palladium): rockLayer to hellLayer
        if oreID in (COBALT, PALLADIUM):
            return (rl, hl)
        # Hardmode tier 2 (Mythril/Orichalcum): deeper cavern
        if oreID in (MYTHRIL, ORICHALCUM):
            return (rl + (hl - rl) // 5, hl)
        # Hardmode tier 3 (Adamantite/Titanium): deepest cavern
        if oreID in (ADAMANTITE, TITANIUM):
            return (rl + 2 * (hl - rl) // 5, hl)
        # Chlorophyte: rockLayer to hellLayer (jungle underground)
        if oreID == CHLOROPHYTE:
            return (rl, hl)
        return (0, maxY)

    # ---------------------------------------------------------------
    # Grid initialization
    # ---------------------------------------------------------------

    def _initGrid(self) -> np.ndarray:
        """Create stone-filled grid section with surface air and dirt layers."""
        grid = np.full((self.worldHeight, self.sectionWidth), STONE, dtype=np.int32)
        ws = int(self.layers.worldSurface)
        rl = int(self.layers.rockLayer)
        grid[:ws, :] = AIR
        grid[ws:rl, :] = DIRT
        return grid

    # ---------------------------------------------------------------
    # Vein placement
    # ---------------------------------------------------------------

    def placeOreVein(self, grid: np.ndarray, x: int, y: int,
                     oreType: int, strength: float, steps: int) -> np.ndarray:
        """Place a single ore vein using TileRunner diamond-brush random walk."""
        return tileRunner(
            grid, x, y,
            strength=strength,
            steps=steps,
            tileType=oreType,
            overRide=True,
        )

    def generatePreHardmodeOres(self) -> np.ndarray:
        """Generate pre-hardmode ores using TileRunner with area-proportional count.

        Formula: int(sectionArea * 6E-05) TileRunner invocations per ore type.
        """
        self.grid = self._initGrid()
        sectionArea = self.sectionWidth * self.worldHeight
        scaledCount = OreConfig.loopCount(sectionArea)

        print(f"World area: {self.area:,}  |  Section area: {sectionArea:,}")
        print(f"Full-world loop count: {self.loopCount}  |  Section loop count: {scaledCount}")
        print(f"Selected ores: {[ORE_META[o][0] for o in self.preHardmodeOres]}")

        for oreID in self.preHardmodeOres:
            meta = ORE_META[oreID]
            strengthRange = meta[2]
            stepsRange = meta[3]
            yMin, yMax = self._depthBounds(oreID)

            for _ in range(scaledCount):
                x = int(self.rng.integers(10, self.sectionWidth - 10))
                y = int(self.rng.integers(max(yMin, 1), max(yMax, yMin + 2)))
                strength = float(self.rng.uniform(*strengthRange))
                steps = int(self.rng.integers(*stepsRange))
                self.placeOreVein(self.grid, x, y, oreID, strength, steps)

            print(f"  {meta[0]:>10s}: {scaledCount} veins  (depth {yMin}-{yMax})")

        # Hellstone in the underworld
        meta = ORE_META[HELLSTONE]
        yMin, yMax = self._depthBounds(HELLSTONE)
        hellCount = scaledCount // 2
        for _ in range(hellCount):
            x = int(self.rng.integers(10, self.sectionWidth - 10))
            y = int(self.rng.integers(yMin, yMax))
            strength = float(self.rng.uniform(*meta[2]))
            steps = int(self.rng.integers(*meta[3]))
            self.placeOreVein(self.grid, x, y, HELLSTONE, strength, steps)

        print(f"  {'Hellstone':>10s}: {hellCount} veins  (depth {yMin}-{yMax})")
        return self.grid

    def generateHardmodeOres(self, altarsSmashed: int = 3) -> np.ndarray:
        """Generate hardmode ores from altar smashing.

        Altars cycle through tiers: 1st/4th/7th -> tier 1,
        2nd/5th/8th -> tier 2, 3rd/6th/9th -> tier 3.
        """
        if self.grid is None:
            self.generatePreHardmodeOres()

        sectionArea = self.sectionWidth * self.worldHeight
        baseCount = OreConfig.loopCount(sectionArea)

        print(f"\nSmashing {altarsSmashed} altars...")
        print(f"Hardmode ores: {[ORE_META[o][0] for o in self.hardmodeOres]}")

        for tierIdx, oreID in enumerate(self.hardmodeOres):
            veinsForTier = sum(1 for a in range(altarsSmashed) if a % 3 == tierIdx)
            numVeins = max(1, int(baseCount * veinsForTier * 0.3))
            meta = ORE_META[oreID]
            yMin, yMax = self._depthBounds(oreID)

            for _ in range(numVeins):
                x = int(self.rng.integers(10, self.sectionWidth - 10))
                y = int(self.rng.integers(max(yMin, 1), max(yMax, yMin + 2)))
                strength = float(self.rng.uniform(*meta[2]))
                steps = int(self.rng.integers(*meta[3]))
                self.placeOreVein(self.grid, x, y, oreID, strength, steps)

            print(f"  {meta[0]:>12s}: {numVeins} veins  ({veinsForTier} altar hits)")

        return self.grid


# ===================================================================
# Visualization helpers
# ===================================================================

def _gridToRGB(grid: np.ndarray, oreIDs: List[int]) -> np.ndarray:
    """Convert tile grid to an RGB image array for matplotlib."""
    h, w = grid.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    # Background tiles
    for tileID, hexColor in BACKGROUND_COLORS.items():
        mask = grid == tileID
        r, g, b = to_rgb(hexColor)
        rgb[mask] = [r, g, b]

    # Ore tiles (drawn on top of background)
    for oreID in oreIDs:
        if oreID in ORE_META:
            mask = grid == oreID
            r, g, b = to_rgb(ORE_META[oreID][1])
            rgb[mask] = [r, g, b]

    return rgb


def _countOreByDepth(grid: np.ndarray, oreID: int,
                     binSize: int = 40) -> Tuple[np.ndarray, np.ndarray]:
    """Count ore tiles in horizontal depth bins."""
    h, _ = grid.shape
    binEdges = np.arange(0, h + binSize, binSize)
    counts = np.zeros(len(binEdges) - 1)

    for i in range(len(binEdges) - 1):
        yStart, yEnd = binEdges[i], min(binEdges[i + 1], h)
        counts[i] = np.sum(grid[yStart:yEnd, :] == oreID)

    binCenters = (binEdges[:-1] + np.minimum(binEdges[1:], h)) / 2.0
    return binCenters, counts


# ===================================================================
# Visualization functions
# ===================================================================

def visualizeCrossSection(dist: TerrariaOreDistribution,
                          savePath: str | None = None) -> None:
    """Render ore cross-section from TileRunner-generated grid."""
    if dist.grid is None:
        dist.generatePreHardmodeOres()

    allOres = dist.preHardmodeOres + [HELLSTONE] + dist.hardmodeOres
    rgb = _gridToRGB(dist.grid, allOres)

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.imshow(rgb, aspect="auto", interpolation="nearest")

    # Layer boundary lines
    for depth, label, color in [
        (dist.layers.worldSurface, "Surface (340)", "#FFD700"),
        (dist.layers.rockLayer, "Rock Layer (880)", "#FF8C00"),
        (dist.layers.hellLayer, "Hell Layer (2200)", "#FF0000"),
    ]:
        ax.axhline(y=depth, color=color, linestyle="--", linewidth=1.5, alpha=0.8)
        ax.text(5, depth - 10, label, color=color, fontsize=9, fontweight="bold")

    # Ore count legend
    legendLines = []
    for oreID in allOres:
        if oreID in ORE_META:
            count = int(np.sum(dist.grid == oreID))
            if count > 0:
                legendLines.append(f"{ORE_META[oreID][0]:>10s}: {count:>6,} tiles")

    ax.text(
        0.98, 0.02, "\n".join(legendLines),
        transform=ax.transAxes, fontsize=8,
        verticalalignment="bottom", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
        color="white", family="monospace",
    )

    ax.set_xlabel("X (tiles)", fontweight="bold")
    ax.set_ylabel("Depth (tiles)", fontweight="bold")
    ax.set_title(
        f"Terraria Ore Cross-Section (TileRunner, {dist.sectionWidth}x{dist.worldHeight})",
        fontsize=14, fontweight="bold",
    )

    plt.tight_layout()
    if savePath:
        plt.savefig(savePath, dpi=200, bbox_inches="tight")
        print(f"Saved: {savePath}")
    plt.show()


def visualizeDepthDensity(dist: TerrariaOreDistribution,
                          savePath: str | None = None) -> None:
    """Line chart showing ore tile density vs depth for each ore type."""
    if dist.grid is None:
        dist.generatePreHardmodeOres()

    allOres = dist.preHardmodeOres + [HELLSTONE]

    fig, ax = plt.subplots(figsize=(12, 8))

    for oreID in allOres:
        if oreID not in ORE_META:
            continue
        depths, counts = _countOreByDepth(dist.grid, oreID, binSize=40)
        if np.sum(counts) == 0:
            continue
        name, hexColor = ORE_META[oreID][0], ORE_META[oreID][1]
        ax.plot(counts, depths, label=name, color=hexColor, linewidth=2, alpha=0.85)

    ax.invert_yaxis()

    # Layer markers
    for depth, label in [
        (dist.layers.worldSurface, "Surface"),
        (dist.layers.rockLayer, "Rock Layer"),
        (dist.layers.hellLayer, "Hell Layer"),
    ]:
        ax.axhline(y=depth, color="white", linestyle=":", linewidth=1, alpha=0.5)
        ax.text(
            ax.get_xlim()[1] * 0.95, depth - 12, label,
            color="white", fontsize=8, ha="right", alpha=0.7,
        )

    ax.set_xlabel("Ore Tiles per 40-row Band", fontweight="bold")
    ax.set_ylabel("Depth (tiles)", fontweight="bold")
    ax.set_title("Ore Density by Depth (Pre-Hardmode)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    if savePath:
        plt.savefig(savePath, dpi=200, bbox_inches="tight")
        print(f"Saved: {savePath}")
    plt.show()


def visualizeComparison(dist: TerrariaOreDistribution,
                        altarsSmashed: int = 6,
                        savePath: str | None = None) -> None:
    """Side-by-side comparison: pre-hardmode only vs post-altar-smash."""
    if dist.grid is None:
        dist.generatePreHardmodeOres()

    # Snapshot pre-hardmode grid before hardmode ores are added
    preGrid = dist.grid.copy()
    dist.generateHardmodeOres(altarsSmashed)
    postGrid = dist.grid.copy()

    preOres = dist.preHardmodeOres + [HELLSTONE]
    postOres = preOres + dist.hardmodeOres

    preRGB = _gridToRGB(preGrid, preOres)
    postRGB = _gridToRGB(postGrid, postOres)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))

    ax1.imshow(preRGB, aspect="auto", interpolation="nearest")
    ax1.set_title("Pre-Hardmode Ores", fontsize=13, fontweight="bold")
    ax1.set_xlabel("X (tiles)")
    ax1.set_ylabel("Depth (tiles)")

    ax2.imshow(postRGB, aspect="auto", interpolation="nearest")
    ax2.set_title(f"After {altarsSmashed} Altars Smashed", fontsize=13, fontweight="bold")
    ax2.set_xlabel("X (tiles)")

    for ax in (ax1, ax2):
        for depth, color in [
            (dist.layers.worldSurface, "#FFD700"),
            (dist.layers.rockLayer, "#FF8C00"),
            (dist.layers.hellLayer, "#FF0000"),
        ]:
            ax.axhline(y=depth, color=color, linestyle="--", linewidth=1, alpha=0.6)

    # Hardmode ore legend on right panel
    hmLegend = []
    for oreID in dist.hardmodeOres:
        count = int(np.sum(postGrid == oreID))
        if count > 0:
            hmLegend.append(f"{ORE_META[oreID][0]:>12s}: {count:>5,} tiles")
    if hmLegend:
        ax2.text(
            0.98, 0.02, "\n".join(hmLegend),
            transform=ax2.transAxes, fontsize=8,
            verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
            color="white", family="monospace",
        )

    plt.suptitle(
        "Pre-Hardmode vs Hardmode Ore Distribution (TileRunner)",
        fontsize=15, fontweight="bold", color="white",
    )

    plt.tight_layout()
    if savePath:
        plt.savefig(savePath, dpi=200, bbox_inches="tight")
        print(f"Saved: {savePath}")
    plt.show()


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    """Run all ore distribution visualizations."""
    plotsDir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Plots"
    )
    os.makedirs(plotsDir, exist_ok=True)

    print("=" * 60)
    print("Terraria Ore Distribution (TileRunner + area formula)")
    print(f"Formula: int(area * 6E-05) = {OreConfig.loopCount(LARGE.area)} veins/ore for Large world")
    print("=" * 60)

    dist = TerrariaOreDistribution(
        worldWidth=LARGE.width, worldHeight=LARGE.height, seed=42,
    )

    dist.generatePreHardmodeOres()

    visualizeCrossSection(
        dist, savePath=os.path.join(plotsDir, "ore_cross_section.png"),
    )
    visualizeDepthDensity(
        dist, savePath=os.path.join(plotsDir, "ore_depth_density.png"),
    )
    visualizeComparison(
        dist, altarsSmashed=6,
        savePath=os.path.join(plotsDir, "ore_prehardmode_vs_hardmode.png"),
    )

    # Summary statistics
    print("\n" + "=" * 60)
    print("Ore Tile Counts:")
    allOres = dist.preHardmodeOres + [HELLSTONE] + dist.hardmodeOres
    for oreID in allOres:
        if oreID in ORE_META:
            count = int(np.sum(dist.grid == oreID))
            print(f"  {ORE_META[oreID][0]:>12s}: {count:>6,} tiles")


if __name__ == "__main__":
    main()
