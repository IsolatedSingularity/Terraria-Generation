"""Terraria Ore Distribution -- TileRunner-based Vein Generation.

Implements ore vein placement using the game's actual TileRunner algorithm
with area-proportional formula: ``int(area * 6E-05)`` invocations per ore type.

Each world picks ONE ore from each alternating pair:
Copper/Tin, Iron/Lead, Silver/Tungsten, Gold/Platinum.

Depth bounds derived from LayerDepths constants (worldSurface, rockLayer,
hellLayer). Hardmode ores unlocked by breaking Demon/Crimson Altars in a
3-tier cycle.

Output: two plots — ``ore_distribution.png`` (3-panel: pre-HM, post-HM,
cross-section detail) and ``ore_depth_density.png`` (line chart).
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from Engine.algorithms import (
    AIR, DIRT, STONE,
    COPPER, TIN, IRON, LEAD, SILVER, TUNGSTEN, GOLD, PLATINUM,
    COBALT, PALLADIUM, MYTHRIL, ORICHALCUM, ADAMANTITE, TITANIUM,
    CHLOROPHYTE, HELLSTONE,
    tileRunner,
)
from Engine.constants import FEATURE_PLOT, LARGE, LayerDepths, OreConfig
from Engine.spriteRenderer import drawTileGrid
from Engine.theme import COLORS, ORE_COLORS, PALETTE, applyTokyoNight

applyTokyoNight()


# ---------------------------------------------------------------------------
# Ore metadata: tileID -> (name, hex, strengthRange, stepsRange)
# ---------------------------------------------------------------------------
ORE_META: dict[int, tuple[str, str, tuple[int, int], tuple[int, int]]] = {
    COPPER:      ("Copper",      ORE_COLORS["copper"],      (2, 5), (10, 20)),
    TIN:         ("Tin",         ORE_COLORS["tin"],         (2, 5), (10, 20)),
    IRON:        ("Iron",        ORE_COLORS["iron"],        (2, 5), (10, 20)),
    LEAD:        ("Lead",        ORE_COLORS["lead"],        (2, 5), (10, 20)),
    SILVER:      ("Silver",      ORE_COLORS["silver"],      (2, 4), (8, 15)),
    TUNGSTEN:    ("Tungsten",    ORE_COLORS["tungsten"],    (2, 4), (8, 15)),
    GOLD:        ("Gold",        ORE_COLORS["gold"],        (2, 4), (8, 15)),
    PLATINUM:    ("Platinum",    ORE_COLORS["platinum"],    (2, 4), (8, 15)),
    COBALT:      ("Cobalt",      ORE_COLORS["cobalt"],      (2, 5), (10, 25)),
    PALLADIUM:   ("Palladium",   ORE_COLORS["palladium"],   (2, 5), (10, 25)),
    MYTHRIL:     ("Mythril",     ORE_COLORS["mythril"],     (2, 4), (8, 20)),
    ORICHALCUM:  ("Orichalcum",  ORE_COLORS["orichalcum"],  (2, 4), (8, 20)),
    ADAMANTITE:  ("Adamantite",  ORE_COLORS["adamantite"],  (2, 4), (8, 18)),
    TITANIUM:    ("Titanium",    ORE_COLORS["titanium"],    (2, 4), (8, 18)),
    CHLOROPHYTE: ("Chlorophyte", ORE_COLORS["chlorophyte"], (2, 5), (12, 25)),
    HELLSTONE:   ("Hellstone",   ORE_COLORS["hellstone"],   (3, 6), (10, 20)),
}


# ===================================================================
# Core class
# ===================================================================
class TerrariaOreDistribution:
    """Ore distribution using TileRunner with area-proportional formula.

    Defaults to the FEATURE_PLOT canvas (500x300) so individual veins
    render at visible pixel scale. Layer depths are scaled proportionally
    from a Large reference world so depth tiers remain game-consistent.
    """

    def __init__(
        self,
        worldWidth: int = FEATURE_PLOT.width,
        worldHeight: int = FEATURE_PLOT.height,
        seed: int = 42,
    ) -> None:
        self.worldWidth = worldWidth
        self.worldHeight = worldHeight
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        ref = LayerDepths.forLarge()
        yScale = worldHeight / ref.maxTilesY
        self.layers = LayerDepths(
            worldSurface=ref.worldSurface * yScale,
            rockLayer=ref.rockLayer * yScale,
            hellLayer=int(ref.hellLayer * yScale),
            maxTilesY=worldHeight,
        )

        self.area = worldWidth * worldHeight
        self.loopCount = OreConfig.loopCount(self.area)

        self.preHardmodeOres: list[int] = self._pickOrePairs()
        self.hardmodeOres: list[int] = self._pickHardmodeTiers()

        self.grid: np.ndarray | None = None

    # ------------------------------------------------------------------
    def _pickOrePairs(self) -> list[int]:
        nameToId = {
            "Copper": COPPER, "Tin": TIN, "Iron": IRON, "Lead": LEAD,
            "Silver": SILVER, "Tungsten": TUNGSTEN,
            "Gold": GOLD, "Platinum": PLATINUM,
        }
        return [nameToId[pair[self.rng.integers(0, 2)]]
                for pair in OreConfig.PRE_HARDMODE_PAIRS]

    def _pickHardmodeTiers(self) -> list[int]:
        nameToId = {
            "Cobalt": COBALT, "Palladium": PALLADIUM,
            "Mythril": MYTHRIL, "Orichalcum": ORICHALCUM,
            "Adamantite": ADAMANTITE, "Titanium": TITANIUM,
        }
        return [nameToId[pair[self.rng.integers(0, 2)]]
                for pair in OreConfig.HARDMODE_TIERS]

    # ------------------------------------------------------------------
    def _depthBounds(self, oreId: int) -> tuple[int, int]:
        ws = int(self.layers.worldSurface)
        rl = int(self.layers.rockLayer)
        hl = self.layers.hellLayer
        maxY = self.layers.maxTilesY

        if oreId in (COPPER, TIN):
            return (0, rl)
        if oreId in (IRON, LEAD):
            return (ws, hl)
        if oreId in (SILVER, TUNGSTEN):
            return (rl // 2, hl)
        if oreId in (GOLD, PLATINUM):
            return (rl, hl)
        if oreId == HELLSTONE:
            return (hl, maxY)
        if oreId in (COBALT, PALLADIUM):
            return (rl, hl)
        if oreId in (MYTHRIL, ORICHALCUM):
            return (rl + (hl - rl) // 5, hl)
        if oreId in (ADAMANTITE, TITANIUM):
            return (rl + 2 * (hl - rl) // 5, hl)
        if oreId == CHLOROPHYTE:
            return (rl, hl)
        return (0, maxY)

    def _initGrid(self) -> np.ndarray:
        grid = np.full((self.worldHeight, self.worldWidth), STONE, dtype=np.int32)
        ws = int(self.layers.worldSurface)
        rl = int(self.layers.rockLayer)
        grid[:ws, :] = AIR
        grid[ws:rl, :] = DIRT
        return grid

    def _placeVein(
        self, grid: np.ndarray, x: int, y: int,
        oreId: int, strength: float, steps: int,
    ) -> None:
        tileRunner(
            grid, x, y,
            strength=strength, steps=steps,
            tileType=oreId, overRide=True,
        )

    def generatePreHardmodeOres(self) -> np.ndarray:
        self.grid = self._initGrid()
        scaledCount = OreConfig.loopCount(self.area)
        margin = max(2, self.worldWidth // 50)

        print(f"World: {self.worldWidth}x{self.worldHeight} "
              f"({self.area:,} tiles). Veins/ore: {scaledCount}")
        print(f"Pre-HM ores: {[ORE_META[o][0] for o in self.preHardmodeOres]}")

        for oreId in self.preHardmodeOres:
            meta = ORE_META[oreId]
            yMin, yMax = self._depthBounds(oreId)
            for _ in range(scaledCount):
                x = int(self.rng.integers(margin, self.worldWidth - margin))
                y = int(self.rng.integers(max(yMin, 1), max(yMax, yMin + 2)))
                self._placeVein(
                    self.grid, x, y, oreId,
                    strength=float(self.rng.uniform(*meta[2])),
                    steps=int(self.rng.integers(*meta[3])),
                )

        meta = ORE_META[HELLSTONE]
        yMin, yMax = self._depthBounds(HELLSTONE)
        hellCount = max(1, scaledCount // 2)
        for _ in range(hellCount):
            x = int(self.rng.integers(margin, self.worldWidth - margin))
            y = int(self.rng.integers(yMin, yMax))
            self._placeVein(
                self.grid, x, y, HELLSTONE,
                strength=float(self.rng.uniform(*meta[2])),
                steps=int(self.rng.integers(*meta[3])),
            )
        return self.grid

    def generateHardmodeOres(self, altarsSmashed: int = 6) -> np.ndarray:
        if self.grid is None:
            self.generatePreHardmodeOres()
        baseCount = OreConfig.loopCount(self.area)
        margin = max(2, self.worldWidth // 50)

        print(f"\nSmashing {altarsSmashed} altars...")
        for tierIdx, oreId in enumerate(self.hardmodeOres):
            veinsForTier = sum(1 for a in range(altarsSmashed) if a % 3 == tierIdx)
            numVeins = max(1, int(baseCount * veinsForTier * 0.3))
            meta = ORE_META[oreId]
            yMin, yMax = self._depthBounds(oreId)
            for _ in range(numVeins):
                x = int(self.rng.integers(margin, self.worldWidth - margin))
                y = int(self.rng.integers(max(yMin, 1), max(yMax, yMin + 2)))
                self._placeVein(
                    self.grid, x, y, oreId,
                    strength=float(self.rng.uniform(*meta[2])),
                    steps=int(self.rng.integers(*meta[3])),
                )
        return self.grid


# ===================================================================
# Plotting helpers
# ===================================================================
def _drawLayerLines(ax: plt.Axes, layers: LayerDepths) -> None:
    for y, label, color in [
        (layers.worldSurface, "Surface", PALETTE["yellow"]),
        (layers.rockLayer, "Rock Layer", PALETTE["orange"]),
        (layers.hellLayer, "Hell Layer", PALETTE["red"]),
    ]:
        ax.axhline(y=y, color=color, linestyle="--", linewidth=1.0, alpha=0.7)
        ax.text(2, y - 3, label, color=color, fontsize=7,
                fontweight="bold", alpha=0.9)


def _renderOrePanel(
    ax: plt.Axes, dist: TerrariaOreDistribution, grid: np.ndarray,
    title: str,
) -> None:
    drawTileGrid(ax, grid)
    _drawLayerLines(ax, dist.layers)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("X (tiles)")
    ax.set_ylabel("Depth (tiles)")
    ax.set_xlim(0, dist.worldWidth)
    ax.set_ylim(dist.worldHeight, 0)


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


# ===================================================================
# Visualizations
# ===================================================================
def visualizeDistribution(
    dist: TerrariaOreDistribution,
    altarsSmashed: int = 6,
    savePath: str | None = None,
) -> None:
    """3-panel ore figure: pre-HM, post-altar, vein-detail crop with luster."""
    if dist.grid is None:
        dist.generatePreHardmodeOres()
    preGrid = dist.grid.copy()
    dist.generateHardmodeOres(altarsSmashed)
    postGrid = dist.grid.copy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    _renderOrePanel(axes[0], dist, preGrid, "Pre-Hardmode")
    _renderOrePanel(axes[1], dist, postGrid, f"After {altarsSmashed} Altars")

    # Detail crop centered on rock layer.
    cy = int(dist.layers.rockLayer)
    cx = dist.worldWidth // 2
    halfW, halfH = 100, 60
    y0, y1 = max(0, cy - halfH), min(dist.worldHeight, cy + halfH)
    x0, x1 = max(0, cx - halfW), min(dist.worldWidth, cx + halfW)
    detail = postGrid[y0:y1, x0:x1]

    drawTileGrid(axes[2], detail, extent=(x0, x1, y1, y0))
    # Luster: white dot offset on each ore tile in the detail crop.
    postOres = dist.preHardmodeOres + [HELLSTONE] + dist.hardmodeOres
    for oreId in postOres:
        if oreId in ORE_META:
            ys, xs = np.where(detail == oreId)
            if ys.size:
                axes[2].scatter(
                    xs + x0 + 0.3, ys + y0 + 0.3,
                    s=8, c="#ffffff", alpha=0.45, marker="o",
                    edgecolors="none", zorder=2,
                )
    axes[2].set_title("Vein Detail (luster)", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("X (tiles)")
    axes[2].set_ylabel("Depth (tiles)")
    axes[2].set_xlim(x0, x1)
    axes[2].set_ylim(y1, y0)

    fig.suptitle(
        "Ore Distribution: TileRunner + 3-Cycle Altar System",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    if savePath:
        plt.savefig(savePath, dpi=200, bbox_inches="tight",
                    facecolor=COLORS["bg"])
        print(f"Saved: {savePath}")
    plt.close(fig)


def visualizeDepthDensity(
    dist: TerrariaOreDistribution,
    savePath: str | None = None,
) -> None:
    if dist.grid is None:
        dist.generatePreHardmodeOres()

    fig, ax = plt.subplots(figsize=(10, 6))
    for oreId in dist.preHardmodeOres + [HELLSTONE]:
        if oreId not in ORE_META:
            continue
        depths, counts = _countOreByDepth(dist.grid, oreId, binSize=8)
        if counts.sum() == 0:
            continue
        name, color = ORE_META[oreId][0], ORE_META[oreId][1]
        ax.plot(counts, depths, label=name, color=color,
                linewidth=2.0, alpha=0.9)
    ax.invert_yaxis()

    for y, label in [
        (dist.layers.worldSurface, "Surface"),
        (dist.layers.rockLayer, "Rock"),
        (dist.layers.hellLayer, "Hell"),
    ]:
        ax.axhline(y=y, color=PALETTE["subtle"],
                   linestyle=":", linewidth=1.0, alpha=0.6)
        ax.text(ax.get_xlim()[1] * 0.95, y - 4, label,
                color=PALETTE["muted"], fontsize=8, ha="right", alpha=0.8)

    ax.set_xlabel("Ore tiles per 8-row band", fontweight="bold")
    ax.set_ylabel("Depth (tiles)", fontweight="bold")
    ax.set_title("Pre-Hardmode Ore Density vs Depth",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    if savePath:
        plt.savefig(savePath, dpi=200, bbox_inches="tight",
                    facecolor=COLORS["bg"])
        print(f"Saved: {savePath}")
    plt.close(fig)


# ===================================================================
# Main
# ===================================================================
def main() -> None:
    plotsDir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Plots"
    )
    os.makedirs(plotsDir, exist_ok=True)

    print("=" * 60)
    print("Terraria Ore Distribution (TileRunner + area formula)")
    print(f"Reference loopCount(LARGE) = {OreConfig.loopCount(LARGE.area)}")
    print("=" * 60)

    dist = TerrariaOreDistribution(
        worldWidth=FEATURE_PLOT.width,
        worldHeight=FEATURE_PLOT.height,
        seed=42,
    )
    dist.generatePreHardmodeOres()
    visualizeDistribution(
        dist, altarsSmashed=6,
        savePath=os.path.join(plotsDir, "ore_distribution.png"),
    )
    visualizeDepthDensity(
        dist,
        savePath=os.path.join(plotsDir, "ore_depth_density.png"),
    )


if __name__ == "__main__":
    main()
