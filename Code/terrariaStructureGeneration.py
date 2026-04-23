"""Terraria Structure Generation -- Macro Density + Sprite Detail.

Game-accurate structure placement using StructureMap exclusion zones,
LayerDepths, and StructureQuotas. Two output plots:

1. ``terraria_structure_density.png`` -- LARGE world (8400x2400)
   density heatmap showing where each structure type clusters.
2. ``terraria_structure_detail.png`` -- DETAIL_PLOT (600x400) sprite
   render of representative structures: dungeon, cabin, floating
   island, pyramid.
"""

from __future__ import annotations

import os
import random

import numpy as np
import matplotlib.pyplot as plt

from Engine.algorithms import (
    AIR, DIRT, STONE, EBONSTONE, SAND,
)
from Engine.constants import DETAIL_PLOT, LARGE, LayerDepths, StructureQuotas
from Engine.spriteRenderer import (
    Rect as SpriteRect,
    drawCabin, drawDungeon, drawFloatingIsland, drawPyramid, drawTileGrid,
)
from Engine.structureMap import Rectangle, StructureMap
from Engine.theme import COLORS, PALETTE, applyTokyoNight

applyTokyoNight()


# ===================================================================
# Generator
# ===================================================================
class TerrariaStructureGenerator:
    """Generates large-world structures via attempt-loop placement.

    Uses instance-scoped ``random.Random`` and ``np.random.Generator``
    to avoid mutating global RNG state.
    """

    def __init__(
        self,
        worldWidth: int = LARGE.width,
        worldHeight: int = LARGE.height,
        seed: int = 12345,
    ) -> None:
        self.worldWidth = worldWidth
        self.worldHeight = worldHeight
        self.seed = seed

        self.rng = random.Random(seed)
        self.npRng = np.random.default_rng(seed)

        self.layers: LayerDepths = LayerDepths.forLarge()
        self.quotas: StructureQuotas = StructureQuotas.forLarge()
        self.borderBuffer: int = LARGE.borderBuffer

        self.structureMap = StructureMap()
        self.structures: dict[str, list[tuple[int, int, int, int]]] = {}
        self.dungeonOnRight: bool = self.rng.random() < 0.5

    # ------------------------------------------------------------------
    def _attemptPlace(
        self, xMin: int, xMax: int, yMin: int, yMax: int,
        structW: int, structH: int,
        padding: int = 5, maxAttempts: int = 1000,
    ) -> Rectangle | None:
        for _ in range(maxAttempts):
            x = self.rng.randint(xMin, xMax - structW)
            y = self.rng.randint(yMin, yMax - structH)
            rect = Rectangle(x, y, structW, structH)
            if self.structureMap.canPlace(rect, padding):
                self.structureMap.addProtectedStructure(rect, padding)
                return rect
        return None

    # ------------------------------------------------------------------
    def placeFloatingIslands(self) -> list[tuple[int, int, int, int]]:
        count = self.quotas.floatingIslands
        islandW, islandH = 80, 40
        yMin = 80
        yMax = int(self.layers.worldSurface) - islandH - 20
        xMin, xMax = self.borderBuffer, self.worldWidth - self.borderBuffer

        placed: list[tuple[int, int, int, int]] = []
        for _ in range(count):
            r = self._attemptPlace(xMin, xMax, yMin, yMax, islandW, islandH,
                                   padding=60, maxAttempts=2000)
            if r is not None:
                placed.append((r.x, r.y, r.width, r.height))
        self.structures["floatingIslands"] = placed
        return placed

    def placeUndergroundCabins(self) -> list[tuple[int, int, int, int]]:
        count = self.rng.randint(self.quotas.undergroundCabinsMin,
                                 self.quotas.undergroundCabinsMax)
        cabinW, cabinH = 24, 18
        yMin = int(self.layers.rockLayer)
        yMax = self.layers.hellLayer - cabinH
        xMin, xMax = self.borderBuffer, self.worldWidth - self.borderBuffer

        placed: list[tuple[int, int, int, int]] = []
        for _ in range(count):
            r = self._attemptPlace(xMin, xMax, yMin, yMax, cabinW, cabinH,
                                   padding=8, maxAttempts=1000)
            if r is not None:
                placed.append((r.x, r.y, r.width, r.height))
        self.structures["undergroundCabins"] = placed
        return placed

    def placeLifeCrystals(self) -> list[tuple[int, int, int, int]]:
        maxCount = self.quotas.lifeCrystalsMax
        crystalW, crystalH = 2, 2
        yMin = int(self.layers.worldSurface) + 20
        yMax = self.layers.hellLayer - crystalH
        xMin, xMax = self.borderBuffer, self.worldWidth - self.borderBuffer

        placed: list[tuple[int, int, int, int]] = []
        for _ in range(maxCount):
            r = self._attemptPlace(xMin, xMax, yMin, yMax,
                                   crystalW, crystalH,
                                   padding=2, maxAttempts=200)
            if r is not None:
                placed.append((r.x, r.y, r.width, r.height))
        self.structures["lifeCrystals"] = placed
        return placed

    def placeSurfaceChests(self) -> list[tuple[int, int, int, int]]:
        count = self.quotas.surfaceChests
        chestW, chestH = 2, 2
        yMin = int(self.layers.worldSurface) - 30
        yMax = int(self.layers.worldSurface) + 60
        xMin, xMax = self.borderBuffer, self.worldWidth - self.borderBuffer

        placed: list[tuple[int, int, int, int]] = []
        for _ in range(count):
            r = self._attemptPlace(xMin, xMax, yMin, yMax,
                                   chestW, chestH,
                                   padding=20, maxAttempts=1000)
            if r is not None:
                placed.append((r.x, r.y, r.width, r.height))
        self.structures["surfaceChests"] = placed
        return placed

    def placeDungeon(self) -> list[tuple[int, int, int, int]]:
        dungeonW, dungeonH = 200, 300
        yMin = int(self.layers.worldSurface) - 20
        yMax = int(self.layers.rockLayer)
        if self.dungeonOnRight:
            xMin = int(self.worldWidth * 0.75)
            xMax = self.worldWidth - self.borderBuffer
        else:
            xMin = self.borderBuffer
            xMax = int(self.worldWidth * 0.25)
        placed: list[tuple[int, int, int, int]] = []
        r = self._attemptPlace(xMin, xMax, yMin, yMax, dungeonW, dungeonH,
                               padding=30, maxAttempts=2000)
        if r is not None:
            placed.append((r.x, r.y, r.width, r.height))
        self.structures["dungeon"] = placed
        return placed

    def placeJungleTemple(self) -> list[tuple[int, int, int, int]]:
        templeW, templeH = 150, 120
        yMin = int(self.layers.rockLayer) + 50
        yMax = self.layers.hellLayer - templeH - 50
        if self.dungeonOnRight:
            xMin, xMax = self.borderBuffer, int(self.worldWidth * 0.35)
        else:
            xMin = int(self.worldWidth * 0.65)
            xMax = self.worldWidth - self.borderBuffer
        placed: list[tuple[int, int, int, int]] = []
        r = self._attemptPlace(xMin, xMax, yMin, yMax, templeW, templeH,
                               padding=40, maxAttempts=2000)
        if r is not None:
            placed.append((r.x, r.y, r.width, r.height))
        self.structures["jungleTemple"] = placed
        return placed

    def generateAllStructures(self) -> dict[str, list[tuple[int, int, int, int]]]:
        self.structureMap.clear()
        self.structures.clear()
        self.placeDungeon()
        self.placeJungleTemple()
        self.placeFloatingIslands()
        self.placeUndergroundCabins()
        self.placeSurfaceChests()
        self.placeLifeCrystals()
        return self.structures


# ===================================================================
# Visualization 1: macro density (LARGE world)
# ===================================================================
STRUCTURE_COLOR = {
    "floatingIslands":   PALETTE["cyan"],
    "undergroundCabins": PALETTE["yellow"],
    "lifeCrystals":      PALETTE["red"],
    "surfaceChests":     PALETTE["orange"],
    "dungeon":           PALETTE["purple"],
    "jungleTemple":      PALETTE["green"],
}

STRUCTURE_MARKER = {
    "floatingIslands":   ("o", 60),
    "undergroundCabins": ("s", 14),
    "lifeCrystals":      ("D", 6),
    "surfaceChests":     ("^", 28),
    "dungeon":           ("D", 200),
    "jungleTemple":      ("^", 160),
}


def createMacroDensityPlot(
    gen: TerrariaStructureGenerator, savePath: str | None = None,
) -> None:
    """LARGE-world overview: scatter density of every placed structure."""
    if not gen.structures:
        gen.generateAllStructures()

    fig, ax = plt.subplots(figsize=(18, 6))

    ws, rl, hl = gen.layers.worldSurface, gen.layers.rockLayer, gen.layers.hellLayer
    ax.axhline(y=ws, color=PALETTE["yellow"], lw=1.5,
               label=f"worldSurface ({int(ws)})")
    ax.axhline(y=rl, color=PALETTE["orange"], ls="--", lw=1.5,
               label=f"rockLayer ({int(rl)})")
    ax.axhline(y=hl, color=PALETTE["red"], ls="-.", lw=1.5,
               label=f"hellLayer ({hl})")

    for key, rects in gen.structures.items():
        if not rects:
            continue
        color = STRUCTURE_COLOR.get(key, PALETTE["fg"])
        marker, size = STRUCTURE_MARKER.get(key, (".", 20))
        xs = [r[0] + r[2] / 2 for r in rects]
        ys = [r[1] + r[3] / 2 for r in rects]
        ax.scatter(xs, ys, c=color, s=size, marker=marker,
                   alpha=0.75, edgecolors=PALETTE["bg"], linewidth=0.4,
                   label=f"{key} ({len(rects)})", zorder=5)

    ax.set_xlim(0, gen.worldWidth)
    ax.set_ylim(0, gen.worldHeight)
    ax.invert_yaxis()
    ax.set_xlabel("X (tiles)", fontweight="bold")
    ax.set_ylabel("Y (tiles)", fontweight="bold")
    ax.set_title(
        "Large-World Structure Density (game-accurate quotas)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=8, ncol=2, frameon=True)
    ax.grid(True, alpha=0.20, ls="--")

    plt.tight_layout()
    if savePath:
        os.makedirs(os.path.dirname(savePath), exist_ok=True)
        plt.savefig(savePath, dpi=200, bbox_inches="tight",
                    facecolor=COLORS["bg"])
        print(f"Saved: {savePath}")
    plt.close(fig)


# ===================================================================
# Visualization 2: detail panel (DETAIL_PLOT 600x400 sprites)
# ===================================================================
def _buildDungeonGrid(w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Stone background + brick room cutouts for dungeon detail."""
    grid = np.full((h, w), STONE, dtype=np.int32)
    walls = np.full((h, w), 0, dtype=np.int32)
    walls[2:h - 2, 2:w - 2] = EBONSTONE
    # Carve a few rooms.
    for (x0, y0, rw, rh) in [(8, 12, 28, 18), (40, 16, 24, 14),
                             (10, 38, 30, 20), (44, 42, 28, 16)]:
        grid[y0:y0 + rh, x0:x0 + rw] = AIR
    return grid, walls


def _dungeonRooms() -> list[SpriteRect]:
    return [
        SpriteRect(8, 12, 28, 18),
        SpriteRect(40, 16, 24, 14),
        SpriteRect(10, 38, 30, 20),
        SpriteRect(44, 42, 28, 16),
    ]


def _buildIslandGrid(w: int, h: int) -> np.ndarray:
    grid = np.full((h, w), AIR, dtype=np.int32)
    # Lens-shaped dirt island.
    cx, cy = w // 2, h // 2
    for j in range(h):
        for i in range(w):
            r = ((i - cx) / (w * 0.45)) ** 2 + ((j - cy) / (h * 0.35)) ** 2
            if r <= 1.0:
                grid[j, i] = DIRT
    return grid


def _buildPyramidGrid(w: int, h: int) -> np.ndarray:
    grid = np.full((h, w), AIR, dtype=np.int32)
    # Triangular sandstone silhouette.
    for j in range(h):
        rowWidth = int((j / h) * w)
        x0 = (w - rowWidth) // 2
        grid[j, x0:x0 + rowWidth] = SAND
    return grid


def createStructureDetailPlot(
    savePath: str | None = None, seed: int = 42,
) -> None:
    """4-panel sprite detail: dungeon, cabin, floating island, pyramid."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # --- Dungeon ---
    dgW, dgH = 80, 64
    dgGrid, dgWalls = _buildDungeonGrid(dgW, dgH)
    drawDungeon(axes[0, 0], dgGrid, dgWalls, _dungeonRooms())
    axes[0, 0].set_xlim(0, dgW)
    axes[0, 0].set_ylim(dgH, 0)
    axes[0, 0].set_title("Dungeon (rooms + doors + chests)",
                         fontsize=11, fontweight="bold")

    # --- Underground Cabin ---
    cbW, cbH = 24, 16
    cabinBg = np.full((cbH, cbW), STONE, dtype=np.int32)
    drawTileGrid(axes[0, 1], cabinBg)
    drawCabin(axes[0, 1], 6, 4, w=12, h=8)
    axes[0, 1].set_xlim(0, cbW)
    axes[0, 1].set_ylim(cbH, 0)
    axes[0, 1].set_title("Underground Cabin (door + chest + torch)",
                         fontsize=11, fontweight="bold")

    # --- Floating Island ---
    fiW, fiH = 60, 40
    fiGrid = _buildIslandGrid(fiW, fiH)
    drawFloatingIsland(axes[1, 0], fiGrid, 0, 0, fiW, fiH)
    axes[1, 0].set_xlim(0, fiW)
    axes[1, 0].set_ylim(fiH, 0)
    axes[1, 0].set_title("Floating Island (sky house)",
                         fontsize=11, fontweight="bold")

    # --- Pyramid ---
    pyW, pyH = 60, 50
    pyGrid = _buildPyramidGrid(pyW, pyH)
    drawPyramid(axes[1, 1], pyGrid, 0, 0, pyW, pyH)
    axes[1, 1].set_xlim(0, pyW)
    axes[1, 1].set_ylim(pyH, 0)
    axes[1, 1].set_title("Pyramid (chamber + chest)",
                         fontsize=11, fontweight="bold")

    for ax in axes.flat:
        ax.set_xlabel("X (tiles)", fontsize=8)
        ax.set_ylabel("Y (tiles)", fontsize=8)

    fig.suptitle("Structure Detail (DETAIL_PLOT sprite render)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if savePath:
        os.makedirs(os.path.dirname(savePath), exist_ok=True)
        plt.savefig(savePath, dpi=200, bbox_inches="tight",
                    facecolor=COLORS["bg"])
        print(f"Saved: {savePath}")
    plt.close(fig)


# ===================================================================
# Main
# ===================================================================
def main() -> None:
    print("Terraria Structure Generation")
    print("=" * 42)

    gen = TerrariaStructureGenerator(seed=12345)
    gen.generateAllStructures()
    for key, rects in gen.structures.items():
        print(f"  {key}: {len(rects)}")
    print(f"  StructureMap zones: {len(gen.structureMap.protectedZones)}")

    plotDir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Plots"
    )
    createMacroDensityPlot(
        gen, savePath=os.path.join(plotDir, "terraria_structure_density.png"),
    )
    createStructureDetailPlot(
        savePath=os.path.join(plotDir, "terraria_structure_detail.png"),
    )


if __name__ == "__main__":
    main()
