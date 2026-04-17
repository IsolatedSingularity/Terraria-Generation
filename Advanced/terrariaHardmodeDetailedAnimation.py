"""
Terraria Hardmode Transformation Animation
==========================================

Matplotlib FuncAnimation of the full hardmode transition at 1/10 scale
(840x240). V-pattern carved via TileRunner, altar smashing with
proportional ore density (6E-05 formula), and tile-update-cycle
infection spread (surface ~140s, underground ~830s).

Frame sequence:
  0        Pre-hardmode world
  1-3      V-pattern carving (progressive, TileRunner strength ~6)
  4-6      Altar smashing + ore placement (one tier per frame)
  7-21     Infection spread simulation (tile update cycle)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import List, Tuple, Optional

from Engine.algorithms import (
    tileRunner, AIR, STONE, DIRT, MUD, GRASS, SAND, HELLSTONE,
    COBALT, PALLADIUM, MYTHRIL, ORICHALCUM,
    ADAMANTITE, TITANIUM, CHLOROPHYTE,
    EBONSTONE, CRIMSTONE, CORRUPT_DIRT, CRIMSON_DIRT,
    PEARLSTONE, HALLOW_DIRT,
)
from Engine.constants import (
    LARGE, LayerDepths, OreConfig,
    INFECTION_GAP_TILES, SURFACE_UPDATE_RATE, UNDERGROUND_UPDATE_RATE,
    HALLOW_GRASS,
)
from Engine.theme import applyDarkTheme, COLORS

applyDarkTheme()

# Tile ID -> RGB colour (0-255)
TILE_COLORS: dict[int, tuple[int, int, int]] = {
    AIR:          (0, 0, 34),
    DIRT:         (139, 69, 19),
    STONE:        (112, 128, 144),
    GRASS:        (50, 205, 50),
    SAND:         (244, 164, 96),
    HELLSTONE:    (255, 69, 0),
    MUD:          (100, 80, 40),
    CORRUPT_DIRT: (106, 13, 173),
    EBONSTONE:    (72, 61, 139),
    CRIMSON_DIRT: (160, 20, 40),
    CRIMSTONE:    (139, 0, 0),
    PEARLSTONE:   (255, 182, 193),
    HALLOW_DIRT:  (255, 105, 180),
    HALLOW_GRASS: (238, 130, 238),
    COBALT:       (0, 102, 204),
    PALLADIUM:    (255, 140, 0),
    MYTHRIL:      (0, 200, 0),
    ORICHALCUM:   (200, 0, 200),
    ADAMANTITE:   (255, 0, 102),
    TITANIUM:     (180, 180, 200),
    CHLOROPHYTE:  (127, 255, 0),
}


# ======================================================================
class TerrariaHardmodeAnimation:
    """Produces a matplotlib FuncAnimation of the hardmode transition."""

    def __init__(
        self,
        worldWidth: int = 840,
        worldHeight: int = 240,
        seed: int = 42,
    ) -> None:
        self.worldWidth = worldWidth
        self.worldHeight = worldHeight
        self.rng = np.random.default_rng(seed)
        self.grid = np.full((worldHeight, worldWidth), AIR, dtype=np.int32)

        # Layer depths at 1/10 scale of Large world
        depths = LayerDepths.forLarge()
        self.surfaceLevel = int(depths.worldSurface / 10)   # ~34
        self.cavernLevel = int(depths.rockLayer / 10)        # ~88
        self.hellLevel = int(depths.hellLayer / 10)          # ~220

        self.worldCenter = worldWidth // 2
        self.area = worldWidth * worldHeight

        # Pick one ore per alternating hardmode pair (seed-dependent)
        self.hardmodeOres = [
            COBALT if self.rng.integers(2) == 0 else PALLADIUM,
            MYTHRIL if self.rng.integers(2) == 0 else ORICHALCUM,
            ADAMANTITE if self.rng.integers(2) == 0 else TITANIUM,
        ]

        self.surfaceHeights: np.ndarray = np.array([])

    # ------------------------------------------------------------------
    # World initialization
    # ------------------------------------------------------------------
    def initializeWorld(self) -> np.ndarray:
        """Build pre-hardmode terrain, evil biome patch, and caves."""
        grid = self.grid
        rng = self.rng
        grid[:] = AIR

        # Surface heights via sine superposition
        xs = np.arange(self.worldWidth, dtype=float)
        heights = (
            self.surfaceLevel
            + 5.0 * np.sin(xs * 0.02)
            + 3.0 * np.sin(xs * 0.05)
            + 2.0 * rng.standard_normal(self.worldWidth)
        )
        heights = np.clip(heights, 10, self.surfaceLevel + 15).astype(int)
        self.surfaceHeights = heights

        # Fill terrain layers per column
        for x in range(self.worldWidth):
            sy = heights[x]
            grid[sy, x] = GRASS
            dirtEnd = min(sy + 12, self.cavernLevel)
            grid[sy + 1 : dirtEnd, x] = DIRT
            grid[dirtEnd : self.hellLevel, x] = STONE
            grid[self.hellLevel :, x] = HELLSTONE

        # Evil biome patch (corruption on left quarter)
        cxS = self.worldWidth // 6
        cxE = self.worldWidth // 4
        cyS = self.surfaceLevel
        cyE = self.cavernLevel
        region = grid[cyS:cyE, cxS:cxE]
        region[region == DIRT] = CORRUPT_DIRT
        region[region == GRASS] = CORRUPT_DIRT
        region[region == STONE] = EBONSTONE

        # Carve caves with tileRunner
        for _ in range(60):
            cx = rng.integers(20, self.worldWidth - 20)
            cy = rng.integers(self.surfaceLevel + 5, self.hellLevel - 10)
            tileRunner(
                grid, cx, cy,
                strength=rng.uniform(3.0, 8.0),
                steps=rng.integers(10, 30),
                tileType=-1,
            )

        return grid

    # ------------------------------------------------------------------
    # V-pattern
    # ------------------------------------------------------------------
    def carveVPattern(self, progress: float = 1.0) -> None:
        """Carve two diagonal strips from world center surface to hell.

        Left strip: Corruption (EBONSTONE / CORRUPT_DIRT).
        Right strip: Hallow (PEARLSTONE / HALLOW_DIRT).
        Each strip carved by TileRunner (strength ~6) along diagonal vectors.
        Post-converts dirt-origin tiles to the appropriate dirt variant.
        """
        grid = self.grid
        original = grid.copy()

        totalDepth = self.hellLevel - self.surfaceLevel
        reachDepth = int(totalDepth * np.clip(progress, 0.0, 1.0))
        numCalls = max(1, reachDepth // 10)

        for i in range(numCalls):
            frac = i / max(1, numCalls - 1)
            dy = int(frac * reachDepth)

            # Left strip (corruption) - diagonal left-down
            lx = self.worldCenter - dy
            ly = self.surfaceLevel + dy
            if 2 < lx < self.worldWidth - 2 and 2 < ly < self.worldHeight - 2:
                tileRunner(
                    grid, lx, ly,
                    strength=6.0, steps=12,
                    tileType=EBONSTONE,
                    speedX=-0.7, speedY=0.7,
                    overRide=False,
                )

            # Right strip (hallow) - diagonal right-down
            rx = self.worldCenter + dy
            ry = self.surfaceLevel + dy
            if 2 < rx < self.worldWidth - 2 and 2 < ry < self.worldHeight - 2:
                tileRunner(
                    grid, rx, ry,
                    strength=6.0, steps=12,
                    tileType=PEARLSTONE,
                    speedX=0.7, speedY=0.7,
                    overRide=False,
                )

        # Post-convert: where original was dirt/grass, use the dirt variant
        dirtOrigin = np.isin(original, [DIRT, GRASS, MUD])

        newCorrupt = (grid == EBONSTONE) & (original != EBONSTONE)
        grid[newCorrupt & dirtOrigin] = CORRUPT_DIRT

        newHallow = (grid == PEARLSTONE) & (original != PEARLSTONE)
        grid[newHallow & dirtOrigin] = HALLOW_DIRT

    # ------------------------------------------------------------------
    # Altar smashing + ore placement
    # ------------------------------------------------------------------
    def smashAltarsAndPlaceOre(self, numAltars: int = 6) -> None:
        """Place hardmode ores via 3-cycle altar smashing.

        Cycle: altar 1->Tier1, altar 2->Tier2, altar 3->Tier3,
               altar 4->Tier1 (fewer loops), altar 5->Tier2, altar 6->Tier3.
        Loop count = OreConfig.loopCount(area) / cycleNum.
        Each vein placed by TileRunner (strength ~3-5).
        """
        grid = self.grid
        rng = self.rng
        baseLoops = OreConfig.loopCount(self.area)

        tierDepths = [
            (self.surfaceLevel + 5, self.hellLevel - 2),
            (self.cavernLevel, self.hellLevel - 2),
            (self.cavernLevel + (self.hellLevel - self.cavernLevel) // 3,
             self.hellLevel - 2),
        ]

        for altarIdx in range(numAltars):
            tier = altarIdx % 3
            cycleNum = altarIdx // 3 + 1
            oreType = self.hardmodeOres[tier]
            loops = max(1, baseLoops // cycleNum)
            minY, maxY = tierDepths[tier]

            for _ in range(loops):
                ox = rng.integers(20, self.worldWidth - 20)
                oy = rng.integers(minY, maxY)
                tileRunner(
                    grid, ox, oy,
                    strength=rng.uniform(2.5, 4.5),
                    steps=rng.integers(4, 10),
                    tileType=oreType,
                    overRide=False,
                )

    def _placeOreTier(self, tier: int, cycleNum: int) -> None:
        """Place ores for a single tier/cycle pair."""
        grid = self.grid
        rng = self.rng
        baseLoops = OreConfig.loopCount(self.area)
        oreType = self.hardmodeOres[tier]
        loops = max(1, baseLoops // cycleNum)

        tierDepths = [
            (self.surfaceLevel + 5, self.hellLevel - 2),
            (self.cavernLevel, self.hellLevel - 2),
            (self.cavernLevel + (self.hellLevel - self.cavernLevel) // 3,
             self.hellLevel - 2),
        ]
        minY, maxY = tierDepths[tier]

        for _ in range(loops):
            ox = rng.integers(20, self.worldWidth - 20)
            oy = rng.integers(minY, maxY)
            tileRunner(
                grid, ox, oy,
                strength=rng.uniform(2.5, 4.5),
                steps=rng.integers(4, 10),
                tileType=oreType,
                overRide=False,
            )

    # ------------------------------------------------------------------
    # Infection spread
    # ------------------------------------------------------------------
    def simulateSpreadStep(self, elapsedSeconds: float) -> None:
        """One tick of tile-update-cycle infection spread.

        Surface tiles update every ~140 s, underground every ~830 s.
        Spread checks 8-connected neighbours (radius 1). AIR tiles are
        not convertible, so any air gap naturally blocks propagation per
        step. In the actual game spread checks radius 3, requiring a
        4-tile air gap (INFECTION_GAP_TILES) to fully block.
        """
        grid = self.grid
        h, w = grid.shape
        rng = self.rng

        # Per-row update probability
        surfaceProb = min(1.0, elapsedSeconds / SURFACE_UPDATE_RATE)
        undergroundProb = min(1.0, elapsedSeconds / UNDERGROUND_UPDATE_RATE)

        rowDepths = np.arange(h)
        probPerRow = np.where(
            rowDepths < self.surfaceLevel, 0.0,
            np.where(rowDepths < self.cavernLevel, surfaceProb, undergroundProb),
        )
        updateMask = rng.random((h, w)) < probPerRow[:, None]

        # Infection source masks
        corruptSet = [CORRUPT_DIRT, EBONSTONE, CRIMSTONE, CRIMSON_DIRT]
        hallowSet = [PEARLSTONE, HALLOW_DIRT, HALLOW_GRASS]

        corruptMask = np.isin(grid, corruptSet)
        hallowMask = np.isin(grid, hallowSet)

        # Dilate by 1 tile (8-connected)
        padC = np.pad(corruptMask, 1, constant_values=False)
        padH = np.pad(hallowMask, 1, constant_values=False)

        neighborCorrupt = np.zeros((h, w), dtype=bool)
        neighborHallow = np.zeros((h, w), dtype=bool)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue
                neighborCorrupt |= padC[1 + dy : h + 1 + dy, 1 + dx : w + 1 + dx]
                neighborHallow |= padH[1 + dy : h + 1 + dy, 1 + dx : w + 1 + dx]

        # Convertible tile mask
        convertible = np.isin(grid, [DIRT, STONE, GRASS, SAND, MUD])

        corruptCand = neighborCorrupt & convertible & updateMask & ~corruptMask
        hallowCand = neighborHallow & convertible & updateMask & ~hallowMask

        # Hallow wins where both overlap (game behaviour)
        overlap = corruptCand & hallowCand
        corruptCand = corruptCand & ~overlap

        # Tile-type conversions
        dirtLike = np.isin(grid, [DIRT, GRASS, MUD])
        stoneLike = (grid == STONE)
        sandLike = (grid == SAND)

        grid[corruptCand & dirtLike] = CORRUPT_DIRT
        grid[corruptCand & stoneLike] = EBONSTONE
        grid[corruptCand & sandLike] = EBONSTONE

        grid[hallowCand & dirtLike] = HALLOW_DIRT
        grid[hallowCand & stoneLike] = PEARLSTONE
        grid[hallowCand & sandLike] = PEARLSTONE

    # ------------------------------------------------------------------
    # Frame generation
    # ------------------------------------------------------------------
    def generateFrames(self, spreadSteps: int = 15) -> List[Tuple[str, np.ndarray]]:
        """Produce all (label, gridCopy) pairs for the animation."""
        frames: List[Tuple[str, np.ndarray]] = []

        # Frame 0: pre-hardmode
        self.initializeWorld()
        preHardmode = self.grid.copy()
        frames.append(("Pre-Hardmode World (1/10 scale)", preHardmode.copy()))

        # Frames 1-3: V-pattern carving (progressive)
        for pct in (0.33, 0.66, 1.0):
            self.grid = preHardmode.copy()
            self.carveVPattern(progress=pct)
            frames.append((
                f"V-Pattern Carving ({int(pct * 100)}%)",
                self.grid.copy(),
            ))

        # Frames 4-6: one ore tier per frame (cumulative)
        baseGrid = frames[-1][1].copy()
        tierNames = ["Cobalt/Palladium", "Mythril/Orichalcum", "Adamantite/Titanium"]
        for tier in range(3):
            self.grid = baseGrid.copy()
            self._placeOreTier(tier, cycleNum=1)
            self._placeOreTier(tier, cycleNum=2)
            frames.append((
                f"Altar Smash: {tierNames[tier]}",
                self.grid.copy(),
            ))
            baseGrid = self.grid.copy()

        # Frames 7+: infection spread
        self.grid = baseGrid.copy()
        secondsPerStep = 300.0  # each frame ~ 5 in-game minutes
        for step in range(spreadSteps):
            self.simulateSpreadStep(secondsPerStep)
            totalTime = (step + 1) * secondsPerStep
            frames.append((
                f"Infection Spread (t={totalTime:.0f}s)",
                self.grid.copy(),
            ))

        return frames

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    @staticmethod
    def buildColorImage(grid: np.ndarray) -> np.ndarray:
        """Convert tile-ID grid to an (H, W, 3) uint8 RGB image."""
        h, w = grid.shape
        image = np.zeros((h, w, 3), dtype=np.uint8)
        for tileId, rgb in TILE_COLORS.items():
            mask = (grid == tileId)
            if mask.any():
                image[mask] = rgb
        return image

    def animate(self, savePath: Optional[str] = None) -> None:
        """Render FuncAnimation and save as .gif (~400 ms/frame)."""
        frames = self.generateFrames()
        totalFrames = len(frames)

        fig, ax = plt.subplots(figsize=(16, 5))
        fig.patch.set_facecolor("#0D1117")

        def update(frameIdx: int) -> list:
            ax.clear()
            label, grid = frames[frameIdx]
            image = self.buildColorImage(grid)
            im = ax.imshow(
                image, aspect="auto",
                extent=[0, self.worldWidth, self.worldHeight, 0],
            )

            # Layer guides
            ax.axhline(y=self.surfaceLevel, color="cyan", ls=":", lw=0.8, alpha=0.5)
            ax.axhline(y=self.cavernLevel, color="yellow", ls=":", lw=0.8, alpha=0.5)
            ax.axhline(y=self.hellLevel, color="red", ls=":", lw=0.8, alpha=0.5)

            ax.set_title(
                f"{label}  [Frame {frameIdx + 1}/{totalFrames}]",
                color="white", fontsize=12, fontweight="bold",
            )
            ax.set_xlabel("X (blocks)", color="white", fontsize=9)
            ax.set_ylabel("Depth (blocks)", color="white", fontsize=9)
            ax.tick_params(colors="white", labelsize=8)

            # Stats overlay
            corruptCount = int(np.isin(grid, [CORRUPT_DIRT, EBONSTONE]).sum())
            hallowCount = int(
                np.isin(grid, [PEARLSTONE, HALLOW_DIRT, HALLOW_GRASS]).sum()
            )
            total = self.area
            stats = (
                f"Corrupt: {corruptCount / total * 100:.1f}%\n"
                f"Hallow:  {hallowCount / total * 100:.1f}%"
            )
            ax.text(
                0.01, 0.97, stats, transform=ax.transAxes, fontsize=8,
                color="white", va="top", family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.7),
            )
            return [im]

        anim = FuncAnimation(
            fig, update, frames=totalFrames,
            interval=400, repeat=True, blit=False,
        )

        if savePath is None:
            savePath = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "Plots", "Advanced", "terraria_hardmode_animation.gif",
            )
        os.makedirs(os.path.dirname(savePath), exist_ok=True)

        print(f"Saving {totalFrames}-frame animation to {savePath} ...")
        writer = PillowWriter(fps=3)
        anim.save(savePath, writer=writer, dpi=100)
        plt.close(fig)
        print("Done.")


# ======================================================================
if __name__ == "__main__":
    sim = TerrariaHardmodeAnimation(worldWidth=840, worldHeight=240, seed=42)
    sim.animate()
