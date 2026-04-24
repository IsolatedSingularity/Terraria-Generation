"""
Terraria Hardmode Transformation System
========================================
Models altar-breaking ore generation, Life Crystal placement, and Chlorophyte
growth using TileRunner from Engine.

All algorithms match decompiled WorldGen.cs behavior:
- Ore via TileRunner diamond-brush random walk (area-proportional loop count)
- Life Crystals as 2x2 FrameImportant tiles (403 max for large world)
- Hardmode ore 3-cycle tier system (Cobalt->Mythril->Adamantite or alternates)
- Altar breaking side effect: chance to convert random stone to evil stone
- Chlorophyte restricted to jungle cavern layer
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from Engine.algorithms import (tileRunner, AIR, STONE, DIRT, MUD, GRASS,
                                COBALT, PALLADIUM, MYTHRIL, ORICHALCUM,
                                ADAMANTITE, TITANIUM, CHLOROPHYTE,
                                EBONSTONE, CRIMSTONE, CORRUPT_DIRT, CRIMSON_DIRT)
from Engine.constants import LARGE, LayerDepths, StructureQuotas, OreConfig, LIFE_CRYSTAL, ALTAR
from Engine.theme import applyTokyoNight, COLORS, TILE_COLORS as _ENGINE_TILE_COLORS

applyTokyoNight()


# ---------------------------------------------------------------------------
# Name lookup
# ---------------------------------------------------------------------------
_ORE_NAMES = {
    COBALT: "Cobalt", PALLADIUM: "Palladium",
    MYTHRIL: "Mythril", ORICHALCUM: "Orichalcum",
    ADAMANTITE: "Adamantite", TITANIUM: "Titanium",
    CHLOROPHYTE: "Chlorophyte",
}

_TILE_NAMES = {
    AIR: "Air", DIRT: "Dirt", STONE: "Stone", MUD: "Mud", GRASS: "Grass",
    CORRUPT_DIRT: "Corrupt Dirt", EBONSTONE: "Ebonstone",
    CRIMSON_DIRT: "Crimson Dirt", CRIMSTONE: "Crimstone",
    LIFE_CRYSTAL: "Life Crystal", ALTAR: "Altar",
    **_ORE_NAMES,
}


def _tileName(tileId: int) -> str:
    return _TILE_NAMES.get(tileId, f"Tile {tileId}")


# ---------------------------------------------------------------------------
# Color palette (delegated to Engine.theme -- single source of truth).
# Local _TILE_COLORS retained as a hex-keyed view used by legend builders.
# ---------------------------------------------------------------------------
_TILE_COLORS = dict(_ENGINE_TILE_COLORS)


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------
class TerrariaHardmodeTransformation:
    """Hardmode world transformation: altar breaking, ore generation,
    Life Crystal placement, and Chlorophyte growth.

    Uses TileRunner with area-proportional loop count (6E-05 formula)
    and proper 3-cycle hardmode ore tier system.
    """

    def __init__(self, worldWidth: int = 8400, worldHeight: int = 2400,
                 seed: int = 12345):
        self.worldWidth = worldWidth
        self.worldHeight = worldHeight
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.grid = np.zeros((worldHeight, worldWidth), dtype=np.int32)

        # Scale layer depths proportionally from large-world reference
        refLayers = LayerDepths.forLarge()
        yScale = worldHeight / refLayers.maxTilesY
        self.layers = LayerDepths(
            worldSurface=refLayers.worldSurface * yScale,
            rockLayer=refLayers.rockLayer * yScale,
            hellLayer=int(refLayers.hellLayer * yScale),
            maxTilesY=worldHeight,
        )

        # Full-scale quotas (used as reference; placement naturally limited by area)
        self.quotas = StructureQuotas.forLarge()
        self._areaScale = (worldWidth * worldHeight) / LARGE.area

        self.altarsSmashed = 0
        self.hardmodeOreTiers: list[int] = []

        # Pick one from each hardmode alternating pair (per-world seed choice)
        self.orePairs = [
            (COBALT, PALLADIUM),
            (MYTHRIL, ORICHALCUM),
            (ADAMANTITE, TITANIUM),
        ]
        self.selectedOres = [
            pair[self.rng.integers(0, 2)] for pair in self.orePairs
        ]

        # World evil type (affects altar side effect)
        self.isCorruption = bool(self.rng.integers(0, 2))

        # Evil biome x-range (opposite side from dungeon)
        self.evilXMin = int(worldWidth * 0.15)
        self.evilXMax = int(worldWidth * 0.35)

        # Jungle x-range
        self.jungleXMin = int(worldWidth * 0.65)
        self.jungleXMax = int(worldWidth * 0.85)

        # History snapshots for visualization
        self.history: list[tuple[str, np.ndarray]] = []

    # ------------------------------------------------------------------
    # Pre-hardmode setup
    # ------------------------------------------------------------------
    def initializePreHardmodeWorld(self) -> None:
        """Build base terrain with dirt/stone layers, evil biome, jungle,
        and altars in evil regions."""
        grid = self.grid
        layers = self.layers
        rng = self.rng

        # Sinusoidal surface profile
        x = np.linspace(0, 8 * np.pi, self.worldWidth)
        surfaceProfile = (
            0.40 * np.sin(x * 0.3 + rng.uniform(0, 2 * np.pi))
            + 0.25 * np.sin(x * 0.7 + rng.uniform(0, 2 * np.pi))
            + 0.15 * np.sin(x * 1.5 + rng.uniform(0, 2 * np.pi))
        )
        surfaceY = (surfaceProfile * 15 + layers.worldSurface).astype(int)
        surfaceY = np.clip(surfaceY, 5, self.worldHeight - 5)

        # Fill terrain layers per column (vectorized row slicing)
        rockRow = int(layers.rockLayer)
        hellRow = layers.hellLayer
        for col in range(self.worldWidth):
            sY = int(surfaceY[col])
            dirtEnd = min(rockRow, hellRow)
            grid[sY:dirtEnd, col] = DIRT
            grid[dirtEnd:hellRow, col] = STONE

        # Carve caves with TileRunner
        numCaves = max(8, self.worldWidth // 80)
        for _ in range(numCaves):
            cx = rng.integers(40, self.worldWidth - 40)
            cy = rng.integers(int(layers.worldSurface + 20), hellRow - 20)
            strength = rng.uniform(4.0, 12.0)
            steps = rng.integers(25, 70)
            tileRunner(grid, int(cx), int(cy), strength, int(steps), tileType=-1)

        # Paint evil biome (Ebonstone/Crimstone replaces Stone in cavern,
        # Corrupt/Crimson Dirt replaces Dirt above)
        evilStone = EBONSTONE if self.isCorruption else CRIMSTONE
        evilDirtTile = CORRUPT_DIRT if self.isCorruption else CRIMSON_DIRT

        cavSlice = grid[rockRow:hellRow, self.evilXMin:self.evilXMax]
        cavSlice[cavSlice == STONE] = evilStone

        dirtSlice = grid[int(layers.worldSurface):rockRow,
                         self.evilXMin:self.evilXMax]
        dirtSlice[dirtSlice == DIRT] = evilDirtTile

        # Paint jungle (MUD replaces Dirt)
        jungleDirt = grid[int(layers.worldSurface):rockRow,
                          self.jungleXMin:self.jungleXMax]
        jungleDirt[jungleDirt == DIRT] = MUD

        # Place altars in evil cavern region
        self._placeAltars()

    def _placeAltars(self) -> None:
        """Place Demon/Crimson altars in the evil biome cavern layer."""
        numAltars = max(6, int(18 * self._areaScale))
        placed = 0
        attempts = 0
        minSpacing = max(20, int(80 * (self.worldWidth / LARGE.width)))
        positions: list[tuple[int, int]] = []
        evilStone = EBONSTONE if self.isCorruption else CRIMSTONE

        while placed < numAltars and attempts < 3000:
            ax = self.rng.integers(self.evilXMin + 5, self.evilXMax - 5)
            ay = self.rng.integers(int(self.layers.rockLayer) + 5,
                                   self.layers.hellLayer - 20)
            if self.grid[ay, ax] == evilStone:
                tooClose = any(
                    abs(ax - px) + abs(ay - py) < minSpacing
                    for px, py in positions
                )
                if not tooClose:
                    self.grid[ay, ax] = ALTAR
                    positions.append((ax, ay))
                    placed += 1
            attempts += 1

    # ------------------------------------------------------------------
    # Altar breaking and ore generation
    # ------------------------------------------------------------------
    def smashAltar(self) -> int | None:
        """Break one altar: generate a hardmode ore tier, with 66% chance
        to convert a random stone tile to evil stone.

        Returns the ore type placed, or None if no altars remain.
        3-cycle: 1st->Tier1, 2nd->Tier2, 3rd->Tier3, 4th->Tier1 (halved), ...
        """
        altarPos = np.argwhere(self.grid == ALTAR)
        if len(altarPos) == 0:
            return None

        # Pick and remove a random altar
        idx = self.rng.integers(0, len(altarPos))
        ay, ax = altarPos[idx]
        self.grid[ay, ax] = AIR

        self.altarsSmashed += 1

        # 3-cycle tier selection
        tierIndex = (self.altarsSmashed - 1) % 3
        oreType = self.selectedOres[tierIndex]
        self.hardmodeOreTiers.append(oreType)

        # Cycle number determines loop-count divisor
        cycleNum = (self.altarsSmashed - 1) // 3 + 1

        # Depth bounds per tier (deeper for rarer ores)
        depthBounds = [
            (int(self.layers.rockLayer), self.layers.hellLayer),
            (int(self.layers.rockLayer + 30 * (self.worldHeight / LARGE.height)),
             self.layers.hellLayer),
            (int(self.layers.rockLayer + 60 * (self.worldHeight / LARGE.height)),
             self.layers.hellLayer),
        ]
        depthMin, depthMax = depthBounds[tierIndex]

        self.placeHardmodeOre(oreType, depthMin, depthMax, cycleNum)

        # 66% chance to corrupt a random stone tile
        if self.rng.random() < 0.66:
            self._corruptRandomStone()

        return oreType

    def placeHardmodeOre(self, oreType: int, depthMin: int, depthMax: int,
                         cycleNum: int = 1) -> None:
        """Place hardmode ore via TileRunner with game formula.

        loopCount = int(area * 6E-05) / cycleNum
        Each invocation picks a random stone tile in the depth range.
        """
        area = self.worldWidth * self.worldHeight
        baseLoopCount = OreConfig.loopCount(area)
        loopCount = max(1, baseLoopCount // cycleNum)

        # Strength/steps per tier (rarer ores get slightly larger veins)
        oreParams = {
            COBALT: (3.0, 8), PALLADIUM: (3.0, 8),
            MYTHRIL: (4.0, 10), ORICHALCUM: (4.0, 10),
            ADAMANTITE: (5.0, 12), TITANIUM: (5.0, 12),
        }
        strength, steps = oreParams.get(oreType, (3.0, 8))

        if depthMin >= depthMax:
            return

        for _ in range(loopCount):
            rx = int(self.rng.integers(40, self.worldWidth - 40))
            ry = int(self.rng.integers(depthMin, depthMax))
            if self.grid[ry, rx] == STONE:
                tileRunner(self.grid, rx, ry, strength, steps,
                           oreType, overRide=False)

    def _corruptRandomStone(self) -> None:
        """Convert a random stone tile to Ebonstone/Crimstone (altar side effect).
        Uses random sampling instead of argwhere for performance."""
        evilStone = EBONSTONE if self.isCorruption else CRIMSTONE
        rockRow = int(self.layers.rockLayer)
        hellRow = self.layers.hellLayer

        for _ in range(200):
            rx = int(self.rng.integers(10, self.worldWidth - 10))
            ry = int(self.rng.integers(rockRow, hellRow))
            if self.grid[ry, rx] == STONE:
                self.grid[ry, rx] = evilStone
                return

    # ------------------------------------------------------------------
    # Life Crystals
    # ------------------------------------------------------------------
    def placeLifeCrystals(self) -> int:
        """Place Life Crystals as 2x2 FrameImportant tiles.

        Max count from StructureQuotas (403 for large), scaled by area.
        Placed between worldSurface and hellLayer where 2x2 stone exists
        near a cave (AIR neighbor within 2 tiles).

        Returns the number placed.
        """
        maxCrystals = max(1, int(self.quotas.lifeCrystalsMax * self._areaScale))
        placed = 0
        maxAttempts = maxCrystals * 30

        surfaceY = int(self.layers.worldSurface) + 10
        hellY = self.layers.hellLayer - 5

        if surfaceY >= hellY:
            return 0

        for _ in range(maxAttempts):
            if placed >= maxCrystals:
                break

            cx = int(self.rng.integers(10, self.worldWidth - 12))
            cy = int(self.rng.integers(surfaceY, hellY))

            # Need 2x2 stone footprint near a cave (AIR neighbor within 2 tiles)
            if cy < 1 or cy + 1 >= self.worldHeight or cx + 1 >= self.worldWidth:
                continue

            if (self.grid[cy, cx] == STONE
                    and self.grid[cy, cx + 1] == STONE
                    and self.grid[cy - 1, cx] == STONE
                    and self.grid[cy - 1, cx + 1] == STONE):
                # Check for nearby air (cave adjacency) within 2 tiles
                hasAir = False
                for dy in range(-2, 3):
                    for dx in range(-1, 3):
                        ny, nx = cy + dy, cx + dx
                        if (0 <= ny < self.worldHeight
                                and 0 <= nx < self.worldWidth
                                and self.grid[ny, nx] == AIR):
                            hasAir = True
                            break
                    if hasAir:
                        break
                if not hasAir:
                    continue

                self.grid[cy, cx] = LIFE_CRYSTAL
                self.grid[cy, cx + 1] = LIFE_CRYSTAL
                self.grid[cy - 1, cx] = LIFE_CRYSTAL
                self.grid[cy - 1, cx + 1] = LIFE_CRYSTAL
                placed += 1

        return placed

    # ------------------------------------------------------------------
    # Chlorophyte
    # ------------------------------------------------------------------
    def placeChlorophyte(self, jungleXMin: int, jungleXMax: int) -> None:
        """Place Chlorophyte ore in jungle cavern layer using TileRunner.

        Restricted to below rockLayer within the given jungle x-range.
        Uses a reduced loop count (1/3 of standard formula).
        """
        area = self.worldWidth * self.worldHeight
        loopCount = max(1, OreConfig.loopCount(area) // 3)

        cavernTop = int(self.layers.rockLayer)
        cavernBot = self.layers.hellLayer

        if cavernTop >= cavernBot or jungleXMin >= jungleXMax:
            return

        for _ in range(loopCount):
            rx = int(self.rng.integers(jungleXMin, jungleXMax))
            ry = int(self.rng.integers(cavernTop, cavernBot))
            if self.grid[ry, rx] in (STONE, MUD):
                tileRunner(self.grid, rx, ry, 3.0, 6,
                           CHLOROPHYTE, overRide=False)

    # ------------------------------------------------------------------
    # Full sequence
    # ------------------------------------------------------------------
    def runHardmodeTransformation(self, numAltars: int = 12) -> None:
        """Execute the full hardmode transformation sequence.

        1. Build pre-hardmode world (terrain, evil biome, altars)
        2. Place Life Crystals (2x2, max 403 scaled by area)
        3. Smash altars (3-cycle ore tiers + corruption side effect)
        4. Place Chlorophyte in jungle cavern layer
        """
        # Phase 1: pre-hardmode world
        self.initializePreHardmodeWorld()
        crystalsPlaced = self.placeLifeCrystals()
        self.history.append(("Pre-Hardmode + Life Crystals", self.grid.copy()))

        # Phase 2: smash altars
        for _ in range(numAltars):
            result = self.smashAltar()
            if result is None:
                break
        self.history.append(("Post-Altar Smashing", self.grid.copy()))

        # Phase 3: Chlorophyte
        self.placeChlorophyte(self.jungleXMin, self.jungleXMax)
        self.history.append(("Post-Chlorophyte", self.grid.copy()))

        # Statistics
        self._printStats(crystalsPlaced)

    def _printStats(self, crystalsPlaced: int) -> None:
        """Print summary statistics."""
        area = self.worldWidth * self.worldHeight
        formulaCount = OreConfig.loopCount(area)
        print(f"World: {self.worldWidth}x{self.worldHeight} (area={area:,})")
        print(f"TileRunner loops/ore (6E-05): {formulaCount:,}")
        print(f"Altars smashed: {self.altarsSmashed}")
        print(f"Life Crystals placed: {crystalsPlaced} "
              f"/ {max(1, int(self.quotas.lifeCrystalsMax * self._areaScale))} "
              f"scaled max ({self.quotas.lifeCrystalsMax} full-scale)")
        print(f"Evil type: {'Corruption' if self.isCorruption else 'Crimson'}")
        print(f"Selected ore tiers: "
              f"{[_ORE_NAMES.get(o, '?') for o in self.selectedOres]}")

        for oreId in [COBALT, PALLADIUM, MYTHRIL, ORICHALCUM,
                      ADAMANTITE, TITANIUM, CHLOROPHYTE,
                      LIFE_CRYSTAL, EBONSTONE, CRIMSTONE]:
            count = int(np.sum(self.grid == oreId))
            if count > 0:
                print(f"  {_tileName(oreId)}: {count:,} tiles")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def _remapGrid(grid: np.ndarray, knownIds: list[int]) -> np.ndarray:
    """Remap tile IDs to sequential 0..N indices for colormap display."""
    idToIdx = {tid: i for i, tid in enumerate(knownIds)}
    out = np.zeros_like(grid, dtype=np.int32)
    for tid, idx in idToIdx.items():
        out[grid == tid] = idx
    return out


def visualize(sim: TerrariaHardmodeTransformation,
              savePath: str | None = None) -> None:
    """3-panel 600x500 SMALL-crop hardmode transformation figure.

    Panels:
      1. Pre-HM baseline
      2. Post-V-pattern (altar x0)
      3. Post-altar-smashing (full HM ore tiers + Chlorophyte in jungle mud)
    """
    from Engine.spriteRenderer import applyMapDecorations, cropSmallWorld, drawTileGrid
    from Engine.worldgen import generateSmallWorld

    plotsDir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Plots", "Advanced",
    )
    os.makedirs(plotsDir, exist_ok=True)
    outFile = os.path.join(plotsDir, "terraria_hardmode_transformation.png")

    print("Generating SMALL world for hardmode transformation (seed=20260425)...")
    worldBase = generateSmallWorld(seed=20260425, evilType="corruption",
                                    altarsSmashed=0, compactBiomes=True)
    worldV = generateSmallWorld(seed=20260425, evilType="corruption",
                                 altarsSmashed=0, compactBiomes=True)
    worldHM = generateSmallWorld(seed=20260425, evilType="corruption",
                                  altarsSmashed=9, compactBiomes=True)

    layers = worldBase.layers
    centerX = worldBase.spawnX
    centerY = int((layers.worldSurface + layers.rockLayer) / 2)

    # Apply V-pattern to worldV grid via TerrariaCorruptionEvolution
    from Advanced.terrariaCorruptionEvolution import TerrariaCorruptionEvolution
    simV = TerrariaCorruptionEvolution(
        worldWidth=worldV.grid.shape[1], worldHeight=worldV.grid.shape[0],
        evilType="corruption", seed=20260425,
    )
    simV.grid = worldV.grid.copy()
    simV.layers = layers
    simV.triggerHardmode()

    panels = [
        (worldBase.grid, "Panel 1: Pre-Hardmode Baseline"),
        (simV.grid, "Panel 2: Post-V-Pattern (WoF)"),
        (worldHM.grid, "Panel 3: Post-Altar x9 (Full HM Ores)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(24, 9))
    for ax, (snap, title) in zip(axes, panels):
        cropped, bounds = cropSmallWorld(snap, centerX=centerX, centerY=centerY,
                                          width=600, height=500)
        drawTileGrid(ax, cropped)
        applyMapDecorations(ax, cropped, layers, cropBounds=bounds)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("X (tiles, crop-local)")
        ax.set_ylabel("Depth (tiles, crop-local)")

    fig.suptitle(
        "Hardmode Transformation (600x500 SMALL-World Crop)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(outFile, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"Saved: {outFile}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sim = TerrariaHardmodeTransformation(seed=42)
    sim.runHardmodeTransformation(numAltars=12)
    visualize(sim)
