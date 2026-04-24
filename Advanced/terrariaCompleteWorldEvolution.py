"""
Terraria Complete World Evolution
=================================

Full lifecycle visualization from empty grid to late Hardmode.
Seven phases: terrain, caves, biomes, pre-HM ores, V-pattern,
altar smashing, and infection spread.

Grid is 840x240 (1/10 scale of Large world 8400x2400).
Layer depths scaled proportionally from decompiled constants.

All algorithms use Engine.algorithms (TileRunner diamond-brush,
cellularAutomataSmooth). Ore counts use int(area * 6E-05) formula.
Infection spread uses tile update cycle rates with 4-tile air gap blocking.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from Engine.algorithms import (
    tileRunner, cellularAutomataSmooth,
    AIR, STONE, DIRT, MUD, GRASS, SAND, ASH, HELLSTONE,
    COBALT, PALLADIUM, MYTHRIL, ORICHALCUM,
    ADAMANTITE, TITANIUM, CHLOROPHYTE,
    EBONSTONE, CRIMSTONE, CORRUPT_DIRT, CRIMSON_DIRT,
    PEARLSTONE, HALLOW_DIRT,
    COPPER, TIN, IRON, LEAD, SILVER, TUNGSTEN, GOLD, PLATINUM,
)
from Engine.constants import (
    LARGE, LayerDepths, StructureQuotas, OreConfig,
    INFECTION_GAP_TILES, SURFACE_UPDATE_RATE, UNDERGROUND_UPDATE_RATE,
    INFECTION_SPREAD_RADIUS, LIFE_CRYSTAL,
)
from Engine.theme import applyTokyoNight, COLORS

applyTokyoNight()

# ---------------------------------------------------------------------------
# RGB color table for rendering sparse tile IDs to an image
# ---------------------------------------------------------------------------
TILE_COLORS: dict[int, tuple[int, int, int]] = {
    AIR: (0, 0, 17),
    DIRT: (139, 69, 19),
    STONE: (105, 105, 105),
    GRASS: (34, 139, 34),
    SAND: (244, 164, 96),
    ASH: (50, 50, 50),
    HELLSTONE: (255, 69, 0),
    MUD: (85, 60, 30),
    CORRUPT_DIRT: (147, 112, 219),
    EBONSTONE: (80, 40, 120),
    CRIMSON_DIRT: (220, 20, 60),
    CRIMSTONE: (139, 0, 0),
    PEARLSTONE: (255, 182, 193),
    HALLOW_DIRT: (255, 200, 220),
    COPPER: (184, 115, 51),
    TIN: (165, 165, 140),
    IRON: (160, 160, 160),
    LEAD: (80, 80, 120),
    SILVER: (210, 210, 210),
    TUNGSTEN: (140, 180, 140),
    GOLD: (255, 215, 0),
    PLATINUM: (220, 230, 240),
    COBALT: (0, 71, 171),
    PALLADIUM: (245, 130, 48),
    MYTHRIL: (0, 255, 127),
    ORICHALCUM: (255, 105, 180),
    ADAMANTITE: (255, 0, 0),
    TITANIUM: (128, 128, 128),
    CHLOROPHYTE: (127, 255, 0),
    LIFE_CRYSTAL: (255, 0, 128),
}

# ---------------------------------------------------------------------------
# Infection spread lookup tables
# ---------------------------------------------------------------------------
CORRUPTION_TILES = frozenset({CORRUPT_DIRT, EBONSTONE})
CRIMSON_TILES = frozenset({CRIMSON_DIRT, CRIMSTONE})
HALLOW_TILES = frozenset({PEARLSTONE, HALLOW_DIRT})

CORRUPTION_CONVERT: dict[int, int] = {
    STONE: EBONSTONE, DIRT: CORRUPT_DIRT, GRASS: CORRUPT_DIRT, SAND: EBONSTONE,
}
CRIMSON_CONVERT: dict[int, int] = {
    STONE: CRIMSTONE, DIRT: CRIMSON_DIRT, GRASS: CRIMSON_DIRT, SAND: CRIMSTONE,
}
HALLOW_CONVERT: dict[int, int] = {
    STONE: PEARLSTONE, DIRT: HALLOW_DIRT, GRASS: HALLOW_DIRT, SAND: PEARLSTONE,
}

ORE_NAME_TO_ID: dict[str, int] = {
    "Copper": COPPER, "Tin": TIN, "Iron": IRON, "Lead": LEAD,
    "Silver": SILVER, "Tungsten": TUNGSTEN, "Gold": GOLD, "Platinum": PLATINUM,
    "Cobalt": COBALT, "Palladium": PALLADIUM,
    "Mythril": MYTHRIL, "Orichalcum": ORICHALCUM,
    "Adamantite": ADAMANTITE, "Titanium": TITANIUM,
}


# ---------------------------------------------------------------------------
# Rendering helper
# ---------------------------------------------------------------------------
def renderGrid(grid: np.ndarray) -> np.ndarray:
    """Convert a tile-ID grid to an (H, W, 3) uint8 RGB image."""
    h, w = grid.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for tileId, color in TILE_COLORS.items():
        mask = grid == tileId
        img[mask] = color
    return img


# ===================================================================
# Main simulation class
# ===================================================================
class TerrariaCompleteWorldEvolution:
    """Complete lifecycle simulation from empty grid to late Hardmode."""

    def __init__(self, worldWidth: int = 840, worldHeight: int = 240, seed: int = 42):
        self.worldWidth = worldWidth
        self.worldHeight = worldHeight
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Scale factors relative to Large world (8400x2400)
        self.xScale = worldWidth / LARGE.width
        self.yScale = worldHeight / LARGE.height
        self.scaledArea = worldWidth * worldHeight
        self.areaRatio = self.scaledArea / LARGE.area

        # Layer depths (scaled from Large constants)
        layers = LayerDepths.forLarge()
        self.worldSurface: int = int(layers.worldSurface * self.yScale)
        self.rockLayer: int = int(layers.rockLayer * self.yScale)
        self.hellLayer: int = int(layers.hellLayer * self.yScale)

        # Structure quotas (scaled)
        quotas = StructureQuotas.forLarge()
        self.lifeCrystalsMax: int = max(1, int(quotas.lifeCrystalsMax * self.areaRatio))

        # Ore loop count: int(area * 6E-05)
        self.oreLoopCount: int = OreConfig.loopCount(self.scaledArea)

        # Evil type selection
        self.evilType: str = self.rng.choice(["corruption", "crimson"])

        # Pre-HM ore selection (one from each alternating pair)
        self.preHmOres: list[str] = []
        for pair in OreConfig.PRE_HARDMODE_PAIRS:
            self.preHmOres.append(pair[int(self.rng.integers(0, 2))])

        # HM ore selection (one from each tier)
        self.hmOres: list[str] = []
        for tier in OreConfig.HARDMODE_TIERS:
            self.hmOres.append(tier[int(self.rng.integers(0, 2))])

        # World grid
        self.grid: np.ndarray = np.full(
            (worldHeight, worldWidth), AIR, dtype=np.int32
        )

        # Surface height profile (populated by buildTerrain)
        self.surfaceY: np.ndarray = np.full(worldWidth, self.worldSurface, dtype=np.int32)

        # Evil biome extents (populated by paintBiomes)
        self.evilXStart: int = 0
        self.evilXEnd: int = 0

        # Snapshot history for visualization
        self.history: list[tuple[str, np.ndarray]] = []

    # ------------------------------------------------------------------
    # Phase 1: Base terrain
    # ------------------------------------------------------------------
    def buildTerrain(self) -> None:
        """Empty world -> layered terrain with surface noise."""
        w, h = self.worldWidth, self.worldHeight

        # Surface height profile with multi-octave sine noise
        x = np.arange(w, dtype=np.float64)
        surface = np.full(w, float(self.worldSurface))
        surface += 3.0 * np.sin(x * 0.015)
        surface += 2.0 * np.sin(x * 0.04) * np.cos(x * 0.025)
        surface += 1.0 * np.sin(x * 0.10)
        self.surfaceY = np.clip(surface, 10, self.worldSurface + 12).astype(np.int32)

        # Vectorized layer fill using full-shape boolean masks
        rowIdx = np.broadcast_to(np.arange(h)[:, None], (h, w))
        surfIdx = np.broadcast_to(self.surfaceY[None, :], (h, w))

        self.grid[rowIdx == surfIdx] = GRASS
        self.grid[(rowIdx > surfIdx) & (rowIdx < self.rockLayer)] = DIRT
        self.grid[(rowIdx >= self.rockLayer) & (rowIdx < self.hellLayer)] = STONE
        self.grid[rowIdx >= self.hellLayer] = ASH

        # Scatter hellstone within the hell layer
        hellMask = self.grid == ASH
        hellCoords = np.argwhere(hellMask)
        nHellstone = int(hellCoords.shape[0] * 0.15)
        if nHellstone > 0:
            idxs = self.rng.choice(hellCoords.shape[0], size=nHellstone, replace=False)
            self.grid[hellCoords[idxs, 0], hellCoords[idxs, 1]] = HELLSTONE

    # ------------------------------------------------------------------
    # Phase 2: Cave systems
    # ------------------------------------------------------------------
    def carveCaves(self) -> None:
        """TileRunner random-walk caves + cellularAutomataSmooth."""
        w = self.worldWidth

        # Surface-level caves
        nSurface = max(4, int(80 * self.xScale))
        for _ in range(nSurface):
            cx = int(self.rng.integers(10, w - 10))
            cy = int(self.rng.integers(self.worldSurface + 5, self.rockLayer))
            strength = float(self.rng.uniform(3.0, 6.0))
            steps = int(self.rng.integers(15, 40))
            tileRunner(self.grid, cx, cy, strength, steps, tileType=-1)

        # Deep caves
        nDeep = max(4, int(60 * self.xScale))
        for _ in range(nDeep):
            cx = int(self.rng.integers(10, w - 10))
            cy = int(self.rng.integers(self.rockLayer, self.hellLayer - 5))
            strength = float(self.rng.uniform(5.0, 10.0))
            steps = int(self.rng.integers(20, 60))
            tileRunner(self.grid, cx, cy, strength, steps, tileType=-1)

        # Smooth cave edges (organic rounded look)
        cellularAutomataSmooth(self.grid, iterations=2)

    # ------------------------------------------------------------------
    # Phase 3: Biome painting
    # ------------------------------------------------------------------
    def paintBiomes(self) -> None:
        """Evil biome, jungle, and desert surface/subsurface painting."""
        w = self.worldWidth

        # Evil biome on one side
        evilSide = str(self.rng.choice(["left", "right"]))
        if evilSide == "left":
            self.evilXStart, self.evilXEnd = int(w * 0.15), int(w * 0.30)
        else:
            self.evilXStart, self.evilXEnd = int(w * 0.70), int(w * 0.85)

        if self.evilType == "corruption":
            stoneTile, dirtTile = EBONSTONE, CORRUPT_DIRT
        else:
            stoneTile, dirtTile = CRIMSTONE, CRIMSON_DIRT

        yMin = max(0, self.worldSurface - 5)
        yMax = min(self.worldHeight, self.rockLayer + 20)
        region = self.grid[yMin:yMax, self.evilXStart:self.evilXEnd]
        region[region == STONE] = stoneTile
        region[np.isin(region, [DIRT, GRASS])] = dirtTile

        # Jungle on opposite side
        if evilSide == "left":
            jStart, jEnd = int(w * 0.70), int(w * 0.90)
        else:
            jStart, jEnd = int(w * 0.10), int(w * 0.30)

        jYMin = max(0, self.worldSurface - 2)
        jYMax = min(self.worldHeight, self.hellLayer)
        jungleRegion = self.grid[jYMin:jYMax, jStart:jEnd]
        jungleRegion[np.isin(jungleRegion, [DIRT, GRASS])] = MUD

        # Small desert near center
        dStart, dEnd = int(w * 0.42), int(w * 0.50)
        dYMin = max(0, self.worldSurface - 3)
        dYMax = min(self.worldHeight, self.rockLayer - 10)
        desertRegion = self.grid[dYMin:dYMax, dStart:dEnd]
        desertRegion[np.isin(desertRegion, [DIRT, GRASS])] = SAND

    # ------------------------------------------------------------------
    # Phase 4: Pre-Hardmode placement
    # ------------------------------------------------------------------
    def placePreHardmode(self) -> None:
        """Life Crystals (403 max scaled) + pre-HM ores via TileRunner."""
        w, h = self.worldWidth, self.worldHeight

        # Life Crystals between rockLayer and hellLayer
        stoneCoords = np.argwhere(
            (self.grid == STONE)
            & (np.arange(h)[:, None] >= self.rockLayer)
            & (np.arange(h)[:, None] < self.hellLayer - 5)
        )
        if stoneCoords.shape[0] > 0:
            nCrystals = min(self.lifeCrystalsMax, stoneCoords.shape[0])
            chosen = self.rng.choice(stoneCoords.shape[0], size=nCrystals, replace=False)
            for idx in chosen:
                self.grid[stoneCoords[idx, 0], stoneCoords[idx, 1]] = LIFE_CRYSTAL

        # Pre-HM ores: int(area * 6E-05) TileRunner calls per type
        depthRanges = [
            (self.worldSurface, self.rockLayer),                      # Copper/Tin
            (self.worldSurface, self.hellLayer),                      # Iron/Lead
            (self.rockLayer, self.hellLayer),                         # Silver/Tungsten
            (min(int(self.rockLayer * 1.2), h - 3), self.hellLayer),  # Gold/Platinum
        ]

        for i, oreName in enumerate(self.preHmOres):
            tileId = ORE_NAME_TO_ID[oreName]
            yMin = max(2, depthRanges[i][0])
            yMax = min(h - 2, depthRanges[i][1])
            if yMax <= yMin:
                continue
            for _ in range(self.oreLoopCount):
                ox = int(self.rng.integers(10, w - 10))
                oy = int(self.rng.integers(yMin, yMax))
                strength = float(self.rng.uniform(2.0, 4.0))
                steps = int(self.rng.integers(5, 15))
                tileRunner(self.grid, ox, oy, strength, steps,
                           tileType=tileId, overRide=False)

    # ------------------------------------------------------------------
    # Phase 5: Hardmode V-pattern
    # ------------------------------------------------------------------
    def carveVPattern(self) -> None:
        """TileRunner along 2 diagonal vectors from center surface.

        Left arm: evil tiles. Right arm: hallow tiles.
        Strength ~4, overRide=False so only solid tiles are converted.
        """
        centerX = self.worldWidth // 2
        startY = self.worldSurface
        endY = self.hellLayer
        deltaY = endY - startY

        evilTile = EBONSTONE if self.evilType == "corruption" else CRIMSTONE
        hallowTile = PEARLSTONE

        numSegments = 20
        for i in range(numSegments):
            t = i / numSegments
            yPos = int(startY + t * deltaY)
            xOffset = int(t * deltaY * 0.6)

            # Left arm (evil)
            lx = max(5, centerX - xOffset)
            tileRunner(
                self.grid, lx, yPos,
                strength=4.0, steps=15,
                tileType=evilTile,
                speedX=-0.5, speedY=0.8,
                overRide=False,
            )

            # Right arm (hallow)
            rx = min(self.worldWidth - 5, centerX + xOffset)
            tileRunner(
                self.grid, rx, yPos,
                strength=4.0, steps=15,
                tileType=hallowTile,
                speedX=0.5, speedY=0.8,
                overRide=False,
            )

    # ------------------------------------------------------------------
    # Phase 6: Altar smashing
    # ------------------------------------------------------------------
    def smashAltars(self, numAltars: int = 6) -> None:
        """3-cycle hardmode ores via TileRunner, int(area*6E-05)/cycle."""
        w, h = self.worldWidth, self.worldHeight

        hmDepthRanges = [
            (self.rockLayer, self.hellLayer),                          # Tier 1
            (min(int(self.rockLayer * 1.2), h - 3), self.hellLayer),   # Tier 2
            (min(int(self.rockLayer * 1.5), h - 3), self.hellLayer),   # Tier 3
        ]

        for altar in range(numAltars):
            cycle = (altar % 3) + 1
            tierIdx = cycle - 1
            oreName = self.hmOres[tierIdx]
            tileId = ORE_NAME_TO_ID[oreName]
            yMin = max(2, hmDepthRanges[tierIdx][0])
            yMax = min(h - 2, hmDepthRanges[tierIdx][1])
            if yMax <= yMin:
                continue

            loopsThisCycle = max(1, self.oreLoopCount // cycle)
            for _ in range(loopsThisCycle):
                ox = int(self.rng.integers(10, w - 10))
                oy = int(self.rng.integers(yMin, yMax))
                strength = float(self.rng.uniform(2.0, 3.5))
                steps = int(self.rng.integers(5, 12))
                tileRunner(self.grid, ox, oy, strength, steps,
                           tileType=tileId, overRide=False)

    # ------------------------------------------------------------------
    # Phase 7: Infection spread
    # ------------------------------------------------------------------
    def simulateSpread(self, steps: int = 15) -> list[np.ndarray]:
        """Tile update cycle infection spread with air gap blocking.

        Each step, infected tiles pick a random neighbor within
        INFECTION_SPREAD_RADIUS and convert it if the target is
        convertible and no air gap of INFECTION_GAP_TILES blocks the path.
        Surface tiles spread faster than underground tiles.
        """
        h, w = self.worldHeight, self.worldWidth
        snapshots: list[np.ndarray] = []

        # Select evil infection set
        if self.evilType == "corruption":
            evilTiles, evilConvert = CORRUPTION_TILES, CORRUPTION_CONVERT
        else:
            evilTiles, evilConvert = CRIMSON_TILES, CRIMSON_CONVERT

        infectionGroups = [
            (evilTiles, evilConvert),
            (HALLOW_TILES, HALLOW_CONVERT),
        ]

        # Depth-based probability grid
        # Ratio preserves surface:underground ~ 830:140 ~ 6:1
        surfaceProb = 0.30
        undergroundProb = 0.05
        probGrid = np.full((h, w), undergroundProb, dtype=np.float64)
        probGrid[: self.rockLayer, :] = surfaceProb

        # Carve a quarantine trench to demonstrate air gap blocking
        trenchX = self.worldWidth // 3
        for col in range(trenchX, trenchX + INFECTION_GAP_TILES):
            self.grid[self.worldSurface - 5 : self.hellLayer, col] = AIR

        allInfTiles = frozenset().union(*[g[0] for g in infectionGroups])
        convertible = frozenset().union(*[frozenset(g[1].keys()) for g in infectionGroups])
        radius = INFECTION_SPREAD_RADIUS

        for step in range(steps):
            newGrid = self.grid.copy()

            # Find all infected positions
            infMask = np.isin(self.grid, list(allInfTiles))
            infPos = np.argwhere(infMask)
            if len(infPos) == 0:
                if step == steps // 3 or step == steps - 1:
                    snapshots.append(self.grid.copy())
                continue

            # Sample a subset of infected tiles to update
            nUpdates = min(len(infPos), max(500, len(infPos) // 3))
            chosen = self.rng.choice(len(infPos), size=nUpdates, replace=False)

            for idx in chosen:
                sy, sx = int(infPos[idx, 0]), int(infPos[idx, 1])
                srcTile = self.grid[sy, sx]

                # Determine which infection group this source belongs to
                convertMap = None
                for infTiles, cMap in infectionGroups:
                    if srcTile in infTiles:
                        convertMap = cMap
                        break
                if convertMap is None:
                    continue

                # Probability check (surface vs underground rate)
                if self.rng.random() > probGrid[sy, sx]:
                    continue

                # Pick random neighbor within INFECTION_SPREAD_RADIUS
                dy = int(self.rng.integers(-radius, radius + 1))
                dx = int(self.rng.integers(-radius, radius + 1))
                if dy == 0 and dx == 0:
                    continue
                ny, nx = sy + dy, sx + dx
                if not (0 <= nx < w and 0 <= ny < h):
                    continue

                target = self.grid[ny, nx]
                if target not in convertMap:
                    continue

                # Air gap check along path
                dist = max(abs(dy), abs(dx))
                blocked = False
                if dist > 1:
                    consecutive = 0
                    for i in range(1, dist):
                        t = i / dist
                        cx = int(sx + dx * t)
                        cy = int(sy + dy * t)
                        if 0 <= cx < w and 0 <= cy < h and self.grid[cy, cx] == AIR:
                            consecutive += 1
                            if consecutive >= INFECTION_GAP_TILES:
                                blocked = True
                                break
                        else:
                            consecutive = 0
                if not blocked:
                    newGrid[ny, nx] = convertMap[target]

            self.grid = newGrid

            # Record early and late snapshots
            if step == steps // 3 or step == steps - 1:
                snapshots.append(self.grid.copy())

        return snapshots

    # ------------------------------------------------------------------
    # Run all phases
    # ------------------------------------------------------------------
    def runEvolution(self) -> list[tuple[str, np.ndarray]]:
        """Execute the full 7-phase lifecycle, returning snapshot history."""
        self.buildTerrain()
        self.history.append(("Phase 1: Base Terrain", self.grid.copy()))

        self.carveCaves()
        self.history.append(("Phase 2: Cave Systems", self.grid.copy()))

        self.paintBiomes()
        self.history.append(("Phase 3: Biome Painting", self.grid.copy()))

        self.placePreHardmode()
        self.history.append(("Phase 4: Pre-Hardmode", self.grid.copy()))

        self.carveVPattern()
        self.history.append(("Phase 5: V-Pattern", self.grid.copy()))

        self.smashAltars()
        self.history.append(("Phase 6: Altar Smashing", self.grid.copy()))

        spreadSnapshots = self.simulateSpread(steps=15)
        if len(spreadSnapshots) >= 2:
            self.history.append(("Phase 7a: Early Spread", spreadSnapshots[0]))
            self.history.append(("Phase 7b: Late Spread", spreadSnapshots[1]))
        else:
            self.history.append(("Phase 7: Infection Spread", self.grid.copy()))
            self.history.append(("Final State", self.grid.copy()))

        return self.history


# ===================================================================
# Visualization
# ===================================================================
def plotEvolution(
    history: list[tuple[str, np.ndarray]],
    evolution: TerrariaCompleteWorldEvolution,
) -> None:
    """Render 2x4 multi-panel evolution grid and save as PNG."""
    from Engine.spriteRenderer import applyMapDecorations, drawTileGrid
    from Engine.constants import LayerDepths

    w = evolution.worldWidth
    h = evolution.worldHeight
    layers = LayerDepths(
        worldSurface=float(evolution.worldSurface),
        rockLayer=float(evolution.rockLayer),
        hellLayer=int(evolution.hellLayer),
        maxTilesY=h,
    )
    cropBounds = (0, w, 0, h)

    from Engine.theme import COLORS, applyTokyoNight
    applyTokyoNight()

    fig, axes = plt.subplots(2, 4, figsize=(28, 8), facecolor=COLORS["bg"])

    for idx in range(min(8, len(history))):
        label, grid = history[idx]
        row, col = divmod(idx, 4)
        ax = axes[row, col]
        drawTileGrid(ax, grid)
        applyMapDecorations(ax, grid, layers, cropBounds=cropBounds,
                            grassBand=True, hellstoneBand=False, layerMarkers=True)
        ax.set_title(label, fontsize=11, fontweight="bold", pad=6)
        ax.set_facecolor(COLORS["bg"])

    plt.suptitle(
        "Terraria Complete World Evolution | 1/10 Scale (840x240)",
        fontsize=14, fontweight="bold", y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    outputDir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Plots", "Advanced",
    )
    os.makedirs(outputDir, exist_ok=True)
    outputPath = os.path.join(outputDir, "terraria_complete_world_evolution.png")
    plt.savefig(outputPath, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"Saved: {outputPath}")


# ===================================================================
# Entry point
# ===================================================================
if __name__ == "__main__":
    print("Terraria Complete World Evolution")
    print("=" * 40)
    print(f"World: {LARGE.width}x{LARGE.height} large, rendered at 1/10 scale")
    layers = LayerDepths.forLarge()
    print(f"Layers: worldSurface={layers.worldSurface}, "
          f"rockLayer={layers.rockLayer}, hellLayer={layers.hellLayer}")
    print(f"Ore loops per type: {OreConfig.loopCount(840 * 240)} "
          f"(area={840 * 240}, factor={OreConfig.DENSITY_FACTOR})")
    print(f"Life Crystals max (scaled): "
          f"{max(1, int(StructureQuotas.forLarge().lifeCrystalsMax * (840*240) / LARGE.area))}")
    print(f"Infection gap: {INFECTION_GAP_TILES} tiles")
    print(f"Spread rates: surface ~{SURFACE_UPDATE_RATE}s, "
          f"underground ~{UNDERGROUND_UPDATE_RATE}s")
    print()

    evo = TerrariaCompleteWorldEvolution(worldWidth=840, worldHeight=240, seed=42)
    history = evo.runEvolution()

    print(f"Phases captured: {len(history)}")
    for label, grid in history:
        print(f"  {label}: {grid.shape}")

    plotEvolution(history, evo)
    print("Done.")
