"""
Terraria World Generation - 19-Pass Pipeline
=============================================

Implements a faithful subset of Terraria's 103-pass world generation system
using diamond-brush TileRunner, cellular automata smoothing, and proper
layer depth calculations derived from decompiled WorldGen.cs.

All algorithms imported from Engine.algorithms; constants from Engine.constants.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')

from Engine.algorithms import (
    tileRunner, digTunnel, cellularAutomataSmooth, settleLiquids,
    AIR, DIRT, STONE, GRASS, SAND, ASH, MUD, SNOW, ICE,
    HELLSTONE, LAVA, WATER, COPPER, IRON, SILVER, GOLD,
    EBONSTONE, CORRUPT_DIRT,
)
from Engine.constants import LARGE, LayerDepths, StructureQuotas, OreConfig
from Engine.structureMap import StructureMap, Rectangle

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Tile IDs not in Engine (used locally for visualization grouping)
_DUNGEON_BRICK = 200
_LIFE_CRYSTAL = 201

# Ordered color map: index = tile ID -> hex color
TILE_COLORS = {
    AIR: '#87CEEB',
    DIRT: '#8B4513',
    STONE: '#696969',
    GRASS: '#90EE90',
    SAND: '#F4A460',
    ASH: '#4A4A4A',
    HELLSTONE: '#FF4500',
    MUD: '#6B4226',
    SNOW: '#E8E8F0',
    ICE: '#ADD8E6',
    WATER: '#1E90FF',
    LAVA: '#FF2400',
    COPPER: '#B87333',
    IRON: '#A9A9A9',
    SILVER: '#C0C0C0',
    GOLD: '#FFD700',
    EBONSTONE: '#9370DB',
    CORRUPT_DIRT: '#7B68AE',
    _DUNGEON_BRICK: '#2F4F4F',
    _LIFE_CRYSTAL: '#FF69B4',
}


def _buildColormap() -> ListedColormap:
    """Build a ListedColormap covering all tile IDs up to max used."""
    maxId = max(TILE_COLORS.keys()) + 1
    colors = []
    defaultColor = '#000000'
    for i in range(maxId):
        hexColor = TILE_COLORS.get(i, defaultColor)
        colors.append(hexColor)
    return ListedColormap(colors)


TERRAIN_CMAP = _buildColormap()


class TerrariaWorldGenerator:
    """Simulates Terraria's world generation using a 19-pass pipeline.

    Supports full-size (8400x2400) and reduced-resolution modes.
    Layer depths, cave counts, and ore density scale proportionally.
    """

    PASS_LIST = [
        "Reset",
        "Terrain",
        "Stone Layer",
        "Sand Patches",
        "Surface Caves",
        "Dirt Layer Caves",
        "Rock Layer Caves",
        "Smooth World",
        "Snow Biome",
        "Jungle",
        "Corruption",
        "Floating Islands",
        "Underworld",
        "Shinies",
        "Dungeon",
        "Settle Liquids",
        "Life Crystals",
        "Grass",
        "Border Buffer",
    ]

    def __init__(self, worldWidth: int = 8400, worldHeight: int = 2400, seed: int = 12345):
        self.worldWidth = worldWidth
        self.worldHeight = worldHeight
        self.seed = seed
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)

        # Scale factor relative to Large world
        self.scaleX = worldWidth / LARGE.width
        self.scaleY = worldHeight / LARGE.height
        self.areaScale = self.scaleX * self.scaleY

        # Layer depths (scaled from Large defaults)
        refLayers = LayerDepths.forLarge()
        self.worldSurface = int(refLayers.worldSurface * self.scaleY)
        self.rockLayer = int(refLayers.rockLayer * self.scaleY)
        self.hellLayer = int((refLayers.maxTilesY - 200) * self.scaleY)

        # Structure quotas (full-size; passes scale counts internally)
        self.quotas = StructureQuotas.forLarge()
        self.structureMap = StructureMap()
        self.borderBuffer = max(1, int(LARGE.borderBuffer * min(self.scaleX, self.scaleY)))

        # Primary grid
        self.grid = np.zeros((worldHeight, worldWidth), dtype=np.int32)
        self.passLog: list[str] = []
        self.snapshots: list[tuple[str, np.ndarray]] = []

        # Biome layout state (set during generation)
        self._dungeonLeft: bool = True
        self._evilCenter: int = 0

    # ------------------------------------------------------------------
    # Backward-compat: master_evolution.py reads .world
    # ------------------------------------------------------------------
    @property
    def world(self) -> np.ndarray:
        return self.grid

    @world.setter
    def world(self, value: np.ndarray) -> None:
        self.grid = value

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def generate(self) -> np.ndarray:
        """Run all 19 passes in order and return the completed grid."""
        passes = [
            ("Reset", self._passReset),
            ("Terrain", self._passTerrain),
            ("Stone Layer", self._passStoneLayer),
            ("Sand Patches", self._passSandPatches),
            ("Surface Caves", self._passSurfaceCaves),
            ("Dirt Layer Caves", self._passDirtLayerCaves),
            ("Rock Layer Caves", self._passRockLayerCaves),
            ("Smooth World", self._passSmoothWorld),
            ("Snow Biome", self._passSnowBiome),
            ("Jungle", self._passJungle),
            ("Corruption", self._passCorruption),
            ("Floating Islands", self._passFloatingIslands),
            ("Underworld", self._passUnderworld),
            ("Shinies", self._passShinies),
            ("Dungeon", self._passDungeon),
            ("Settle Liquids", self._passSettleLiquids),
            ("Life Crystals", self._passLifeCrystals),
            ("Grass", self._passGrass),
            ("Border Buffer", self._passBorderBuffer),
        ]
        for name, fn in passes:
            fn()
            self.passLog.append(name)
            self.snapshots.append((name, self.grid.copy()))
        return self.grid

    # ------------------------------------------------------------------
    # Backward-compatible API for terraria_master_evolution.py
    # ------------------------------------------------------------------
    def generate_surface_terrain(self) -> None:
        """Legacy: run Reset + Terrain + Stone Layer + Sand Patches."""
        for fn in [self._passReset, self._passTerrain, self._passStoneLayer, self._passSandPatches]:
            fn()

    def carve_caves(self) -> None:
        """Legacy: run cave passes + smoothing."""
        for fn in [self._passSurfaceCaves, self._passDirtLayerCaves,
                    self._passRockLayerCaves, self._passSmoothWorld]:
            fn()

    def place_biomes(self) -> None:
        """Legacy: run biome passes."""
        for fn in [self._passSnowBiome, self._passJungle, self._passCorruption,
                    self._passFloatingIslands]:
            fn()

    def place_structures(self) -> None:
        """Legacy: run structure and finishing passes."""
        for fn in [self._passUnderworld, self._passShinies, self._passDungeon,
                    self._passSettleLiquids, self._passLifeCrystals,
                    self._passGrass, self._passBorderBuffer]:
            fn()

    # ------------------------------------------------------------------
    # Helper: 1D surface heightmap via multi-octave sine noise
    # ------------------------------------------------------------------
    def _surfaceHeight(self, x: int) -> int:
        """Return surface Y for column x using layered sine noise."""
        base = self.worldSurface
        h = 0.0
        h += 30.0 * self.scaleY * np.sin(x * 0.005 + self.seed * 0.1)
        h += 15.0 * self.scaleY * np.sin(x * 0.012 + self.seed * 0.3)
        h += 7.0 * self.scaleY * np.sin(x * 0.03 + self.seed * 0.7)
        return int(np.clip(base + h, 10, self.worldSurface + 50 * self.scaleY))

    # ------------------------------------------------------------------
    # Pass implementations
    # ------------------------------------------------------------------
    def _passReset(self) -> None:
        """Pass 0: Clear the grid to AIR."""
        self.grid[:] = AIR
        self.structureMap.clear()

    def _passTerrain(self) -> None:
        """Pass 1: Generate surface heightmap and fill dirt below."""
        for x in range(self.worldWidth):
            surfaceY = self._surfaceHeight(x)
            if surfaceY < self.worldHeight:
                self.grid[surfaceY:, x] = DIRT

    def _passStoneLayer(self) -> None:
        """Pass 2: Fill stone below the dirt-to-stone transition."""
        for x in range(self.worldWidth):
            surfaceY = self._surfaceHeight(x)
            # Stone starts ~50 tiles below surface (scaled), blending into rockLayer
            stoneStart = surfaceY + max(1, int(50 * self.scaleY))
            stoneStart = min(stoneStart, self.worldHeight - 1)
            mask = self.grid[stoneStart:, x] == DIRT
            self.grid[stoneStart:, x][mask] = STONE

    def _passSandPatches(self) -> None:
        """Pass 3: Place sand patches on the surface (desert/beach areas)."""
        # Beaches: leftmost and rightmost 5% of world
        beachWidth = max(10, int(self.worldWidth * 0.05))
        for x in list(range(beachWidth)) + list(range(self.worldWidth - beachWidth, self.worldWidth)):
            surfaceY = self._surfaceHeight(x)
            depth = max(2, int(20 * self.scaleY))
            yEnd = min(surfaceY + depth, self.worldHeight)
            mask = self.grid[surfaceY:yEnd, x] != AIR
            self.grid[surfaceY:yEnd, x][mask] = SAND

        # Desert patch (random location in middle third)
        desertCenter = self.rng.integers(self.worldWidth // 3, 2 * self.worldWidth // 3)
        desertHalf = max(20, int(150 * self.scaleX))
        desertDepth = max(5, int(60 * self.scaleY))
        for x in range(max(0, desertCenter - desertHalf), min(self.worldWidth, desertCenter + desertHalf)):
            surfaceY = self._surfaceHeight(x)
            yEnd = min(surfaceY + desertDepth, self.worldHeight)
            mask = self.grid[surfaceY:yEnd, x] != AIR
            self.grid[surfaceY:yEnd, x][mask] = SAND

    def _passSurfaceCaves(self) -> None:
        """Pass 4: Small caves near the surface using TileRunner."""
        count = max(5, int(self.worldWidth * self.worldHeight * 1.5e-05))
        for _ in range(count):
            sx = self.rng.integers(0, self.worldWidth)
            sy = self.rng.integers(self.worldSurface, min(self.worldSurface + int(100 * self.scaleY), self.rockLayer))
            strength = self.rng.uniform(4, 8) * min(self.scaleX, self.scaleY)
            steps = self.rng.integers(10, 30)
            tileRunner(self.grid, int(sx), int(sy), float(strength), int(steps), tileType=-1)

    def _passDirtLayerCaves(self) -> None:
        """Pass 5: Medium caves in the dirt/transition layer."""
        count = max(10, int(self.worldWidth * self.worldHeight * 3e-05))
        for _ in range(count):
            sx = self.rng.integers(0, self.worldWidth)
            sy = self.rng.integers(self.worldSurface, self.rockLayer)
            strength = self.rng.uniform(6, 14) * min(self.scaleX, self.scaleY)
            steps = self.rng.integers(20, 60)
            tileRunner(self.grid, int(sx), int(sy), float(strength), int(steps), tileType=-1)

    def _passRockLayerCaves(self) -> None:
        """Pass 6: Large caves in the rock/cavern layer."""
        count = max(10, int(self.worldWidth * self.worldHeight * 4e-05))
        for _ in range(count):
            sx = self.rng.integers(0, self.worldWidth)
            sy = self.rng.integers(self.rockLayer, self.hellLayer)
            strength = self.rng.uniform(10, 22) * min(self.scaleX, self.scaleY)
            steps = self.rng.integers(30, 100)
            tileRunner(self.grid, int(sx), int(sy), float(strength), int(steps), tileType=-1)

    def _passSmoothWorld(self) -> None:
        """Pass 7: Cellular automata smoothing for organic cave edges."""
        cellularAutomataSmooth(self.grid, iterations=3, birthThreshold=5, deathThreshold=3)

    def _passSnowBiome(self) -> None:
        """Pass 8: Convert a lateral section to snow/ice biome."""
        self._dungeonLeft = bool(self.rng.integers(0, 2))
        if self._dungeonLeft:
            snowStart = 0
            snowEnd = max(1, int(self.worldWidth * 0.18))
        else:
            snowStart = int(self.worldWidth * 0.82)
            snowEnd = self.worldWidth

        for x in range(snowStart, snowEnd):
            surfaceY = self._surfaceHeight(x)
            for y in range(surfaceY, self.hellLayer):
                tile = self.grid[y, x]
                if tile == DIRT:
                    self.grid[y, x] = SNOW
                elif tile == STONE and y > self.rockLayer:
                    self.grid[y, x] = ICE

    def _passJungle(self) -> None:
        """Pass 9: Convert opposite side to jungle (dirt->mud)."""
        if self._dungeonLeft:
            jungleStart = int(self.worldWidth * 0.72)
            jungleEnd = self.worldWidth
        else:
            jungleStart = 0
            jungleEnd = max(1, int(self.worldWidth * 0.28))

        for x in range(jungleStart, jungleEnd):
            surfaceY = self._surfaceHeight(x)
            for y in range(surfaceY, self.hellLayer):
                if self.grid[y, x] == DIRT:
                    self.grid[y, x] = MUD

    def _passCorruption(self) -> None:
        """Pass 10: Evil biome using TileRunner chasms (Corruption)."""
        # Place evil in the middle-ish area, opposite side from jungle
        if self._dungeonLeft:
            self._evilCenter = self.rng.integers(int(self.worldWidth * 0.30), int(self.worldWidth * 0.55))
        else:
            self._evilCenter = self.rng.integers(int(self.worldWidth * 0.45), int(self.worldWidth * 0.70))

        evilHalf = max(20, int(100 * self.scaleX))

        # TileRunner conversion passes (replace stone/dirt with ebonstone/corrupt_dirt)
        conversionRuns = max(5, int(30 * self.areaScale))
        for _ in range(conversionRuns):
            sx = self.rng.integers(max(0, self._evilCenter - evilHalf),
                                   min(self.worldWidth, self._evilCenter + evilHalf))
            sy = self.rng.integers(self.worldSurface, self.rockLayer)
            tileRunner(self.grid, int(sx), int(sy), strength=8.0 * min(self.scaleX, self.scaleY),
                       steps=40, tileType=EBONSTONE, overRide=False)

        # Carve 3-6 vertical chasms via TileRunner
        chasmCount = self.rng.integers(3, 7)
        for _ in range(chasmCount):
            cx = self.rng.integers(max(0, self._evilCenter - evilHalf),
                                   min(self.worldWidth, self._evilCenter + evilHalf))
            surfaceY = self._surfaceHeight(cx)
            tileRunner(self.grid, cx, surfaceY,
                       strength=6.0 * min(self.scaleX, self.scaleY),
                       steps=int(80 * self.scaleY),
                       tileType=-1, speedX=0.0, speedY=1.5)

    def _passFloatingIslands(self) -> None:
        """Pass 11: Place floating islands above the surface."""
        count = max(1, int(self.quotas.floatingIslands * self.areaScale))
        minY = max(5, int(50 * self.scaleY))
        maxY = max(minY + 5, self.worldSurface - int(30 * self.scaleY))
        spacing = self.worldWidth // (count + 1)

        for i in range(count):
            ix = spacing * (i + 1) + self.rng.integers(-spacing // 4, spacing // 4)
            ix = int(np.clip(ix, self.borderBuffer, self.worldWidth - self.borderBuffer))
            iy = self.rng.integers(minY, maxY)
            islandW = max(10, int(self.rng.integers(60, 100) * self.scaleX))
            islandH = max(4, int(self.rng.integers(15, 25) * self.scaleY))

            rect = Rectangle(ix - islandW // 2, iy, islandW, islandH)
            if not self.structureMap.canPlace(rect, padding=5):
                continue
            self.structureMap.addProtectedStructure(rect, padding=5)

            # Fill island body
            for dx in range(islandW):
                for dy in range(islandH):
                    wx = ix - islandW // 2 + dx
                    wy = iy + dy
                    if 0 <= wx < self.worldWidth and 0 <= wy < self.worldHeight:
                        # Ellipse check for rounded shape
                        rx = (dx - islandW / 2) / (islandW / 2)
                        ry = (dy - islandH / 2) / (islandH / 2)
                        if rx * rx + ry * ry <= 1.0:
                            self.grid[wy, wx] = DIRT if dy < islandH // 3 else STONE

    def _passUnderworld(self) -> None:
        """Pass 12: Fill hell layer with ash and lava."""
        for x in range(self.worldWidth):
            for y in range(self.hellLayer, self.worldHeight):
                if self.grid[y, x] == AIR:
                    # Lava pools in the lower half of hell
                    if y > self.hellLayer + (self.worldHeight - self.hellLayer) // 2:
                        self.grid[y, x] = LAVA
                else:
                    self.grid[y, x] = ASH

        # Scatter hellstone via TileRunner
        hellRuns = max(3, int(20 * self.areaScale))
        hellYMin = self.hellLayer + 2
        hellYMax = self.worldHeight - 2
        if hellYMin < hellYMax:
            for _ in range(hellRuns):
                sx = self.rng.integers(self.borderBuffer, self.worldWidth - self.borderBuffer)
                sy = self.rng.integers(hellYMin, hellYMax)
                tileRunner(self.grid, int(sx), int(sy), strength=5.0, steps=15, tileType=HELLSTONE)

    def _passShinies(self) -> None:
        """Pass 13: Ore generation using TileRunner (area-proportional density)."""
        area = self.worldWidth * self.worldHeight
        loopCount = max(1, int(area * OreConfig.DENSITY_FACTOR))

        oreSpecs = [
            (COPPER, self.worldSurface, self.rockLayer, 4.0, 15),
            (IRON, self.worldSurface, self.rockLayer, 3.5, 14),
            (SILVER, self.rockLayer, self.hellLayer, 3.0, 12),
            (GOLD, self.rockLayer, self.hellLayer, 2.5, 10),
        ]

        for oreType, yMin, yMax, strength, steps in oreSpecs:
            yMax = max(yMin + 1, yMax)
            runs = max(1, loopCount // 4)
            for _ in range(runs):
                sx = self.rng.integers(self.borderBuffer, self.worldWidth - self.borderBuffer)
                sy = self.rng.integers(yMin, yMax)
                tileRunner(self.grid, int(sx), int(sy), strength=strength, steps=steps,
                           tileType=oreType, overRide=False)

    def _passDungeon(self) -> None:
        """Pass 14: Simple dungeon using digTunnel for rooms/corridors."""
        if self._dungeonLeft:
            dungeonX = max(self.borderBuffer + 10, int(80 * self.scaleX))
        else:
            dungeonX = min(self.worldWidth - self.borderBuffer - 10,
                           self.worldWidth - int(80 * self.scaleX))

        surfaceY = self._surfaceHeight(dungeonX)
        dungeonW = max(15, int(60 * self.scaleX))
        dungeonH = max(30, int(120 * self.scaleY))

        rect = Rectangle(dungeonX - dungeonW // 2, surfaceY, dungeonW, dungeonH)
        if not self.structureMap.canPlace(rect, padding=10):
            # Force-place if nothing else fits
            pass
        self.structureMap.addProtectedStructure(rect, padding=10)

        # Fill dungeon shell with dungeon brick
        x0 = max(0, dungeonX - dungeonW // 2)
        x1 = min(self.worldWidth, dungeonX + dungeonW // 2)
        y0 = surfaceY
        y1 = min(self.worldHeight, surfaceY + dungeonH)
        self.grid[y0:y1, x0:x1] = _DUNGEON_BRICK

        # Carve rooms using digTunnel (eating algorithm)
        roomCount = max(3, int(8 * self.areaScale))
        for _ in range(roomCount):
            rx = self.rng.integers(x0 + 2, max(x0 + 3, x1 - 2))
            ry = self.rng.integers(y0 + 2, max(y0 + 3, y1 - 2))
            digTunnel(self.grid, float(rx), float(ry),
                      xDir=self.rng.uniform(-0.5, 0.5),
                      yDir=self.rng.uniform(0.3, 1.0),
                      steps=max(3, int(15 * self.scaleY)),
                      size=max(2, int(4 * min(self.scaleX, self.scaleY))))

    def _passSettleLiquids(self) -> None:
        """Pass 15: Settle water and lava via gravity."""
        settleLiquids(self.grid, maxPasses=10)

    def _passLifeCrystals(self) -> None:
        """Pass 16: Scatter life crystals in cavern layer."""
        maxCrystals = max(1, int(self.quotas.lifeCrystalsMax * self.areaScale))
        placed = 0
        attempts = maxCrystals * 10
        crystalYMin = self.rockLayer
        crystalYMax = max(self.rockLayer + 1, self.hellLayer)
        for _ in range(attempts):
            if placed >= maxCrystals:
                break
            cx = self.rng.integers(self.borderBuffer, self.worldWidth - self.borderBuffer)
            cy = self.rng.integers(crystalYMin, crystalYMax)
            # Place only in air with solid below
            if (0 <= cy < self.worldHeight - 1
                    and self.grid[cy, cx] == AIR
                    and self.grid[cy + 1, cx] not in (AIR, WATER, LAVA)):
                self.grid[cy, cx] = _LIFE_CRYSTAL
                placed += 1

    def _passGrass(self) -> None:
        """Pass 17: Convert surface dirt tiles exposed to air into grass."""
        for x in range(self.worldWidth):
            for y in range(max(0, self.worldSurface - int(60 * self.scaleY)),
                           min(self.worldHeight - 1, self.worldSurface + int(80 * self.scaleY))):
                if self.grid[y, x] != DIRT:
                    continue
                # Check if any neighbor is AIR
                hasAir = False
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < self.worldHeight and 0 <= nx < self.worldWidth:
                            if self.grid[ny, nx] == AIR:
                                hasAir = True
                                break
                    if hasAir:
                        break
                if hasAir:
                    self.grid[y, x] = GRASS

    def _passBorderBuffer(self) -> None:
        """Pass 18: Fill border edges with impassable stone."""
        b = self.borderBuffer
        self.grid[:, :b] = STONE
        self.grid[:, -b:] = STONE
        self.grid[:b, :] = STONE
        self.grid[-b:, :] = STONE


# ======================================================================
# Visualization
# ======================================================================

def _savePath(filename: str) -> str:
    """Return full path under Plots/Code+/, creating directory if needed."""
    baseDir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Plots", "Advanced")
    os.makedirs(baseDir, exist_ok=True)
    return os.path.join(baseDir, filename)


def _renderGrid(ax, grid: np.ndarray, title: str, maxId: int) -> None:
    """Render a grid onto a matplotlib axes."""
    ax.imshow(grid, cmap=TERRAIN_CMAP, aspect='auto', vmin=0, vmax=maxId, interpolation='nearest')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('X (blocks)')
    ax.set_ylabel('Y (blocks)')


def createWorldGenerationAnimation(saveName: str = "world_generation_animation.gif") -> None:
    """Create a GIF animation stepping through all 19 generation passes.

    Uses 840x240 (1/10 scale of Large) for performance. Labels indicate
    this is a scaled visualization, not a real Terraria world size.
    """
    print("Generating world at 1/10 scale for animation...")
    gen = TerrariaWorldGenerator(worldWidth=840, worldHeight=240, seed=12345)
    gen.generate()
    snapshots = gen.snapshots

    maxId = max(TILE_COLORS.keys())
    fig, ax = plt.subplots(figsize=(16, 5))

    def animate(frame: int):
        ax.clear()
        idx = frame % len(snapshots)
        name, grid = snapshots[idx]
        _renderGrid(ax, grid, f"Pass {idx}: {name}  (1/10 scale visualization)", maxId)

    anim = animation.FuncAnimation(fig, animate, frames=len(snapshots), interval=1500, repeat=True)
    path = _savePath(saveName)
    print(f"Saving animation to {path}")
    anim.save(path, writer='pillow', fps=1, dpi=100)
    plt.close(fig)
    print("Animation saved.")


def createGenerationStagesPlot(saveName: str = "world_generation_stages.png") -> None:
    """Static plot showing key generation milestones (4 panels)."""
    print("Generating world at 1/10 scale for stage plot...")
    gen = TerrariaWorldGenerator(worldWidth=840, worldHeight=240, seed=12345)
    gen.generate()

    keyIndices = [1, 7, 10, len(gen.snapshots) - 1]  # Terrain, Smooth, Corruption, Final
    keyIndices = [min(i, len(gen.snapshots) - 1) for i in keyIndices]
    maxId = max(TILE_COLORS.keys())

    fig, axes = plt.subplots(2, 2, figsize=(18, 8))
    fig.suptitle("Terraria World Generation Stages (1/10 scale)", fontsize=16, fontweight='bold')

    for ax, idx in zip(axes.flat, keyIndices):
        name, grid = gen.snapshots[idx]
        _renderGrid(ax, grid, f"After: {name}", maxId)

    plt.tight_layout()
    path = _savePath(saveName)
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Stage plot saved to {path}")


def createFullPassGrid(saveName: str = "world_generation_all_passes.png") -> None:
    """Grid showing every pass as a thumbnail."""
    print("Generating world at 1/10 scale for full pass grid...")
    gen = TerrariaWorldGenerator(worldWidth=840, worldHeight=240, seed=12345)
    gen.generate()

    n = len(gen.snapshots)
    cols = 4
    rows = (n + cols - 1) // cols
    maxId = max(TILE_COLORS.keys())

    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 3))
    fig.suptitle("All 19 Generation Passes (1/10 scale)", fontsize=16, fontweight='bold')

    for idx, ax in enumerate(axes.flat):
        if idx < n:
            name, grid = gen.snapshots[idx]
            _renderGrid(ax, grid, f"{idx}: {name}", maxId)
        else:
            ax.axis('off')

    plt.tight_layout()
    path = _savePath(saveName)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Full pass grid saved to {path}")


if __name__ == "__main__":
    createGenerationStagesPlot()
    createFullPassGrid()
    createWorldGenerationAnimation()
