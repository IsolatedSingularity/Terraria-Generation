"""
Terraria corruption/crimson/hallow evolution simulation.

Models pre-hardmode evil biome placement, hardmode V-pattern creation via
TileRunner, and tile update cycle biome spread with air gap infection blocking.
All tile conversion rules match decompiled WorldGen.cs behavior.
"""

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from Engine.algorithms import (
    AIR,
    CORRUPT_DIRT,
    CORRUPT_ICE,
    CRIMSON_DIRT,
    CRIMSON_ICE,
    CRIMSTONE,
    DIRT,
    EBONSTONE,
    GRASS,
    HALLOW_DIRT,
    HALLOW_ICE,
    ICE,
    MUD,
    PEARLSAND,
    PEARLSTONE,
    SAND,
    SNOW,
    STONE,
    tileRunner,
)
from Engine.constants import (
    INFECTION_GAP_TILES,
    INFECTION_SPREAD_RADIUS,
    SURFACE_UPDATE_RATE,
    UNDERGROUND_UPDATE_RATE,
    LayerDepths,
)
from Engine.theme import COLORS, applyTokyoNight

applyTokyoNight()


# ---------------------------------------------------------------------------
# Tile conversion rules by infection type
# ---------------------------------------------------------------------------
CORRUPTION_CONVERSIONS: dict[int, int] = {
    DIRT: CORRUPT_DIRT, STONE: EBONSTONE, ICE: CORRUPT_ICE, GRASS: CORRUPT_DIRT,
}
CRIMSON_CONVERSIONS: dict[int, int] = {
    DIRT: CRIMSON_DIRT, STONE: CRIMSTONE, ICE: CRIMSON_ICE, GRASS: CRIMSON_DIRT,
}
HALLOW_CONVERSIONS: dict[int, int] = {
    STONE: PEARLSTONE, DIRT: HALLOW_DIRT, SAND: PEARLSAND,
    ICE: HALLOW_ICE, GRASS: HALLOW_DIRT,
}

CORRUPTION_TILES = frozenset({CORRUPT_DIRT, EBONSTONE, CORRUPT_ICE})
CRIMSON_TILES = frozenset({CRIMSON_DIRT, CRIMSTONE, CRIMSON_ICE})
HALLOW_TILES = frozenset({PEARLSTONE, HALLOW_DIRT, PEARLSAND, HALLOW_ICE})
ALL_INFECTED_TILES = CORRUPTION_TILES | CRIMSON_TILES | HALLOW_TILES
CONVERTIBLE_TILES = frozenset({DIRT, STONE, ICE, SAND, GRASS})


def _getConversions(infectionType: str) -> dict[int, int]:
    """Return tile conversion dict for the given infection category."""
    if infectionType == "corruption":
        return CORRUPTION_CONVERSIONS
    if infectionType == "crimson":
        return CRIMSON_CONVERSIONS
    if infectionType == "hallow":
        return HALLOW_CONVERSIONS
    return {}


# Shared color palette
TILE_COLORS: dict[int, tuple[float, float, float]] = {
    AIR: (0.08, 0.08, 0.12),
    DIRT: (0.45, 0.32, 0.22),
    STONE: (0.50, 0.50, 0.50),
    GRASS: (0.20, 0.70, 0.20),
    SAND: (0.90, 0.80, 0.50),
    ICE: (0.70, 0.85, 1.00),
    MUD: (0.35, 0.25, 0.15),
    SNOW: (0.88, 0.90, 0.95),
    CORRUPT_DIRT: (0.40, 0.00, 0.60),
    EBONSTONE: (0.30, 0.00, 0.45),
    CORRUPT_ICE: (0.50, 0.20, 0.70),
    CRIMSON_DIRT: (0.70, 0.05, 0.15),
    CRIMSTONE: (0.60, 0.00, 0.10),
    CRIMSON_ICE: (0.80, 0.20, 0.30),
    PEARLSTONE: (0.95, 0.85, 1.00),
    HALLOW_DIRT: (0.90, 0.80, 0.95),
    PEARLSAND: (1.00, 0.95, 0.80),
    HALLOW_ICE: (0.85, 0.90, 1.00),
}


def _gridToRgb(grid: np.ndarray) -> np.ndarray:
    """Convert tile-ID grid to (H, W, 3) float32 RGB image."""
    h, w = grid.shape
    rgb = np.full((h, w, 3), TILE_COLORS[AIR], dtype=np.float32)
    for tileId, color in TILE_COLORS.items():
        mask = grid == tileId
        if np.any(mask):
            rgb[mask] = color
    return rgb


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class TerrariaCorruptionEvolution:
    """Simulates corruption/crimson/hallow placement and spread.

    Implements pre-hardmode evil pockets (TileRunner), hardmode V-pattern
    (TileRunner along diagonals), tile-update-cycle biome spread with
    air gap infection blocking, and tile-type-specific conversion rules.
    """

    def __init__(
        self,
        worldWidth: int = 8400,
        worldHeight: int = 2400,
        evilType: str = "corruption",
        seed: int = 12345,
    ):
        self.worldWidth = worldWidth
        self.worldHeight = worldHeight
        self.evilType = evilType
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Layer depths scaled to actual world size
        if worldHeight >= 2400:
            base = LayerDepths.forLarge()
        elif worldHeight >= 1800:
            base = LayerDepths.forMedium()
        else:
            base = LayerDepths.forSmall()
        scale = worldHeight / base.maxTilesY
        self.layers = LayerDepths(
            worldSurface=base.worldSurface * scale,
            rockLayer=base.rockLayer * scale,
            hellLayer=int(worldHeight - 200 * scale),
            maxTilesY=worldHeight,
        )

        self.grid = np.zeros((worldHeight, worldWidth), dtype=np.int32)

        # Backwards-compat aliases (terraria_master_evolution.py reads these)
        self.EMPTY = AIR
        self.DIRT = DIRT
        self.STONE = STONE
        self.GRASS = GRASS
        self.SAND = SAND
        self.SNOW = SNOW
        self.JUNGLE = MUD
        self.MUD = MUD
        self.ICE = ICE
        self.CORRUPTION = CORRUPT_DIRT
        self.CRIMSON = CRIMSON_DIRT
        self.HALLOW = HALLOW_DIRT

        self.corruptionHistory: list[int] = []
        self.corruption_history = self.corruptionHistory  # alias
        self.world = self.grid  # alias
        self.tileColors = TILE_COLORS

    # ------------------------------------------------------------------
    # World setup
    # ------------------------------------------------------------------
    def initializeWorld(self) -> None:
        """Set up base terrain with layer strata, surface grass, biome strips."""
        w, h = self.worldWidth, self.worldHeight
        surface = int(self.layers.worldSurface)
        rock = int(self.layers.rockLayer)
        nScale = h / 2400.0  # proportional noise/depth scaling

        # Multi-octave surface profile
        xNorm = np.linspace(0, 12 * np.pi, w)
        surfaceProfile = (
            surface
            + (30 * nScale) * np.sin(xNorm * 0.3)
            + (15 * nScale) * np.sin(xNorm * 0.8 + 1.2)
            + (8 * nScale) * np.sin(xNorm * 1.7 + 2.5)
            + (4 * nScale) * np.sin(xNorm * 3.5 + 4.0)
        ).astype(int)
        surfaceProfile = np.clip(surfaceProfile, 1, h - 2)

        # Vectorized strata fill
        rowIdx = np.arange(h)[:, None]
        surfaceLine = surfaceProfile[None, :]
        self.grid[:] = AIR
        belowSurface = rowIdx >= surfaceLine
        self.grid[belowSurface] = DIRT
        self.grid[belowSurface & (rowIdx >= rock)] = STONE

        # Grass at surface
        cols = np.arange(w)
        self.grid[surfaceProfile, cols] = GRASS

        # Biome strips with scaled depth
        self._fillBiomeStrip(surfaceProfile, int(w * 0.05), int(w * 0.18),
                             ICE, max(4, int(80 * nScale)), {DIRT})
        self._fillBiomeStrip(surfaceProfile, int(w * 0.72), int(w * 0.88),
                             MUD, max(4, int(100 * nScale)), {DIRT})
        self._fillBiomeStrip(surfaceProfile, int(w * 0.40), int(w * 0.48),
                             SAND, max(4, int(60 * nScale)), {DIRT, GRASS})

        self.world = self.grid

    def _fillBiomeStrip(
        self,
        surfaceProfile: np.ndarray,
        colStart: int,
        colEnd: int,
        tile: int,
        depth: int,
        replaceable: set[int],
    ) -> None:
        """Overwrite tiles in a strip below the surface with a biome tile."""
        h = self.worldHeight
        for c in range(colStart, min(colEnd, self.worldWidth)):
            sy = surfaceProfile[c]
            seg = self.grid[sy : min(sy + depth, h), c]
            for old in replaceable:
                seg[seg == old] = tile

    # ------------------------------------------------------------------
    # Pre-hardmode evil
    # ------------------------------------------------------------------
    def placePreHardmodeEvil(self) -> None:
        """Place 3-5 evil pockets using TileRunner passes."""
        evilTile = CORRUPT_DIRT if self.evilType == "corruption" else CRIMSON_DIRT
        surface = int(self.layers.worldSurface)
        rock = int(self.layers.rockLayer)
        sScale = max(0.5, self.worldHeight / 2400.0)
        numPockets = int(self.rng.integers(3, 6))

        for _ in range(numPockets):
            px = int(self.rng.integers(self.worldWidth // 4, 3 * self.worldWidth // 4))
            py = int(self.rng.integers(surface + 5, max(surface + 6, rock)))

            numPasses = int(self.rng.integers(3, 8))
            for _ in range(numPasses):
                tileRunner(
                    self.grid, px, py,
                    strength=float(self.rng.uniform(8 * sScale, 20 * sScale)),
                    steps=int(self.rng.integers(
                        max(5, int(15 * sScale)),
                        max(6, int(40 * sScale)),
                    )),
                    tileType=evilTile,
                    overRide=False,
                )
                px = int(np.clip(
                    px + self.rng.integers(-30, 31), 0, self.worldWidth - 1))
                py = int(np.clip(
                    py + self.rng.integers(-10, 11), 0, self.worldHeight - 1))

        self.world = self.grid

    # ------------------------------------------------------------------
    # Hardmode V-pattern
    # ------------------------------------------------------------------
    def triggerHardmode(self) -> None:
        """Create V-pattern from world center via TileRunner along diagonals.

        Two arms extend from (centerX, worldSurface) diagonally down to
        the hell layer. One arm carries evil, the other hallow. Each arm
        is constructed from multiple TileRunner passes along the diagonal
        vector, not fixed-width strips.
        """
        centerX = self.worldWidth // 2
        surface = int(self.layers.worldSurface)
        hell = self.layers.hellLayer
        sScale = max(0.5, self.worldHeight / 2400.0)

        evilTile = CORRUPT_DIRT if self.evilType == "corruption" else CRIMSON_DIRT
        hallowTile = PEARLSTONE

        # 50/50 swap which side is evil vs hallow
        if self.rng.random() < 0.5:
            evilTile, hallowTile = hallowTile, evilTile

        vertDrop = hell - surface
        horzSpread = self.worldWidth // 4
        leftDx = -horzSpread / max(vertDrop, 1)
        rightDx = horzSpread / max(vertDrop, 1)

        numPasses = max(30, vertDrop // 20)

        for i in range(numPasses):
            t = i / numPasses
            y = int(surface + vertDrop * t)
            lx = int(centerX + leftDx * vertDrop * t)
            rx = int(centerX + rightDx * vertDrop * t)

            tileRunner(
                self.grid, lx, y,
                strength=float(self.rng.uniform(10 * sScale, 25 * sScale)),
                steps=int(self.rng.integers(
                    max(5, int(10 * sScale)),
                    max(6, int(30 * sScale)),
                )),
                tileType=evilTile, overRide=False,
                speedX=float(leftDx * 2), speedY=0.5,
            )
            tileRunner(
                self.grid, rx, y,
                strength=float(self.rng.uniform(10 * sScale, 25 * sScale)),
                steps=int(self.rng.integers(
                    max(5, int(10 * sScale)),
                    max(6, int(30 * sScale)),
                )),
                tileType=hallowTile, overRide=False,
                speedX=float(rightDx * 2), speedY=0.5,
            )

        self.world = self.grid

    # ------------------------------------------------------------------
    # Biome spread simulation
    # ------------------------------------------------------------------
    def simulateSpread(self, gameSeconds: float) -> None:
        """Tile-update-cycle spread with asymmetric surface/underground rates.

        Surface tiles update every ~140 s, underground every ~830 s.
        Each sampled infected tile picks one random neighbor within radius 3
        and converts it if the target is convertible and no air gap blocks.
        """
        surface = int(self.layers.worldSurface)

        infectedMask = np.isin(self.grid, list(ALL_INFECTED_TILES))
        infectedPos = np.argwhere(infectedMask)

        if len(infectedPos) == 0:
            self.corruptionHistory.append(0)
            return

        isSurface = infectedPos[:, 0] < surface
        surfCount = int(np.sum(isSurface))
        underCount = len(infectedPos) - surfCount

        surfUpdates = (
            int(surfCount * gameSeconds / SURFACE_UPDATE_RATE) if surfCount > 0 else 0
        )
        underUpdates = (
            int(underCount * gameSeconds / UNDERGROUND_UPDATE_RATE)
            if underCount > 0 else 0
        )
        totalUpdates = min(surfUpdates + underUpdates, 50000)

        if totalUpdates == 0:
            self.corruptionHistory.append(int(np.sum(infectedMask)))
            return

        # Build proportional sample array from surface and underground
        chunks: list[np.ndarray] = []
        if surfUpdates > 0 and surfCount > 0:
            surfPos = infectedPos[isSurface]
            n = min(surfUpdates, totalUpdates)
            chunks.append(surfPos[self.rng.integers(0, len(surfPos), size=n)])
        if underUpdates > 0 and underCount > 0:
            underPos = infectedPos[~isSurface]
            remaining = totalUpdates - (len(chunks[0]) if chunks else 0)
            if remaining > 0:
                n = min(underUpdates, remaining)
                chunks.append(underPos[self.rng.integers(0, len(underPos), size=n)])

        if not chunks:
            self.corruptionHistory.append(int(np.sum(infectedMask)))
            return

        samples = np.concatenate(chunks)

        for k in range(len(samples)):
            sy, sx = int(samples[k, 0]), int(samples[k, 1])
            srcTile = self.grid[sy, sx]

            if srcTile in CORRUPTION_TILES:
                infType = "corruption"
            elif srcTile in CRIMSON_TILES:
                infType = "crimson"
            elif srcTile in HALLOW_TILES:
                infType = "hallow"
            else:
                continue

            # Pick one random neighbor within INFECTION_SPREAD_RADIUS
            dy = int(self.rng.integers(-INFECTION_SPREAD_RADIUS, INFECTION_SPREAD_RADIUS + 1))
            dx = int(self.rng.integers(-INFECTION_SPREAD_RADIUS, INFECTION_SPREAD_RADIUS + 1))
            if dy == 0 and dx == 0:
                continue
            ny, nx = sy + dy, sx + dx
            if 0 <= nx < self.worldWidth and 0 <= ny < self.worldHeight:
                if self._canInfect(nx, ny, sx, sy, infType):
                    self._convertTile(nx, ny, infType)

        count = int(np.count_nonzero(np.isin(self.grid, list(ALL_INFECTED_TILES))))
        self.corruptionHistory.append(count)
        self.world = self.grid

    # ------------------------------------------------------------------
    # Infection checks
    # ------------------------------------------------------------------
    def _canInfect(
        self, x: int, y: int, sourceX: int, sourceY: int, infectionType: str,
    ) -> bool:
        """Check tile convertibility and air gap blocking."""
        target = self.grid[y, x]
        if target not in CONVERTIBLE_TILES:
            return False
        if target in ALL_INFECTED_TILES:
            return False
        conversions = _getConversions(infectionType)
        if target not in conversions:
            return False
        if self._hasAirGap(x, y, sourceX, sourceY):
            return False
        return True

    def _hasAirGap(self, x: int, y: int, sourceX: int, sourceY: int) -> bool:
        """Return True if INFECTION_GAP_TILES consecutive air tiles on the path."""
        dx = x - sourceX
        dy = y - sourceY
        steps = max(abs(dx), abs(dy))
        if steps <= 1:
            return False

        consecutive = 0
        for i in range(1, steps):
            t = i / steps
            cx = int(sourceX + dx * t)
            cy = int(sourceY + dy * t)
            if not (0 <= cx < self.worldWidth and 0 <= cy < self.worldHeight):
                return True
            if self.grid[cy, cx] == AIR:
                consecutive += 1
                if consecutive >= INFECTION_GAP_TILES:
                    return True
            else:
                consecutive = 0
        return False

    def _convertTile(self, x: int, y: int, infectionType: str) -> None:
        """Apply tile-type-specific conversion."""
        conversions = _getConversions(infectionType)
        target = self.grid[y, x]
        if target in conversions:
            self.grid[y, x] = conversions[target]

    # ------------------------------------------------------------------
    # Legacy shims (terraria_master_evolution.py compatibility)
    # ------------------------------------------------------------------
    def initialize_corruption_points(self) -> None:
        """Legacy: initializeWorld + placePreHardmodeEvil."""
        if np.count_nonzero(self.grid) == 0:
            self.initializeWorld()
        self.placePreHardmodeEvil()

    def trigger_hardmode_spread(self) -> None:
        """Legacy: triggerHardmode."""
        self.triggerHardmode()

    def simulate_spread_step(self, hardmode: bool = False) -> None:
        """Legacy: one spread step (~500 s pre-hardmode, ~2000 s hardmode)."""
        self.simulateSpread(500.0 if not hardmode else 2000.0)
        self.corruption_history = self.corruptionHistory


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------
def _addLegend(ax: plt.Axes) -> None:
    """Add compact biome color legend to an axis."""
    entries = {
        "Air": TILE_COLORS[AIR],
        "Dirt": TILE_COLORS[DIRT],
        "Stone": TILE_COLORS[STONE],
        "Grass": TILE_COLORS[GRASS],
        "Sand": TILE_COLORS[SAND],
        "Ice": TILE_COLORS[ICE],
        "Mud": TILE_COLORS[MUD],
        "Corrupt": TILE_COLORS[CORRUPT_DIRT],
        "Crimson": TILE_COLORS[CRIMSON_DIRT],
        "Hallow": TILE_COLORS[PEARLSTONE],
    }
    handles = [mpatches.Patch(color=c, label=lbl) for lbl, c in entries.items()]
    ax.legend(handles=handles, loc="upper right", fontsize=5, ncol=2, framealpha=0.5)


def _buildAirGapDemo() -> np.ndarray:
    """Return a small grid demonstrating air gap infection blocking.

    Top half: no barrier, corruption spreads freely through stone.
    Bottom half: 4-tile air trench blocks corruption from crossing.
    """
    dW, dH = 120, 80
    sim = TerrariaCorruptionEvolution(worldWidth=dW, worldHeight=dH, seed=99)
    sim.grid[:] = STONE

    # Top half: seed corruption at left, no barrier
    sim.grid[5:35, 10:28] = CORRUPT_DIRT

    # Bottom half: seed corruption at left, air trench at x=55
    sim.grid[45:75, 10:28] = CORRUPT_DIRT
    sim.grid[40:80, 55 : 55 + INFECTION_GAP_TILES] = AIR

    for _ in range(20):
        sim.simulateSpread(3000.0)

    return sim.grid.copy()


# ---------------------------------------------------------------------------
# Public figure builders
# ---------------------------------------------------------------------------
def createEvolutionFigure(savePath: str | None = None) -> plt.Figure:
    """4-panel 600x500 SMALL-crop figure: pre-HM pockets, V-pattern, spread T+1, T+10."""
    from Engine.spriteRenderer import applyMapDecorations, cropSmallWorld, drawTileGrid
    from Engine.worldgen import generateSmallWorld

    if savePath is None:
        savePath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "Plots", "Advanced", "corruption_evolution.png",
        )

    print("Generating SMALL world for corruption evolution (seed=20260423)...")
    world = generateSmallWorld(seed=20260423, evilType="corruption", compactBiomes=True)
    layers = world.layers
    baseGrid = world.grid.copy()

    centerX = world.spawnX
    centerY = int((layers.worldSurface + layers.rockLayer) / 2)

    # Build snapshots by running a reduced-res simulation on the SMALL grid
    # then cropping at the same center for all 4 panels.
    sim = TerrariaCorruptionEvolution(
        worldWidth=baseGrid.shape[1], worldHeight=baseGrid.shape[0],
        evilType="corruption", seed=20260423,
    )
    # Seed with the actual world geometry rather than sim's default.
    sim.grid = baseGrid.copy()
    sim.layers = layers
    sim.world = sim.grid

    # Phase 1: pre-HM pockets only (use world's pre-placed evil)
    snapPreHM = sim.grid.copy()

    # Phase 2: V-pattern
    sim.triggerHardmode()
    snapV = sim.grid.copy()

    # Phase 3: T+1 (one spread step, ~5000 in-game seconds)
    sim.simulateSpread(5000.0)
    snapSpread1 = sim.grid.copy()

    # Phase 4: T+10 (nine more spread steps)
    for _ in range(9):
        sim.simulateSpread(5000.0)
    snapSpread2 = sim.grid.copy()

    titles = [
        "Phase 1: Pre-Hardmode Evil Pockets",
        "Phase 2: Hardmode V-Pattern (WoF Defeated)",
        "Phase 3: Spread T+1 (~5 000 s)",
        "Phase 4: Spread T+10 (~50 000 s)",
    ]
    snaps = [snapPreHM, snapV, snapSpread1, snapSpread2]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for idx, (ax, snap, title) in enumerate(zip(axes.flat, snaps, titles)):
        cropped, bounds = cropSmallWorld(snap, centerX=centerX, centerY=centerY,
                                         width=130, height=90)
        drawTileGrid(ax, cropped)
        applyMapDecorations(ax, cropped, layers, cropBounds=bounds)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        "Corruption Evolution (130x90 tight crops)",
        fontsize=13, fontweight="bold", y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(os.path.dirname(os.path.abspath(savePath)), exist_ok=True)
    fig.savefig(savePath, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    print(f"Saved: {savePath}")
    plt.close(fig)
    return fig


def createSpreadAnimation(savePath: str | None = None) -> None:
    """Animated GIF showing corruption spread (SMALL world crop, ~600px wide)."""
    from matplotlib.animation import FuncAnimation

    from Engine.worldgen import generateSmallWorld

    if savePath is None:
        savePath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "Plots", "Advanced", "corruption_spread.gif",
        )

    print("Generating SMALL world for corruption spread GIF...")
    world = generateSmallWorld(seed=20260423, evilType="corruption", compactBiomes=True)
    layers = world.layers
    centerX = world.grid.shape[1] // 2
    # Center near surface so both V-arms stay within the crop window.
    centerY = int(layers.worldSurface) + 60
    h0, w0 = world.grid.shape

    # Crop helper (no sprite decorations -- fast for animation frames).
    x0 = max(0, centerX - 130)
    x1 = min(w0, centerX + 130)
    y0 = max(0, centerY - 100)
    y1 = min(h0, centerY + 100)

    def _crop(grid: np.ndarray) -> np.ndarray:
        return grid[y0:y1, x0:x1].copy()

    sim = TerrariaCorruptionEvolution(
        worldWidth=w0, worldHeight=h0,
        evilType="corruption", seed=20260423,
    )
    sim.grid = world.grid.copy()
    sim.layers = layers
    sim.world = sim.grid

    frames: list[np.ndarray] = [_crop(sim.grid)]

    for _ in range(10):
        sim.simulateSpread(2500.0)
        frames.append(_crop(sim.grid))

    sim.triggerHardmode()
    frames.append(_crop(sim.grid))
    hmFrame = len(frames) - 1

    for _ in range(30):
        sim.simulateSpread(12000.0)
        frames.append(_crop(sim.grid))

    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(_gridToRgb(frames[0]), aspect="equal", interpolation="nearest",
                   origin="upper")
    ax.set_xticks([]); ax.set_yticks([])
    titleObj = ax.set_title("", fontsize=11, fontweight="bold")

    def _update(f: int):
        im.set_data(_gridToRgb(frames[f]))
        phase = "HARDMODE" if f >= hmFrame else "Pre-Hardmode"
        titleObj.set_text(
            f"Corruption Spread -- Frame {f}/{len(frames) - 1} [{phase}]"
        )
        return [im, titleObj]

    anim = FuncAnimation(fig, _update, frames=len(frames), interval=200, blit=False)

    os.makedirs(os.path.dirname(os.path.abspath(savePath)), exist_ok=True)
    anim.save(savePath, writer="pillow", fps=5)
    print(f"Saved: {savePath}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Terraria Corruption Evolution")
    print("=" * 40)
    createEvolutionFigure()
    createSpreadAnimation()
    print("Done.")
