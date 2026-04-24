"""
Core Terraria world generation algorithms.

Implements TileRunner (diamond-brush random walk), digTunnel (sphere-cutter),
Cavinator (macro cave carver with immune-tile checks), cellular automata
cave smoothing, and SettleLiquids (bottom-up gravity scan).

All algorithms derived from decompiled WorldGen.cs source analysis.
"""

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Tile type constants used across the engine
# ---------------------------------------------------------------------------
AIR = 0
DIRT = 1
STONE = 2
GRASS = 3
SAND = 4
ASH = 5
HELLSTONE = 6
MUD = 7
SNOW = 8
ICE = 9
WATER = 50
LAVA = 51
HONEY = 52
OBSIDIAN = 53
CRISPY_HONEY_BLOCK = 54
CORRUPT_DIRT = 60
EBONSTONE = 61
CRIMSON_DIRT = 62
CRIMSTONE = 63
PEARLSTONE = 64
PEARLSAND = 65
HALLOW_DIRT = 66
CORRUPT_ICE = 67
CRIMSON_ICE = 68
HALLOW_ICE = 69

# Ore tile IDs
COPPER = 100
TIN = 101
IRON = 102
LEAD = 103
SILVER = 104
TUNGSTEN = 105
GOLD = 106
PLATINUM = 107
COBALT = 110
PALLADIUM = 111
MYTHRIL = 112
ORICHALCUM = 113
ADAMANTITE = 114
TITANIUM = 115
CHLOROPHYTE = 116

# Tiles immune to Cavinator destruction (CanBeClearedDuringGeneration = False)
# Includes: biome shells, dungeon/temple bricks, granite, hardened sand/sandstone
DUNGEON_BRICK = 120
LIHZAHRD_BRICK = 121
GRANITE_BLOCK = 123
HARDENED_SAND = 124
SANDSTONE_BLOCK = 125

HONEY_BLOCK = 145

# Tiles the cavinator and tileRunner refuse to carve. Biome shells (MUD,
# HARDENED_SAND, SANDSTONE_BLOCK) are deliberately NOT immune so caves can
# tunnel through every biome. Only structure bricks and hellstone resist.
IMMUNE_TILES = frozenset({
    ASH, HELLSTONE,
    DUNGEON_BRICK, LIHZAHRD_BRICK,
    GRANITE_BLOCK,
    CHLOROPHYTE,
})


def tileRunner(
    grid: npt.NDArray[np.int32],
    x: int,
    y: int,
    strength: float,
    steps: int,
    tileType: int,
    addTile: bool = False,
    speedX: float = 0.0,
    speedY: float = 0.0,
    noYChange: bool = False,
    overRide: bool = True,
    seed: int | None = None,
) -> npt.NDArray[np.int32]:
    """WorldGen.TileRunner: diamond-brush random walk algorithm.

    The most frequently called method in Terraria's worldgen. Places or
    removes tiles using a diamond-shaped brush that decays in strength
    as it walks along a drifting vector path.

    Args:
        grid: 2D tile array (height x width), modified in place.
        x, y: Starting coordinates.
        strength: Initial brush radius (decays each step).
        steps: Number of walk iterations.
        tileType: Tile ID to place. Use -1 to carve (remove tiles),
                  -2 to carve and fill with lava below hell threshold.
        addTile: If True, only place where grid == AIR.
        speedX, speedY: Initial directional drift vectors.
                        If both 0, randomized per step.
        noYChange: Clamp vertical drift (for flat horizontal tunnels).
        overRide: If True, replace any existing tile.

    Returns:
        The modified grid (same reference as input).
    """
    maxY, maxX = grid.shape
    rng = np.random.default_rng(seed)

    # Pre-compute hell threshold (used inside hot loop for tileType == -2).
    hellThreshold = max(maxY - 200, int(maxY * 0.917))

    # Initialize drift vectors if not provided
    if speedX == 0.0 and speedY == 0.0:
        speedX = rng.uniform(-1.0, 1.0)
        speedY = rng.uniform(-1.0, 1.0)

    cx, cy = float(x), float(y)
    currentStrength = strength

    for _ in range(steps):
        if currentStrength <= 0:
            break

        # Diamond-shaped brush with per-tile noise for organic edges
        radius = int(currentStrength / 2.0)
        radiusNoise = rng.integers(0, 2)  # +0 or +1 noise per step
        effectiveRadius = radius + radiusNoise

        for dx in range(-effectiveRadius, effectiveRadius + 1):
            for dy in range(-effectiveRadius, effectiveRadius + 1):
                # Diamond check with per-tile jitter for jagged edges
                tileNoise = rng.integers(0, 2)  # 0 or 1
                if abs(dx) + abs(dy) > effectiveRadius + tileNoise:
                    continue

                tx = int(cx) + dx
                ty = int(cy) + dy

                # Bounds check with border buffer
                if tx < 1 or tx >= maxX - 1 or ty < 1 or ty >= maxY - 1:
                    continue

                if tileType == -1:
                    # Carve mode: remove tile (set to AIR)
                    if grid[ty, tx] not in IMMUNE_TILES:
                        grid[ty, tx] = AIR
                elif tileType == -2:
                    # Carve + lava mode (use hellLayer = maxY - 200 for full grids)
                    if grid[ty, tx] not in IMMUNE_TILES:
                        grid[ty, tx] = LAVA if ty >= hellThreshold else AIR
                else:
                    # Place mode
                    if addTile:
                        if grid[ty, tx] == AIR:
                            grid[ty, tx] = tileType
                    elif overRide:
                        grid[ty, tx] = tileType
                    else:
                        if grid[ty, tx] != AIR:
                            grid[ty, tx] = tileType

        # Linear/step strength decay (matches decompiled source)
        currentStrength -= rng.integers(0, 3)

        # Update drift vectors with randomization (drunkard's walk)
        speedX += rng.uniform(-0.5, 0.5)
        speedY += rng.uniform(-0.5, 0.5) if not noYChange else 0.0

        # Clamp speed to +-1.0 for gentle drift (game default)
        speedX = np.clip(speedX, -1.0, 1.0)
        speedY = np.clip(speedY, -1.0, 1.0)

        # Move center point
        cx += speedX
        cy += speedY

    return grid


def digTunnel(
    grid: npt.NDArray[np.int32],
    x: float,
    y: float,
    xDir: float,
    yDir: float,
    steps: int,
    size: int,
    wet: bool = False,
    seed: int | None = None,
) -> npt.NDArray[np.int32]:
    """WorldGen.digTunnel: sphere-cutter for smooth shafts.

    Unlike TileRunner's diamond brush, digTunnel uses a circular/spherical
    radius to carve clean, geometric tunnels along a precise vector line.

    Args:
        grid: 2D tile array (height x width), modified in place.
        x, y: Starting coordinates (float for sub-tile precision).
        xDir, yDir: Directional vector (normalized or scaled).
        steps: Number of carving iterations.
        size: Radius of the spherical cutter.
        wet: If True, fill carved space with WATER instead of AIR.

    Returns:
        The modified grid.
    """
    maxY, maxX = grid.shape
    cx, cy = x, y
    rng = np.random.default_rng(seed)
    fillType = WATER if wet else AIR

    for _ in range(steps):
        # Circular brush: euclidean distance
        for dx in range(-size, size + 1):
            for dy in range(-size, size + 1):
                if dx * dx + dy * dy > size * size:
                    continue

                tx = int(cx) + dx
                ty = int(cy) + dy

                if tx < 1 or tx >= maxX - 1 or ty < 1 or ty >= maxY - 1:
                    continue

                if grid[ty, tx] not in IMMUNE_TILES:
                    grid[ty, tx] = fillType

        # Advance along the direction vector with slight randomization
        cx += xDir + rng.uniform(-0.2, 0.2)
        cy += yDir + rng.uniform(-0.2, 0.2)

    return grid


def cavinator(
    grid: npt.NDArray[np.int32],
    x: int,
    y: int,
    strength: float,
    steps: int,
    immuneTiles: frozenset[int] | None = None,
    seed: int | None = None,
) -> npt.NDArray[np.int32]:
    """WorldGen.Cavinator: macro cave carver with immune-tile checks.

    Behaves like a massive TileRunner but checks each tile against
    TileID.Sets.CanBeClearedDuringGeneration. If a tile type is in the
    immune set, destruction is aborted for that tile.

    Args:
        grid: 2D tile array, modified in place.
        x, y: Starting coordinates.
        strength: Initial carving radius.
        steps: Number of walk iterations.
        immuneTiles: Set of tile IDs that cannot be destroyed.
                     Defaults to IMMUNE_TILES.

    Returns:
        The modified grid.
    """
    if immuneTiles is None:
        immuneTiles = IMMUNE_TILES

    maxY, maxX = grid.shape
    rng = np.random.default_rng(seed)
    cx, cy = float(x), float(y)
    currentStrength = strength
    speedX = rng.uniform(-1.0, 1.0)
    speedY = rng.uniform(-1.0, 1.0)

    for _ in range(steps):
        if currentStrength <= 0:
            break

        radius = int(currentStrength / 2.0)
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if abs(dx) + abs(dy) > radius:
                    continue

                tx = int(cx) + dx
                ty = int(cy) + dy

                if tx < 1 or tx >= maxX - 1 or ty < 1 or ty >= maxY - 1:
                    continue

                if grid[ty, tx] in immuneTiles:
                    continue

                grid[ty, tx] = AIR

        currentStrength *= rng.uniform(0.93, 0.98)
        speedX += rng.uniform(-0.5, 0.5)
        speedY += rng.uniform(-0.5, 0.5)
        speedX = np.clip(speedX, -2.0, 2.0)
        speedY = np.clip(speedY, -2.0, 2.0)
        cx += speedX
        cy += speedY

    return grid


def cellularAutomataSmooth(
    grid: npt.NDArray[np.int32],
    iterations: int = 3,
    birthThreshold: int = 5,
    deathThreshold: int = 3,
    affectedTiles: frozenset[int] | None = None,
    fillTile: int = STONE,
) -> npt.NDArray[np.int32]:
    """Post-carving cellular automata smoothing pass.

    Iterates over the grid and applies neighbor-count rules:
    - If a solid tile has fewer than deathThreshold solid neighbors, destroy it.
    - If an air tile has more than birthThreshold solid neighbors, fill it.

    Uses vectorized 8-neighbor counting via shifted arrays for performance.

    Args:
        grid: 2D tile array, modified in place.
        iterations: Number of smoothing passes.
        birthThreshold: Neighbor count above which air becomes solid.
        deathThreshold: Neighbor count below which solid becomes air.
        affectedTiles: If provided, only smooth tiles of these types.
                       None means all non-air tiles are eligible.
        fillTile: Tile type to use when filling air cells.

    Returns:
        The modified grid.
    """
    maxY, maxX = grid.shape

    for _ in range(iterations):
        snapshot = grid.copy()
        solidMap = (snapshot != AIR).astype(np.int8)

        # Count 8-connected solid neighbors via shifted sums
        neighborCount = np.zeros((maxY, maxX), dtype=np.int8)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                neighborCount[1:-1, 1:-1] += solidMap[
                    1 + dy : maxY - 1 + dy,
                    1 + dx : maxX - 1 + dx,
                ]

        airMask = snapshot == AIR
        solidMask = ~airMask

        # Build the eligible mask for affected tiles
        if affectedTiles is not None:
            eligible = np.isin(snapshot, list(affectedTiles))
            solidMask = solidMask & eligible

        # Solid tiles with too few neighbors become air
        deathMask = solidMask & (neighborCount < deathThreshold)
        # Air tiles with too many solid neighbors become fill
        birthMask = airMask & (neighborCount > birthThreshold)

        # Only apply within the interior (skip border)
        interior = np.zeros((maxY, maxX), dtype=bool)
        interior[1:-1, 1:-1] = True
        grid[deathMask & interior] = AIR
        grid[birthMask & interior] = fillTile

    return grid


def settleLiquids(
    grid: npt.NDArray[np.int32],
    maxPasses: int = 50,
    seed: int | None = None,
) -> npt.NDArray[np.int32]:
    """SettleLiquids: bottom-up gravity scan for liquid settling.

    Scans from the bottom row upward. When a liquid tile has AIR below,
    it moves down. When blocked below, it spreads left and right.
    Runs until equilibrium or maxPasses reached.

    Liquid interaction rules:
    - Water + Lava = Obsidian (lava consumed)
    - Honey + Water and Honey + Lava interactions are not modeled here.

    Args:
        grid: 2D tile array, modified in place.
        maxPasses: Maximum settling iterations.
        seed: Random seed for reproducibility.

    Returns:
        The modified grid.
    """
    maxY, maxX = grid.shape
    liquidTypes = {WATER, LAVA, HONEY}
    rng = np.random.default_rng(seed)

    for _ in range(maxPasses):
        moved = False

        # Bottom-up scan (from second-to-last row up to row 1)
        for y in range(maxY - 2, 0, -1):
            for x in range(1, maxX - 1):
                currentTile = grid[y, x]
                if currentTile not in liquidTypes:
                    continue

                # Try to move down
                below = grid[y + 1, x]
                if below == AIR:
                    grid[y + 1, x] = currentTile
                    grid[y, x] = AIR
                    moved = True
                elif currentTile == WATER and below == LAVA:
                    # Water meets lava below: create obsidian
                    grid[y + 1, x] = OBSIDIAN
                    grid[y, x] = AIR
                    moved = True
                elif currentTile == LAVA and below == WATER:
                    grid[y + 1, x] = OBSIDIAN
                    grid[y, x] = AIR
                    moved = True
                elif currentTile == HONEY and below == LAVA:
                    grid[y + 1, x] = CRISPY_HONEY_BLOCK
                    grid[y, x] = AIR
                    moved = True
                elif currentTile == LAVA and below == HONEY:
                    grid[y + 1, x] = CRISPY_HONEY_BLOCK
                    grid[y, x] = AIR
                    moved = True
                elif currentTile == HONEY and below == WATER:
                    # Honey + Water = HoneyBlock
                    grid[y + 1, x] = HONEY_BLOCK
                    grid[y, x] = AIR
                    moved = True
                elif currentTile == WATER and below == HONEY:
                    grid[y + 1, x] = HONEY_BLOCK
                    grid[y, x] = AIR
                    moved = True
                elif below in liquidTypes or below != AIR:
                    # Blocked below: try to spread horizontally
                    leftOpen = x > 1 and grid[y, x - 1] == AIR
                    rightOpen = x < maxX - 2 and grid[y, x + 1] == AIR

                    if leftOpen and rightOpen:
                        # Spread to both sides (alternate to prevent bias)
                        if rng.random() < 0.5:
                            grid[y, x - 1] = currentTile
                        else:
                            grid[y, x + 1] = currentTile
                        grid[y, x] = AIR
                        moved = True
                    elif leftOpen:
                        grid[y, x - 1] = currentTile
                        grid[y, x] = AIR
                        moved = True
                    elif rightOpen:
                        grid[y, x + 1] = currentTile
                        grid[y, x] = AIR
                        moved = True

        if not moved:
            break

    return grid
