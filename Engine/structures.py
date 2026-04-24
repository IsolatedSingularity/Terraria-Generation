"""
Structure generation algorithms for Terraria worldgen.

Implements the Dungeon eating algorithm, Jungle Temple, Living Trees,
Pyramids, Spider Caves, Gem Caves, Shimmer/Aether biome, Underground
Desert (ant-hive), Marble/Granite cave generation, and decorative
placement passes (pots, traps, chests, sunflowers, etc.).

All algorithms derived from decompiled WorldGen.cs source analysis.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

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
    HELLSTONE,
    ICE,
    MUD,
    PEARLSAND,
    PEARLSTONE,
    SAND,
    STONE,
    tileRunner,
)
from Engine.constants import (
    BOULDER_TRAP,
    CHEST,
    CLAY,
    COBWEB,
    DART_TRAP,
    DUNGEON_BRICK,
    GRANITE_BLOCK,
    HARDENED_SAND,
    LEAF,
    LIHZAHRD_BRICK,
    LIVING_WOOD,
    MARBLE_BLOCK,
    MINECART_TRACK,
    MUSHROOM_GRASS,
    POT,
    PYRAMID_BRICK,
    SANDSTONE,
    SHIMMER,
    SILT,
    SUNFLOWER,
    WALL_GRANITE,
    WALL_LIHZAHRD,
    WALL_MARBLE,
    WALL_MUSHROOM,
    WALL_SPIDER,
    DungeonConfig,
    LivingTreeConfig,
    PyramidConfig,
    ShimmerConfig,
    TempleConfig,
)
from Engine.structureMap import Rectangle, StructureMap

# ---------------------------------------------------------------------------
# Dungeon eating algorithm
# ---------------------------------------------------------------------------


def generateDungeon(
    grid: npt.NDArray[np.int32],
    walls: npt.NDArray[np.int32],
    wallDungeon: npt.NDArray[np.bool_],
    startX: int,
    startY: int,
    structureMap: StructureMap,
    config: DungeonConfig | None = None,
    seed: int | None = None,
) -> npt.NDArray[np.int32]:
    """Generate dungeon using the eating algorithm with interlocking rooms.

    Places a shell of DUNGEON_BRICK, then sequentially carves rectangular
    rooms that interlock. Each room gets colored walls. The WallDungeon
    boolean array tracks dungeon interior tiles.

    Args:
        grid: 2D tile array, modified in place.
        walls: 2D wall array (parallel to grid), modified in place.
        wallDungeon: 2D boolean array tracking dungeon interior.
        startX, startY: Top-left corner of dungeon entrance.
        structureMap: Structure exclusion zone manager.
        config: Dungeon parameters. Defaults to DungeonConfig().
        seed: RNG seed for reproducibility.

    Returns:
        The modified grid.
    """
    if config is None:
        config = DungeonConfig()
    rng = np.random.default_rng(seed)
    maxY, maxX = grid.shape

    # Determine wall color for this dungeon
    wallColor = rng.choice(config.wallTypes)

    # Place dungeon shell (brick rectangle)
    numRooms = rng.integers(config.minRooms, config.maxRooms + 1)
    rooms: list[Rectangle] = []

    # First room anchored at start position
    w = rng.integers(config.minRoomWidth, config.maxRoomWidth + 1)
    h = rng.integers(config.minRoomHeight, config.maxRoomHeight + 1)
    firstRoom = Rectangle(startX, startY, w, h)
    rooms.append(firstRoom)

    # Fill first room shell
    _fillDungeonRoom(grid, walls, wallDungeon, firstRoom, wallColor, maxX, maxY)

    # Eat outward: each subsequent room overlaps the previous by 2-4 tiles
    for _ in range(numRooms - 1):
        parentRoom = rooms[rng.integers(0, len(rooms))]
        w = rng.integers(config.minRoomWidth, config.maxRoomWidth + 1)
        h = rng.integers(config.minRoomHeight, config.maxRoomHeight + 1)

        # Pick a side to attach to (0=right, 1=left, 2=below, 3=above)
        side = rng.integers(0, 4)
        overlap = rng.integers(2, 5)

        if side == 0:  # right
            nx = parentRoom.x + parentRoom.width - overlap
            ny = parentRoom.y + rng.integers(-h // 2, parentRoom.height // 2)
        elif side == 1:  # left
            nx = parentRoom.x - w + overlap
            ny = parentRoom.y + rng.integers(-h // 2, parentRoom.height // 2)
        elif side == 2:  # below
            nx = parentRoom.x + rng.integers(-w // 2, parentRoom.width // 2)
            ny = parentRoom.y + parentRoom.height - overlap
        else:  # above
            nx = parentRoom.x + rng.integers(-w // 2, parentRoom.width // 2)
            ny = parentRoom.y - h + overlap

        # Clamp to world bounds
        nx = max(5, min(nx, maxX - w - 5))
        ny = max(5, min(ny, maxY - h - 5))

        newRoom = Rectangle(nx, ny, w, h)

        # Check no critical overlap with protected structures
        if structureMap.canPlace(newRoom):
            rooms.append(newRoom)
            _fillDungeonRoom(grid, walls, wallDungeon, newRoom, wallColor, maxX, maxY)

    # Protect entire dungeon bounding box
    if rooms:
        minRx = min(r.x for r in rooms)
        minRy = min(r.y for r in rooms)
        maxRx = max(r.x + r.width for r in rooms)
        maxRy = max(r.y + r.height for r in rooms)
        structureMap.addProtectedStructure(
            Rectangle(minRx, minRy, maxRx - minRx, maxRy - minRy), padding=10
        )

    return grid


def _fillDungeonRoom(
    grid: npt.NDArray[np.int32],
    walls: npt.NDArray[np.int32],
    wallDungeon: npt.NDArray[np.bool_],
    room: Rectangle,
    wallColor: int,
    maxX: int,
    maxY: int,
) -> None:
    """Fill a dungeon room: brick walls, air interior, colored wall tiles."""
    for y in range(room.y, room.y + room.height):
        for x in range(room.x, room.x + room.width):
            if x < 0 or x >= maxX or y < 0 or y >= maxY:
                continue

            # Outer 1-tile border = brick
            if (
                x == room.x
                or x == room.x + room.width - 1
                or y == room.y
                or y == room.y + room.height - 1
            ):
                grid[y, x] = DUNGEON_BRICK
            else:
                grid[y, x] = AIR
                walls[y, x] = wallColor
                wallDungeon[y, x] = True


# ---------------------------------------------------------------------------
# Jungle Temple (Lihzahrd)
# ---------------------------------------------------------------------------


def generateJungleTemple(
    grid: npt.NDArray[np.int32],
    walls: npt.NDArray[np.int32],
    centerX: int,
    centerY: int,
    width: int,
    height: int,
    structureMap: StructureMap,
    config: TempleConfig | None = None,
    seed: int | None = None,
) -> npt.NDArray[np.int32]:
    """Generate Jungle Temple with Lihzahrd brick rooms and traps.

    Places a protected rectangle of LIHZAHRD_BRICK, then carves rooms
    with pressure plates and dart/boulder traps inside.

    Returns:
        The modified grid.
    """
    if config is None:
        config = TempleConfig()
    rng = np.random.default_rng(seed)
    maxY, maxX = grid.shape

    x0 = max(1, centerX - width // 2)
    y0 = max(1, centerY - height // 2)
    x1 = min(maxX - 1, x0 + width)
    y1 = min(maxY - 1, y0 + height)

    # Fill with Lihzahrd brick
    grid[y0:y1, x0:x1] = LIHZAHRD_BRICK
    walls[y0:y1, x0:x1] = WALL_LIHZAHRD

    # Carve rooms inside
    numRooms = rng.integers(config.minRooms, config.maxRooms + 1)
    for _ in range(numRooms):
        rw = rng.integers(config.minRoomWidth, config.maxRoomWidth + 1)
        rh = rng.integers(config.minRoomHeight, config.maxRoomHeight + 1)
        rx = rng.integers(x0 + 2, max(x0 + 3, x1 - rw - 2))
        ry = rng.integers(y0 + 2, max(y0 + 3, y1 - rh - 2))

        # Carve interior (keep 1-tile brick border)
        for ty in range(ry + 1, min(ry + rh - 1, y1 - 1)):
            for tx in range(rx + 1, min(rx + rw - 1, x1 - 1)):
                grid[ty, tx] = AIR

        # Place traps with configured density
        for ty in range(ry + 1, min(ry + rh - 1, y1 - 1)):
            for tx in range(rx + 1, min(rx + rw - 1, x1 - 1)):
                if grid[ty, tx] == AIR and rng.random() < config.trapDensity:
                    trapType = rng.choice([DART_TRAP, BOULDER_TRAP])
                    grid[ty, tx] = trapType

    # Protect the temple
    structureMap.addProtectedStructure(Rectangle(x0, y0, x1 - x0, y1 - y0), padding=10)

    return grid


# ---------------------------------------------------------------------------
# Living Trees
# ---------------------------------------------------------------------------


def generateLivingTree(
    grid: npt.NDArray[np.int32],
    baseX: int,
    surfaceY: int,
    config: LivingTreeConfig | None = None,
    seed: int | None = None,
) -> npt.NDArray[np.int32]:
    """Generate a Living Tree with trunk, canopy, hollow interior, and branches.

    Args:
        grid: 2D tile array, modified in place.
        baseX: X coordinate of the tree base.
        surfaceY: Y coordinate of the surface at this X position.
        config: Tree parameters.
        seed: RNG seed.

    Returns:
        The modified grid.
    """
    if config is None:
        config = LivingTreeConfig()
    rng = np.random.default_rng(seed)
    maxY, maxX = grid.shape

    height = rng.integers(config.minHeight, config.maxHeight + 1)
    topY = max(5, surfaceY - height)
    halfTrunk = config.trunkWidth // 2

    # Draw trunk (Living Wood)
    for y in range(topY, surfaceY + 10):
        for dx in range(-halfTrunk, halfTrunk + 1):
            tx = baseX + dx
            if 0 <= tx < maxX and 0 <= y < maxY:
                grid[y, tx] = LIVING_WOOD

    # Hollow interior if chance permits
    if rng.random() < config.hollowChance:
        for y in range(topY + 5, surfaceY + 8):
            for dx in range(-halfTrunk + 1, halfTrunk):
                tx = baseX + dx
                if 0 <= tx < maxX and 0 <= y < maxY:
                    grid[y, tx] = AIR

    # Canopy (leaf dome)
    canopyCenterY = topY
    r = config.canopyRadius
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            dist = dx * dx + dy * dy
            if dist <= r * r:
                tx = baseX + dx
                ty = canopyCenterY + dy
                if 0 <= tx < maxX and 0 <= ty < maxY:
                    if grid[ty, tx] == AIR:
                        grid[ty, tx] = LEAF

    # Branches (horizontal wood lines)
    for i in range(config.branchCount):
        branchY = rng.integers(topY + 3, max(topY + 4, surfaceY - 5))
        direction = rng.choice([-1, 1])
        branchLen = rng.integers(8, 16)
        for bx in range(branchLen):
            tx = baseX + direction * (halfTrunk + bx)
            if 0 <= tx < maxX and 0 <= branchY < maxY:
                grid[branchY, tx] = LIVING_WOOD
                # Leaf fringe
                if 0 <= branchY - 1 < maxY and grid[branchY - 1, tx] == AIR:
                    grid[branchY - 1, tx] = LEAF
                if branchY + 1 < maxY and grid[branchY + 1, tx] == AIR:
                    grid[branchY + 1, tx] = LEAF

    # Roots extending below surface
    for dx in range(-halfTrunk - 2, halfTrunk + 3):
        rootDepth = rng.integers(3, 10)
        tx = baseX + dx
        for dy in range(rootDepth):
            ty = surfaceY + dy
            if 0 <= tx < maxX and 0 <= ty < maxY:
                if grid[ty, tx] in (DIRT, STONE):
                    grid[ty, tx] = LIVING_WOOD

    return grid


# ---------------------------------------------------------------------------
# Pyramids
# ---------------------------------------------------------------------------


def generatePyramid(
    grid: npt.NDArray[np.int32],
    baseX: int,
    surfaceY: int,
    config: PyramidConfig | None = None,
    seed: int | None = None,
) -> npt.NDArray[np.int32]:
    """Generate a desert Pyramid with randomized interior corridors.

    Builds a triangular structure of PYRAMID_BRICK with internal
    corridors leading to a treasure room.

    Returns:
        The modified grid.
    """
    if config is None:
        config = PyramidConfig()
    rng = np.random.default_rng(seed)
    maxY, maxX = grid.shape

    width = rng.integers(config.minWidth, config.maxWidth + 1)
    halfW = width // 2
    height = halfW  # Isoceles triangle: height = half width

    # Build pyramid shell
    for row in range(height):
        leftEdge = baseX - halfW + row
        rightEdge = baseX + halfW - row
        ty = surfaceY + row

        if ty >= maxY:
            break

        for tx in range(max(0, leftEdge), min(maxX, rightEdge + 1)):
            grid[ty, tx] = PYRAMID_BRICK

    # Carve interior corridor (diagonal shaft)
    direction = rng.choice([-1, 1])
    corridorY = surfaceY + 3
    corridorX = baseX

    for step in range(height - 8):
        for dy in range(config.corridorWidth):
            for dx in range(config.corridorWidth):
                ty = corridorY + dy
                tx = corridorX + dx
                if 0 <= tx < maxX and 0 <= ty < maxY:
                    grid[ty, tx] = AIR

        corridorY += 1
        corridorX += direction

    # Treasure room at bottom
    roomX = corridorX - 4
    roomY = corridorY
    for dy in range(6):
        for dx in range(8):
            ty = roomY + dy
            tx = roomX + dx
            if 0 <= tx < maxX and 0 <= ty < maxY:
                if dy == 0 or dy == 5 or dx == 0 or dx == 7:
                    grid[ty, tx] = PYRAMID_BRICK
                else:
                    grid[ty, tx] = AIR

    # Place a chest in the treasure room
    chestX = roomX + 3
    chestY = roomY + 4
    if 0 <= chestX < maxX and 0 <= chestY < maxY:
        grid[chestY, chestX] = CHEST

    return grid


# ---------------------------------------------------------------------------
# Spider Caves
# ---------------------------------------------------------------------------


def generateSpiderCave(
    grid: npt.NDArray[np.int32],
    walls: npt.NDArray[np.int32],
    centerX: int,
    centerY: int,
    radius: int = 12,
    seed: int | None = None,
) -> npt.NDArray[np.int32]:
    """Generate a Spider Cave (cobweb-filled cavity with spider wall).

    Returns:
        The modified grid.
    """
    rng = np.random.default_rng(seed)
    maxY, maxX = grid.shape

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            tx = centerX + dx
            ty = centerY + dy
            if 0 <= tx < maxX and 0 <= ty < maxY:
                if rng.random() < 0.7:
                    grid[ty, tx] = AIR
                    walls[ty, tx] = WALL_SPIDER
                if rng.random() < 0.3 and grid[ty, tx] == AIR:
                    grid[ty, tx] = COBWEB

    return grid


# ---------------------------------------------------------------------------
# Gem Caves
# ---------------------------------------------------------------------------

# Gem tile IDs (visualization markers)
GEM_AMETHYST = 150
GEM_TOPAZ = 151
GEM_SAPPHIRE = 152
GEM_EMERALD = 153
GEM_RUBY = 154
GEM_DIAMOND = 155
GEM_TYPES = (GEM_AMETHYST, GEM_TOPAZ, GEM_SAPPHIRE, GEM_EMERALD, GEM_RUBY, GEM_DIAMOND)


def generateGemCave(
    grid: npt.NDArray[np.int32],
    centerX: int,
    centerY: int,
    radius: int = 8,
    seed: int | None = None,
) -> npt.NDArray[np.int32]:
    """Generate a Gem Cave micro-biome with gem crystals on walls.

    Returns:
        The modified grid.
    """
    rng = np.random.default_rng(seed)
    maxY, maxX = grid.shape
    gemType = rng.choice(GEM_TYPES)

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            tx = centerX + dx
            ty = centerY + dy
            if 0 <= tx < maxX and 0 <= ty < maxY:
                edgeDist = abs(dx * dx + dy * dy - radius * radius)
                if edgeDist < radius * 3:  # Near the edge
                    if rng.random() < 0.4:
                        grid[ty, tx] = gemType
                elif rng.random() < 0.6:
                    grid[ty, tx] = AIR

    return grid


# ---------------------------------------------------------------------------
# Underground Desert (ant-hive)
# ---------------------------------------------------------------------------


def generateUndergroundDesert(
    grid: npt.NDArray[np.int32],
    centerX: int,
    centerY: int,
    radiusX: int = 60,
    radiusY: int = 80,
    seed: int | None = None,
) -> npt.NDArray[np.int32]:
    """Generate Underground Desert circular ant-hive with Hardened Sand and Sandstone.

    The game stores this in WorldGen.UndergroundDesertLocation.
    Uses an elliptical shape with internal cavity carving.

    Returns:
        The modified grid.
    """
    rng = np.random.default_rng(seed)
    maxY, maxX = grid.shape

    for dy in range(-radiusY, radiusY + 1):
        for dx in range(-radiusX, radiusX + 1):
            # Ellipse check
            normDist = (dx / radiusX) ** 2 + (dy / radiusY) ** 2
            if normDist > 1.0:
                continue

            tx = centerX + dx
            ty = centerY + dy
            if tx < 0 or tx >= maxX or ty < 0 or ty >= maxY:
                continue

            # Outer shell: Hardened Sand, inner: Sandstone, cavities in center
            if normDist > 0.85:
                grid[ty, tx] = HARDENED_SAND
            elif normDist > 0.5:
                grid[ty, tx] = SANDSTONE
            else:
                # Inner area: mix of sandstone and air cavities
                if rng.random() < 0.35:
                    grid[ty, tx] = AIR
                else:
                    grid[ty, tx] = SANDSTONE

    # TileRunner carving for organic tunnels inside
    numTunnels = rng.integers(8, 16)
    for _ in range(numTunnels):
        tx = centerX + rng.integers(-radiusX // 2, radiusX // 2)
        ty = centerY + rng.integers(-radiusY // 2, radiusY // 2)
        tileRunner(
            grid, tx, ty, strength=rng.uniform(4, 8), steps=rng.integers(20, 50), tileType=-1
        )

    return grid


# ---------------------------------------------------------------------------
# Marble Cave Generation
# ---------------------------------------------------------------------------


def generateMarbleCave(
    grid: npt.NDArray[np.int32],
    walls: npt.NDArray[np.int32],
    centerX: int,
    centerY: int,
    seed: int | None = None,
) -> npt.NDArray[np.int32]:
    """Generate an elongated Marble cave with Marble blocks and walls.

    Returns:
        The modified grid.
    """
    rng = np.random.default_rng(seed)
    maxY, maxX = grid.shape

    # Elongated cavity via multiple TileRunner passes
    for _ in range(rng.integers(3, 7)):
        sx = centerX + rng.integers(-10, 11)
        sy = centerY + rng.integers(-5, 6)
        strength = rng.uniform(5, 12)
        steps = rng.integers(15, 35)
        tileRunner(
            grid, sx, sy, strength, steps, tileType=MARBLE_BLOCK, overRide=True, noYChange=True
        )

    # Fill walls
    for dy in range(-25, 26):
        for dx in range(-40, 41):
            tx = centerX + dx
            ty = centerY + dy
            if 0 <= tx < maxX and 0 <= ty < maxY:
                if grid[ty, tx] == MARBLE_BLOCK:
                    walls[ty, tx] = WALL_MARBLE
                elif grid[ty, tx] == AIR:
                    # Check if near marble for wall assignment
                    for ndy in range(-1, 2):
                        for ndx in range(-1, 2):
                            ntx, nty = tx + ndx, ty + ndy
                            if 0 <= ntx < maxX and 0 <= nty < maxY:
                                if grid[nty, ntx] == MARBLE_BLOCK:
                                    walls[ty, tx] = WALL_MARBLE
                                    break

    return grid


# ---------------------------------------------------------------------------
# Granite Cave Generation
# ---------------------------------------------------------------------------


def generateGraniteCave(
    grid: npt.NDArray[np.int32],
    walls: npt.NDArray[np.int32],
    centerX: int,
    centerY: int,
    seed: int | None = None,
) -> npt.NDArray[np.int32]:
    """Generate a Granite cave (similar to Marble but with granite blocks).

    Returns:
        The modified grid.
    """
    rng = np.random.default_rng(seed)
    maxY, maxX = grid.shape

    # Granite caves are more circular than marble
    for _ in range(rng.integers(4, 8)):
        sx = centerX + rng.integers(-8, 9)
        sy = centerY + rng.integers(-8, 9)
        strength = rng.uniform(4, 10)
        steps = rng.integers(10, 30)
        tileRunner(grid, sx, sy, strength, steps, tileType=GRANITE_BLOCK, overRide=True)

    # Fill walls
    for dy in range(-25, 26):
        for dx in range(-25, 26):
            tx = centerX + dx
            ty = centerY + dy
            if 0 <= tx < maxX and 0 <= ty < maxY:
                if grid[ty, tx] == GRANITE_BLOCK:
                    walls[ty, tx] = WALL_GRANITE

    return grid


# ---------------------------------------------------------------------------
# Shimmer/Aether biome (post-1.4.4)
# ---------------------------------------------------------------------------


def generateShimmerBiome(
    grid: npt.NDArray[np.int32],
    centerX: int,
    centerY: int,
    config: ShimmerConfig | None = None,
    seed: int | None = None,
) -> npt.NDArray[np.int32]:
    """Generate Shimmer/Aether biome pool.

    Placed in the cavern layer, correlated with Jungle side.
    Creates a shimmer-filled lake surrounded by unique terrain.

    Returns:
        The modified grid.
    """
    if config is None:
        config = ShimmerConfig()
    _ = np.random.default_rng(seed)
    maxY, maxX = grid.shape
    r = config.radius

    # Carve a cavity
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            dist = (dx / r) ** 2 + (dy / (r * 0.6)) ** 2
            if dist > 1.0:
                continue
            tx = centerX + dx
            ty = centerY + dy
            if 0 <= tx < maxX and 0 <= ty < maxY:
                grid[ty, tx] = AIR

    # Fill lower portion with shimmer liquid
    shimmerTop = centerY + int(r * 0.1)
    for dy in range(0, r):
        for dx in range(-r, r + 1):
            dist = (dx / r) ** 2 + (dy / (r * 0.5)) ** 2
            if dist > 1.0:
                continue
            tx = centerX + dx
            ty = shimmerTop + dy
            if 0 <= tx < maxX and 0 <= ty < maxY:
                if grid[ty, tx] == AIR:
                    grid[ty, tx] = SHIMMER

    return grid


# ---------------------------------------------------------------------------
# Floating Island Houses
# ---------------------------------------------------------------------------


def generateFloatingIslandHouse(
    grid: npt.NDArray[np.int32],
    islandX: int,
    islandTopY: int,
    seed: int | None = None,
) -> npt.NDArray[np.int32]:
    """Place a small house structure on top of a floating island.

    Returns:
        The modified grid.
    """
    rng = np.random.default_rng(seed)
    maxY, maxX = grid.shape

    houseW = rng.integers(8, 14)
    houseH = rng.integers(6, 10)
    hx = islandX - houseW // 2
    hy = islandTopY - houseH

    for dy in range(houseH):
        for dx in range(houseW):
            tx = hx + dx
            ty = hy + dy
            if 0 <= tx < maxX and 0 <= ty < maxY:
                if dy == 0 or dy == houseH - 1 or dx == 0 or dx == houseW - 1:
                    grid[ty, tx] = LIVING_WOOD  # Walls
                else:
                    grid[ty, tx] = AIR

    # Door (opening in bottom wall)
    doorX = hx + houseW // 2
    if 0 <= doorX < maxX:
        doorY = hy + houseH - 1
        if 0 <= doorY < maxY:
            grid[doorY, doorX] = AIR

    # Place chest inside
    chestX = hx + houseW // 2 - 1
    chestY = hy + houseH - 2
    if 0 <= chestX < maxX and 0 <= chestY < maxY:
        grid[chestY, chestX] = CHEST

    return grid


# ---------------------------------------------------------------------------
# Terrain injection passes
# ---------------------------------------------------------------------------


def rocksInDirt(
    grid: npt.NDArray[np.int32],
    count: int,
    worldSurface: int,
    rockLayer: int,
    seed: int | None = None,
) -> npt.NDArray[np.int32]:
    """Inject stone pockets into the dirt layer using TileRunner.

    GenPass 'Rocks In Dirt': places stone blobs in the dirt transition zone.

    Returns:
        The modified grid.
    """
    rng = np.random.default_rng(seed)
    maxY, maxX = grid.shape

    for _ in range(count):
        x = rng.integers(10, maxX - 10)
        y = rng.integers(worldSurface, rockLayer)
        tileRunner(
            grid,
            x,
            y,
            strength=rng.uniform(4, 10),
            steps=rng.integers(10, 30),
            tileType=STONE,
            overRide=False,
        )

    return grid


def dirtInRocks(
    grid: npt.NDArray[np.int32],
    count: int,
    rockLayer: int,
    hellLayer: int,
    seed: int | None = None,
) -> npt.NDArray[np.int32]:
    """Inject dirt pockets into the stone/cavern layer using TileRunner.

    GenPass 'Dirt In Rocks': places dirt blobs in the rock zone.

    Returns:
        The modified grid.
    """
    rng = np.random.default_rng(seed)
    maxY, maxX = grid.shape

    for _ in range(count):
        x = rng.integers(10, maxX - 10)
        y = rng.integers(rockLayer, hellLayer)
        tileRunner(
            grid,
            x,
            y,
            strength=rng.uniform(4, 10),
            steps=rng.integers(10, 30),
            tileType=DIRT,
            overRide=False,
        )

    return grid


def placeClay(
    grid: npt.NDArray[np.int32],
    count: int,
    worldSurface: int,
    rockLayer: int,
    seed: int | None = None,
) -> npt.NDArray[np.int32]:
    """Place clay pockets near the surface using TileRunner.

    Returns:
        The modified grid.
    """
    rng = np.random.default_rng(seed)
    maxY, maxX = grid.shape

    for _ in range(count):
        x = rng.integers(10, maxX - 10)
        y = rng.integers(max(1, worldSurface - 20), rockLayer)
        tileRunner(
            grid,
            x,
            y,
            strength=rng.uniform(3, 7),
            steps=rng.integers(8, 20),
            tileType=CLAY,
            overRide=False,
        )

    return grid


def placeSilt(
    grid: npt.NDArray[np.int32],
    count: int,
    rockLayer: int,
    hellLayer: int,
    seed: int | None = None,
) -> npt.NDArray[np.int32]:
    """Place silt deposits in the cavern layer using TileRunner.

    Returns:
        The modified grid.
    """
    rng = np.random.default_rng(seed)
    maxY, maxX = grid.shape

    for _ in range(count):
        x = rng.integers(10, maxX - 10)
        y = rng.integers(rockLayer, hellLayer)
        tileRunner(
            grid,
            x,
            y,
            strength=rng.uniform(3, 6),
            steps=rng.integers(8, 18),
            tileType=SILT,
            overRide=False,
        )

    return grid


# ---------------------------------------------------------------------------
# Decorative placement passes
# ---------------------------------------------------------------------------


def placeSunflowers(
    grid: npt.NDArray[np.int32],
    surfaceHeights: npt.NDArray[np.int32],
    evilBiomeXRanges: list[tuple[int, int]],
    count: int = 20,
    seed: int | None = None,
) -> npt.NDArray[np.int32]:
    """Place sunflowers at evil biome borders to slow spread.

    Returns:
        The modified grid.
    """
    _ = np.random.default_rng(seed)
    maxY, maxX = grid.shape

    for xStart, xEnd in evilBiomeXRanges:
        # Place sunflowers near borders
        for borderX in [xStart - 3, xStart - 2, xEnd + 1, xEnd + 2]:
            if 0 <= borderX < maxX:
                sy = surfaceHeights[borderX]
                if 0 <= sy - 1 < maxY:
                    grid[sy - 1, borderX] = SUNFLOWER

    return grid


def placeTraps(
    grid: npt.NDArray[np.int32],
    count: int,
    rockLayer: int,
    hellLayer: int,
    seed: int | None = None,
) -> npt.NDArray[np.int32]:
    """Place dart traps and boulder traps in the cavern layer.

    Returns:
        The modified grid.
    """
    rng = np.random.default_rng(seed)
    maxY, maxX = grid.shape

    for _ in range(count):
        x = rng.integers(10, maxX - 10)
        y = rng.integers(rockLayer, hellLayer)

        # Only place in air tiles adjacent to stone
        if 0 <= y < maxY and 0 <= x < maxX and grid[y, x] == AIR:
            hasStoneNeighbor = False
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < maxY and 0 <= nx < maxX:
                        if grid[ny, nx] == STONE:
                            hasStoneNeighbor = True
                            break
            if hasStoneNeighbor:
                grid[y, x] = rng.choice([DART_TRAP, BOULDER_TRAP])

    return grid


def placePots(
    grid: npt.NDArray[np.int32],
    count: int,
    worldSurface: int,
    hellLayer: int,
    seed: int | None = None,
) -> npt.NDArray[np.int32]:
    """Place pots on flat surfaces in caves.

    Returns:
        The modified grid.
    """
    rng = np.random.default_rng(seed)
    maxY, maxX = grid.shape

    placed = 0
    maxAttempts = count * 5
    for _ in range(maxAttempts):
        if placed >= count:
            break
        x = rng.integers(10, maxX - 10)
        y = rng.integers(worldSurface, hellLayer)

        # Need air at position and solid below
        if 0 <= y < maxY - 1 and 0 <= x < maxX and grid[y, x] == AIR and grid[y + 1, x] != AIR:
            grid[y, x] = POT
            placed += 1

    return grid


def placeMinecartTracks(
    grid: npt.NDArray[np.int32],
    count: int,
    rockLayer: int,
    hellLayer: int,
    seed: int | None = None,
) -> npt.NDArray[np.int32]:
    """Place minecart tracks in the cavern layer.

    Each track is a horizontal run of MINECART_TRACK tiles.

    Returns:
        The modified grid.
    """
    rng = np.random.default_rng(seed)
    maxY, maxX = grid.shape

    for _ in range(count):
        x = rng.integers(50, maxX - 50)
        y = rng.integers(rockLayer, hellLayer)
        trackLen = rng.integers(30, 80)

        # Find a flat surface to place tracks
        for tx in range(x, min(x + trackLen, maxX)):
            if 0 <= y < maxY and grid[y, tx] == AIR and y + 1 < maxY and grid[y + 1, tx] != AIR:
                grid[y, tx] = MINECART_TRACK

    return grid


# ---------------------------------------------------------------------------
# Meteorite collision
# ---------------------------------------------------------------------------


def dropMeteor(
    grid: npt.NDArray[np.int32],
    surfaceHeights: npt.NDArray[np.int32],
    x: int | None = None,
    seed: int | None = None,
) -> npt.NDArray[np.int32]:
    """Drop a meteorite at the given X position using TileRunner.

    Replaces terrain with HELLSTONE in a roughly circular impact zone.

    Returns:
        The modified grid.
    """
    rng = np.random.default_rng(seed)
    maxY, maxX = grid.shape

    if x is None:
        x = rng.integers(50, maxX - 50)

    y = int(surfaceHeights[min(x, len(surfaceHeights) - 1)])

    # Meteorite impact: TileRunner with hellstone
    tileRunner(
        grid,
        x,
        y,
        strength=rng.uniform(15, 25),
        steps=rng.integers(30, 60),
        tileType=HELLSTONE,
        overRide=True,
    )

    return grid


# ---------------------------------------------------------------------------
# Purification / Clentaminator
# ---------------------------------------------------------------------------

# Convertible tile mapping for purification
PURIFICATION_MAP = {
    CORRUPT_DIRT: DIRT,
    EBONSTONE: STONE,
    CRIMSON_DIRT: DIRT,
    CRIMSTONE: STONE,
    PEARLSTONE: STONE,
    PEARLSAND: SAND,
    HALLOW_DIRT: DIRT,
    CORRUPT_ICE: ICE,
    CRIMSON_ICE: ICE,
    HALLOW_ICE: ICE,
}

# Corruption conversion map
CORRUPTION_MAP = {
    DIRT: CORRUPT_DIRT,
    STONE: EBONSTONE,
    GRASS: CORRUPT_DIRT,
    SAND: EBONSTONE,
    ICE: CORRUPT_ICE,
}

# Crimson conversion map
CRIMSON_MAP = {
    DIRT: CRIMSON_DIRT,
    STONE: CRIMSTONE,
    GRASS: CRIMSON_DIRT,
    SAND: CRIMSTONE,
    ICE: CRIMSON_ICE,
}

# Hallow conversion map
HALLOW_MAP = {
    DIRT: HALLOW_DIRT,
    STONE: PEARLSTONE,
    GRASS: HALLOW_DIRT,
    SAND: PEARLSAND,
    ICE: HALLOW_ICE,
}


def clentaminatorSpray(
    grid: npt.NDArray[np.int32],
    startX: int,
    startY: int,
    directionX: float,
    directionY: float,
    sprayRange: int = 60,
    sprayWidth: int = 2,
    solutionType: str = "green",
) -> npt.NDArray[np.int32]:
    """Simulate Clentaminator spray pattern for purification/corruption.

    Args:
        grid: 2D tile array, modified in place.
        startX, startY: Spray origin.
        directionX, directionY: Spray direction (normalized).
        sprayRange: How far the spray reaches.
        sprayWidth: Width of the spray cone.
        solutionType: 'green' (purify), 'purple' (corrupt), 'red' (crimson),
                      'blue' (hallow), 'dark_blue' (mushroom).

    Returns:
        The modified grid.
    """
    maxY, maxX = grid.shape

    # Normalize direction
    mag = np.sqrt(directionX**2 + directionY**2)
    if mag == 0:
        return grid
    dx, dy = directionX / mag, directionY / mag

    for step in range(sprayRange):
        cx = int(startX + dx * step)
        cy = int(startY + dy * step)

        for w in range(-sprayWidth, sprayWidth + 1):
            # Perpendicular offset
            tx = cx + int(-dy * w)
            ty = cy + int(dx * w)

            if 0 <= tx < maxX and 0 <= ty < maxY:
                currentTile = grid[ty, tx]

                if solutionType == "green":
                    if currentTile in PURIFICATION_MAP:
                        grid[ty, tx] = PURIFICATION_MAP[currentTile]
                elif solutionType == "purple":
                    if currentTile in CORRUPTION_MAP:
                        grid[ty, tx] = CORRUPTION_MAP[currentTile]
                elif solutionType == "red":
                    if currentTile in CRIMSON_MAP:
                        grid[ty, tx] = CRIMSON_MAP[currentTile]
                elif solutionType == "blue":
                    if currentTile in HALLOW_MAP:
                        grid[ty, tx] = HALLOW_MAP[currentTile]
                elif solutionType == "dark_blue":
                    if currentTile in (DIRT, GRASS, MUD):
                        grid[ty, tx] = MUSHROOM_GRASS

    return grid


# ---------------------------------------------------------------------------
# Underground Mushroom Biome
# ---------------------------------------------------------------------------


def generateMushroomBiome(
    grid: npt.NDArray[np.int32],
    walls: npt.NDArray[np.int32],
    centerX: int,
    centerY: int,
    radius: int = 30,
    seed: int | None = None,
) -> npt.NDArray[np.int32]:
    """Generate underground Mushroom biome (mud + mushroom grass).

    Returns:
        The modified grid.
    """
    _ = np.random.default_rng(seed)
    maxY, maxX = grid.shape

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            tx = centerX + dx
            ty = centerY + dy
            if 0 <= tx < maxX and 0 <= ty < maxY:
                if grid[ty, tx] in (STONE, DIRT):
                    grid[ty, tx] = MUD
                    walls[ty, tx] = WALL_MUSHROOM

    # Convert exposed mud surfaces to mushroom grass
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            tx = centerX + dx
            ty = centerY + dy
            if 0 <= tx < maxX and 0 <= ty < maxY:
                if grid[ty, tx] == MUD:
                    # Check if adjacent to air
                    for ndy in [-1, 0, 1]:
                        for ndx in [-1, 0, 1]:
                            ntx, nty = tx + ndx, ty + ndy
                            if 0 <= ntx < maxX and 0 <= nty < maxY:
                                if grid[nty, ntx] == AIR:
                                    grid[ty, tx] = MUSHROOM_GRASS
                                    break

    return grid


# ---------------------------------------------------------------------------
# Full grass spreading (all depths)
# ---------------------------------------------------------------------------


def spreadGrass(
    grid: npt.NDArray[np.int32],
) -> npt.NDArray[np.int32]:
    """Spread grass to all dirt tiles adjacent to air at any depth.

    Game does this as a post-generation pass, not just at the surface.
    Uses vectorized 8-neighbor air detection via shifted boolean arrays.

    Returns:
        The modified grid.
    """
    dirtMask = grid == DIRT
    airMask = grid == AIR

    # Check all 8 neighbors for air via shifted arrays
    hasAirNeighbor = np.zeros_like(dirtMask)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.roll(np.roll(airMask, -dy, axis=0), -dx, axis=1)
            hasAirNeighbor |= shifted

    # Zero out the border to avoid wrap-around artifacts from np.roll
    hasAirNeighbor[0, :] = False
    hasAirNeighbor[-1, :] = False
    hasAirNeighbor[:, 0] = False
    hasAirNeighbor[:, -1] = False

    grid[dirtMask & hasAirNeighbor] = GRASS
    return grid
