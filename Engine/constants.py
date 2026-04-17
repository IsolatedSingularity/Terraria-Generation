"""
Terraria world generation constants derived from decompiled C# source.

All values sourced from WorldGen.cs, Main.cs, and tModLoader documentation.
World sizes, layer depths, structure quotas, ore distribution parameters,
tile IDs, wall types, and biome detection thresholds.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class WorldSize:
    """Tile dimensions for each world size class."""
    width: int
    height: int
    area: int

    @property
    def borderBuffer(self) -> int:
        """40-50 tile buffer enforced by RandomWorldPoint fluff parameter."""
        return 45


SMALL = WorldSize(width=4200, height=1200, area=4200 * 1200)
MEDIUM = WorldSize(width=6400, height=1800, area=6400 * 1800)
LARGE = WorldSize(width=8400, height=2400, area=8400 * 2400)


@dataclass(frozen=True)
class LayerDepths:
    """Vertical strata boundaries for a given world.

    In the actual game, worldSurface and rockLayer are dynamically computed
    during terrain generation. These defaults approximate a Large world.
    hellLayer is always maxTilesY - 200.
    """
    worldSurface: float
    rockLayer: float
    hellLayer: int
    maxTilesY: int

    @classmethod
    def forLarge(cls) -> "LayerDepths":
        return cls(
            worldSurface=340.0,
            rockLayer=880.0,
            hellLayer=2400 - 200,
            maxTilesY=2400,
        )

    @classmethod
    def forMedium(cls) -> "LayerDepths":
        return cls(
            worldSurface=280.0,
            rockLayer=720.0,
            hellLayer=1800 - 200,
            maxTilesY=1800,
        )

    @classmethod
    def forSmall(cls) -> "LayerDepths":
        return cls(
            worldSurface=220.0,
            rockLayer=560.0,
            hellLayer=1200 - 200,
            maxTilesY=1200,
        )


@dataclass(frozen=True)
class StructureQuotas:
    """Min/max placement counts per world size, from decompiled source."""
    floatingIslands: int
    undergroundCabinsMin: int
    undergroundCabinsMax: int
    livingMahoganyMin: int
    livingMahoganyMax: int
    marbleCavesMin: int
    marbleCavesMax: int
    minecartTracksMin: int
    minecartTracksMax: int
    surfaceChests: int
    lifeCrystalsMax: int

    @classmethod
    def forLarge(cls) -> "StructureQuotas":
        return cls(
            floatingIslands=6,
            undergroundCabinsMin=140,
            undergroundCabinsMax=160,
            livingMahoganyMin=12,
            livingMahoganyMax=22,
            marbleCavesMin=16,
            marbleCavesMax=32,
            minecartTracksMin=16,
            minecartTracksMax=28,
            surfaceChests=42,
            lifeCrystalsMax=403,
        )

    @classmethod
    def forMedium(cls) -> "StructureQuotas":
        return cls(
            floatingIslands=5,
            undergroundCabinsMin=80,
            undergroundCabinsMax=91,
            livingMahoganyMin=9,
            livingMahoganyMax=16,
            marbleCavesMin=9,
            marbleCavesMax=18,
            minecartTracksMin=9,
            minecartTracksMax=15,
            surfaceChests=32,
            lifeCrystalsMax=230,
        )

    @classmethod
    def forSmall(cls) -> "StructureQuotas":
        return cls(
            floatingIslands=3,
            undergroundCabinsMin=35,
            undergroundCabinsMax=40,
            livingMahoganyMin=6,
            livingMahoganyMax=11,
            marbleCavesMin=4,
            marbleCavesMax=8,
            minecartTracksMin=4,
            minecartTracksMax=7,
            surfaceChests=21,
            lifeCrystalsMax=100,
        )


@dataclass(frozen=True)
class OreConfig:
    """Ore distribution parameters from the Shinies pass.

    The game picks ONE from each alternating pair per world:
    Copper/Tin, Iron/Lead, Silver/Tungsten, Gold/Platinum.
    """
    # Loop count formula: int(area * 6E-05) per ore type
    DENSITY_FACTOR: float = 6e-05

    @staticmethod
    def loopCount(worldArea: int) -> int:
        """Number of TileRunner invocations per ore type."""
        return int(worldArea * 6e-05)

    # Alternating ore pairs (game picks one per world based on seed)
    PRE_HARDMODE_PAIRS: tuple = (
        ("Copper", "Tin"),
        ("Iron", "Lead"),
        ("Silver", "Tungsten"),
        ("Gold", "Platinum"),
    )

    # Hardmode ore tiers (unlocked by breaking altars)
    HARDMODE_TIERS: tuple = (
        ("Cobalt", "Palladium"),
        ("Mythril", "Orichalcum"),
        ("Adamantite", "Titanium"),
    )

    # Depth bounds (as fraction of maxTilesY)
    # Surface ores: above rockLayer
    # Deep ores: below rockLayer
    # Hellstone: only in hell layer (maxTilesY - 200)


# Air gap required to block biome infection spread (3-4 tiles)
INFECTION_GAP_TILES: int = 4

# Biome spread tile update rates (seconds per tile per update)
SURFACE_UPDATE_RATE: float = 140.0
UNDERGROUND_UPDATE_RATE: float = 830.0

# Meteorite threshold: 400 * (maxTilesX / 4200) total tiles allowed
METEORITE_TILES_PER_SMALL: int = 400

# Biome spread radius (tiles checked per update cycle)
INFECTION_SPREAD_RADIUS: int = 3

# ---------------------------------------------------------------------------
# Tile IDs for structures, terrain, and special blocks
# ---------------------------------------------------------------------------
DUNGEON_BRICK: int = 120
LIHZAHRD_BRICK: int = 121
MARBLE_BLOCK: int = 122
GRANITE_BLOCK: int = 123
HARDENED_SAND: int = 124
SANDSTONE: int = 125
COBWEB: int = 126
SHIMMER: int = 127
MUSHROOM_GRASS: int = 128
JUNGLE_GRASS: int = 129
LIVING_WOOD: int = 130
LEAF: int = 131
CLAY: int = 132
SILT: int = 133
MINECART_TRACK: int = 134
CHEST: int = 135
POT: int = 136
SUNFLOWER: int = 137
FALLEN_LOG: int = 138
ALTAR: int = 139
LIFE_CRYSTAL: int = 140
DART_TRAP: int = 141
BOULDER_TRAP: int = 142
PYRAMID_BRICK: int = 143
HIVE: int = 144
HONEY_BLOCK: int = 145

# ---------------------------------------------------------------------------
# Wall IDs (parallel wall array)
# ---------------------------------------------------------------------------
WALL_NONE: int = 0
WALL_DIRT: int = 1
WALL_STONE: int = 2
WALL_DUNGEON_BLUE: int = 7
WALL_DUNGEON_GREEN: int = 8
WALL_DUNGEON_PINK: int = 9
WALL_WOOD: int = 4
WALL_LIHZAHRD: int = 10
WALL_SPIDER: int = 11
WALL_MARBLE: int = 12
WALL_GRANITE: int = 13
WALL_SANDSTONE: int = 14
WALL_MUSHROOM: int = 15
WALL_JUNGLE: int = 16
WALL_HALLOW: int = 17
WALL_CORRUPTION: int = 18
WALL_CRIMSON: int = 19

# ---------------------------------------------------------------------------
# FrameImportant tile IDs (multi-tile objects that must not be overwritten)
# ---------------------------------------------------------------------------
FRAME_IMPORTANT_TILES: frozenset = frozenset({
    LIFE_CRYSTAL, CHEST, POT, DART_TRAP, BOULDER_TRAP,
    ALTAR, SUNFLOWER, FALLEN_LOG, MINECART_TRACK,
})

# ---------------------------------------------------------------------------
# Full immune tile set for Cavinator (CanBeClearedDuringGeneration = False)
# ---------------------------------------------------------------------------
IMMUNE_TILES_FULL: frozenset = frozenset({
    5,    # ASH
    6,    # HELLSTONE
    7,    # MUD
    DUNGEON_BRICK,
    LIHZAHRD_BRICK,
    GRANITE_BLOCK,
    HARDENED_SAND,
    SANDSTONE,
    116,  # CHLOROPHYTE
})

# ---------------------------------------------------------------------------
# Biome detection thresholds (tile counts in 150-tile rectangle)
# ---------------------------------------------------------------------------
BIOME_THRESHOLDS = {
    "hallow": 200,
    "corruption": 200,
    "crimson": 200,
    "jungle": 140,
    "snow": 1500,
    "desert": 1500,
    "mushroom": 100,
    "dungeon": 250,
    "marble": 150,
    "granite": 150,
    "spider": 200,
    "shimmer": 300,
}

# ---------------------------------------------------------------------------
# Structure placement specific constants
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DungeonConfig:
    """Dungeon eating algorithm parameters."""
    minRooms: int = 15
    maxRooms: int = 30
    minRoomWidth: int = 10
    maxRoomWidth: int = 20
    minRoomHeight: int = 8
    maxRoomHeight: int = 16
    wallTypes: tuple = (WALL_DUNGEON_BLUE, WALL_DUNGEON_GREEN, WALL_DUNGEON_PINK)

@dataclass(frozen=True)
class TempleConfig:
    """Jungle Temple (Lihzahrd) parameters."""
    minRooms: int = 10
    maxRooms: int = 20
    minRoomWidth: int = 8
    maxRoomWidth: int = 16
    minRoomHeight: int = 6
    maxRoomHeight: int = 12
    trapDensity: float = 0.15

@dataclass(frozen=True)
class PyramidConfig:
    """Desert pyramid parameters."""
    minWidth: int = 80
    maxWidth: int = 120
    corridorWidth: int = 4
    maxPerWorld: int = 2

@dataclass(frozen=True)
class LivingTreeConfig:
    """Living Tree parameters."""
    trunkWidth: int = 4
    minHeight: int = 40
    maxHeight: int = 80
    canopyRadius: int = 15
    hollowChance: float = 0.5
    branchCount: int = 4

@dataclass(frozen=True)
class ShimmerConfig:
    """Shimmer/Aether biome parameters (post-1.4.4)."""
    radius: int = 30
    correlatedToJungle: bool = True
    depth: str = "cavern"

# ---------------------------------------------------------------------------
# Seed format: size.difficulty.evil.special.identifier
# ---------------------------------------------------------------------------
SEED_SIZES = {"1": "Small", "2": "Medium", "3": "Large"}
SEED_DIFFICULTIES = {"1": "Classic", "2": "Expert", "3": "Master", "4": "Journey"}
SEED_EVILS = {"1": "Corruption", "2": "Crimson", "3": "Random"}
SECRET_SEEDS = {
    "05162020": "drunkWorld",
    "5162020": "drunkWorld",
    "celebrationmk10": "celebrationmk10",
    "getfixedboi": "zenithWorld",
    "not the bees": "notTheBees",
    "for the worthy": "forTheWorthy",
    "don't dig up": "remixWorld",
    "no traps": "noTraps",
}

# ---------------------------------------------------------------------------
# Clentaminator purification parameters
# ---------------------------------------------------------------------------
CLENTAMINATOR_SPRAY_RANGE: int = 60
CLENTAMINATOR_SPRAY_WIDTH: int = 2
PURIFICATION_POWDER_RANGE: int = 4
