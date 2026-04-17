"""
Terraria world generation constants derived from decompiled C# source.

All values sourced from WorldGen.cs, Main.cs, and tModLoader documentation.
World sizes, layer depths, structure quotas, and ore distribution parameters.
"""

from dataclasses import dataclass


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
