"""Canonical TerraExplorer tile, wall, liquid, and biome registries.

IDs in this module are simulation IDs, not Terraria's proprietary internal
IDs. Keeping one registry eliminates the conflicting meanings found in the
legacy engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Tile(IntEnum):
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
    CLAY = 10
    SILT = 11
    HARDENED_SAND = 12
    SANDSTONE = 13
    JUNGLE_GRASS = 14
    MUSHROOM_GRASS = 15
    EBONSTONE = 16
    CRIMSTONE = 17
    PEARLSTONE = 18
    CORRUPT_GRASS = 19
    CRIMSON_GRASS = 20
    HALLOW_GRASS = 21
    COPPER = 22
    TIN = 23
    IRON = 24
    LEAD = 25
    SILVER = 26
    TUNGSTEN = 27
    GOLD = 28
    PLATINUM = 29
    DUNGEON_BRICK = 30
    LIVING_WOOD = 31
    LEAF = 32
    COBWEB = 33
    MARBLE = 34
    GRANITE = 35
    HIVE = 36
    ALTAR = 37
    CHEST = 38
    LIFE_CRYSTAL = 39
    POT = 40
    TRAP = 41
    GEM = 42
    CACTUS = 43
    TREE = 44
    PLATFORM = 45
    LIHZAHRD_BRICK = 46
    PYRAMID_BRICK = 47
    STALACTITE = 48
    FLOWER = 49
    VINE = 50
    SKY_BRICK = 51
    OBSIDIAN_BRICK = 52
    HELLFORGE = 53
    CLOUD = 54
    RAIN_CLOUD = 55
    METEORITE = 56
    OBSIDIAN = 57
    HONEY_BLOCK = 58
    CRISPY_HONEY_BLOCK = 59
    AETHERIUM = 60
    GEM_TREE = 61
    HELLSTONE_BRICK = 62
    CHLOROPHYTE = 63


class Wall(IntEnum):
    NONE = 0
    DIRT = 1
    STONE = 2
    DUNGEON = 3
    JUNGLE = 4
    SANDSTONE = 5
    MUSHROOM = 6
    SPIDER = 7
    HIVE = 8
    LIHZAHRD = 9
    SKY = 10
    OBSIDIAN = 11
    HELLSTONE = 12


class Liquid(IntEnum):
    NONE = 0
    WATER = 1
    LAVA = 2
    HONEY = 3
    SHIMMER = 4


class Biome(IntEnum):
    SKY = 0
    FOREST = 1
    SNOW = 2
    JUNGLE = 3
    DESERT = 4
    CORRUPTION = 5
    CRIMSON = 6
    HALLOW = 7
    OCEAN = 8
    UNDERWORLD = 9
    MUSHROOM = 10
    DUNGEON = 11
    SHIMMER = 12


@dataclass(frozen=True, slots=True)
class TileStyle:
    name: str
    color: str
    texture: str = "solid"


TILE_STYLES: dict[Tile, TileStyle] = {
    Tile.AIR: TileStyle("Air", "#111827"),
    Tile.DIRT: TileStyle("Dirt", "#765037", "earth"),
    Tile.STONE: TileStyle("Stone", "#6f7680", "stone"),
    Tile.GRASS: TileStyle("Forest grass", "#58a84f", "grass"),
    Tile.SAND: TileStyle("Sand", "#d8bd75", "sand"),
    Tile.ASH: TileStyle("Ash", "#48404c", "stone"),
    Tile.HELLSTONE: TileStyle("Hellstone", "#cb4b36", "ore"),
    Tile.MUD: TileStyle("Mud", "#59452e", "earth"),
    Tile.SNOW: TileStyle("Snow", "#dbe9f4", "snow"),
    Tile.ICE: TileStyle("Ice", "#82c7dc", "ice"),
    Tile.CLAY: TileStyle("Clay", "#ae6c54", "earth"),
    Tile.SILT: TileStyle("Silt", "#80705f", "sand"),
    Tile.HARDENED_SAND: TileStyle("Hardened sand", "#bb945d", "sand"),
    Tile.SANDSTONE: TileStyle("Sandstone", "#9f7548", "stone"),
    Tile.JUNGLE_GRASS: TileStyle("Jungle grass", "#3f9b4f", "grass"),
    Tile.MUSHROOM_GRASS: TileStyle("Mushroom grass", "#5964d8", "grass"),
    Tile.EBONSTONE: TileStyle("Ebonstone", "#67428d", "stone"),
    Tile.CRIMSTONE: TileStyle("Crimstone", "#9b354b", "stone"),
    Tile.PEARLSTONE: TileStyle("Pearlstone", "#d594c8", "stone"),
    Tile.CORRUPT_GRASS: TileStyle("Corrupt grass", "#8b4fb0", "grass"),
    Tile.CRIMSON_GRASS: TileStyle("Crimson grass", "#cc4352", "grass"),
    Tile.HALLOW_GRASS: TileStyle("Hallow grass", "#78d5ce", "grass"),
    Tile.COPPER: TileStyle("Copper", "#c27645", "ore"),
    Tile.TIN: TileStyle("Tin", "#b8c2c9", "ore"),
    Tile.IRON: TileStyle("Iron", "#a99c91", "ore"),
    Tile.LEAD: TileStyle("Lead", "#65747e", "ore"),
    Tile.SILVER: TileStyle("Silver", "#d6d9df", "ore"),
    Tile.TUNGSTEN: TileStyle("Tungsten", "#7fa45c", "ore"),
    Tile.GOLD: TileStyle("Gold", "#e5b844", "ore"),
    Tile.PLATINUM: TileStyle("Platinum", "#d8e5e3", "ore"),
    Tile.DUNGEON_BRICK: TileStyle("Dungeon brick", "#3e5f8d", "brick"),
    Tile.LIVING_WOOD: TileStyle("Living wood", "#70452f", "wood"),
    Tile.LEAF: TileStyle("Leaf", "#3d873f", "leaf"),
    Tile.COBWEB: TileStyle("Cobweb", "#c7ced9", "web"),
    Tile.MARBLE: TileStyle("Marble", "#c9c4bc", "stone"),
    Tile.GRANITE: TileStyle("Granite", "#35415f", "stone"),
    Tile.HIVE: TileStyle("Hive", "#bc7d33", "hive"),
    Tile.ALTAR: TileStyle("Altar", "#9b4ea3", "symbol"),
    Tile.CHEST: TileStyle("Chest", "#d9a82e", "symbol"),
    Tile.LIFE_CRYSTAL: TileStyle("Life crystal", "#f06292", "crystal"),
    Tile.POT: TileStyle("Pot", "#b86f50", "symbol"),
    Tile.TRAP: TileStyle("Trap", "#b44343", "symbol"),
    Tile.GEM: TileStyle("Gem", "#62d6c7", "crystal"),
    Tile.CACTUS: TileStyle("Cactus", "#4a9849", "plant"),
    Tile.TREE: TileStyle("Tree", "#6b412b", "wood"),
    Tile.PLATFORM: TileStyle("Platform", "#9b6d43", "wood"),
    Tile.LIHZAHRD_BRICK: TileStyle("Lihzahrd brick", "#9d6f37", "brick"),
    Tile.PYRAMID_BRICK: TileStyle("Pyramid brick", "#b48a51", "brick"),
    Tile.STALACTITE: TileStyle("Stalactite", "#9ba0a7", "stone"),
    Tile.FLOWER: TileStyle("Flower", "#f0cc59", "plant"),
    Tile.VINE: TileStyle("Vine", "#4a9b45", "plant"),
    Tile.SKY_BRICK: TileStyle("Sky brick", "#c8d7cf", "brick"),
    Tile.OBSIDIAN_BRICK: TileStyle("Obsidian brick", "#3d314c", "brick"),
    Tile.HELLFORGE: TileStyle("Hellforge", "#f18a36", "symbol"),
    Tile.CLOUD: TileStyle("Cloud", "#e8edf2", "cloud"),
    Tile.RAIN_CLOUD: TileStyle("Rain cloud", "#91a4b5", "cloud"),
    Tile.METEORITE: TileStyle("Meteorite", "#77546f", "ore"),
    Tile.OBSIDIAN: TileStyle("Obsidian", "#2f2943", "stone"),
    Tile.HONEY_BLOCK: TileStyle("Honey block", "#d7a434", "solid"),
    Tile.CRISPY_HONEY_BLOCK: TileStyle("Crispy honey block", "#b9682f", "solid"),
    Tile.AETHERIUM: TileStyle("Aetherium", "#9777bd", "crystal"),
    Tile.GEM_TREE: TileStyle("Gem tree", "#62d6c7", "crystal"),
    Tile.HELLSTONE_BRICK: TileStyle("Hellstone brick", "#8f3e37", "brick"),
    Tile.CHLOROPHYTE: TileStyle("Chlorophyte", "#60b84f", "ore"),
}

WALL_COLORS: dict[Wall, str] = {
    Wall.NONE: "#111827",
    Wall.DIRT: "#44362f",
    Wall.STONE: "#3e424b",
    Wall.DUNGEON: "#263a5b",
    Wall.JUNGLE: "#273f2d",
    Wall.SANDSTONE: "#6e563d",
    Wall.MUSHROOM: "#333a80",
    Wall.SPIDER: "#4a424b",
    Wall.HIVE: "#6a4728",
    Wall.LIHZAHRD: "#5b422a",
    Wall.SKY: "#596d76",
    Wall.OBSIDIAN: "#281f33",
    Wall.HELLSTONE: "#542c2c",
}

BIOME_NAMES = {biome: biome.name.replace("_", " ").title() for biome in Biome}
