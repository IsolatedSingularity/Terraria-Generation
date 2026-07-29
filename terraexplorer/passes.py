"""Canonical named-pass catalogue and fidelity metadata.

The public tModLoader 1.4.4 vanilla list contains 107 named steps. Older
TerraExplorer documentation called this a 103-pass pipeline; this module replaces
that stale count with the complete public list and labels every implementation
honestly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Fidelity(StrEnum):
    MODELED = "modeled"
    APPROXIMATED = "approximated"
    DOCUMENTED = "documented"


class Phase(StrEnum):
    TERRAIN = "Terrain baseline"
    BIOMES = "Caves and biomes"
    STRUCTURES = "Structures"
    SIMULATION = "Simulation"
    POLISH = "Polish"


@dataclass(frozen=True, slots=True)
class PassSpec:
    index: int
    name: str
    phase: Phase
    fidelity: Fidelity
    handler: str | None = None
    note: str = ""


VANILLA_PASS_ORDER = (
    "Reset",
    "Terrain",
    "Dunes",
    "Ocean Sand",
    "Sand Patches",
    "Tunnels",
    "Mount Caves",
    "Dirt Wall Backgrounds",
    "Rocks In Dirt",
    "Dirt In Rocks",
    "Clay",
    "Small Holes",
    "Dirt Layer Caves",
    "Rock Layer Caves",
    "Surface Caves",
    "Wavy Caves",
    "Generate Ice Biome",
    "Grass",
    "Jungle",
    "Mud Caves To Grass",
    "Full Desert",
    "Floating Islands",
    "Mushroom Patches",
    "Marble",
    "Granite",
    "Dirt To Mud",
    "Silt",
    "Shinies",
    "Webs",
    "Underworld",
    "Corruption",
    "Lakes",
    "Dungeon",
    "Slush",
    "Mountain Caves",
    "Beaches",
    "Gems",
    "Gravitating Sand",
    "Create Ocean Caves",
    "Shimmer",
    "Clean Up Dirt",
    "Pyramids",
    "Dirt Rock Wall Runner",
    "Living Trees",
    "Wood Tree Walls",
    "Altars",
    "Wet Jungle",
    "Jungle Temple",
    "Hives",
    "Jungle Chests",
    "Settle Liquids",
    "Remove Water From Sand",
    "Oasis",
    "Shell Piles",
    "Smooth World",
    "Waterfalls",
    "Ice",
    "Wall Variety",
    "Life Crystals",
    "Statues",
    "Buried Chests",
    "Surface Chests",
    "Jungle Chests Placement",
    "Water Chests",
    "Spider Caves",
    "Gem Caves",
    "Moss",
    "Temple",
    "Cave Walls",
    "Jungle Trees",
    "Floating Island Houses",
    "Quick Cleanup",
    "Pots",
    "Hellforge",
    "Spreading Grass",
    "Surface Ore and Stone",
    "Place Fallen Log",
    "Traps",
    "Piles",
    "Spawn Point",
    "Grass Wall",
    "Guide",
    "Sunflowers",
    "Planting Trees",
    "Herbs",
    "Dye Plants",
    "Webs And Honey",
    "Weeds",
    "Glowing Mushrooms and Jungle Plants",
    "Jungle Plants",
    "Vines",
    "Flowers",
    "Mushrooms",
    "Gems In Ice Biome",
    "Random Gems",
    "Moss Grass",
    "Muds Walls In Jungle",
    "Larva",
    "Settle Liquids Again",
    "Cactus, Palm Trees, & Coral",
    "Tile Cleanup",
    "Lihzahrd Altars",
    "Micro Biomes",
    "Water Plants",
    "Stalac",
    "Remove Broken Traps",
    "Final Cleanup",
)


_MODELED_HANDLERS: dict[str, str] = {
    "Reset": "reset",
    "Terrain": "terrain",
    "Dunes": "dunes",
    "Ocean Sand": "ocean_sand",
    "Sand Patches": "sand_patches",
    "Tunnels": "tunnels",
    "Mount Caves": "mount_caves",
    "Dirt Wall Backgrounds": "dirt_walls",
    "Rocks In Dirt": "rocks_in_dirt",
    "Dirt In Rocks": "dirt_in_rocks",
    "Clay": "clay",
    "Small Holes": "small_holes",
    "Dirt Layer Caves": "dirt_caves",
    "Rock Layer Caves": "rock_caves",
    "Surface Caves": "surface_caves",
    "Wavy Caves": "wavy_caves",
    "Generate Ice Biome": "ice_biome",
    "Grass": "grass",
    "Jungle": "jungle",
    "Mud Caves To Grass": "jungle_grass",
    "Full Desert": "full_desert",
    "Floating Islands": "floating_islands",
    "Mushroom Patches": "mushroom_patches",
    "Marble": "marble",
    "Granite": "granite",
    "Dirt To Mud": "dirt_to_mud",
    "Silt": "silt",
    "Shinies": "shinies",
    "Webs": "webs",
    "Underworld": "underworld",
    "Corruption": "evil_biome",
    "Lakes": "lakes",
    "Dungeon": "dungeon",
    "Beaches": "beaches",
    "Gems": "gems",
    "Create Ocean Caves": "ocean_caves",
    "Shimmer": "shimmer",
    "Pyramids": "pyramids",
    "Living Trees": "living_trees",
    "Altars": "altars",
    "Jungle Temple": "jungle_temple",
    "Hives": "hives",
    "Settle Liquids": "settle_liquids",
    "Smooth World": "smooth_world",
    "Waterfalls": "waterfalls",
    "Life Crystals": "life_crystals",
    "Buried Chests": "buried_chests",
    "Surface Chests": "surface_chests",
    "Spider Caves": "spider_caves",
    "Gem Caves": "gem_caves",
    "Cave Walls": "cave_walls",
    "Temple": "temple_polish",
    "Floating Island Houses": "island_houses",
    "Pots": "pots",
    "Hellforge": "hellforge",
    "Spreading Grass": "spreading_grass",
    "Traps": "traps",
    "Spawn Point": "spawn_point",
    "Planting Trees": "planting_trees",
    "Vines": "vines",
    "Flowers": "flowers",
    "Settle Liquids Again": "settle_liquids",
    "Cactus, Palm Trees, & Coral": "coastal_plants",
    "Tile Cleanup": "tile_cleanup",
    "Stalac": "stalactites",
    "Remove Broken Traps": "remove_broken_traps",
    "Final Cleanup": "final_cleanup",
}

_APPROXIMATED_HANDLERS: dict[str, str] = {
    "Slush": "slush",
    "Mountain Caves": "mount_caves",
    "Gravitating Sand": "gravity_sand",
    "Clean Up Dirt": "cleanup_dirt",
    "Dirt Rock Wall Runner": "cave_walls",
    "Wood Tree Walls": "wood_tree_walls",
    "Wet Jungle": "wet_jungle",
    "Jungle Chests": "jungle_chests",
    "Remove Water From Sand": "remove_sand_water",
    "Oasis": "oasis",
    "Shell Piles": "shell_piles",
    "Ice": "ice_polish",
    "Wall Variety": "wall_variety",
    "Statues": "statues",
    "Jungle Chests Placement": "jungle_chests",
    "Water Chests": "water_chests",
    "Moss": "moss",
    "Jungle Trees": "jungle_trees",
    "Quick Cleanup": "tile_cleanup",
    "Surface Ore and Stone": "surface_ore",
    "Place Fallen Log": "fallen_log",
    "Piles": "piles",
    "Grass Wall": "grass_wall",
    "Sunflowers": "sunflowers",
    "Herbs": "flowers",
    "Dye Plants": "flowers",
    "Webs And Honey": "webs_honey",
    "Weeds": "flowers",
    "Glowing Mushrooms and Jungle Plants": "jungle_plants",
    "Jungle Plants": "jungle_plants",
    "Mushrooms": "mushrooms",
    "Gems In Ice Biome": "ice_gems",
    "Random Gems": "gems",
    "Moss Grass": "moss_grass",
    "Muds Walls In Jungle": "mud_walls",
    "Larva": "larva",
    "Lihzahrd Altars": "lihzahrd_altars",
    "Micro Biomes": "micro_biomes",
    "Water Plants": "water_plants",
}


def _phase_for(index: int) -> Phase:
    if index <= 16:
        return Phase.TERRAIN
    if index <= 32:
        return Phase.BIOMES
    if index <= 50:
        return Phase.STRUCTURES
    if index <= 69:
        return Phase.SIMULATION
    return Phase.POLISH


def _build_specs() -> tuple[PassSpec, ...]:
    specs: list[PassSpec] = []
    for index, name in enumerate(VANILLA_PASS_ORDER, start=1):
        if name in _MODELED_HANDLERS:
            fidelity = Fidelity.MODELED
            handler = _MODELED_HANDLERS[name]
        elif name in _APPROXIMATED_HANDLERS:
            fidelity = Fidelity.APPROXIMATED
            handler = _APPROXIMATED_HANDLERS[name]
        else:
            fidelity = Fidelity.DOCUMENTED
            handler = None
        specs.append(PassSpec(index, name, _phase_for(index), fidelity, handler))
    return tuple(specs)


PASS_SPECS = _build_specs()

assert len(VANILLA_PASS_ORDER) == 107
assert len(PASS_SPECS) == 107
