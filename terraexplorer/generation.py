"""Modeled TerraExplorer pass handlers.

The handlers favor deterministic, readable approximations over claims of
bit-for-bit compatibility. Every random decision comes from the per-pass RNG
provided by :mod:`terraexplorer.pipeline`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from terraexplorer.config import Evil, WorldScale
from terraexplorer.geometry import smooth_noise_1d, stamp_ellipse, stamp_walk, surface_candidates
from terraexplorer.model import GeneratedWorld, StructureMarker
from terraexplorer.tiles import Biome, Liquid, Tile, Wall

PassHandler = Callable[[GeneratedWorld, np.random.Generator], None]

_CARVABLE = (
    Tile.DIRT,
    Tile.STONE,
    Tile.GRASS,
    Tile.MUD,
    Tile.SNOW,
    Tile.ICE,
    Tile.CLAY,
    Tile.SILT,
    Tile.HARDENED_SAND,
    Tile.SANDSTONE,
    Tile.JUNGLE_GRASS,
    Tile.MUSHROOM_GRASS,
    Tile.EBONSTONE,
    Tile.CRIMSTONE,
    Tile.PEARLSTONE,
    Tile.CORRUPT_GRASS,
    Tile.CRIMSON_GRASS,
    Tile.HALLOW_GRASS,
)
_ORE_HOSTS = (
    Tile.DIRT,
    Tile.STONE,
    Tile.MUD,
    Tile.ICE,
    Tile.SNOW,
    Tile.EBONSTONE,
    Tile.CRIMSTONE,
    Tile.PEARLSTONE,
)


def _pick(world: GeneratedWorld, preview: int | float, small: int | float):
    return preview if world.config.scale is WorldScale.PREVIEW else small


def _band(world: GeneratedWorld, center: int, half_width: int) -> tuple[int, int]:
    return max(1, center - half_width), min(world.config.width - 1, center + half_width)


def _place_marker(
    world: GeneratedWorld,
    kind: str,
    x: int,
    y: int,
    width: int,
    height: int,
    symbol: str,
) -> None:
    world.structures.append(StructureMarker(kind, x, y, width, height, symbol))


def _annotate_only(world: GeneratedWorld, rng: np.random.Generator) -> None:
    del world, rng


def reset(world: GeneratedWorld, rng: np.random.Generator) -> None:
    width = world.config.width
    dungeon_side = -1 if rng.integers(0, 2) == 0 else 1
    snow_x = round(width * (0.14 if dungeon_side < 0 else 0.86))
    jungle_x = round(width * (0.78 if dungeon_side < 0 else 0.22))
    desert_x = round(width * (0.68 if dungeon_side < 0 else 0.32))
    evil_x = round(width * (0.30 if rng.integers(0, 2) == 0 else 0.72))
    if abs(evil_x - width // 2) < round(width * 0.12):
        evil_x = round(width * 0.78)

    ore_pairs = (
        (Tile.COPPER, Tile.TIN),
        (Tile.IRON, Tile.LEAD),
        (Tile.SILVER, Tile.TUNGSTEN),
        (Tile.GOLD, Tile.PLATINUM),
    )
    selected_ores = tuple(pair[int(rng.integers(0, 2))] for pair in ore_pairs)
    world.metadata.update(
        {
            "seed": str(world.config.seed),
            "seed_value": world.config.seed_value,
            "scale": world.config.scale.value,
            "evil": world.config.evil.value,
            "difficulty": world.config.difficulty.value,
            "hardmode_requested": world.config.hardmode,
            "dungeon_side": "left" if dungeon_side < 0 else "right",
            "dungeon_x": round(width * (0.07 if dungeon_side < 0 else 0.93)),
            "spawn_x": width // 2,
            "snow_x": snow_x,
            "jungle_x": jungle_x,
            "desert_x": desert_x,
            "evil_x": evil_x,
            "selected_ores": [ore.name.title() for ore in selected_ores],
            "selected_ore_ids": selected_ores,
        }
    )


def terrain(world: GeneratedWorld, rng: np.random.Generator) -> None:
    height, width = world.shape
    layers = world.layers
    spacing = int(_pick(world, 18, 150))
    broad = smooth_noise_1d(width, rng, float(_pick(world, 4.2, 23.0)), spacing)
    fine = smooth_noise_1d(width, rng, float(_pick(world, 1.2, 6.0)), max(4, spacing // 3))
    phase = rng.uniform(0, np.pi * 2)
    wave = np.sin(np.linspace(phase, phase + np.pi * 5, width)) * _pick(world, 1.5, 8.0)
    surface = np.clip(
        layers.world_surface + broad + fine + wave,
        round(height * 0.10),
        round(height * 0.30),
    ).astype(np.int16)
    world.surface[:] = surface

    rows = np.arange(height, dtype=np.int16)[:, None]
    world.tiles[:] = Tile.STONE
    world.tiles[rows < surface[None, :]] = Tile.AIR
    dirt = (rows >= surface[None, :]) & (rows < layers.rock_layer)
    world.tiles[dirt] = Tile.DIRT
    world.biomes[:] = Biome.FOREST
    world.biomes[rows < surface[None, :]] = Biome.SKY
    world.biomes[layers.underworld :, :] = Biome.UNDERWORLD


def dunes(world: GeneratedWorld, rng: np.random.Generator) -> None:
    center = int(world.metadata["desert_x"])
    half = int(_pick(world, 20, 280))
    x0, x1 = _band(world, center, half)
    x = np.arange(x0, x1)
    shape = np.sin(np.linspace(0, np.pi * 3, x.size)) ** 2
    noise = rng.integers(0, int(_pick(world, 2, 8)) + 1, size=x.size)
    heights = (shape * _pick(world, 5, 28) + noise).astype(int)
    for local, column in enumerate(x):
        top = max(1, int(world.surface[column]) - int(heights[local]))
        bottom = min(world.shape[0], int(world.surface[column]) + int(_pick(world, 5, 35)))
        world.tiles[top:bottom, column] = Tile.SAND
        world.surface[column] = top


def ocean_sand(world: GeneratedWorld, rng: np.random.Generator) -> None:
    del rng
    height, width = world.shape
    coast = int(_pick(world, 22, 300))
    sea_level = world.layers.world_surface + int(_pick(world, 2, 12))
    for start, end in ((0, coast), (width - coast, width)):
        for x in range(start, end):
            edge_distance = min(x, width - 1 - x)
            floor = min(height - 2, sea_level + max(2, (coast - edge_distance) // 3))
            world.tiles[floor:, x][world.tiles[floor:, x] == Tile.DIRT] = Tile.SAND
            world.tiles[sea_level:floor, x] = Tile.AIR
            world.liquid_kind[sea_level:floor, x] = Liquid.WATER
            world.liquid_amount[sea_level:floor, x] = 255
            world.biomes[:, x] = Biome.OCEAN
            world.surface[x] = floor


def sand_patches(world: GeneratedWorld, rng: np.random.Generator) -> None:
    count = int(_pick(world, 5, 80))
    for _ in range(count):
        stamp_ellipse(
            world.tiles,
            int(rng.integers(5, world.config.width - 5)),
            int(rng.integers(world.layers.world_surface, world.layers.rock_layer)),
            int(rng.integers(*_pick(world, (2, 5), (5, 15)))),
            int(rng.integers(*_pick(world, (1, 3), (3, 8)))),
            Tile.SAND,
            (Tile.DIRT, Tile.STONE),
        )


def _carve_walks(
    world: GeneratedWorld,
    rng: np.random.Generator,
    count: int,
    y_min: int,
    y_max: int,
    radius: tuple[int, int],
    steps: tuple[int, int],
    drift_y: float = 0.2,
) -> None:
    for _ in range(count):
        stamp_walk(
            world.tiles,
            rng,
            int(rng.integers(3, world.config.width - 3)),
            int(rng.integers(y_min, max(y_min + 1, y_max))),
            int(rng.integers(*steps)),
            radius,
            Tile.AIR,
            (float(rng.uniform(-0.9, 0.9)), drift_y),
            _CARVABLE,
        )


def tunnels(world: GeneratedWorld, rng: np.random.Generator) -> None:
    _carve_walks(
        world,
        rng,
        int(_pick(world, 8, 100)),
        world.layers.world_surface + 3,
        world.layers.underworld - 5,
        tuple(_pick(world, (3, 1), (9, 3))),
        tuple(_pick(world, (12, 24), (40, 90))),
    )


def mount_caves(world: GeneratedWorld, rng: np.random.Generator) -> None:
    _carve_walks(
        world,
        rng,
        int(_pick(world, 3, 32)),
        world.layers.world_surface,
        world.layers.rock_layer,
        tuple(_pick(world, (3, 1), (7, 2))),
        tuple(_pick(world, (8, 18), (25, 55))),
        0.5,
    )


def dirt_walls(world: GeneratedWorld, rng: np.random.Generator) -> None:
    del rng
    rows = np.arange(world.shape[0])[:, None]
    world.walls[(rows >= world.surface[None, :]) & (rows < world.layers.rock_layer)] = Wall.DIRT
    world.walls[world.layers.rock_layer : world.layers.underworld, :] = Wall.STONE


def _inject_walks(
    world: GeneratedWorld,
    rng: np.random.Generator,
    count: int,
    y0: int,
    y1: int,
    tile: Tile,
    replace: tuple[Tile, ...],
) -> None:
    for _ in range(count):
        stamp_walk(
            world.tiles,
            rng,
            int(rng.integers(5, world.config.width - 5)),
            int(rng.integers(y0, y1)),
            int(rng.integers(*_pick(world, (2, 5), (5, 14)))),
            tuple(_pick(world, (2, 1), (5, 2))),
            tile,
            replace=replace,
        )


def rocks_in_dirt(world: GeneratedWorld, rng: np.random.Generator) -> None:
    _inject_walks(
        world,
        rng,
        int(_pick(world, 10, 180)),
        world.layers.world_surface,
        world.layers.rock_layer,
        Tile.STONE,
        (Tile.DIRT,),
    )


def dirt_in_rocks(world: GeneratedWorld, rng: np.random.Generator) -> None:
    _inject_walks(
        world,
        rng,
        int(_pick(world, 8, 150)),
        world.layers.rock_layer,
        world.layers.underworld,
        Tile.DIRT,
        (Tile.STONE,),
    )


def clay(world: GeneratedWorld, rng: np.random.Generator) -> None:
    _inject_walks(
        world,
        rng,
        int(_pick(world, 5, 60)),
        world.layers.world_surface,
        world.layers.rock_layer,
        Tile.CLAY,
        (Tile.DIRT,),
    )


def small_holes(world: GeneratedWorld, rng: np.random.Generator) -> None:
    _carve_walks(
        world,
        rng,
        int(_pick(world, 18, 260)),
        world.layers.world_surface + 2,
        world.layers.underworld,
        tuple(_pick(world, (2, 1), (4, 1))),
        tuple(_pick(world, (2, 6), (4, 12))),
    )


def dirt_caves(world: GeneratedWorld, rng: np.random.Generator) -> None:
    _carve_walks(
        world,
        rng,
        int(_pick(world, 16, 210)),
        world.layers.world_surface + 2,
        world.layers.rock_layer,
        tuple(_pick(world, (4, 2), (10, 4))),
        tuple(_pick(world, (8, 20), (20, 55))),
    )


def rock_caves(world: GeneratedWorld, rng: np.random.Generator) -> None:
    _carve_walks(
        world,
        rng,
        int(_pick(world, 28, 430)),
        world.layers.rock_layer - int(_pick(world, 5, 30)),
        world.layers.underworld - 4,
        tuple(_pick(world, (5, 2), (13, 4))),
        tuple(_pick(world, (10, 28), (28, 70))),
    )


def surface_caves(world: GeneratedWorld, rng: np.random.Generator) -> None:
    count = int(_pick(world, 3, 14))
    coast = int(_pick(world, 25, 350))
    for _ in range(count):
        x = int(rng.integers(coast, world.config.width - coast))
        stamp_walk(
            world.tiles,
            rng,
            x,
            int(world.surface[x]) - 1,
            int(_pick(world, 18, 55)),
            tuple(_pick(world, (3, 1), (8, 2))),
            Tile.AIR,
            (float(rng.uniform(-0.25, 0.25)), 1.2),
            _CARVABLE,
        )


def wavy_caves(world: GeneratedWorld, rng: np.random.Generator) -> None:
    _carve_walks(
        world,
        rng,
        int(_pick(world, 6, 65)),
        world.layers.rock_layer,
        world.layers.underworld,
        tuple(_pick(world, (3, 1), (7, 2))),
        tuple(_pick(world, (20, 40), (70, 140))),
        0.0,
    )


def ice_biome(world: GeneratedWorld, rng: np.random.Generator) -> None:
    del rng
    center = int(world.metadata["snow_x"])
    half = int(_pick(world, 24, 430))
    x0, x1 = _band(world, center, half)
    region = world.tiles[:, x0:x1]
    region[region == Tile.DIRT] = Tile.SNOW
    region[region == Tile.STONE] = Tile.ICE
    region[region == Tile.GRASS] = Tile.SNOW
    rows = np.arange(world.shape[0])[:, None]
    mask = rows >= world.surface[None, x0:x1]
    world.biomes[:, x0:x1][mask] = Biome.SNOW


def grass(world: GeneratedWorld, rng: np.random.Generator) -> None:
    del rng
    exposed = surface_candidates(world.tiles, (Tile.DIRT,))
    world.tiles[exposed] = Tile.GRASS


def jungle(world: GeneratedWorld, rng: np.random.Generator) -> None:
    del rng
    center = int(world.metadata["jungle_x"])
    half = int(_pick(world, 28, 520))
    x0, x1 = _band(world, center, half)
    region = world.tiles[:, x0:x1]
    region[np.isin(region, (Tile.DIRT, Tile.GRASS))] = Tile.MUD
    rows = np.arange(world.shape[0])[:, None]
    mask = (rows >= world.surface[None, x0:x1]) & (rows < world.layers.underworld)
    world.biomes[:, x0:x1][mask] = Biome.JUNGLE
    world.walls[:, x0:x1][world.walls[:, x0:x1] != Wall.NONE] = Wall.JUNGLE


def jungle_grass(world: GeneratedWorld, rng: np.random.Generator) -> None:
    del rng
    exposed = surface_candidates(world.tiles, (Tile.MUD,))
    world.tiles[exposed] = Tile.JUNGLE_GRASS


def full_desert(world: GeneratedWorld, rng: np.random.Generator) -> None:
    center = int(world.metadata["desert_x"])
    center_y = round((world.layers.world_surface + world.layers.rock_layer) * 0.65)
    rx, ry = tuple(_pick(world, (24, 30), (330, 280)))
    stamp_ellipse(
        world.tiles,
        center,
        center_y,
        int(rx),
        int(ry),
        Tile.HARDENED_SAND,
        _CARVABLE,
    )
    stamp_ellipse(
        world.tiles,
        center,
        center_y + int(ry * 0.15),
        max(2, int(rx * 0.82)),
        max(2, int(ry * 0.72)),
        Tile.SANDSTONE,
        (Tile.HARDENED_SAND,),
    )
    for _ in range(int(_pick(world, 5, 25))):
        stamp_walk(
            world.tiles,
            rng,
            int(center + rng.integers(-rx // 2, rx // 2 + 1)),
            int(center_y + rng.integers(-ry // 2, ry // 2 + 1)),
            int(_pick(world, 12, 45)),
            tuple(_pick(world, (3, 1), (8, 3))),
            Tile.AIR,
            (float(rng.uniform(-0.5, 0.5)), float(rng.uniform(-0.2, 0.6))),
            (Tile.HARDENED_SAND, Tile.SANDSTONE),
        )
    x0, x1 = _band(world, center, int(rx))
    world.biomes[world.layers.world_surface : world.layers.rock_layer + int(ry), x0:x1] = (
        Biome.DESERT
    )
    world.walls[:, x0:x1][world.walls[:, x0:x1] != Wall.NONE] = Wall.SANDSTONE


def floating_islands(world: GeneratedWorld, rng: np.random.Generator) -> None:
    count = int(_pick(world, 2, 3))
    coast = int(_pick(world, 25, 420))
    for _ in range(count):
        x = int(rng.integers(coast, world.config.width - coast))
        y = int(
            rng.integers(max(8, world.layers.world_surface // 4), world.layers.world_surface - 5)
        )
        rx, ry = tuple(_pick(world, (9, 4), (45, 16)))
        stamp_ellipse(world.tiles, x, y, int(rx), int(ry), Tile.DIRT, (Tile.AIR,))
        top = max(1, y - int(ry))
        world.tiles[top, max(0, x - int(rx) + 2) : min(world.config.width, x + int(rx) - 1)] = (
            Tile.GRASS
        )
        _place_marker(
            world, "Floating island", x - int(rx), y - int(ry), int(rx * 2), int(ry * 2), "◇"
        )


def mushroom_patches(world: GeneratedWorld, rng: np.random.Generator) -> None:
    count = int(_pick(world, 2, 9))
    for _ in range(count):
        x = int(rng.integers(8, world.config.width - 8))
        y = int(rng.integers(world.layers.rock_layer, world.layers.underworld - 5))
        radius = int(_pick(world, rng.integers(6, 11), rng.integers(25, 55)))
        stamp_ellipse(world.tiles, x, y, radius, max(3, radius // 2), Tile.MUD, _CARVABLE)
        stamp_ellipse(
            world.tiles, x, y, max(2, radius - 2), max(2, radius // 2 - 1), Tile.AIR, (Tile.MUD,)
        )
        local = world.tiles[
            max(0, y - radius) : y + radius + 1, max(0, x - radius) : x + radius + 1
        ]
        exposed = surface_candidates(local, (Tile.MUD,))
        local[exposed] = Tile.MUSHROOM_GRASS
        world.biomes[max(0, y - radius) : y + radius + 1, max(0, x - radius) : x + radius + 1] = (
            Biome.MUSHROOM
        )


def _stone_biome(world: GeneratedWorld, rng: np.random.Generator, tile: Tile) -> None:
    count = int(_pick(world, 2, 6))
    for _ in range(count):
        x = int(rng.integers(10, world.config.width - 10))
        y = int(rng.integers(world.layers.rock_layer, world.layers.underworld - 5))
        rx, ry = tuple(_pick(world, (7, 4), (35, 18)))
        stamp_ellipse(world.tiles, x, y, int(rx), int(ry), tile, _CARVABLE)
        stamp_ellipse(
            world.tiles, x, y, max(2, int(rx) - 2), max(2, int(ry) - 2), Tile.AIR, (tile,)
        )


def marble(world: GeneratedWorld, rng: np.random.Generator) -> None:
    _stone_biome(world, rng, Tile.MARBLE)


def granite(world: GeneratedWorld, rng: np.random.Generator) -> None:
    _stone_biome(world, rng, Tile.GRANITE)


def dirt_to_mud(world: GeneratedWorld, rng: np.random.Generator) -> None:
    center = int(world.metadata["jungle_x"])
    half = int(_pick(world, 30, 600))
    x0, x1 = _band(world, center, half)
    region = world.tiles[world.layers.rock_layer : world.layers.underworld, x0:x1]
    chance = rng.random(region.shape) < 0.08
    region[chance & (region == Tile.DIRT)] = Tile.MUD


def silt(world: GeneratedWorld, rng: np.random.Generator) -> None:
    _inject_walks(
        world,
        rng,
        int(_pick(world, 5, 90)),
        world.layers.rock_layer,
        world.layers.underworld,
        Tile.SILT,
        (Tile.STONE, Tile.DIRT),
    )


def _place_ore(
    world: GeneratedWorld,
    rng: np.random.Generator,
    tile: Tile,
    count: int,
    y0: int,
    y1: int,
) -> None:
    for _ in range(count):
        x = int(rng.integers(4, world.config.width - 4))
        y = int(rng.integers(y0, y1))
        stamp_walk(
            world.tiles,
            rng,
            x,
            y,
            int(_pick(world, rng.integers(2, 4), rng.integers(3, 7))),
            tuple(_pick(world, (2, 1), (4, 2))),
            tile,
            replace=_ORE_HOSTS,
        )


def shinies(world: GeneratedWorld, rng: np.random.Generator) -> None:
    selected = tuple(world.metadata["selected_ore_ids"])
    base_count = max(int(_pick(world, 5, world.config.width * world.config.height * 0.00006)), 1)
    ranges = (
        (world.layers.world_surface, world.layers.rock_layer),
        (world.layers.world_surface + 4, round(world.layers.underworld * 0.70)),
        (world.layers.rock_layer, round(world.layers.underworld * 0.90)),
        (round(world.layers.rock_layer * 1.10), world.layers.underworld),
    )
    for tile, (y0, y1) in zip(selected, ranges, strict=True):
        _place_ore(world, rng, tile, base_count, y0, y1)


def webs(world: GeneratedWorld, rng: np.random.Generator) -> None:
    candidates = np.argwhere(
        (world.tiles == Tile.AIR)
        & (np.arange(world.shape[0])[:, None] > world.layers.world_surface)
        & (world.walls != Wall.NONE)
    )
    if candidates.size == 0:
        return
    count = min(len(candidates), int(_pick(world, 30, 950)))
    chosen = candidates[rng.choice(len(candidates), count, replace=False)]
    world.tiles[chosen[:, 0], chosen[:, 1]] = Tile.COBWEB


def underworld(world: GeneratedWorld, rng: np.random.Generator) -> None:
    height, width = world.shape
    top = world.layers.underworld
    depth = height - top
    columns = np.arange(width)
    noise = rng.normal(0.0, 1.0, width)
    smoothing = max(3, int(_pick(world, 5, 55)))
    noise = np.convolve(noise, np.ones(smoothing) / smoothing, mode="same")
    floor_profile = (
        top
        + depth * 0.74
        + np.sin(columns / max(4, _pick(world, 10, 130))) * depth * 0.09
        + np.sin(columns / max(3, _pick(world, 4, 47))) * depth * 0.04
        + noise * depth * 0.12
    ).astype(np.int16)
    floor_profile = np.clip(
        floor_profile,
        top + max(5, round(depth * 0.56)),
        height - 3,
    )
    lava_level = top + round(depth * 0.58)
    world.metadata["underworld_lava_level"] = lava_level

    world.tiles[top:, :] = Tile.AIR
    world.biomes[top:, :] = Biome.UNDERWORLD
    world.walls[top:, :] = Wall.STONE
    world.liquid_kind[top:, :] = Liquid.NONE
    world.liquid_amount[top:, :] = 0
    for x, floor_y in enumerate(floor_profile):
        world.tiles[floor_y:, x] = Tile.ASH
        hellstone_y = floor_y + max(2, (height - int(floor_y)) // 2)
        world.tiles[hellstone_y:, x] = Tile.HELLSTONE

    for _ in range(int(_pick(world, 8, 65))):
        x = int(rng.integers(4, width - 4))
        y = int(rng.integers(top + 3, max(top + 4, lava_level + 1)))
        rx, ry = tuple(_pick(world, (4, 2), (22, 10)))
        material = Tile.HELLSTONE if rng.random() < 0.25 else Tile.ASH
        stamp_ellipse(world.tiles, x, y, int(rx), int(ry), material, (Tile.AIR,))

    for _ in range(int(_pick(world, 12, 120))):
        x = int(rng.integers(4, width - 4))
        y = int(rng.integers(lava_level, height - 3))
        rx, ry = tuple(_pick(world, (5, 3), (18, 9)))
        stamp_ellipse(world.tiles, x, y, int(rx), int(ry), Tile.AIR, (Tile.ASH, Tile.HELLSTONE))

    lava_region = world.tiles[lava_level:, :] == Tile.AIR
    world.liquid_kind[lava_level:, :][lava_region] = Liquid.LAVA
    world.liquid_amount[lava_level:, :][lava_region] = 255

    city_count = int(_pick(world, 2, 8))
    city_span = int(_pick(world, 42, 360))
    for _ in range(city_count):
        start_x = int(rng.integers(4, max(5, width - city_span - 4)))
        house_count = int(rng.integers(3, 6))
        cursor = start_x
        city_top = height
        city_bottom = top
        street_y = int(
            np.clip(
                lava_level + rng.integers(-int(_pick(world, 1, 7)), int(_pick(world, 2, 9))),
                top + 8,
                height - 5,
            )
        )
        for _house in range(house_count):
            room_w = int(_pick(world, rng.integers(9, 15), rng.integers(28, 48)))
            room_h = int(_pick(world, rng.integers(7, 11), rng.integers(16, 28)))
            if cursor + room_w >= min(width - 2, start_x + city_span):
                break
            room_y = max(top + 2, street_y - room_h)
            _carve_room(
                world,
                cursor,
                room_y,
                room_w,
                room_h,
                Tile.OBSIDIAN_BRICK,
                Wall.OBSIDIAN,
            )
            world.liquid_kind[room_y:street_y, cursor : cursor + room_w] = Liquid.NONE
            world.liquid_amount[room_y:street_y, cursor : cursor + room_w] = 0
            if room_h >= int(_pick(world, 9, 22)):
                shelf_y = room_y + room_h // 2
                world.tiles[shelf_y, cursor + 1 : cursor + room_w - 1] = Tile.PLATFORM
                stair_x = cursor + room_w // 2
                world.tiles[shelf_y, stair_x - 1 : stair_x + 2] = Tile.AIR
            merlon_width = max(1, int(_pick(world, 1, 3)))
            for merlon_x in range(cursor, cursor + room_w, merlon_width * 3):
                world.tiles[
                    max(top + 1, room_y - merlon_width) : room_y,
                    merlon_x : merlon_x + merlon_width,
                ] = Tile.OBSIDIAN_BRICK
            doorway_y = room_y + room_h - int(_pick(world, 3, 5))
            world.tiles[doorway_y : room_y + room_h - 1, cursor] = Tile.AIR
            bridge_y = room_y + room_h - 1
            bridge_end = min(width - 1, cursor + room_w + int(_pick(world, 3, 16)))
            world.tiles[bridge_y, cursor + room_w : bridge_end] = Tile.PLATFORM
            world.liquid_kind[bridge_y, cursor + room_w : bridge_end] = Liquid.NONE
            world.liquid_amount[bridge_y, cursor + room_w : bridge_end] = 0
            support_columns = (cursor + 1, cursor + room_w // 2, cursor + room_w - 2)
            for support_x in support_columns:
                support_bottom = max(street_y + 2, int(floor_profile[support_x]) + 1)
                support_bottom = min(height - 1, support_bottom)
                world.tiles[street_y:support_bottom, support_x] = Tile.OBSIDIAN_BRICK
                world.liquid_kind[street_y:support_bottom, support_x] = Liquid.NONE
                world.liquid_amount[street_y:support_bottom, support_x] = 0
                city_bottom = max(city_bottom, support_bottom)
            city_top = min(city_top, room_y)
            city_bottom = max(city_bottom, room_y + room_h)
            cursor = bridge_end
        if cursor > start_x and city_top < height:
            _place_marker(
                world,
                "Underworld city",
                start_x,
                city_top,
                cursor - start_x,
                city_bottom - city_top,
                "▥",
            )
    solid = world.tiles != Tile.AIR
    world.liquid_kind[solid] = Liquid.NONE
    world.liquid_amount[solid] = 0


def evil_biome(world: GeneratedWorld, rng: np.random.Generator) -> None:
    center = int(world.metadata["evil_x"])
    half = int(_pick(world, 22, 350))
    x0, x1 = _band(world, center, half)
    region = world.tiles[:, x0:x1]
    evil_stone = Tile.EBONSTONE if world.config.evil is Evil.CORRUPTION else Tile.CRIMSTONE
    evil_grass = Tile.CORRUPT_GRASS if world.config.evil is Evil.CORRUPTION else Tile.CRIMSON_GRASS
    region[np.isin(region, (Tile.STONE, Tile.DIRT))] = evil_stone
    region[np.isin(region, (Tile.GRASS, Tile.JUNGLE_GRASS))] = evil_grass
    rows = np.arange(world.shape[0])[:, None]
    biome_id = Biome.CORRUPTION if world.config.evil is Evil.CORRUPTION else Biome.CRIMSON
    mask = (rows >= world.surface[None, x0:x1]) & (rows < world.layers.underworld)
    world.biomes[:, x0:x1][mask] = biome_id

    if world.config.evil is Evil.CORRUPTION:
        for offset in (-half // 2, half // 3):
            x = center + offset
            stamp_walk(
                world.tiles,
                rng,
                x,
                int(world.surface[x]) - 1,
                int(_pick(world, 48, 360)),
                tuple(_pick(world, (4, 2), (12, 4))),
                Tile.AIR,
                (float(rng.uniform(-0.12, 0.12)), float(_pick(world, 1.6, 2.1))),
                _CARVABLE,
            )
            branch_y = int(rng.integers(world.layers.rock_layer, world.layers.underworld - 5))
            stamp_walk(
                world.tiles,
                rng,
                x,
                branch_y,
                int(_pick(world, 22, 95)),
                tuple(_pick(world, (3, 1), (8, 3))),
                Tile.AIR,
                (float(rng.choice((-1.2, 1.2))), 0.1),
                _CARVABLE,
            )
    else:
        bulbs = int(_pick(world, 4, 13))
        previous: tuple[int, int] | None = None
        for index in range(bulbs):
            x = int(center + rng.integers(-half, half + 1))
            y = int(
                world.layers.world_surface
                + (world.layers.underworld - world.layers.world_surface) * (index + 1) / (bulbs + 1)
            )
            rx, ry = tuple(_pick(world, (7, 5), (28, 20)))
            stamp_ellipse(world.tiles, x, y, int(rx), int(ry), Tile.AIR, _CARVABLE)
            if previous is not None:
                px, py = previous
                stamp_walk(
                    world.tiles,
                    rng,
                    px,
                    py,
                    int(_pick(world, 16, 70)),
                    tuple(_pick(world, (3, 1), (7, 2))),
                    Tile.AIR,
                    (
                        (x - px) / max(8, _pick(world, 16, 70)),
                        (y - py) / max(8, _pick(world, 16, 70)),
                    ),
                    _CARVABLE,
                )
            previous = (x, y)

    fringe_length = int(_pick(world, 18, 160))
    for side, start_x in ((-1, x0), (1, x1 - 1)):
        for _ in range(int(_pick(world, 3, 8))):
            y = int(rng.integers(world.layers.world_surface, world.layers.underworld - 4))
            stamp_walk(
                world.tiles,
                rng,
                start_x,
                y,
                fringe_length,
                tuple(_pick(world, (2, 1), (7, 3))),
                evil_stone,
                (side * float(rng.uniform(0.45, 0.9)), float(rng.uniform(-0.2, 0.3))),
                _ORE_HOSTS,
            )
    evil_tiles = np.isin(world.tiles, (evil_stone, evil_grass))
    world.biomes[evil_tiles] = biome_id
    advance_biome_spread(world, rng, iterations=int(_pick(world, 2, 4)))


def advance_biome_spread(
    world: GeneratedWorld,
    rng: np.random.Generator,
    *,
    iterations: int = 1,
) -> None:
    """Advance deterministic evil and Hallow growth into adjacent natural tiles."""

    evil_biome_id = Biome.CORRUPTION if world.config.evil is Evil.CORRUPTION else Biome.CRIMSON
    evil_stone = Tile.EBONSTONE if world.config.evil is Evil.CORRUPTION else Tile.CRIMSTONE
    evil_grass = Tile.CORRUPT_GRASS if world.config.evil is Evil.CORRUPTION else Tile.CRIMSON_GRASS
    natural = np.isin(
        world.tiles,
        (
            Tile.DIRT,
            Tile.STONE,
            Tile.GRASS,
            Tile.SAND,
            Tile.HARDENED_SAND,
            Tile.SANDSTONE,
            Tile.ICE,
            Tile.SNOW,
        ),
    )
    for _ in range(max(0, iterations)):
        evil = world.biomes == evil_biome_id
        hallow = world.biomes == Biome.HALLOW
        evil_edge = np.zeros(world.shape, dtype=bool)
        hallow_edge = np.zeros(world.shape, dtype=bool)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            evil_edge |= np.roll(np.roll(evil, dy, axis=0), dx, axis=1)
            hallow_edge |= np.roll(np.roll(hallow, dy, axis=0), dx, axis=1)
        chance = rng.random(world.shape)
        evil_growth = natural & evil_edge & (chance < 0.24)
        hallow_growth = natural & hallow_edge & (chance >= 0.24) & (chance < 0.46)
        evil_growth[[0, -1], :] = False
        evil_growth[:, [0, -1]] = False
        hallow_growth[[0, -1], :] = False
        hallow_growth[:, [0, -1]] = False
        grass = np.isin(world.tiles, (Tile.GRASS, Tile.JUNGLE_GRASS))
        world.tiles[evil_growth & grass] = evil_grass
        world.tiles[evil_growth & ~grass] = evil_stone
        world.tiles[hallow_growth & grass] = Tile.HALLOW_GRASS
        world.tiles[hallow_growth & ~grass] = Tile.PEARLSTONE
        world.biomes[evil_growth] = evil_biome_id
        world.biomes[hallow_growth] = Biome.HALLOW


def lakes(world: GeneratedWorld, rng: np.random.Generator) -> None:
    count = int(_pick(world, 3, 18))
    coast = int(_pick(world, 25, 400))
    for _ in range(count):
        x = int(rng.integers(coast, world.config.width - coast))
        y = int(world.surface[x] + _pick(world, 4, 15))
        rx, ry = tuple(_pick(world, (6, 3), (28, 10)))
        stamp_ellipse(world.tiles, x, y, int(rx), int(ry), Tile.AIR, _CARVABLE)
        y0, y1 = y, min(world.shape[0], y + int(ry) + 1)
        x0, x1 = max(0, x - int(rx)), min(world.shape[1], x + int(rx) + 1)
        air = world.tiles[y0:y1, x0:x1] == Tile.AIR
        world.liquid_kind[y0:y1, x0:x1][air] = Liquid.WATER
        world.liquid_amount[y0:y1, x0:x1][air] = 255


def _carve_room(
    world: GeneratedWorld,
    x: int,
    y: int,
    width: int,
    height: int,
    brick: Tile,
    wall: Wall,
) -> None:
    max_y, max_x = world.shape
    x0, x1 = max(1, x), min(max_x - 1, x + width)
    y0, y1 = max(1, y), min(max_y - 1, y + height)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return
    world.tiles[y0:y1, x0:x1] = brick
    world.tiles[y0 + 1 : y1 - 1, x0 + 1 : x1 - 1] = Tile.AIR
    world.walls[y0 + 1 : y1 - 1, x0 + 1 : x1 - 1] = wall


def dungeon(world: GeneratedWorld, rng: np.random.Generator) -> None:
    entrance_x = int(world.metadata["dungeon_x"])
    surface_y = int(world.surface[entrance_x])
    tower_w = int(_pick(world, 11, 38))
    tower_h = int(_pick(world, 12, 52))
    tower_x = int(np.clip(entrance_x - tower_w // 2, 2, world.shape[1] - tower_w - 2))
    tower_y = max(2, surface_y - tower_h + int(_pick(world, 3, 10)))
    _carve_room(
        world,
        tower_x,
        tower_y,
        tower_w,
        surface_y - tower_y + int(_pick(world, 5, 16)),
        Tile.DUNGEON_BRICK,
        Wall.DUNGEON,
    )
    battlement = max(2, int(_pick(world, 2, 5)))
    for battlement_x in range(tower_x, tower_x + tower_w, battlement * 2):
        world.tiles[
            max(1, tower_y - battlement) : tower_y,
            battlement_x : battlement_x + battlement,
        ] = Tile.DUNGEON_BRICK
    wing_w = max(6, tower_w // 3)
    wing_h = max(6, tower_h // 3)
    for wing_x in (tower_x - wing_w + 1, tower_x + tower_w - 1):
        clipped_x = int(np.clip(wing_x, 2, world.shape[1] - wing_w - 2))
        wing_y = max(2, surface_y - wing_h + 2)
        _carve_room(
            world,
            clipped_x,
            wing_y,
            wing_w,
            wing_h + 3,
            Tile.DUNGEON_BRICK,
            Wall.DUNGEON,
        )
        for merlon_x in range(clipped_x, clipped_x + wing_w, battlement * 2):
            world.tiles[
                max(1, wing_y - battlement) : wing_y,
                merlon_x : merlon_x + battlement,
            ] = Tile.DUNGEON_BRICK
    for floor_y in (
        tower_y + (surface_y - tower_y) // 3,
        tower_y + 2 * (surface_y - tower_y) // 3,
    ):
        world.tiles[floor_y, tower_x + 1 : tower_x + tower_w - 1] = Tile.DUNGEON_BRICK
        world.tiles[floor_y, entrance_x - 1 : entrance_x + 2] = Tile.AIR
    door_top = max(tower_y + 2, surface_y - int(_pick(world, 3, 8)))
    door_x = tower_x + tower_w // 2
    world.tiles[door_top : surface_y + 2, door_x - 1 : door_x + 2] = Tile.AIR
    world.walls[door_top : surface_y + 2, door_x - 1 : door_x + 2] = Wall.DUNGEON

    start_y = surface_y + int(_pick(world, 3, 12))
    room_w, room_h = tuple(_pick(world, (11, 7), (28, 16)))
    room_count = int(_pick(world, 10, 28))
    x, y = entrance_x - room_w // 2, start_y
    min_x, min_y = min(x, tower_x), tower_y
    max_x, max_y = x + room_w, y + room_h
    for index in range(room_count):
        _carve_room(world, x, y, room_w, room_h, Tile.DUNGEON_BRICK, Wall.DUNGEON)
        world.biomes[
            max(0, y) : min(world.shape[0], y + room_h), max(0, x) : min(world.shape[1], x + room_w)
        ] = Biome.DUNGEON
        if index > 0:
            corridor_x = x + room_w // 2
            world.tiles[
                max(1, y - room_h // 2) : min(world.shape[0] - 1, y + 2),
                corridor_x - 1 : corridor_x + 2,
            ] = Tile.AIR
            world.walls[
                max(1, y - room_h // 2) : min(world.shape[0] - 1, y + 2),
                corridor_x - 1 : corridor_x + 2,
            ] = Wall.DUNGEON
        direction = -1 if rng.integers(0, 2) == 0 else 1
        x = int(
            np.clip(
                x + direction * rng.integers(room_w // 3, room_w), 2, world.shape[1] - room_w - 2
            )
        )
        y += int(rng.integers(max(3, room_h // 2), room_h + 2))
        if y + room_h >= world.layers.underworld:
            break
        min_x, min_y = min(min_x, x), min(min_y, y)
        max_x, max_y = max(max_x, x + room_w), max(max_y, y + room_h)
    _place_marker(world, "Dungeon", min_x, min_y, max_x - min_x, max_y - min_y, "▦")


def beaches(world: GeneratedWorld, rng: np.random.Generator) -> None:
    del rng
    coast = int(_pick(world, 24, 340))
    for x0, x1 in ((0, coast), (world.shape[1] - coast, world.shape[1])):
        region = world.tiles[:, x0:x1]
        region[np.isin(region, (Tile.DIRT, Tile.GRASS, Tile.STONE))] = Tile.SAND


def gems(world: GeneratedWorld, rng: np.random.Generator) -> None:
    _inject_walks(
        world,
        rng,
        int(_pick(world, 8, 110)),
        world.layers.rock_layer,
        world.layers.underworld,
        Tile.GEM,
        (Tile.STONE,),
    )


def ocean_caves(world: GeneratedWorld, rng: np.random.Generator) -> None:
    coast = int(_pick(world, 20, 300))
    for x in (coast // 2, world.shape[1] - coast // 2):
        stamp_walk(
            world.tiles,
            rng,
            x,
            world.layers.world_surface + int(_pick(world, 6, 25)),
            int(_pick(world, 18, 70)),
            tuple(_pick(world, (4, 2), (12, 4))),
            Tile.AIR,
            (float(rng.choice((-0.6, 0.6))), 0.7),
            _CARVABLE,
        )


def shimmer(world: GeneratedWorld, rng: np.random.Generator) -> None:
    center = int(world.metadata["jungle_x"])
    direction = -1 if center > world.shape[1] // 2 else 1
    x = int(np.clip(center + direction * _pick(world, 18, 260), 10, world.shape[1] - 10))
    y = int(
        rng.integers(world.layers.rock_layer, world.layers.underworld - int(_pick(world, 10, 70)))
    )
    rx, ry = tuple(_pick(world, (11, 7), (65, 34)))
    stamp_ellipse(world.tiles, x, y, int(rx), int(ry), Tile.STONE, _CARVABLE)
    stamp_ellipse(
        world.tiles,
        x,
        y,
        max(3, int(rx) - int(_pick(world, 2, 8))),
        max(3, int(ry) - int(_pick(world, 2, 6))),
        Tile.AIR,
        (Tile.STONE,),
    )
    x0, x1 = max(0, x - int(rx)), min(world.shape[1], x + int(rx) + 1)
    y0, y1 = y + max(1, int(ry) // 4), min(world.shape[0], y + int(ry) + 1)
    air = world.tiles[y0:y1, x0:x1] == Tile.AIR
    world.liquid_kind[y0:y1, x0:x1][air] = Liquid.SHIMMER
    world.liquid_amount[y0:y1, x0:x1][air] = 255
    world.biomes[max(0, y - int(ry)) : y1, x0:x1] = Biome.SHIMMER
    for angle in np.linspace(0, np.pi * 2, int(_pick(world, 10, 36)), endpoint=False):
        gx = round(x + np.cos(angle) * max(2, int(rx) - 1))
        gy = round(y + np.sin(angle) * max(2, int(ry) - 1))
        if 0 <= gx < world.shape[1] and 0 <= gy < world.shape[0]:
            world.tiles[gy, gx] = Tile.GEM
    _place_marker(world, "Aether", x0, y - int(ry), x1 - x0, int(ry * 2), "A")


def pyramids(world: GeneratedWorld, rng: np.random.Generator) -> None:
    if rng.random() > 0.70:
        return
    x = int(world.metadata["desert_x"])
    y = int(world.surface[x]) + int(_pick(world, 2, 8))
    height = int(_pick(world, 18, 78))
    for row in range(height):
        half = max(2, round(2 + row * 0.72))
        yy = y + row
        if yy >= world.shape[0]:
            break
        world.tiles[yy, max(0, x - half) : min(world.shape[1], x + half + 1)] = Tile.PYRAMID_BRICK
        if row > int(height * 0.18):
            world.tiles[yy, max(0, x - 1) : min(world.shape[1], x + 2)] = Tile.AIR
            world.walls[yy, max(0, x - 1) : min(world.shape[1], x + 2)] = Wall.SANDSTONE
    chamber_y = min(world.shape[0] - 6, y + round(height * 0.68))
    chamber_w = max(7, round(height * 0.42))
    chamber_h = max(5, round(height * 0.16))
    _carve_room(
        world,
        x - chamber_w // 2,
        chamber_y,
        chamber_w,
        chamber_h,
        Tile.PYRAMID_BRICK,
        Wall.SANDSTONE,
    )
    if chamber_y + chamber_h - 2 < world.shape[0]:
        world.tiles[chamber_y + chamber_h - 2, x] = Tile.CHEST
    pyramid_half = max(2, round(2 + height * 0.72))
    _place_marker(world, "Pyramid", x - pyramid_half, y, pyramid_half * 2, height, "P")


def living_trees(world: GeneratedWorld, rng: np.random.Generator) -> None:
    count = int(_pick(world, 2, 5))
    coast = int(_pick(world, 28, 450))
    for _ in range(count):
        x = int(rng.integers(coast, world.shape[1] - coast))
        if world.biomes[int(world.surface[x]), x] not in (Biome.FOREST, Biome.JUNGLE):
            continue
        y = int(world.surface[x])
        height = int(_pick(world, rng.integers(9, 14), rng.integers(35, 70)))
        trunk_half = int(_pick(world, 1, 3))
        world.tiles[
            max(1, y - height) : min(world.shape[0], y + height // 3),
            x - trunk_half : x + trunk_half + 1,
        ] = Tile.LIVING_WOOD
        stamp_ellipse(
            world.tiles,
            x,
            y - height,
            int(_pick(world, 6, 20)),
            int(_pick(world, 4, 12)),
            Tile.LEAF,
            (Tile.AIR,),
        )
        for direction in (-1, 1):
            stamp_walk(
                world.tiles,
                rng,
                x,
                y,
                int(_pick(world, 8, 28)),
                tuple(_pick(world, (2, 1), (4, 1))),
                Tile.LIVING_WOOD,
                (direction * 0.6, 0.8),
                (Tile.DIRT, Tile.STONE, Tile.GRASS),
            )
        _place_marker(
            world,
            "Living tree",
            x - int(_pick(world, 6, 20)),
            y - height - int(_pick(world, 4, 12)),
            int(_pick(world, 12, 40)),
            height + int(_pick(world, 8, 20)),
            "♜",
        )


def altars(world: GeneratedWorld, rng: np.random.Generator) -> None:
    biome = Biome.CORRUPTION if world.config.evil is Evil.CORRUPTION else Biome.CRIMSON
    candidates = np.argwhere((world.biomes == biome) & surface_candidates(world.tiles, _CARVABLE))
    if candidates.size == 0:
        return
    count = min(len(candidates), int(_pick(world, 3, 18)))
    chosen = candidates[rng.choice(len(candidates), count, replace=False)]
    world.tiles[chosen[:, 0] - 1, chosen[:, 1]] = Tile.ALTAR


def jungle_temple(world: GeneratedWorld, rng: np.random.Generator) -> None:
    center = int(world.metadata["jungle_x"])
    width, height = tuple(_pick(world, (26, 18), (180, 110)))
    x = int(np.clip(center + rng.integers(-width, width + 1), 2, world.shape[1] - width - 2))
    y = int(
        rng.integers(
            world.layers.rock_layer,
            max(world.layers.rock_layer + 1, world.layers.underworld - height - 4),
        )
    )
    corridor_h = max(2, int(_pick(world, 2, 5)))
    level_step = max(5, int(_pick(world, 5, 14)))
    step_width = max(2, int(_pick(world, 2, 8)))
    maximum_inset = max(4, width // 5)
    silhouette = np.zeros((height, width), dtype=bool)
    for local_y in range(height):
        progress = local_y / max(1, height - 1)
        raw_inset = round((1.0 - progress) * maximum_inset)
        inset = (raw_inset // step_width) * step_width
        shift = 0
        if local_y < height // 3:
            shift = step_width
        elif local_y > 2 * height // 3:
            shift = -step_width
        left = int(np.clip(inset + shift, 1, width - 5))
        right = int(np.clip(width - inset + shift, left + 4, width - 1))
        silhouette[local_y, left:right] = True
    silhouette[-max(3, level_step // 2) :, 1 : width - 1] = True
    temple_region = world.tiles[y : y + height, x : x + width]
    temple_region[silhouette] = Tile.LIHZAHRD_BRICK

    direction = 1
    for level_y in range(y + 2, y + height - 3, level_step):
        corridor_bottom = min(y + height - 2, level_y + corridor_h)
        local_level = level_y - y
        row_columns = np.flatnonzero(silhouette[local_level])
        if row_columns.size < 6:
            continue
        corridor_left = int(row_columns[0] + 2)
        corridor_right = int(row_columns[-1] - 1)
        for corridor_y in range(level_y, corridor_bottom):
            local_row = corridor_y - y
            row_columns = np.flatnonzero(silhouette[local_row])
            if row_columns.size >= 6:
                left = max(corridor_left, int(row_columns[0] + 2))
                right = min(corridor_right, int(row_columns[-1] - 1))
                world.tiles[corridor_y, x + left : x + right] = Tile.AIR
        connector_local_x = corridor_right - 2 if direction > 0 else corridor_left
        connector_x = x + connector_local_x
        for connector_y in range(level_y, min(y + height - 2, level_y + level_step + 1)):
            if silhouette[connector_y - y, connector_local_x]:
                world.tiles[connector_y, connector_x : connector_x + 2] = Tile.AIR
        direction *= -1

    chamber_y = y + height - max(7, level_step)
    chamber_left = x + width // 3
    chamber_right = x + 2 * width // 3
    world.tiles[chamber_y : y + height - 3, chamber_left:chamber_right] = Tile.AIR
    interior = world.tiles[y + 1 : y + height - 1, x + 1 : x + width - 1] == Tile.AIR
    world.walls[y + 1 : y + height - 1, x + 1 : x + width - 1][interior] = Wall.LIHZAHRD
    doorway_y = y + height - max(5, level_step)
    doorway_local_x = int(np.flatnonzero(silhouette[doorway_y - y])[0])
    world.tiles[
        doorway_y : y + height - 2,
        x + doorway_local_x : x + doorway_local_x + 3,
    ] = Tile.AIR
    world.walls[
        doorway_y : y + height - 2,
        x + doorway_local_x : x + doorway_local_x + 3,
    ] = Wall.LIHZAHRD
    _place_marker(world, "Jungle temple", x, y, width, height, "T")


def hives(world: GeneratedWorld, rng: np.random.Generator) -> None:
    center = int(world.metadata["jungle_x"])
    half = int(_pick(world, 24, 500))
    count = int(_pick(world, 2, 7))
    for _ in range(count):
        x = int(rng.integers(max(5, center - half), min(world.shape[1] - 5, center + half)))
        y = int(rng.integers(world.layers.rock_layer, world.layers.underworld - 5))
        rx, ry = tuple(_pick(world, (5, 4), (22, 16)))
        stamp_ellipse(world.tiles, x, y, int(rx), int(ry), Tile.HIVE, _CARVABLE)
        stamp_ellipse(
            world.tiles, x, y, max(2, int(rx) - 2), max(2, int(ry) - 2), Tile.AIR, (Tile.HIVE,)
        )
        y0, y1 = y, min(world.shape[0], y + int(ry))
        x0, x1 = max(0, x - int(rx)), min(world.shape[1], x + int(rx) + 1)
        air = world.tiles[y0:y1, x0:x1] == Tile.AIR
        world.liquid_kind[y0:y1, x0:x1][air] = Liquid.HONEY
        world.liquid_amount[y0:y1, x0:x1][air] = 255
        world.walls[max(0, y - int(ry)) : y1, x0:x1] = Wall.HIVE


def settle_liquids(world: GeneratedWorld, rng: np.random.Generator) -> None:
    del rng
    height, _ = world.shape
    for _ in range(4):
        moved = False
        for y in range(height - 2, -1, -1):
            falling = (
                (world.liquid_amount[y] > 0)
                & (world.liquid_amount[y + 1] == 0)
                & (world.tiles[y + 1] == Tile.AIR)
            )
            if not np.any(falling):
                continue
            world.liquid_amount[y + 1, falling] = world.liquid_amount[y, falling]
            world.liquid_kind[y + 1, falling] = world.liquid_kind[y, falling]
            world.liquid_amount[y, falling] = 0
            world.liquid_kind[y, falling] = Liquid.NONE
            moved = True
        if not moved:
            break


def smooth_world(world: GeneratedWorld, rng: np.random.Generator) -> None:
    del rng
    solid = world.tiles != Tile.AIR
    neighbors = np.zeros(world.shape, dtype=np.uint8)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        neighbors += np.roll(np.roll(solid, dy, axis=0), dx, axis=1)
    natural = np.isin(world.tiles, _CARVABLE)
    erode = natural & (neighbors <= 1)
    erode[:2] = False
    erode[-2:] = False
    erode[:, :2] = False
    erode[:, -2:] = False
    world.tiles[erode] = Tile.AIR


def _place_surface_objects(
    world: GeneratedWorld,
    rng: np.random.Generator,
    tile: Tile,
    count: int,
    hosts: tuple[Tile, ...] = _CARVABLE,
    biome: Biome | None = None,
) -> None:
    candidates = surface_candidates(world.tiles, hosts)
    if biome is not None:
        candidates &= world.biomes == biome
    positions = np.argwhere(candidates)
    if positions.size == 0:
        return
    count = min(count, len(positions))
    selected = positions[rng.choice(len(positions), count, replace=False)]
    target_y = selected[:, 0] - 1
    world.tiles[target_y, selected[:, 1]] = tile


def life_crystals(world: GeneratedWorld, rng: np.random.Generator) -> None:
    count = int(_pick(world, 6, 100))
    candidates = surface_candidates(world.tiles, (Tile.STONE, Tile.MUD, Tile.ICE))
    rows = np.arange(world.shape[0])[:, None]
    candidates &= (rows > world.layers.world_surface + 2) & (rows < world.layers.underworld)
    positions = np.argwhere(candidates)
    if positions.size == 0:
        return
    selected = positions[rng.choice(len(positions), min(count, len(positions)), replace=False)]
    world.tiles[selected[:, 0] - 1, selected[:, 1]] = Tile.LIFE_CRYSTAL


def buried_chests(world: GeneratedWorld, rng: np.random.Generator) -> None:
    _place_surface_objects(world, rng, Tile.CHEST, int(_pick(world, 7, 80)))


def surface_chests(world: GeneratedWorld, rng: np.random.Generator) -> None:
    _place_surface_objects(
        world,
        rng,
        Tile.CHEST,
        int(_pick(world, 3, 21)),
        (Tile.GRASS, Tile.SNOW, Tile.SAND, Tile.JUNGLE_GRASS),
    )


def spider_caves(world: GeneratedWorld, rng: np.random.Generator) -> None:
    count = int(_pick(world, 1, 5))
    for _ in range(count):
        x = int(rng.integers(8, world.shape[1] - 8))
        y = int(rng.integers(world.layers.rock_layer, world.layers.underworld - 5))
        rx, ry = tuple(_pick(world, (7, 5), (28, 20)))
        stamp_ellipse(world.tiles, x, y, int(rx), int(ry), Tile.AIR, _CARVABLE)
        x0, x1 = max(0, x - int(rx)), min(world.shape[1], x + int(rx) + 1)
        y0, y1 = max(0, y - int(ry)), min(world.shape[0], y + int(ry) + 1)
        area = world.tiles[y0:y1, x0:x1]
        air = area == Tile.AIR
        scatter = rng.random(area.shape) < 0.10
        area[air & scatter] = Tile.COBWEB
        world.walls[y0:y1, x0:x1][air] = Wall.SPIDER
        _place_marker(world, "Spider cave", x0, y0, x1 - x0, y1 - y0, "✣")


def gem_caves(world: GeneratedWorld, rng: np.random.Generator) -> None:
    count = int(_pick(world, 1, 5))
    for _ in range(count):
        x = int(rng.integers(8, world.shape[1] - 8))
        y = int(rng.integers(world.layers.rock_layer, world.layers.underworld - 5))
        radius = int(_pick(world, 6, 24))
        stamp_ellipse(world.tiles, x, y, radius, max(3, radius // 2), Tile.AIR, _CARVABLE)
        for angle in np.linspace(0, np.pi * 2, int(_pick(world, 8, 22)), endpoint=False):
            gx = round(x + np.cos(angle) * radius)
            gy = round(y + np.sin(angle) * radius * 0.5)
            if 0 <= gx < world.shape[1] and 0 <= gy < world.shape[0]:
                world.tiles[gy, gx] = Tile.GEM
        _place_marker(world, "Gem cave", x - radius, y - radius // 2, radius * 2, radius, "✧")


def cave_walls(world: GeneratedWorld, rng: np.random.Generator) -> None:
    del rng
    rows = np.arange(world.shape[0])[:, None]
    caves = (
        (world.tiles == Tile.AIR)
        & (rows >= world.surface[None, :])
        & (rows < world.layers.underworld)
    )
    missing = caves & (world.walls == Wall.NONE)
    world.walls[missing & (rows < world.layers.rock_layer)] = Wall.DIRT
    world.walls[missing & (rows >= world.layers.rock_layer)] = Wall.STONE


def pots(world: GeneratedWorld, rng: np.random.Generator) -> None:
    _place_surface_objects(world, rng, Tile.POT, int(_pick(world, 18, 850)))


def spreading_grass(world: GeneratedWorld, rng: np.random.Generator) -> None:
    del rng
    exposed_dirt = surface_candidates(world.tiles, (Tile.DIRT,))
    exposed_mud = surface_candidates(world.tiles, (Tile.MUD,))
    world.tiles[exposed_dirt] = Tile.GRASS
    world.tiles[exposed_mud & (world.biomes == Biome.JUNGLE)] = Tile.JUNGLE_GRASS
    world.tiles[exposed_mud & (world.biomes == Biome.MUSHROOM)] = Tile.MUSHROOM_GRASS


def traps(world: GeneratedWorld, rng: np.random.Generator) -> None:
    _place_surface_objects(world, rng, Tile.TRAP, int(_pick(world, 8, 210)))


def spawn_point(world: GeneratedWorld, rng: np.random.Generator) -> None:
    del rng
    center = world.shape[1] // 2
    radius = int(_pick(world, 8, 160))
    columns = np.arange(max(1, center - radius), min(world.shape[1] - 1, center + radius + 1))
    valid = [
        int(x)
        for x in columns
        if world.tiles[int(world.surface[x]), x] in (Tile.GRASS, Tile.DIRT, Tile.SNOW, Tile.SAND)
        and world.liquid_amount[max(0, int(world.surface[x]) - 1), x] == 0
    ]
    spawn_x = min(valid, key=lambda x: abs(x - center)) if valid else center
    world.metadata["spawn_x"] = spawn_x
    world.metadata["spawn_y"] = int(world.surface[spawn_x]) - 2
    _place_marker(world, "Spawn", spawn_x - 1, int(world.surface[spawn_x]) - 3, 3, 3, "⌂")


def planting_trees(world: GeneratedWorld, rng: np.random.Generator) -> None:
    spacing = int(_pick(world, 11, 85))
    offset = int(rng.integers(0, max(1, spacing)))
    for x in range(offset, world.shape[1], spacing):
        y = int(world.surface[x])
        if (
            y < 5
            or y >= world.shape[0]
            or world.tiles[y, x]
            not in (
                Tile.GRASS,
                Tile.JUNGLE_GRASS,
            )
        ):
            continue
        height = int(_pick(world, rng.integers(4, 8), rng.integers(12, 25)))
        top = max(1, y - height)
        world.tiles[top:y, x] = Tile.TREE
        stamp_ellipse(
            world.tiles,
            x,
            top,
            int(_pick(world, 3, 7)),
            int(_pick(world, 2, 4)),
            Tile.LEAF,
            (Tile.AIR,),
        )


def vines(world: GeneratedWorld, rng: np.random.Generator) -> None:
    del rng
    solid = np.isin(world.tiles, (Tile.JUNGLE_GRASS, Tile.MUSHROOM_GRASS))
    hanging = np.zeros_like(solid)
    hanging[:-1] = solid[:-1] & (world.tiles[1:] == Tile.AIR)
    positions = np.argwhere(hanging)
    for y, x in positions[:: max(1, int(_pick(world, 4, 18)))]:
        length = int(_pick(world, 3, 12))
        for step in range(1, length + 1):
            if y + step >= world.shape[0] or world.tiles[y + step, x] != Tile.AIR:
                break
            world.tiles[y + step, x] = Tile.VINE


def flowers(world: GeneratedWorld, rng: np.random.Generator) -> None:
    _place_surface_objects(
        world,
        rng,
        Tile.FLOWER,
        int(_pick(world, 15, 500)),
        (Tile.GRASS, Tile.JUNGLE_GRASS),
    )


def coastal_plants(world: GeneratedWorld, rng: np.random.Generator) -> None:
    _place_surface_objects(
        world,
        rng,
        Tile.CACTUS,
        int(_pick(world, 5, 120)),
        (Tile.SAND,),
        Biome.DESERT,
    )


def stalactites(world: GeneratedWorld, rng: np.random.Generator) -> None:
    del rng
    ceiling = np.zeros(world.shape, dtype=bool)
    ceiling[:-1] = np.isin(world.tiles[:-1], _CARVABLE) & (world.tiles[1:] == Tile.AIR)
    positions = np.argwhere(ceiling)
    stride = max(1, len(positions) // int(_pick(world, 24, 850))) if len(positions) else 1
    selected = positions[::stride]
    world.tiles[selected[:, 0] + 1, selected[:, 1]] = Tile.STALACTITE


def remove_broken_traps(world: GeneratedWorld, rng: np.random.Generator) -> None:
    del rng
    trap_y, trap_x = np.where(world.tiles == Tile.TRAP)
    if trap_y.size == 0:
        return
    unsupported = (trap_y + 1 >= world.shape[0]) | (
        world.tiles[np.minimum(trap_y + 1, world.shape[0] - 1), trap_x] == Tile.AIR
    )
    world.tiles[trap_y[unsupported], trap_x[unsupported]] = Tile.AIR


def tile_cleanup(world: GeneratedWorld, rng: np.random.Generator) -> None:
    del rng
    occupied = world.tiles != Tile.AIR
    world.liquid_amount[occupied] = 0
    world.liquid_kind[occupied] = Liquid.NONE
    world.walls[: max(1, world.layers.world_surface // 2)] = Wall.NONE


def final_cleanup(world: GeneratedWorld, rng: np.random.Generator) -> None:
    tile_cleanup(world, rng)
    world.metadata["tile_count"] = int(np.count_nonzero(world.tiles))
    world.metadata["air_fraction"] = float(np.mean(world.tiles == Tile.AIR))
    world.metadata["liquid_tiles"] = int(np.count_nonzero(world.liquid_amount))
    world.metadata["structure_count"] = len(world.structures)


def waterfalls(world: GeneratedWorld, rng: np.random.Generator) -> None:
    """Cut visible sky-lake outlets from a subset of floating islands."""

    islands = [marker for marker in world.structures if marker.kind == "Floating island"]
    if not islands:
        return
    selected = rng.permutation(len(islands))[: max(1, (len(islands) + 1) // 2)]
    for index in selected:
        marker = islands[int(index)]
        basin_x = int(
            np.clip(
                marker.x + round(marker.width * 0.62),
                2,
                world.shape[1] - 3,
            )
        )
        basin_y = max(2, marker.y + int(_pick(world, 2, 5)))
        basin_half = int(_pick(world, 2, 8))
        basin_bottom = min(world.shape[0] - 1, basin_y + int(_pick(world, 2, 4)))
        world.tiles[
            basin_y:basin_bottom,
            max(1, basin_x - basin_half) : min(world.shape[1] - 1, basin_x + basin_half + 1),
        ] = Tile.AIR
        outlet_width = int(_pick(world, 1, 2))
        channel_bottom = min(
            int(world.surface[basin_x]) - 1,
            marker.y + marker.height + int(_pick(world, 28, 190)),
        )
        for stream_x in range(basin_x, min(world.shape[1], basin_x + outlet_width)):
            world.tiles[basin_y:channel_bottom, stream_x] = Tile.AIR
            world.liquid_kind[basin_y:channel_bottom, stream_x] = Liquid.WATER
            world.liquid_amount[basin_y:channel_bottom, stream_x] = 255
        pool = (
            world.tiles[
                basin_y:basin_bottom,
                max(1, basin_x - basin_half) : min(world.shape[1] - 1, basin_x + basin_half + 1),
            ]
            == Tile.AIR
        )
        liquid_kind = world.liquid_kind[
            basin_y:basin_bottom,
            max(1, basin_x - basin_half) : min(world.shape[1] - 1, basin_x + basin_half + 1),
        ]
        liquid_amount = world.liquid_amount[
            basin_y:basin_bottom,
            max(1, basin_x - basin_half) : min(world.shape[1] - 1, basin_x + basin_half + 1),
        ]
        liquid_kind[pool] = Liquid.WATER
        liquid_amount[pool] = 255


def temple_polish(world: GeneratedWorld, rng: np.random.Generator) -> None:
    """Populate the Temple maze with traps and a deep altar chamber."""

    for marker in (item for item in world.structures if item.kind == "Jungle temple"):
        x0, x1 = max(1, marker.x), min(world.shape[1] - 1, marker.x + marker.width)
        y0, y1 = max(1, marker.y), min(world.shape[0] - 1, marker.y + marker.height)
        floor = surface_candidates(world.tiles[y0:y1, x0:x1], (Tile.LIHZAHRD_BRICK,))
        candidates = np.argwhere(floor)
        if candidates.size == 0:
            continue
        count = min(len(candidates), int(_pick(world, 4, 28)))
        chosen = candidates[rng.choice(len(candidates), count, replace=False)]
        world.tiles[y0 + chosen[:, 0] - 1, x0 + chosen[:, 1]] = Tile.TRAP
        deepest_y, deepest_x = candidates[np.argmax(candidates[:, 0])]
        world.tiles[y0 + deepest_y - 1, x0 + deepest_x] = Tile.ALTAR


def island_houses(world: GeneratedWorld, rng: np.random.Generator) -> None:
    """Build an original sky-brick cabin on most floating islands."""

    islands = [marker for marker in world.structures if marker.kind == "Floating island"]
    for index, marker in enumerate(islands):
        if index > 0 and rng.random() < 0.25:
            continue
        width = min(marker.width - 4, int(_pick(world, 8, 26)))
        height = int(_pick(world, 6, 15))
        if width < 6:
            continue
        x = marker.x + (marker.width - width) // 2
        y = max(2, marker.y - height + 1)
        _carve_room(world, x, y, width, height, Tile.SKY_BRICK, Wall.SKY)
        roof_height = int(_pick(world, 2, 5))
        center = x + width // 2
        for step in range(roof_height):
            left = max(x, center - width // 2 + step)
            right = min(x + width, center + width // 2 - step + 1)
            world.tiles[max(1, y - roof_height + step), left:right] = Tile.SKY_BRICK
        door_x = x + width - 2
        world.tiles[y + height - int(_pick(world, 3, 5)) : y + height - 1, door_x] = Tile.AIR
        world.tiles[y + height - 2, x + 2] = Tile.CHEST


def hellforge(world: GeneratedWorld, rng: np.random.Generator) -> None:
    """Place Hellforges on floors inside generated Underworld cities."""

    floor = surface_candidates(world.tiles, (Tile.OBSIDIAN_BRICK,))
    interior = np.zeros(world.shape, dtype=bool)
    for marker in (item for item in world.structures if item.kind == "Underworld city"):
        x0, x1 = max(1, marker.x), min(world.shape[1] - 1, marker.x + marker.width)
        y0, y1 = max(1, marker.y), min(world.shape[0] - 1, marker.y + marker.height)
        interior[y0:y1, x0:x1] = True
    candidates = np.argwhere(floor & interior & (np.roll(world.walls, 1, axis=0) == Wall.OBSIDIAN))
    if candidates.size == 0:
        return
    city_count = sum(marker.kind == "Underworld city" for marker in world.structures)
    count = min(len(candidates), max(1, city_count))
    chosen = candidates[rng.choice(len(candidates), count, replace=False)]
    world.tiles[chosen[:, 0] - 1, chosen[:, 1]] = Tile.HELLFORGE


def apply_hardmode(world: GeneratedWorld, rng: np.random.Generator) -> None:
    """Apply a visible evil/Hallow V after the vanilla creation passes."""

    center = int(world.metadata.get("spawn_x", world.shape[1] // 2))
    y0, y1 = world.layers.world_surface, world.layers.underworld
    spread = int(_pick(world, 70, 1350))
    evil_tile = Tile.EBONSTONE if world.config.evil is Evil.CORRUPTION else Tile.CRIMSTONE
    evil_biome = Biome.CORRUPTION if world.config.evil is Evil.CORRUPTION else Biome.CRIMSON
    swap = rng.integers(0, 2) == 0
    for y in range(y0, y1, max(1, int(_pick(world, 2, 7)))):
        progress = (y - y0) / max(1, y1 - y0)
        left_x = round(center - spread * progress)
        right_x = round(center + spread * progress)
        branches = (
            (
                left_x,
                evil_tile if not swap else Tile.PEARLSTONE,
                evil_biome if not swap else Biome.HALLOW,
            ),
            (
                right_x,
                Tile.PEARLSTONE if not swap else evil_tile,
                Biome.HALLOW if not swap else evil_biome,
            ),
        )
        for x, tile, biome in branches:
            if not 2 <= x < world.shape[1] - 2:
                continue
            radius = int(_pick(world, 3, 15))
            stamp_ellipse(world.tiles, x, y, radius, radius, tile, _ORE_HOSTS)
            x0, x1b = max(0, x - radius), min(world.shape[1], x + radius + 1)
            y0b, y1b = max(0, y - radius), min(world.shape[0], y + radius + 1)
            world.biomes[y0b:y1b, x0:x1b] = biome
    advance_biome_spread(world, rng, iterations=int(_pick(world, 3, 6)))
    world.metadata["hardmode"] = True


# Lightweight approximations used to keep the complete named-pass timeline
# operational without misrepresenting those steps as source-faithful ports.
slush = gravity_sand = cleanup_dirt = wood_tree_walls = wet_jungle = _annotate_only
jungle_chests = surface_chests
remove_sand_water = shell_piles = ice_polish = wall_variety = _annotate_only
oasis = lakes
statues = piles = _annotate_only
water_chests = buried_chests
moss = moss_grass = mushrooms = jungle_plants = mud_walls = _annotate_only
lihzahrd_altars = temple_polish
jungle_trees = planting_trees
surface_ore = gems
fallen_log = _annotate_only
grass_wall = cave_walls
sunflowers = flowers
webs_honey = webs
ice_gems = gems
larva = _annotate_only
micro_biomes = gem_caves
water_plants = flowers


PASS_HANDLERS: dict[str, PassHandler] = {
    name: _annotate_only
    for name in {
        "reset",
        "terrain",
        "dunes",
        "ocean_sand",
        "sand_patches",
        "tunnels",
        "mount_caves",
        "dirt_walls",
        "rocks_in_dirt",
        "dirt_in_rocks",
        "clay",
        "small_holes",
        "dirt_caves",
        "rock_caves",
        "surface_caves",
        "wavy_caves",
        "ice_biome",
        "grass",
        "jungle",
        "jungle_grass",
        "full_desert",
        "floating_islands",
        "mushroom_patches",
        "marble",
        "granite",
        "dirt_to_mud",
        "silt",
        "shinies",
        "webs",
        "underworld",
        "evil_biome",
        "lakes",
        "dungeon",
        "beaches",
        "gems",
        "ocean_caves",
        "shimmer",
        "pyramids",
        "living_trees",
        "altars",
        "jungle_temple",
        "hives",
        "settle_liquids",
        "smooth_world",
        "life_crystals",
        "buried_chests",
        "surface_chests",
        "spider_caves",
        "gem_caves",
        "cave_walls",
        "pots",
        "spreading_grass",
        "traps",
        "spawn_point",
        "planting_trees",
        "vines",
        "flowers",
        "coastal_plants",
        "tile_cleanup",
        "stalactites",
        "remove_broken_traps",
        "final_cleanup",
    }
}
PASS_HANDLERS.update(
    {
        name: handler
        for name, handler in globals().items()
        if callable(handler) and name in PASS_HANDLERS
    }
)
PASS_HANDLERS.update(
    {
        "slush": slush,
        "gravity_sand": gravity_sand,
        "cleanup_dirt": cleanup_dirt,
        "wood_tree_walls": wood_tree_walls,
        "wet_jungle": wet_jungle,
        "jungle_chests": jungle_chests,
        "remove_sand_water": remove_sand_water,
        "oasis": oasis,
        "shell_piles": shell_piles,
        "waterfalls": waterfalls,
        "ice_polish": ice_polish,
        "wall_variety": wall_variety,
        "statues": statues,
        "water_chests": water_chests,
        "moss": moss,
        "temple_polish": temple_polish,
        "jungle_trees": jungle_trees,
        "island_houses": island_houses,
        "hellforge": hellforge,
        "surface_ore": surface_ore,
        "fallen_log": fallen_log,
        "piles": piles,
        "grass_wall": grass_wall,
        "sunflowers": sunflowers,
        "webs_honey": webs_honey,
        "jungle_plants": jungle_plants,
        "mushrooms": mushrooms,
        "ice_gems": ice_gems,
        "moss_grass": moss_grass,
        "mud_walls": mud_walls,
        "larva": larva,
        "lihzahrd_altars": lihzahrd_altars,
        "micro_biomes": micro_biomes,
        "water_plants": water_plants,
    }
)
