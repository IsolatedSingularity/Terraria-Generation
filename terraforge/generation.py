"""Modeled TerraForge pass handlers.

The handlers favor deterministic, readable approximations over claims of
bit-for-bit compatibility. Every random decision comes from the per-pass RNG
provided by :mod:`terraforge.pipeline`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from terraforge.config import Evil, WorldScale
from terraforge.geometry import smooth_noise_1d, stamp_ellipse, stamp_walk, surface_candidates
from terraforge.model import GeneratedWorld, StructureMarker
from terraforge.tiles import Biome, Liquid, Tile, Wall

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
    world.tiles[top:, :] = Tile.ASH
    world.tiles[height - max(3, (height - top) // 4) :, :] = Tile.HELLSTONE
    world.biomes[top:, :] = Biome.UNDERWORLD
    world.walls[top:, :] = Wall.STONE
    for _ in range(int(_pick(world, 10, 110))):
        x = int(rng.integers(4, width - 4))
        y = int(rng.integers(top + 2, height - 3))
        rx, ry = tuple(_pick(world, (5, 3), (18, 9)))
        stamp_ellipse(world.tiles, x, y, int(rx), int(ry), Tile.AIR, (Tile.ASH, Tile.HELLSTONE))
        if y > top + (height - top) // 2:
            y0, y1 = y, min(height, y + max(1, int(ry) // 2))
            x0, x1 = max(0, x - int(rx) // 2), min(width, x + int(rx) // 2)
            air = world.tiles[y0:y1, x0:x1] == Tile.AIR
            world.liquid_kind[y0:y1, x0:x1][air] = Liquid.LAVA
            world.liquid_amount[y0:y1, x0:x1][air] = 255


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
    start_y = int(world.surface[entrance_x]) - int(_pick(world, 6, 30))
    room_w, room_h = tuple(_pick(world, (11, 7), (28, 16)))
    room_count = int(_pick(world, 10, 28))
    x, y = entrance_x - room_w // 2, start_y
    min_x, min_y = x, y
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
    rx, ry = tuple(_pick(world, (9, 6), (55, 28)))
    stamp_ellipse(world.tiles, x, y, int(rx), int(ry), Tile.AIR, _CARVABLE)
    x0, x1 = max(0, x - int(rx)), min(world.shape[1], x + int(rx) + 1)
    y0, y1 = y, min(world.shape[0], y + int(ry) + 1)
    air = world.tiles[y0:y1, x0:x1] == Tile.AIR
    world.liquid_kind[y0:y1, x0:x1][air] = Liquid.SHIMMER
    world.liquid_amount[y0:y1, x0:x1][air] = 255
    world.biomes[max(0, y - int(ry)) : y1, x0:x1] = Biome.SHIMMER
    _place_marker(world, "Aether", x0, y - int(ry), x1 - x0, int(ry * 2), "✦")


def pyramids(world: GeneratedWorld, rng: np.random.Generator) -> None:
    if rng.random() > 0.70:
        return
    x = int(world.metadata["desert_x"])
    y = int(world.surface[x])
    height = int(_pick(world, 12, 55))
    for row in range(height):
        half = max(1, height - row)
        yy = y + row
        if yy >= world.shape[0]:
            break
        world.tiles[yy, max(0, x - half) : min(world.shape[1], x + half + 1)] = Tile.PYRAMID_BRICK
        if row > 3:
            world.tiles[yy, max(0, x - 1) : min(world.shape[1], x + 2)] = Tile.AIR
    _place_marker(world, "Pyramid", x - height, y, height * 2, height, "△")


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
    width, height = tuple(_pick(world, (18, 12), (150, 90)))
    x = int(np.clip(center + rng.integers(-width, width + 1), 2, world.shape[1] - width - 2))
    y = int(
        rng.integers(
            world.layers.rock_layer,
            max(world.layers.rock_layer + 1, world.layers.underworld - height - 4),
        )
    )
    world.tiles[y : y + height, x : x + width] = Tile.LIHZAHRD_BRICK
    world.tiles[y + 2 : y + height - 2, x + 2 : x + width - 2] = Tile.AIR
    world.walls[y + 1 : y + height - 1, x + 1 : x + width - 1] = Wall.LIHZAHRD
    _place_marker(world, "Jungle temple", x, y, width, height, "▰")


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
    world.metadata["hardmode"] = True


# Lightweight approximations used to keep the complete named-pass timeline
# operational without misrepresenting those steps as source-faithful ports.
slush = gravity_sand = cleanup_dirt = wood_tree_walls = wet_jungle = _annotate_only
jungle_chests = surface_chests
remove_sand_water = shell_piles = waterfalls = ice_polish = wall_variety = _annotate_only
oasis = lakes
statues = piles = _annotate_only
water_chests = buried_chests
moss = moss_grass = mushrooms = jungle_plants = mud_walls = _annotate_only
temple_polish = lihzahrd_altars = _annotate_only
jungle_trees = planting_trees
island_houses = surface_chests
hellforge = surface_chests
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
