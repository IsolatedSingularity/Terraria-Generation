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
    surface_half = int(_pick(world, 27, 470))
    deep_half = int(surface_half * 0.62)
    bottom = world.layers.underworld
    for y in range(max(1, int(world.surface.min())), bottom):
        depth = np.clip(
            (y - world.layers.world_surface)
            / max(1, world.layers.underworld - world.layers.world_surface),
            0.0,
            1.0,
        )
        half = round(surface_half * (1.0 - depth) + deep_half * depth)
        x0, x1 = _band(world, center, half)
        columns = np.arange(x0, x1)
        below_surface = y >= world.surface[columns]
        row = world.tiles[y, x0:x1]
        dirt = below_surface & np.isin(row, (Tile.DIRT, Tile.GRASS))
        stone = below_surface & (row == Tile.STONE)
        row[dirt] = Tile.SNOW
        row[stone] = Tile.ICE
        world.biomes[y, x0:x1][below_surface] = Biome.SNOW


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
    region[np.isin(region, (Tile.DIRT, Tile.GRASS, Tile.STONE))] = Tile.MUD
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
    count = 3
    coast = int(_pick(world, 25, 420))
    available = world.config.width - coast * 2
    for index in range(count):
        segment_center = coast + round(available * (index + 0.5) / count)
        jitter = max(2, available // (count * 7))
        x = int(
            np.clip(
                segment_center + rng.integers(-jitter, jitter + 1),
                coast,
                world.config.width - coast,
            )
        )
        y = int(
            rng.integers(max(8, world.layers.world_surface // 4), world.layers.world_surface - 5)
        )
        rx = int(_pick(world, rng.integers(10, 15), rng.integers(38, 56)))
        ry = int(_pick(world, rng.integers(5, 8), rng.integers(14, 21)))
        stamp_ellipse(world.tiles, x, y, rx, ry, Tile.CLOUD, (Tile.AIR,))
        stamp_ellipse(
            world.tiles,
            x,
            y + max(1, ry // 3),
            max(3, round(rx * 0.72)),
            max(2, round(ry * 0.62)),
            Tile.RAIN_CLOUD,
            (Tile.CLOUD,),
        )
        top = max(1, y - ry)
        x0, x1 = max(0, x - rx), min(world.config.width, x + rx + 1)
        cap = world.tiles[top : y + 1, x0:x1]
        cap[np.isin(cap, (Tile.CLOUD, Tile.RAIN_CLOUD))] = Tile.DIRT
        exposed = surface_candidates(cap, (Tile.DIRT,))
        cap[exposed] = Tile.GRASS
        _place_marker(world, "Floating island", x - rx, y - ry, rx * 2 + 1, ry * 2 + 1, "◇")


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

    house_count = int(_pick(world, 4, 18))
    left_limit, right_limit = width // 4, width * 3 // 4
    centers = np.linspace(left_limit, right_limit, house_count + 2, dtype=int)[1:-1]
    for center_x in centers:
        room_w = int(_pick(world, rng.integers(12, 18), rng.integers(30, 52)))
        floor_count = int(rng.integers(2, 5))
        floor_h = int(_pick(world, rng.integers(5, 8), rng.integers(10, 16)))
        room_h = floor_count * floor_h + 2
        jitter = int(_pick(world, 3, 18))
        x = int(
            np.clip(
                center_x + rng.integers(-jitter, jitter + 1) - room_w // 2, 2, width - room_w - 2
            )
        )
        base_y = int(
            np.clip(
                lava_level + rng.integers(-int(_pick(world, 1, 8)), int(_pick(world, 3, 13))),
                top + room_h + 2,
                height - 4,
            )
        )
        y = base_y - room_h
        rare_hellstone = rng.random() < 0.24
        brick = Tile.HELLSTONE_BRICK if rare_hellstone else Tile.OBSIDIAN_BRICK
        wall = Wall.HELLSTONE if rare_hellstone else Wall.OBSIDIAN

        world.tiles[y : base_y + 1, x : x + room_w] = brick
        world.tiles[y + 1 : base_y, x + 1 : x + room_w - 1] = Tile.AIR
        world.walls[y + 1 : base_y, x + 1 : x + room_w - 1] = wall
        for floor_index in range(1, floor_count):
            floor_y = y + floor_index * floor_h
            world.tiles[floor_y, x + 1 : x + room_w - 1] = Tile.PLATFORM
            opening_x = x + 2 if floor_index % 2 else x + room_w - 4
            world.tiles[floor_y, opening_x : opening_x + 2] = Tile.AIR
        door_y = base_y - max(3, floor_h // 2)
        door_x = x if rng.integers(0, 2) == 0 else x + room_w - 1
        world.tiles[door_y:base_y, door_x] = Tile.AIR

        roof_step = max(1, int(_pick(world, 1, 3)))
        for roof_x in range(x + roof_step, x + room_w - roof_step, roof_step * 3):
            world.tiles[max(top + 1, y - roof_step) : y, roof_x : roof_x + roof_step] = brick

        support_bottom = min(height - 1, max(base_y + 2, int(floor_profile[x + room_w // 2]) + 1))
        for support_x in (x + 1, x + room_w // 2, x + room_w - 2):
            world.tiles[base_y:support_bottom, support_x] = brick

        world.liquid_kind[y:base_y, x : x + room_w] = Liquid.NONE
        world.liquid_amount[y:base_y, x : x + room_w] = 0
        flooded_y = max(y + 1, lava_level)
        flooded = world.tiles[flooded_y:base_y, x + 1 : x + room_w - 1] == Tile.AIR
        world.liquid_kind[flooded_y:base_y, x + 1 : x + room_w - 1][flooded] = Liquid.LAVA
        world.liquid_amount[flooded_y:base_y, x + 1 : x + room_w - 1][flooded] = 255
        _place_marker(world, "Ruined house", x, y, room_w, support_bottom - y, "▥")

    world.metadata["ruined_house_count"] = house_count
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
        chasm_x = (center - half // 2, center + half // 3)
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
        connection_y = min(
            world.layers.underworld - 8,
            world.layers.rock_layer + int(_pick(world, 18, 130)),
        )
        _carve_corridor(
            world,
            (chasm_x[0], connection_y),
            (chasm_x[1], connection_y),
            int(_pick(world, 3, 7)),
            Wall.NONE,
        )
    else:
        bulbs = int(_pick(world, 4, 13))
        previous: tuple[int, int] | None = None
        first_chamber: tuple[int, int] | None = None
        for index in range(bulbs):
            x = int(center + rng.integers(-half, half + 1))
            y = int(
                world.layers.world_surface
                + (world.layers.underworld - world.layers.world_surface) * (index + 1) / (bulbs + 1)
            )
            rx, ry = tuple(_pick(world, (7, 5), (28, 20)))
            stamp_ellipse(world.tiles, x, y, int(rx), int(ry), Tile.AIR, _CARVABLE)
            if first_chamber is None:
                first_chamber = (x, y)
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
        if first_chamber is not None:
            first_x, first_y = first_chamber
            entry_direction = -1 if first_x >= center else 1
            entry_x = int(np.clip(first_x + entry_direction * half // 2, x0 + 3, x1 - 4))
            entry_y = max(2, int(world.surface[entry_x]) - 1)
            length = max(abs(first_x - entry_x), abs(first_y - entry_y)) + 1
            for x_value, y_value in zip(
                np.linspace(entry_x, first_x, length),
                np.linspace(entry_y, first_y, length),
                strict=True,
            ):
                stamp_ellipse(
                    world.tiles,
                    round(x_value),
                    round(y_value),
                    int(_pick(world, 3, 8)),
                    int(_pick(world, 2, 5)),
                    Tile.AIR,
                    _CARVABLE,
                )

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


def _carve_corridor(
    world: GeneratedWorld,
    start: tuple[int, int],
    end: tuple[int, int],
    width: int,
    wall: Wall,
) -> None:
    """Carve a clipped L-shaped passage between room centers."""

    start_x, start_y = start
    end_x, end_y = end
    half = max(1, width // 2)
    x0, x1 = sorted((start_x, end_x))
    y0, y1 = sorted((start_y, end_y))
    horizontal_y0 = max(1, start_y - half)
    horizontal_y1 = min(world.shape[0] - 1, start_y + half + 1)
    horizontal_x0 = max(1, x0)
    horizontal_x1 = min(world.shape[1] - 1, x1 + 1)
    world.tiles[horizontal_y0:horizontal_y1, horizontal_x0:horizontal_x1] = Tile.AIR
    world.walls[horizontal_y0:horizontal_y1, horizontal_x0:horizontal_x1] = wall
    vertical_x0 = max(1, end_x - half)
    vertical_x1 = min(world.shape[1] - 1, end_x + half + 1)
    vertical_y0 = max(1, y0)
    vertical_y1 = min(world.shape[0] - 1, y1 + 1)
    world.tiles[vertical_y0:vertical_y1, vertical_x0:vertical_x1] = Tile.AIR
    world.walls[vertical_y0:vertical_y1, vertical_x0:vertical_x1] = wall


def dungeon(world: GeneratedWorld, rng: np.random.Generator) -> None:
    entrance_x = int(world.metadata["dungeon_x"])
    surface_y = int(world.surface[entrance_x])
    hall_w = int(_pick(world, 13, 34))
    hall_h = int(_pick(world, 9, 26))
    hall_x = int(np.clip(entrance_x - hall_w // 2, 2, world.shape[1] - hall_w - 2))
    hall_y = max(2, surface_y - hall_h + int(_pick(world, 2, 6)))
    _carve_room(world, hall_x, hall_y, hall_w, hall_h, Tile.DUNGEON_BRICK, Wall.DUNGEON)
    roof_y = max(1, hall_y - 1)
    world.tiles[roof_y, hall_x + 2 : hall_x + hall_w - 2] = Tile.DUNGEON_BRICK
    if hall_w >= 10:
        world.tiles[roof_y - 1 : roof_y + 1, hall_x + hall_w // 3 : hall_x + hall_w // 3 + 2] = (
            Tile.DUNGEON_BRICK
        )
    door_x = hall_x + hall_w // 2
    world.tiles[surface_y - 3 : surface_y + 2, door_x - 1 : door_x + 2] = Tile.AIR
    world.walls[surface_y - 3 : surface_y + 2, door_x - 1 : door_x + 2] = Wall.DUNGEON

    room_count = int(_pick(world, 9, 32))
    corridor_w = int(_pick(world, 3, 7))
    previous_center = (door_x, surface_y)
    x = entrance_x
    y = surface_y + int(_pick(world, 5, 18))
    min_x, min_y = hall_x, hall_y
    max_x, max_y = hall_x + hall_w, hall_y + hall_h
    for index in range(room_count):
        room_w = int(_pick(world, rng.integers(10, 17), rng.integers(24, 46)))
        room_h = int(_pick(world, rng.integers(6, 11), rng.integers(12, 24)))
        room_x = int(np.clip(x - room_w // 2, 2, world.shape[1] - room_w - 2))
        room_y = int(np.clip(y, surface_y + 2, world.layers.underworld - room_h - 2))
        _carve_room(world, room_x, room_y, room_w, room_h, Tile.DUNGEON_BRICK, Wall.DUNGEON)
        center = (room_x + room_w // 2, room_y + room_h // 2)
        _carve_corridor(world, previous_center, center, corridor_w, Wall.DUNGEON)
        world.biomes[room_y : room_y + room_h, room_x : room_x + room_w] = Biome.DUNGEON
        if room_w >= int(_pick(world, 13, 34)) and index % 3 == 1:
            shelf_y = room_y + room_h // 2
            world.tiles[shelf_y, room_x + 2 : room_x + room_w - 2] = Tile.PLATFORM
            world.tiles[shelf_y, center[0] - 1 : center[0] + 2] = Tile.AIR
        min_x, min_y = min(min_x, room_x), min(min_y, room_y)
        max_x, max_y = max(max_x, room_x + room_w), max(max_y, room_y + room_h)
        previous_center = center
        direction = -1 if rng.integers(0, 2) == 0 else 1
        x = int(
            np.clip(
                center[0] + direction * rng.integers(room_w // 2, room_w + 3), 8, world.shape[1] - 8
            )
        )
        y = center[1] + int(rng.integers(max(4, room_h // 2), room_h + 3))
        if y + room_h >= world.layers.underworld:
            break
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
    jungle_x = int(world.metadata["jungle_x"])
    jungle_on_left = jungle_x < world.shape[1] // 2
    zone = (0.12, 0.20) if jungle_on_left else (0.80, 0.88)
    x = int(rng.integers(round(world.shape[1] * zone[0]), round(world.shape[1] * zone[1])))
    y = int(
        rng.integers(world.layers.rock_layer, world.layers.underworld - int(_pick(world, 10, 70)))
    )
    rx = int(_pick(world, rng.integers(11, 15), rng.integers(105, 121)))
    ry = int(_pick(world, rng.integers(7, 10), rng.integers(55, 71)))
    stamp_ellipse(world.tiles, x, y, int(rx), int(ry), Tile.STONE, _CARVABLE)
    stamp_ellipse(
        world.tiles,
        x + int(_pick(world, 1, 8)),
        y - int(_pick(world, 1, 5)),
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
    pool_top = y + max(1, ry // 4)
    for tree_x in (x - rx // 2, x + rx // 2):
        ground = None
        for tree_y in range(pool_top - 1, min(world.shape[0] - 1, y + ry)):
            if world.tiles[tree_y, tree_x] != Tile.AIR:
                ground = tree_y
                break
        if ground is None:
            continue
        tree_height = int(_pick(world, 3, 10))
        trunk_top = max(1, ground - tree_height)
        world.tiles[trunk_top:ground, tree_x] = Tile.GEM_TREE
        stamp_ellipse(
            world.tiles, tree_x, trunk_top, int(_pick(world, 2, 5)), 2, Tile.GEM, (Tile.AIR,)
        )
    _place_marker(world, "Aether", x0, y - int(ry), x1 - x0, int(ry * 2), "A")


def pyramids(world: GeneratedWorld, rng: np.random.Generator) -> None:
    if rng.random() > 0.70:
        return
    x = int(world.metadata["desert_x"])
    y = int(world.surface[x]) + int(_pick(world, 2, rng.integers(13, 20)))
    height = int(_pick(world, 24, rng.integers(70, 90)))
    segment_height = int(_pick(world, 5, 14))
    tunnel_half = int(_pick(world, 1, 2))
    for row in range(height):
        half = max(2, round(2 + row * 0.72))
        yy = y + row
        if yy >= world.shape[0]:
            break
        world.tiles[yy, max(0, x - half) : min(world.shape[1], x + half + 1)] = Tile.PYRAMID_BRICK
        if row > int(height * 0.14):
            segment = row // segment_height
            local = (row % segment_height) / max(1, segment_height - 1)
            direction = -1 if segment % 2 == 0 else 1
            maximum_offset = max(0, half - tunnel_half - 2)
            offset = round(direction * maximum_offset * (1.0 - local * 2.0))
            passage_x = x + offset
            x0 = max(0, passage_x - tunnel_half)
            x1 = min(world.shape[1], passage_x + tunnel_half + 1)
            world.tiles[yy, x0:x1] = Tile.AIR
            world.walls[yy, x0:x1] = Wall.SANDSTONE
    chamber_y = min(world.shape[0] - 6, y + round(height * 0.58))
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
    width, height = tuple(_pick(world, (34, 24), (190, 120)))
    x = int(np.clip(center + rng.integers(-width, width + 1), 2, world.shape[1] - width - 2))
    y = int(
        rng.integers(
            world.layers.rock_layer,
            max(world.layers.rock_layer + 1, world.layers.underworld - height - 4),
        )
    )
    silhouette = np.zeros((height, width), dtype=bool)
    for local_y in range(height):
        band = local_y // max(3, height // 7)
        left = 1 + (band % 3)
        right = width - 1 - ((band + 1) % 3)
        silhouette[local_y, left:right] = True
    silhouette[-max(3, height // 10) :, 1 : width - 1] = True
    temple_region = world.tiles[y : y + height, x : x + width]
    temple_region[silhouette] = Tile.LIHZAHRD_BRICK

    rows = int(_pick(world, 3, 5))
    rooms_per_row = int(_pick(world, 2, 3))
    row_height = max(6, (height - 6) // rows)
    previous_center: tuple[int, int] | None = None
    for row in range(rows):
        room_y = y + 3 + row * row_height
        if room_y + row_height >= y + height - 3:
            break
        usable_width = width - 8
        room_width = max(8, usable_width // rooms_per_row)
        order = range(rooms_per_row) if row % 2 == 0 else reversed(range(rooms_per_row))
        for column in order:
            room_x = x + 4 + column * room_width
            carved_width = min(room_width + 1, x + width - 3 - room_x)
            carved_height = min(row_height - 1, y + height - 3 - room_y)
            _carve_room(
                world,
                room_x,
                room_y,
                carved_width,
                carved_height,
                Tile.LIHZAHRD_BRICK,
                Wall.LIHZAHRD,
            )
            room_center = (room_x + carved_width // 2, room_y + carved_height // 2)
            if previous_center is not None:
                _carve_corridor(
                    world,
                    previous_center,
                    room_center,
                    int(_pick(world, 2, 4)),
                    Wall.LIHZAHRD,
                )
            previous_center = room_center

    chamber_y = y + height - max(8, row_height)
    chamber_x = x + width // 4
    chamber_width = width // 2
    _carve_room(
        world,
        chamber_x,
        chamber_y,
        chamber_width,
        y + height - 2 - chamber_y,
        Tile.LIHZAHRD_BRICK,
        Wall.LIHZAHRD,
    )
    chamber_center = (chamber_x + chamber_width // 2, chamber_y + 3)
    if previous_center is not None:
        _carve_corridor(
            world,
            previous_center,
            chamber_center,
            int(_pick(world, 2, 4)),
            Wall.LIHZAHRD,
        )

    door_on_left = center < world.shape[1] // 2
    door_x = x + 1 if door_on_left else x + width - 3
    door_y = y + 4
    world.tiles[door_y : door_y + int(_pick(world, 4, 8)), door_x : door_x + 2] = Tile.AIR
    world.walls[door_y : door_y + int(_pick(world, 4, 8)), door_x : door_x + 2] = Wall.LIHZAHRD
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
    """Place falling water at steep natural surface breaks, not Floating Islands."""

    differences = np.abs(np.diff(world.surface.astype(np.int32)))
    threshold = int(_pick(world, 3, 14))
    candidates = np.flatnonzero(differences >= threshold)
    if candidates.size == 0:
        return
    shuffled = rng.permutation(candidates)
    ranked = shuffled[np.argsort(differences[shuffled], kind="stable")][::-1]
    count = min(len(ranked), int(_pick(world, 2, 12)))
    for boundary in ranked[:count]:
        left_y = int(world.surface[boundary])
        right_y = int(world.surface[boundary + 1])
        if left_y < right_y:
            source_x, fall_x = int(boundary), int(boundary + 1)
            top, bottom = left_y - 1, right_y
        else:
            source_x, fall_x = int(boundary + 1), int(boundary)
            top, bottom = right_y - 1, left_y
        top = max(1, top)
        bottom = min(world.shape[0] - 1, bottom)
        world.tiles[top:bottom, fall_x] = Tile.AIR
        world.liquid_kind[top:bottom, fall_x] = Liquid.WATER
        world.liquid_amount[top:bottom, fall_x] = 255
        source_y = max(1, int(world.surface[source_x]) - 1)
        world.tiles[source_y, source_x] = Tile.AIR
        world.liquid_kind[source_y, source_x] = Liquid.WATER
        world.liquid_amount[source_y, source_x] = 255


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
    """Place Hellforges on floors inside generated Underworld Ruined Houses."""

    floor = surface_candidates(world.tiles, (Tile.OBSIDIAN_BRICK, Tile.HELLSTONE_BRICK))
    interior = np.zeros(world.shape, dtype=bool)
    for marker in (item for item in world.structures if item.kind == "Ruined house"):
        x0, x1 = max(1, marker.x), min(world.shape[1] - 1, marker.x + marker.width)
        y0, y1 = max(1, marker.y), min(world.shape[0] - 1, marker.y + marker.height)
        interior[y0:y1, x0:x1] = True
    house_wall = np.isin(np.roll(world.walls, 1, axis=0), (Wall.OBSIDIAN, Wall.HELLSTONE))
    candidates = np.argwhere(floor & interior & house_wall)
    if candidates.size == 0:
        return
    house_count = sum(marker.kind == "Ruined house" for marker in world.structures)
    count = min(len(candidates), max(1, house_count))
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
