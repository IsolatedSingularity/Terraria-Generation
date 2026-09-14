from functools import lru_cache

import numpy as np
import pytest

from scripts.media_views import crop_world
from terraexplorer.config import Evil, WorldConfig, WorldScale
from terraexplorer.fidelity.structures import components
from terraexplorer.model import GeneratedWorld
from terraexplorer.pipeline import generate_world
from terraexplorer.source_generation import hardmode, terrain
from terraexplorer.spawn_opportunity import ground_candidate_mass
from terraexplorer.tiles import Biome, Liquid, Tile, Wall


def test_hardmode_native_width_and_inward_slope():
    world = GeneratedWorld.empty(WorldConfig(scale=WorldScale.SMALL))
    world.tiles[:] = Tile.STONE
    world.metadata["dungeon_side"] = "left"
    hardmode(world, np.random.default_rng(42))
    left, right = world.metadata["hardmode_origins"]
    assert 840 <= left < 1260 and 2520 < right <= 2940
    for y in (100, 500, 900):
        xs = np.flatnonzero(np.isin(world.biomes[y], (Biome.HALLOW, Biome.CORRUPTION)))
        assert 350 < len(xs) < 550
        assert abs(xs.min() - (left + 0.6 * y - 110)) < 35
        assert abs(xs.max() - (right - 0.6 * y + 110)) < 35


def test_crimson_small_chambers_spawn_and_protected_structures():
    world = generate_world(WorldConfig(seed=17, evil=Evil.CRIMSON, scale=WorldScale.SMALL))
    assert world.metadata["evil_initial_depth"] < world.layers.rock_layer + 250
    assert np.count_nonzero(world.tiles == Tile.CRIMSTONE) > 1000
    x, y = int(world.metadata["spawn_x"]), int(world.metadata["spawn_y"])
    assert world.tiles[y, x] == Tile.AIR
    major = [s for s in world.structures if s.kind in ("Dungeon", "Jungle temple")]
    assert len(major) == 2
    a, b = major
    assert a.x + a.width <= b.x or b.x + b.width <= a.x
    assert not np.any(world.liquid_amount[world.tiles != Tile.AIR])


@lru_cache(maxsize=3)
def small(seed):
    return generate_world(WorldConfig(seed=seed, scale=WorldScale.SMALL))


@pytest.mark.parametrize("seed", [1, 2, 314159265])
def test_native_terrain_bounds_and_extrema(seed):
    world = GeneratedWorld.empty(WorldConfig(seed=seed, scale=WorldScale.SMALL))
    terrain(world, None)
    assert 228 <= world.surface.min() <= world.surface.max() <= 312
    assert world.layers.world_surface == int(world.surface.max()) + 25
    assert (world.layers.rock_layer - world.layers.world_surface) % 6 == 0
    assert world.layers.world_surface < world.layers.rock_layer < world.layers.underworld
    # Broad flat bands and directional runs replace sine-wave terrain.
    assert np.count_nonzero(np.diff(world.surface) == 0) > 1000
    assert not np.shares_memory(world.surface, world.tiles)


@pytest.mark.parametrize("seed", [1, 2, 314159265])
def test_small_major_structures_connected_bounded_and_dry(seed):
    world = small(seed)
    assert len(world.pass_results) == 107
    assert all(
        0 <= s.x < s.x + s.width <= world.shape[1] and 0 <= s.y < s.y + s.height <= world.shape[0]
        for s in world.structures
    )
    assert {"Dungeon", "Jungle temple"} <= {s.kind for s in world.structures}
    assert 10 <= world.metadata["temple_room_count"] <= 15
    for kind, tile, wall in (
        ("Dungeon", Tile.DUNGEON_BRICK, Wall.DUNGEON),
        ("Jungle temple", Tile.LIHZAHRD_BRICK, Wall.LIHZAHRD),
    ):
        marker = next(s for s in world.structures if s.kind == kind)
        assert 0 < marker.x < marker.x + marker.width < world.shape[1]
        assert 0 < marker.y < marker.y + marker.height < world.shape[0]
        mask = (world.tiles == tile) | (world.walls == wall)
        yy, xx = np.nonzero(mask)
        assert xx.min() >= marker.x and xx.max() < marker.x + marker.width
        assert yy.min() >= marker.y and yy.max() < marker.y + marker.height
        groups = components(mask)
        assert max(map(len, groups)) / np.count_nonzero(mask) > 0.99
        traversable = (world.walls == wall) & np.isin(
            world.tiles, (Tile.AIR, Tile.PLATFORM, Tile.CHEST, Tile.POT, Tile.TRAP, Tile.ALTAR)
        )
        groups = components(traversable)
        assert max(map(len, groups)) / np.count_nonzero(traversable) > 0.90
    assert not np.any(world.liquid_amount[world.tiles != Tile.AIR])
    assert np.all(world.liquid_kind[world.liquid_amount == 0] == Liquid.NONE)
    # Loose numerical guard on the measured ordinary-world cavern range.
    assert 0.55 < np.mean(world.tiles[500:1000, 350:-350] != Tile.AIR) < 0.80


def test_small_repeatability_and_crop_independence():
    first, second = small(1), generate_world(WorldConfig(seed=1, scale=WorldScale.SMALL))
    for name in ("tiles", "walls", "liquid_amount", "liquid_kind", "biomes", "surface"):
        assert np.array_equal(getattr(first, name), getattr(second, name))
    crop = crop_world(first, 1000, 200, 240, 140)
    assert crop.metadata["source_crop"] == [1000, 200, 1240, 340]
    assert crop.layers.world_surface == first.layers.world_surface - 200
    assert np.array_equal(crop.tiles, first.tiles[200:340, 1000:1240])
    assert not np.shares_memory(crop.tiles, first.tiles)


def test_spawn_mass_analytic_floor_clearance_liquid_and_no_world_spawn_bubble():
    world = GeneratedWorld.empty(WorldConfig())
    world.tiles[40:] = Tile.STONE
    mass = ground_candidate_mass(world)
    sky = round(world.layers.world_surface * 0.35)
    assert mass[40, 100] == 40 - sky
    assert mass[:40].sum() == 0
    world.metadata.update(spawn_x=100, spawn_y=38)
    assert np.array_equal(mass, ground_candidate_mass(world))
    world.tiles[38, 99] = Tile.STONE
    assert ground_candidate_mass(world)[40, 100] == 0
    world.tiles[38, 99] = Tile.AIR
    world.liquid_amount[39, 100] = 255
    world.liquid_kind[39, 100] = Liquid.LAVA
    assert ground_candidate_mass(world)[40, 100] == 0
    world.liquid_kind[38:40, 100] = Liquid.SHIMMER
    world.liquid_amount[38:40, 100] = 255
    assert ground_candidate_mass(world)[40, 100] == 0
