import numpy as np

from terraforge.config import WorldConfig, WorldScale
from terraforge.model import GeneratedWorld, LayerDepths
from terraforge.tiles import Biome, Liquid, Tile, Wall


def test_layer_depths_small():
    # Small world: 4200 width, 1200 height
    layers = LayerDepths.for_shape(4200, 1200)
    assert layers.world_surface == 228  # 1200 * 0.19
    assert layers.rock_layer == 552  # 1200 * 0.46
    assert layers.underworld == 1000  # 1200 - 200


def test_layer_depths_preview():
    # Preview world: 240 width, 140 height
    layers = LayerDepths.for_shape(240, 140)
    assert layers.world_surface == 27  # round(140 * 0.19)
    assert layers.rock_layer == 64  # round(140 * 0.46)
    assert layers.underworld == 117  # 140 - max(16, round(140/6)) = 140 - 23


def test_generated_world_empty():
    config = WorldConfig(scale=WorldScale.PREVIEW)
    world = GeneratedWorld.empty(config)

    assert world.config == config

    shape = (config.height, config.width)

    assert world.tiles.shape == shape
    assert world.tiles.dtype == np.uint8
    assert np.all(world.tiles == Tile.AIR)

    assert world.walls.shape == shape
    assert world.walls.dtype == np.uint8
    assert np.all(world.walls == Wall.NONE)

    assert world.liquid_amount.shape == shape
    assert world.liquid_amount.dtype == np.uint8
    assert np.all(world.liquid_amount == 0)

    assert world.liquid_kind.shape == shape
    assert world.liquid_kind.dtype == np.uint8
    assert np.all(world.liquid_kind == Liquid.NONE)

    assert world.biomes.shape == shape
    assert world.biomes.dtype == np.uint8
    assert np.all(world.biomes == Biome.SKY)

    assert world.surface.shape == (config.width,)
    assert world.surface.dtype == np.int16
    assert np.all(world.surface == world.layers.world_surface)

    assert world.layers == LayerDepths.for_shape(config.width, config.height)

    assert "target" in world.metadata
