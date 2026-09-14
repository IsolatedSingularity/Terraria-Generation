import pytest

from terraforge.model import LayerDepths


@pytest.mark.parametrize(
    "width, height, expected_surface, expected_rock, expected_underworld_start",
    [
        # height < 1200: underworld height is max(16, round(height / 6))
        # For height = 600:
        #   underworld_height = max(16, round(600 / 6)) = 100
        #   world_surface = round(600 * 0.19) = 114
        #   rock_layer = round(600 * 0.46) = 276
        #   underworld = 600 - 100 = 500
        (4200, 600, 114, 276, 500),
        # For height = 1000:
        #   underworld_height = max(16, round(1000 / 6)) = 167
        #   world_surface = round(1000 * 0.19) = 190
        #   rock_layer = round(1000 * 0.46) = 460
        #   underworld = 1000 - 167 = 833
        (4200, 1000, 190, 460, 833),
    ],
)
def test_layer_depths_small_world(
    width, height, expected_surface, expected_rock, expected_underworld_start
):
    layers = LayerDepths.for_shape(width, height)
    assert layers.world_surface == expected_surface
    assert layers.rock_layer == expected_rock
    assert layers.underworld == expected_underworld_start


@pytest.mark.parametrize(
    "width, height, expected_surface, expected_rock, expected_underworld_start",
    [
        # height >= 1200: underworld height is 200
        # For height = 1200:
        #   underworld_height = 200
        #   world_surface = round(1200 * 0.19) = 228
        #   rock_layer = round(1200 * 0.46) = 552
        #   underworld = 1200 - 200 = 1000
        (4200, 1200, 228, 552, 1000),
        # For height = 2400:
        #   underworld_height = 200
        #   world_surface = round(2400 * 0.19) = 456
        #   rock_layer = round(2400 * 0.46) = 1104
        #   underworld = 2400 - 200 = 2200
        (4200, 2400, 456, 1104, 2200),
    ],
)
def test_layer_depths_large_world(
    width, height, expected_surface, expected_rock, expected_underworld_start
):
    layers = LayerDepths.for_shape(width, height)
    assert layers.world_surface == expected_surface
    assert layers.rock_layer == expected_rock
    assert layers.underworld == expected_underworld_start


@pytest.mark.parametrize(
    "width, height, expected_surface, expected_rock, expected_underworld_start",
    [
        # Edge case: small height where round(height / 6) < 16
        # For height = 50:
        #   underworld_height = max(16, round(50 / 6)) = max(16, 8) = 16
        #   world_surface = round(50 * 0.19) = 10
        #   rock_layer = round(50 * 0.46) = 23
        #   underworld = 50 - 16 = 34
        (4200, 50, 10, 23, 34),
        # Edge case: height = 0
        #   underworld_height = max(16, round(0)) = 16
        #   world_surface = 0
        #   rock_layer = 0
        #   underworld = 0 - 16 = -16
        (4200, 0, 0, 0, -16),
        # Edge case: width doesn't matter
        (1, 1200, 228, 552, 1000),
        (-100, 1200, 228, 552, 1000),
    ],
)
def test_layer_depths_edge_cases(
    width, height, expected_surface, expected_rock, expected_underworld_start
):
    layers = LayerDepths.for_shape(width, height)
    assert layers.world_surface == expected_surface
    assert layers.rock_layer == expected_rock
    assert layers.underworld == expected_underworld_start
