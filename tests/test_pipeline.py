import threading
import time

import numpy as np
import pytest

from terraexplorer.config import Evil, WorldConfig
from terraexplorer.generation import advance_biome_spread
from terraexplorer.passes import Phase
from terraexplorer.pipeline import GenerationCancelledError, TerraExplorerPipeline, generate_world
from terraexplorer.tiles import Biome, Liquid, Tile, Wall


def test_generation_is_deterministic_and_uses_independent_arrays() -> None:
    first = generate_world(WorldConfig(seed="repeatable"))
    second = generate_world(WorldConfig(seed="repeatable"))

    for name in ("tiles", "walls", "liquid_amount", "liquid_kind", "biomes", "surface"):
        assert np.array_equal(getattr(first, name), getattr(second, name)), name
    assert first.tiles.dtype == np.uint8
    assert first.surface.dtype == np.int16
    assert not np.shares_memory(first.tiles, first.walls)
    assert len(first.pass_results) == 107


def test_different_seeds_change_the_world() -> None:
    first = generate_world(WorldConfig(seed="alpha"))
    second = generate_world(WorldConfig(seed="beta"))

    assert not np.array_equal(first.tiles, second.tiles)


def test_one_ore_from_each_alternative_pair_is_selected() -> None:
    world = generate_world(WorldConfig(seed="ores"))
    selected = {int(tile) for tile in world.metadata["selected_ore_ids"]}
    pairs = (
        (Tile.COPPER, Tile.TIN),
        (Tile.IRON, Tile.LEAD),
        (Tile.SILVER, Tile.TUNGSTEN),
        (Tile.GOLD, Tile.PLATINUM),
    )

    for pair in pairs:
        present = {int(tile) for tile in pair if np.any(world.tiles == tile)}
        assert len(selected & {int(tile) for tile in pair}) == 1
        assert present == selected & {int(tile) for tile in pair}


@pytest.mark.parametrize(
    ("evil", "present", "absent"),
    (
        (Evil.CORRUPTION, Biome.CORRUPTION, Biome.CRIMSON),
        (Evil.CRIMSON, Biome.CRIMSON, Biome.CORRUPTION),
    ),
)
def test_world_evil_is_a_real_generation_choice(evil: Evil, present: Biome, absent: Biome) -> None:
    world = generate_world(WorldConfig(seed="evil-layout", evil=evil))

    assert np.any(world.biomes == present)
    assert not np.any(world.biomes == absent)


def test_hardmode_adds_hallow_and_a_post_generation_pass() -> None:
    world = generate_world(WorldConfig(seed="hardmode", hardmode=True))

    assert np.any(world.biomes == Biome.HALLOW)
    assert world.metadata["hardmode"] is True
    assert world.metadata["difficulty"] == "classic"
    assert world.metadata["evil"] == "corruption"
    assert world.metadata["executed_pass_count"] == 108
    assert len(world.pass_results) == 108
    assert world.pass_results[-1].name == "Hardmode V Transformation"


def test_showcase_structures_are_real_world_state() -> None:
    world = generate_world(WorldConfig(seed="TerraExplorer", hardmode=True))
    marker_kinds = {marker.kind for marker in world.structures}

    assert {
        "Aether",
        "Dungeon",
        "Floating island",
        "Jungle temple",
        "Pyramid",
        "Ruined house",
    } <= marker_kinds
    assert np.any(world.tiles == Tile.OBSIDIAN_BRICK)
    assert np.any(world.tiles == Tile.HELLFORGE)
    assert np.any(world.tiles == Tile.SKY_BRICK)
    assert np.any(world.tiles == Tile.CLOUD)
    assert np.any(world.tiles == Tile.RAIN_CLOUD)
    assert np.any(world.tiles == Tile.GEM_TREE)

    for island in (marker for marker in world.structures if marker.kind == "Floating island"):
        island_liquid = world.liquid_amount[
            island.y : island.y + island.height,
            island.x : island.x + island.width,
        ]
        assert not np.any(island_liquid)

    rows = np.arange(world.shape[0])[:, None]
    sky_water = (
        (world.liquid_kind == Liquid.WATER)
        & (world.liquid_amount > 0)
        & (rows < world.surface[None, :])
    )
    assert np.any(sky_water)

    pyramid = next(marker for marker in world.structures if marker.kind == "Pyramid")
    pyramid_x = pyramid.x + pyramid.width // 2
    assert pyramid.y > world.surface[pyramid_x]
    pyramid_tiles = world.tiles[
        pyramid.y : pyramid.y + pyramid.height,
        pyramid.x : pyramid.x + pyramid.width,
    ]
    passage_columns = np.flatnonzero(np.any(pyramid_tiles == Tile.AIR, axis=0))
    assert np.ptp(passage_columns) > 4

    aether = next(marker for marker in world.structures if marker.kind == "Aether")
    aether_x = aether.x + aether.width // 2
    jungle_x = int(world.metadata["jungle_x"])
    assert (aether_x < world.shape[1] // 5) == (jungle_x < world.shape[1] // 2)

    temple = next(marker for marker in world.structures if marker.kind == "Jungle temple")
    temple_state = np.isin(
        world.tiles[
            temple.y : temple.y + temple.height,
            temple.x : temple.x + temple.width,
        ],
        (Tile.LIHZAHRD_BRICK, Tile.TRAP, Tile.ALTAR),
    ) | (
        world.walls[
            temple.y : temple.y + temple.height,
            temple.x : temple.x + temple.width,
        ]
        == Wall.LIHZAHRD
    )
    assert np.count_nonzero(temple_state) > temple.width * 3
    assert np.any(world.tiles == Tile.TRAP)
    assert np.any(world.tiles == Tile.ALTAR)

    lava_level = int(world.metadata["underworld_lava_level"])
    underworld_lava = (world.liquid_kind[lava_level:] == Liquid.LAVA) & (
        world.liquid_amount[lava_level:] > 0
    )
    assert np.count_nonzero(underworld_lava) > world.shape[1]
    ruined_houses = [marker for marker in world.structures if marker.kind == "Ruined house"]
    assert ruined_houses
    assert all(world.shape[1] // 5 < marker.x < world.shape[1] * 4 // 5 for marker in ruined_houses)
    submerged_house = np.isin(
        world.tiles,
        (Tile.OBSIDIAN_BRICK, Tile.HELLSTONE_BRICK),
    ) & (rows >= lava_level)
    assert np.any(submerged_house)


def test_biome_spread_stops_at_world_boundaries() -> None:
    class ZeroRng:
        @staticmethod
        def random(shape):
            return np.zeros(shape)

    world = generate_world(WorldConfig(seed="spread-boundary", evil=Evil.CORRUPTION))
    world.tiles[:] = Tile.STONE
    world.biomes[:] = Biome.FOREST
    world.biomes[10, 0] = Biome.CORRUPTION

    advance_biome_spread(world, ZeroRng())

    assert world.biomes[10, 1] == Biome.CORRUPTION
    assert world.biomes[10, -1] == Biome.FOREST


def test_phase_controls_skip_optional_pass_groups() -> None:
    enabled = tuple(phase.value for phase in Phase if phase is not Phase.STRUCTURES)
    world = generate_world(WorldConfig(seed="phase-control", enabled_phases=enabled))

    assert not any(marker.kind == "Dungeon" for marker in world.structures)
    disabled = [result for result in world.pass_results if result.phase == Phase.STRUCTURES]
    assert disabled
    assert all("disabled" in result.note for result in disabled)


def test_terrain_baseline_cannot_be_disabled_by_api_callers() -> None:
    world = generate_world(WorldConfig(seed="required-terrain", enabled_phases=()))

    assert np.any(world.tiles != Tile.AIR)
    terrain_results = [result for result in world.pass_results if result.phase == Phase.TERRAIN]
    assert terrain_results
    assert all("disabled" not in result.note for result in terrain_results)


def test_progress_and_cancellation_are_cooperative() -> None:
    cancel = threading.Event()
    events = []

    def on_progress(event) -> None:
        events.append(event)
        if event.finished:
            cancel.set()

    with pytest.raises(GenerationCancelledError):
        TerraExplorerPipeline().generate(WorldConfig(seed="cancel"), on_progress, cancel)

    assert events[0].finished is False
    assert events[-1].finished is True


def test_preview_generation_has_no_runaway_regression() -> None:
    started = time.perf_counter()
    generate_world(WorldConfig(seed="performance-smoke"))

    # Coverage and shared CI runners add substantial overhead. Fine-grained
    # timing belongs to `terraexplorer benchmark`; this only catches a runaway.
    assert time.perf_counter() - started < 10.0
