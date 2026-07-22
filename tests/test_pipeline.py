import threading
import time

import numpy as np
import pytest

from terraforge.config import Evil, WorldConfig
from terraforge.passes import Phase
from terraforge.pipeline import GenerationCancelledError, TerraForgePipeline, generate_world
from terraforge.tiles import Biome, Tile


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
        TerraForgePipeline().generate(WorldConfig(seed="cancel"), on_progress, cancel)

    assert events[0].finished is False
    assert events[-1].finished is True


def test_preview_generation_meets_interactive_budget() -> None:
    started = time.perf_counter()
    generate_world(WorldConfig(seed="performance-smoke"))

    assert time.perf_counter() - started < 1.5
