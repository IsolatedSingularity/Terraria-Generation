import numpy as np
import pytest

from terraexplorer import (
    ContainmentStrategy,
    GeneratedWorld,
    WorldConfig,
    WorldScale,
    generate_world,
    simulate_biome_containment,
    simulate_catastrophe_chain,
)
from terraexplorer.tiles import Biome, Tile


def test_containment_interventions_are_deterministic_and_measurable() -> None:
    first = simulate_biome_containment(ContainmentStrategy.OPEN, seed=42)
    second = simulate_biome_containment(ContainmentStrategy.OPEN, seed=42)
    trench = simulate_biome_containment(ContainmentStrategy.TRENCH, seed=42)
    sunflowers = simulate_biome_containment(ContainmentStrategy.SUNFLOWERS, seed=42)
    chlorophyte = simulate_biome_containment(ContainmentStrategy.CHLOROPHYTE, seed=42)

    assert first.infected_counts == second.infected_counts
    assert np.array_equal(first.frames[-1].tiles, second.frames[-1].tiles)
    assert first.infected_counts[-1] > first.infected_counts[0]
    open_beyond = np.count_nonzero(
        first.frames[-1].biomes[:, first.barrier_x + 3 :] == Biome.CORRUPTION
    )
    trench_beyond = np.count_nonzero(
        trench.frames[-1].biomes[:, trench.barrier_x + 3 :] == Biome.CORRUPTION
    )
    assert open_beyond > 0
    assert trench_beyond == 0
    assert sunflowers.infected_counts[-1] < first.infected_counts[-1]
    assert chlorophyte.infected_counts[-1] < first.infected_counts[-1]


def test_catastrophe_couples_impact_collapse_and_all_contact_products() -> None:
    world = generate_world(WorldConfig(seed="catastrophe-test"))
    original_tiles = world.tiles.copy()
    original_liquid_amount = world.liquid_amount.copy()
    result = simulate_catastrophe_chain(world, seed=84)
    final = result.frames[-1]

    assert len(result.frames) >= 4
    assert abs(result.impact_x - int(world.metadata["spawn_x"])) > round(world.shape[1] * 0.08)
    assert np.any(final.tiles == Tile.METEORITE)
    assert {
        "obsidian",
        "honey_block",
        "crispy_honey_block",
        "aetherium",
    } <= result.contact_products.keys()
    assert final.metadata["meteor_impact"] == {"x": result.impact_x, "y": result.impact_y}
    assert final.metadata["contact_products"] == result.contact_products
    assert not np.any((final.liquid_amount > 0) & (final.tiles != Tile.AIR))
    assert np.array_equal(world.tiles, original_tiles)
    assert np.array_equal(world.liquid_amount, original_liquid_amount)


def test_catastrophe_rejects_full_small_worlds() -> None:
    world = GeneratedWorld.empty(WorldConfig(seed="small-catastrophe", scale=WorldScale.SMALL))

    with pytest.raises(ValueError, match="Preview"):
        simulate_catastrophe_chain(world)
