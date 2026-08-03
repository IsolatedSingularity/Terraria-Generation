"""Deterministic post-generation experiments for the TerraExplorer laboratory."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt

from terraexplorer.config import Evil, WorldConfig, WorldScale
from terraexplorer.model import GeneratedWorld
from terraexplorer.pipeline import generate_world
from terraexplorer.tiles import Biome, Liquid, Tile, Wall


class ContainmentStrategy(StrEnum):
    """Interventions compared by :func:`simulate_biome_containment`."""

    OPEN = "open"
    TRENCH = "trench"
    SUNFLOWERS = "sunflowers"
    CHLOROPHYTE = "chlorophyte"


@dataclass(frozen=True, slots=True)
class ContainmentResult:
    strategy: ContainmentStrategy
    frames: tuple[GeneratedWorld, ...]
    infected_counts: tuple[int, ...]
    barrier_x: int
    spread_direction: int


@dataclass(frozen=True, slots=True)
class CatastropheResult:
    frames: tuple[GeneratedWorld, ...]
    impact_x: int
    impact_y: int
    contact_products: dict[str, int]


_VULNERABLE = (
    Tile.DIRT,
    Tile.STONE,
    Tile.GRASS,
    Tile.SAND,
    Tile.HARDENED_SAND,
    Tile.SANDSTONE,
    Tile.ICE,
    Tile.SNOW,
    Tile.MUD,
    Tile.JUNGLE_GRASS,
)


def _containment_baseline(seed: int | str) -> tuple[GeneratedWorld, int, int]:
    """Generate the actual Preview world used by every containment strategy."""

    world = generate_world(WorldConfig(seed=seed, evil=Evil.CORRUPTION))
    infected = (world.biomes == Biome.CORRUPTION) & np.isin(
        world.tiles, (Tile.EBONSTONE, Tile.CORRUPT_GRASS)
    )
    infected_columns = np.flatnonzero(np.any(infected, axis=0))
    spawn_x = int(world.metadata.get("spawn_x", world.shape[1] // 2))
    evil_x = int(world.metadata.get("evil_x", infected_columns.mean()))
    if evil_x < spawn_x:
        spread_direction = 1
        frontier = int(infected_columns[infected_columns < spawn_x].max(initial=evil_x))
        barrier_x = min(spawn_x - 7, frontier + 5)
    else:
        spread_direction = -1
        right_columns = infected_columns[infected_columns > spawn_x]
        frontier = int(right_columns.min(initial=evil_x))
        barrier_x = max(spawn_x + 7, frontier - 5)
    barrier_x = int(np.clip(barrier_x, 8, world.shape[1] - 9))
    return world, barrier_x, spread_direction


def _install_containment(
    world: GeneratedWorld, strategy: ContainmentStrategy, barrier_x: int
) -> None:
    height, width = world.tiles.shape
    if strategy is ContainmentStrategy.TRENCH:
        x0, x1 = max(1, barrier_x - 1), min(width - 1, barrier_x + 2)
        world.tiles[:, x0:x1] = Tile.AIR
        world.walls[:, x0:x1] = Wall.NONE
        world.liquid_amount[:, x0:x1] = 0
        world.liquid_kind[:, x0:x1] = Liquid.NONE
        world.biomes[:, x0:x1] = Biome.FOREST
    elif strategy is ContainmentStrategy.SUNFLOWERS:
        for x in range(max(1, barrier_x - 4), min(width - 1, barrier_x + 5), 2):
            y = int(world.surface[x])
            if y > 0:
                world.tiles[y - 1, x] = Tile.FLOWER
    elif strategy is ContainmentStrategy.CHLOROPHYTE:
        center_y = _chlorophyte_center_y(world, barrier_x)
        world.metadata["containment_chlorophyte_y"] = center_y
        cluster = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
        for dx, dy in cluster:
            world.tiles[center_y + dy, barrier_x + dx] = Tile.CHLOROPHYTE


def _chlorophyte_center_y(world: GeneratedWorld, barrier_x: int) -> int:
    infected_y, infected_x = np.nonzero(
        (world.biomes == Biome.CORRUPTION)
        & np.isin(world.tiles, (Tile.EBONSTONE, Tile.CORRUPT_GRASS))
    )
    if not len(infected_y):
        return (world.layers.rock_layer + world.layers.underworld) // 2
    distance = np.abs(infected_x - barrier_x)
    frontier_y = infected_y[distance <= distance.min() + 3]
    return int(np.clip(np.median(frontier_y), 12, world.shape[0] - 13))


def _containment_protection(
    world: GeneratedWorld,
    strategy: ContainmentStrategy,
    barrier_x: int,
) -> npt.NDArray[np.bool_]:
    protected = np.zeros(world.tiles.shape, dtype=bool)
    if strategy is ContainmentStrategy.SUNFLOWERS:
        for x in range(max(1, barrier_x - 4), min(world.tiles.shape[1] - 1, barrier_x + 5), 2):
            y = int(world.surface[x])
            protected[y : min(world.tiles.shape[0], y + 3), x] = True
    elif strategy is ContainmentStrategy.CHLOROPHYTE:
        center_y = int(
            world.metadata.get(
                "containment_chlorophyte_y",
                _chlorophyte_center_y(world, barrier_x),
            )
        )
        yy, xx = np.ogrid[: world.tiles.shape[0], : world.tiles.shape[1]]
        protected = (xx - barrier_x) ** 2 + (yy - center_y) ** 2 <= 10**2
    return protected


def _advance_containment(
    world: GeneratedWorld,
    strategy: ContainmentStrategy,
    barrier_x: int,
    rng: np.random.Generator,
    attempts: int,
) -> None:
    infected = (world.biomes == Biome.CORRUPTION) & np.isin(
        world.tiles, (Tile.EBONSTONE, Tile.CORRUPT_GRASS)
    )
    vulnerable_grid = np.isin(world.tiles, _VULNERABLE)
    near_vulnerable = np.zeros(world.tiles.shape, dtype=bool)
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            if dy or dx:
                near_vulnerable |= np.roll(np.roll(vulnerable_grid, dy, axis=0), dx, axis=1)
    near_vulnerable[:3] = False
    near_vulnerable[-3:] = False
    near_vulnerable[:, :3] = False
    near_vulnerable[:, -3:] = False
    sources = np.argwhere(infected & near_vulnerable)
    if not len(sources):
        return
    surface_limit = world.surface[sources[:, 1]] + 4
    weights = np.where(sources[:, 0] <= surface_limit, 6.0, 1.0)
    weights /= weights.sum()
    chosen = sources[rng.choice(len(sources), size=attempts, replace=True, p=weights)]
    offsets = rng.integers(-3, 4, size=(attempts, 2))
    targets = chosen + offsets
    valid = (
        (targets[:, 0] >= 1)
        & (targets[:, 0] < world.tiles.shape[0] - 1)
        & (targets[:, 1] >= 1)
        & (targets[:, 1] < world.tiles.shape[1] - 1)
        & np.any(offsets != 0, axis=1)
    )
    targets = targets[valid]
    if not len(targets):
        return
    target_y, target_x = targets[:, 0], targets[:, 1]
    vulnerable = np.isin(world.tiles[target_y, target_x], _VULNERABLE)
    protection = _containment_protection(world, strategy, barrier_x)
    accepted = vulnerable & ~protection[target_y, target_x]
    target_y, target_x = target_y[accepted], target_x[accepted]
    grass = np.isin(world.tiles[target_y, target_x], (Tile.GRASS, Tile.JUNGLE_GRASS))
    world.tiles[target_y[grass], target_x[grass]] = Tile.CORRUPT_GRASS
    world.tiles[target_y[~grass], target_x[~grass]] = Tile.EBONSTONE
    world.biomes[target_y, target_x] = Biome.CORRUPTION


def simulate_biome_containment(
    strategy: ContainmentStrategy,
    *,
    seed: int | str = "Containment Field",
    steps: int = 24,
) -> ContainmentResult:
    """Run a controlled infection experiment with six-times-faster surface sampling.

    Conversion attempts use Terraria's three-tile neighborhood. The interventions
    intentionally isolate one mechanic at a time; they are educational models,
    not claims of tick-for-tick source parity.
    """

    strategy = ContainmentStrategy(strategy)
    world, barrier_x, spread_direction = _containment_baseline(seed)
    _install_containment(world, strategy, barrier_x)
    rng = np.random.default_rng(world.config.seed_value ^ 0x51A7E)
    frames = [_clone_world(world)]
    counts = [int(np.count_nonzero(world.biomes == Biome.CORRUPTION))]
    capture_every = max(1, steps // 6)
    for step in range(1, max(0, steps) + 1):
        _advance_containment(world, strategy, barrier_x, rng, max(240, world.shape[1] * 5))
        if step % capture_every == 0 or step == steps:
            frames.append(_clone_world(world))
            counts.append(int(np.count_nonzero(world.biomes == Biome.CORRUPTION)))
    return ContainmentResult(
        strategy,
        tuple(frames),
        tuple(counts),
        barrier_x,
        spread_direction,
    )


def _clone_world(world: GeneratedWorld) -> GeneratedWorld:
    return GeneratedWorld(
        config=world.config,
        tiles=world.tiles.copy(),
        walls=world.walls.copy(),
        liquid_amount=world.liquid_amount.copy(),
        liquid_kind=world.liquid_kind.copy(),
        biomes=world.biomes.copy(),
        surface=world.surface.copy(),
        layers=world.layers,
        metadata=copy.deepcopy(world.metadata),
        structures=list(world.structures),
        pass_results=list(world.pass_results),
    )


def _meteor_site(world: GeneratedWorld, rng: np.random.Generator) -> int:
    width = world.shape[1]
    spawn = int(world.metadata.get("spawn_x", width // 2))
    candidates = np.arange(round(width * 0.15), round(width * 0.85))
    candidates = candidates[np.abs(candidates - spawn) > round(width * 0.08)]
    protected = [
        marker
        for marker in world.structures
        if marker.kind in {"Dungeon", "Jungle temple", "Floating island", "Aether"}
    ]
    valid = []
    clearance = 10
    for x in candidates:
        if any(
            marker.x - clearance <= x <= marker.x + marker.width + clearance for marker in protected
        ):
            continue
        valid.append(int(x))
    if not valid:
        return int(np.clip(world.metadata.get("desert_x", width // 3), 10, width - 11))
    desert_x = int(world.metadata.get("desert_x", width // 3))
    distances = np.abs(np.asarray(valid) - desert_x)
    nearest = np.flatnonzero(distances == distances.min())
    return valid[int(nearest[int(rng.integers(0, len(nearest)))])]


def _carve_natural_pool(
    world: GeneratedWorld,
    center_x: int,
    center_y: int,
    radius_x: int,
    radius_y: int,
    liquid: Liquid,
) -> None:
    y0, y1 = max(1, center_y - radius_y), min(world.shape[0] - 1, center_y + radius_y + 1)
    x0, x1 = max(1, center_x - radius_x), min(world.shape[1] - 1, center_x + radius_x + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    cave = ((xx - center_x) / radius_x) ** 2 + ((yy - center_y) / radius_y) ** 2 <= 1.0
    local_tiles = world.tiles[y0:y1, x0:x1]
    local_amount = world.liquid_amount[y0:y1, x0:x1]
    local_kind = world.liquid_kind[y0:y1, x0:x1]
    local_tiles[cave] = Tile.AIR
    local_amount[cave] = 0
    local_kind[cave] = Liquid.NONE
    basin = cave & (yy >= center_y + 1)
    local_amount[basin] = 255
    local_kind[basin] = liquid


def _carve_natural_channel(
    world: GeneratedWorld,
    start: tuple[int, int],
    end: tuple[int, int],
) -> None:
    length = max(abs(end[0] - start[0]), abs(end[1] - start[1])) + 1
    for x_value, y_value in zip(
        np.linspace(start[0], end[0], length),
        np.linspace(start[1], end[1], length),
        strict=True,
    ):
        x = int(round(x_value))
        y = int(round(y_value))
        y0, y1 = max(1, y - 1), min(world.shape[0] - 1, y + 2)
        x0, x1 = max(1, x - 1), min(world.shape[1] - 1, x + 2)
        world.tiles[y0:y1, x0:x1] = Tile.AIR
        world.liquid_amount[y0:y1, x0:x1] = 0
        world.liquid_kind[y0:y1, x0:x1] = Liquid.NONE


def _prime_catastrophe_lab(world: GeneratedWorld, impact_x: int, impact_y: int) -> None:
    """Prepare four natural cave pools inside the generated Preview geology."""

    height, width = world.shape
    upper_y = min(height - 36, impact_y + 27)
    lower_y = min(height - 15, impact_y + 50)
    pool_specs = (
        (Liquid.WATER, impact_x - 19, upper_y, 10, 7),
        (Liquid.LAVA, impact_x + 19, upper_y, 10, 7),
        (Liquid.HONEY, impact_x - 16, lower_y, 9, 7),
        (Liquid.SHIMMER, impact_x + 16, lower_y, 9, 7),
    )
    for liquid, center_x, center_y, radius_x, radius_y in pool_specs:
        center_x = int(np.clip(center_x, radius_x + 2, width - radius_x - 3))
        _carve_natural_pool(world, center_x, center_y, radius_x, radius_y, liquid)

    fissure_bottom = min(height - 8, lower_y + 13)
    for y in range(max(2, impact_y + 5), fissure_bottom):
        offset = round(np.sin((y - impact_y) * 0.55) * 1.5)
        x = int(np.clip(impact_x + offset, 2, width - 3))
        world.tiles[y, x - 1 : x + 2] = Tile.AIR
        world.liquid_amount[y, x - 1 : x + 2] = 0
        world.liquid_kind[y, x - 1 : x + 2] = Liquid.NONE

    channels = (
        ((impact_x - 11, upper_y + 2), (impact_x - 1, upper_y + 5)),
        ((impact_x + 11, upper_y + 2), (impact_x + 1, upper_y + 5)),
        ((impact_x - 9, lower_y + 2), (impact_x - 1, lower_y + 5)),
        ((impact_x + 9, lower_y + 2), (impact_x + 1, lower_y + 5)),
    )
    for start, end in channels:
        _carve_natural_channel(world, start, end)

    for index, gate_y in enumerate((upper_y + 5, lower_y + 5)):
        gate = Tile.SAND if index == 0 else Tile.SILT
        world.tiles[gate_y, impact_x - 2 : impact_x + 3] = gate
        world.liquid_amount[gate_y, impact_x - 2 : impact_x + 3] = 0
        world.liquid_kind[gate_y, impact_x - 2 : impact_x + 3] = Liquid.NONE

    contact_y = min(height - 5, lower_y + 11)
    contact_pairs = (
        (Liquid.WATER, Liquid.LAVA),
        (Liquid.WATER, Liquid.HONEY),
        (Liquid.LAVA, Liquid.HONEY),
        (Liquid.SHIMMER, Liquid.WATER),
    )
    pair_x = max(3, impact_x - 23)
    for first, second in contact_pairs:
        world.tiles[contact_y, pair_x : pair_x + 2] = Tile.AIR
        world.liquid_kind[contact_y, pair_x] = first
        world.liquid_kind[contact_y, pair_x + 1] = second
        world.liquid_amount[contact_y, pair_x : pair_x + 2] = 255
        pair_x += 12
    deposit_top = min(height - 2, impact_y + 8)
    world.tiles[deposit_top : deposit_top + 3, impact_x - 10 : impact_x + 11] = Tile.SAND
    world.tiles[deposit_top - 2 : deposit_top, impact_x - 6 : impact_x + 7] = Tile.SILT


def _meteor_impact(
    world: GeneratedWorld, impact_x: int, impact_y: int, rng: np.random.Generator
) -> None:
    radius = 12
    y0, y1 = max(1, impact_y - radius), min(world.shape[0] - 1, impact_y + radius + 1)
    x0, x1 = max(1, impact_x - radius), min(world.shape[1] - 1, impact_x + radius + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    distance = np.sqrt(((xx - impact_x) / radius) ** 2 + ((yy - impact_y) / (radius * 0.90)) ** 2)
    local_tiles = world.tiles[y0:y1, x0:x1]
    inner = distance <= 0.52
    ore = (distance > 0.52) & (distance <= 1.0) & (local_tiles != Tile.AIR)
    scatter = (distance > 1.0) & (distance <= 1.18) & (rng.random(distance.shape) < 0.18)
    local_tiles[inner] = Tile.AIR
    local_tiles[ore | scatter] = Tile.METEORITE
    local_amount = world.liquid_amount[y0:y1, x0:x1]
    local_kind = world.liquid_kind[y0:y1, x0:x1]
    local_amount[inner | ore | scatter] = 0
    local_kind[inner | ore | scatter] = Liquid.NONE


def _fall_granular(world: GeneratedWorld) -> None:
    granular = (Tile.SAND, Tile.SILT)
    for y in range(world.shape[0] - 2, 0, -1):
        falling = (
            np.isin(world.tiles[y], granular)
            & (world.tiles[y + 1] == Tile.AIR)
            & (world.liquid_amount[y + 1] == 0)
        )
        if not np.any(falling):
            continue
        world.tiles[y + 1, falling] = world.tiles[y, falling]
        world.tiles[y, falling] = Tile.AIR


def _transfer_liquid(world: GeneratedWorld, direction: int) -> None:
    height, width = world.shape
    for y in range(height - 2, 0, -1):
        for x in range(1, width - 1):
            amount = int(world.liquid_amount[y, x])
            if amount == 0 or world.tiles[y, x] != Tile.AIR:
                continue
            liquid = int(world.liquid_kind[y, x])
            if world.tiles[y + 1, x] == Tile.AIR and world.liquid_kind[y + 1, x] in (
                Liquid.NONE,
                liquid,
            ):
                capacity = 255 - int(world.liquid_amount[y + 1, x])
                moved = min(amount, capacity)
                if moved:
                    world.liquid_kind[y + 1, x] = liquid
                    world.liquid_amount[y + 1, x] += moved
                    world.liquid_amount[y, x] -= moved
                    amount -= moved
                    if amount == 0:
                        world.liquid_kind[y, x] = Liquid.NONE
            if amount == 0 or world.tiles[y + 1, x] == Tile.AIR:
                continue
            side_x = x + direction
            if world.tiles[y, side_x] != Tile.AIR or world.liquid_kind[y, side_x] not in (
                Liquid.NONE,
                liquid,
            ):
                continue
            side_amount = int(world.liquid_amount[y, side_x])
            if side_amount >= amount - 1:
                continue
            moved = min(amount, max(1, (amount - side_amount) // 2))
            world.liquid_kind[y, side_x] = liquid
            world.liquid_amount[y, side_x] += moved
            world.liquid_amount[y, x] -= moved
            if world.liquid_amount[y, x] == 0:
                world.liquid_kind[y, x] = Liquid.NONE


_CONTACT_PRODUCTS = {
    frozenset((Liquid.WATER, Liquid.LAVA)): Tile.OBSIDIAN,
    frozenset((Liquid.WATER, Liquid.HONEY)): Tile.HONEY_BLOCK,
    frozenset((Liquid.LAVA, Liquid.HONEY)): Tile.CRISPY_HONEY_BLOCK,
    frozenset((Liquid.SHIMMER, Liquid.WATER)): Tile.AETHERIUM,
    frozenset((Liquid.SHIMMER, Liquid.LAVA)): Tile.AETHERIUM,
    frozenset((Liquid.SHIMMER, Liquid.HONEY)): Tile.AETHERIUM,
}


def _react_liquids(world: GeneratedWorld, product_counts: dict[str, int]) -> None:
    consumed = np.zeros(world.shape, dtype=bool)
    for y in range(1, world.shape[0] - 1):
        for x in range(1, world.shape[1] - 1):
            if consumed[y, x] or world.liquid_amount[y, x] == 0:
                continue
            first = Liquid(int(world.liquid_kind[y, x]))
            for dy, dx in ((1, 0), (0, 1)):
                other_y, other_x = y + dy, x + dx
                if consumed[other_y, other_x] or world.liquid_amount[other_y, other_x] == 0:
                    continue
                second = Liquid(int(world.liquid_kind[other_y, other_x]))
                product = _CONTACT_PRODUCTS.get(frozenset((first, second)))
                if product is None:
                    continue
                world.tiles[other_y, other_x] = product
                world.liquid_amount[y, x] = 0
                world.liquid_amount[other_y, other_x] = 0
                world.liquid_kind[y, x] = Liquid.NONE
                world.liquid_kind[other_y, other_x] = Liquid.NONE
                consumed[y, x] = True
                consumed[other_y, other_x] = True
                product_counts[product.name.lower()] = (
                    product_counts.get(product.name.lower(), 0) + 1
                )
                break


def _normalize_liquids(world: GeneratedWorld) -> None:
    trapped = world.tiles != Tile.AIR
    world.liquid_amount[trapped] = 0
    world.liquid_kind[trapped] = Liquid.NONE
    world.liquid_kind[world.liquid_amount == 0] = Liquid.NONE


def simulate_catastrophe_chain(
    world: GeneratedWorld,
    *,
    seed: int = 0xC47A57,
    steps: int = 30,
) -> CatastropheResult:
    """Couple a protected meteor event, granular collapse, liquids, and reactions."""

    if world.config.scale is not WorldScale.PREVIEW:
        raise ValueError("The animated catastrophe laboratory intentionally uses Preview worlds")
    state = _clone_world(world)
    rng = np.random.default_rng(seed)
    impact_x = _meteor_site(state, rng)
    impact_y = int(state.surface[impact_x])
    _prime_catastrophe_lab(state, impact_x, impact_y)
    frames = [_clone_world(state)]
    _meteor_impact(state, impact_x, impact_y, rng)
    frames.append(_clone_world(state))
    products: dict[str, int] = {}
    capture_every = max(1, steps // 6)
    for step in range(1, max(0, steps) + 1):
        _fall_granular(state)
        _transfer_liquid(state, -1)
        _transfer_liquid(state, 1)
        _react_liquids(state, products)
        _normalize_liquids(state)
        if step % capture_every == 0 or step == steps:
            frames.append(_clone_world(state))
    state.metadata["meteor_impact"] = {"x": impact_x, "y": impact_y}
    state.metadata["contact_products"] = dict(products)
    frames[-1] = _clone_world(state)
    return CatastropheResult(tuple(frames), impact_x, impact_y, products)
