"""Deterministic post-generation experiments for the TerraExplorer laboratory."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt

from terraexplorer.config import WorldScale
from terraexplorer.model import GeneratedWorld
from terraexplorer.tiles import Biome, Liquid, Tile, Wall


class ContainmentStrategy(StrEnum):
    """Interventions compared by :func:`simulate_biome_containment`."""

    OPEN = "open"
    TRENCH = "trench"
    SUNFLOWERS = "sunflowers"
    CHLOROPHYTE = "chlorophyte"


@dataclass(slots=True)
class SimulationGrid:
    """Compact tile and biome state used by controlled experiments."""

    tiles: npt.NDArray[np.uint8]
    biomes: npt.NDArray[np.uint8]
    surface: npt.NDArray[np.int16]

    def clone(self) -> SimulationGrid:
        return SimulationGrid(self.tiles.copy(), self.biomes.copy(), self.surface.copy())


@dataclass(frozen=True, slots=True)
class ContainmentResult:
    strategy: ContainmentStrategy
    frames: tuple[SimulationGrid, ...]
    infected_counts: tuple[int, ...]
    barrier_x: int


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


def _containment_baseline(seed: int, width: int, height: int) -> SimulationGrid:
    rng = np.random.default_rng(seed)
    columns = np.arange(width)
    surface = (
        round(height * 0.18)
        + np.sin(columns / 13.0) * 2.0
        + np.sin(columns / 31.0) * 1.5
        + rng.normal(0.0, 0.35, width)
    ).astype(np.int16)
    rows = np.arange(height)[:, None]
    tiles = np.full((height, width), Tile.STONE, dtype=np.uint8)
    tiles[rows < surface[None, :]] = Tile.AIR
    dirt = (rows >= surface[None, :]) & (rows < surface[None, :] + round(height * 0.20))
    tiles[dirt] = Tile.DIRT
    for x, surface_y in enumerate(surface):
        tiles[int(surface_y), x] = Tile.GRASS

    desert = slice(round(width * 0.38), round(width * 0.56))
    jungle = slice(round(width * 0.72), round(width * 0.92))
    tiles[:, desert][np.isin(tiles[:, desert], (Tile.DIRT, Tile.GRASS))] = Tile.SAND
    tiles[:, jungle][tiles[:, jungle] == Tile.DIRT] = Tile.MUD
    tiles[:, jungle][tiles[:, jungle] == Tile.GRASS] = Tile.JUNGLE_GRASS
    biomes = np.full((height, width), Biome.FOREST, dtype=np.uint8)
    biomes[rows < surface[None, :]] = Biome.SKY

    source_x1 = max(10, round(width * 0.30))
    source = (columns[None, :] < source_x1) & (rows >= surface[None, :])
    source &= np.isin(tiles, _VULNERABLE)
    grass = np.isin(tiles, (Tile.GRASS, Tile.JUNGLE_GRASS))
    tiles[source & grass] = Tile.CORRUPT_GRASS
    tiles[source & ~grass] = Tile.EBONSTONE
    biomes[source] = Biome.CORRUPTION
    return SimulationGrid(tiles, biomes, surface)


def _install_containment(
    grid: SimulationGrid, strategy: ContainmentStrategy, barrier_x: int
) -> None:
    height, width = grid.tiles.shape
    if strategy is ContainmentStrategy.TRENCH:
        x0, x1 = max(1, barrier_x - 1), min(width - 1, barrier_x + 2)
        grid.tiles[:, x0:x1] = Tile.AIR
        grid.biomes[:, x0:x1] = Biome.FOREST
    elif strategy is ContainmentStrategy.SUNFLOWERS:
        for x in range(max(1, barrier_x - 4), min(width - 1, barrier_x + 5), 2):
            y = int(grid.surface[x])
            if y > 0:
                grid.tiles[y - 1, x] = Tile.FLOWER
    elif strategy is ContainmentStrategy.CHLOROPHYTE:
        center_y = round(height * 0.62)
        cluster = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
        for dx, dy in cluster:
            grid.tiles[center_y + dy, barrier_x + dx] = Tile.CHLOROPHYTE


def _containment_protection(
    grid: SimulationGrid,
    strategy: ContainmentStrategy,
    barrier_x: int,
) -> npt.NDArray[np.bool_]:
    protected = np.zeros(grid.tiles.shape, dtype=bool)
    if strategy is ContainmentStrategy.SUNFLOWERS:
        for x in range(max(1, barrier_x - 4), min(grid.tiles.shape[1] - 1, barrier_x + 5), 2):
            y = int(grid.surface[x])
            protected[y : min(grid.tiles.shape[0], y + 3), x] = True
    elif strategy is ContainmentStrategy.CHLOROPHYTE:
        center_y = round(grid.tiles.shape[0] * 0.62)
        yy, xx = np.ogrid[: grid.tiles.shape[0], : grid.tiles.shape[1]]
        protected = (xx - barrier_x) ** 2 + (yy - center_y) ** 2 <= 10**2
    return protected


def _advance_containment(
    grid: SimulationGrid,
    strategy: ContainmentStrategy,
    barrier_x: int,
    rng: np.random.Generator,
    attempts: int,
) -> None:
    infected = (grid.biomes == Biome.CORRUPTION) & np.isin(
        grid.tiles, (Tile.EBONSTONE, Tile.CORRUPT_GRASS)
    )
    vulnerable_grid = np.isin(grid.tiles, _VULNERABLE)
    near_vulnerable = np.zeros(grid.tiles.shape, dtype=bool)
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
    surface_limit = grid.surface[sources[:, 1]] + 4
    weights = np.where(sources[:, 0] <= surface_limit, 6.0, 1.0)
    weights /= weights.sum()
    chosen = sources[rng.choice(len(sources), size=attempts, replace=True, p=weights)]
    offsets = rng.integers(-3, 4, size=(attempts, 2))
    targets = chosen + offsets
    valid = (
        (targets[:, 0] >= 1)
        & (targets[:, 0] < grid.tiles.shape[0] - 1)
        & (targets[:, 1] >= 1)
        & (targets[:, 1] < grid.tiles.shape[1] - 1)
        & np.any(offsets != 0, axis=1)
    )
    targets = targets[valid]
    if not len(targets):
        return
    target_y, target_x = targets[:, 0], targets[:, 1]
    vulnerable = np.isin(grid.tiles[target_y, target_x], _VULNERABLE)
    protection = _containment_protection(grid, strategy, barrier_x)
    accepted = vulnerable & ~protection[target_y, target_x]
    target_y, target_x = target_y[accepted], target_x[accepted]
    grass = np.isin(grid.tiles[target_y, target_x], (Tile.GRASS, Tile.JUNGLE_GRASS))
    grid.tiles[target_y[grass], target_x[grass]] = Tile.CORRUPT_GRASS
    grid.tiles[target_y[~grass], target_x[~grass]] = Tile.EBONSTONE
    grid.biomes[target_y, target_x] = Biome.CORRUPTION


def simulate_biome_containment(
    strategy: ContainmentStrategy,
    *,
    seed: int = 0xC01A1E,
    steps: int = 24,
    width: int = 180,
    height: int = 84,
) -> ContainmentResult:
    """Run a controlled infection experiment with six-times-faster surface sampling.

    Conversion attempts use Terraria's three-tile neighborhood. The interventions
    intentionally isolate one mechanic at a time; they are educational models,
    not claims of tick-for-tick source parity.
    """

    strategy = ContainmentStrategy(strategy)
    grid = _containment_baseline(seed, width, height)
    barrier_x = width // 2
    _install_containment(grid, strategy, barrier_x)
    rng = np.random.default_rng(seed ^ 0x51A7E)
    frames = [grid.clone()]
    counts = [int(np.count_nonzero(grid.biomes == Biome.CORRUPTION))]
    capture_every = max(1, steps // 6)
    for step in range(1, max(0, steps) + 1):
        _advance_containment(grid, strategy, barrier_x, rng, max(240, width * 5))
        if step % capture_every == 0 or step == steps:
            frames.append(grid.clone())
            counts.append(int(np.count_nonzero(grid.biomes == Biome.CORRUPTION)))
    return ContainmentResult(strategy, tuple(frames), tuple(counts), barrier_x)


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


def _prime_catastrophe_lab(world: GeneratedWorld, impact_x: int, impact_y: int) -> None:
    height, width = world.shape
    arena_top = min(height - 38, impact_y + 8)
    arena_bottom = min(height - 7, arena_top + 48)
    x0, x1 = max(3, impact_x - 30), min(width - 3, impact_x + 31)
    world.tiles[arena_top:arena_bottom, x0:x1] = Tile.STONE
    world.walls[arena_top:arena_bottom, x0:x1] = Wall.NONE
    world.liquid_amount[arena_top:arena_bottom, x0:x1] = 0
    world.liquid_kind[arena_top:arena_bottom, x0:x1] = Liquid.NONE

    chambers = (
        (Liquid.WATER, x0 + 3, arena_top + 3),
        (Liquid.HONEY, x0 + 3, arena_top + 24),
        (Liquid.LAVA, x1 - 15, arena_top + 3),
        (Liquid.SHIMMER, x1 - 15, arena_top + 24),
    )
    for liquid, chamber_x, chamber_y in chambers:
        chamber_x1 = min(x1 - 1, chamber_x + 12)
        chamber_y1 = min(arena_bottom - 2, chamber_y + 11)
        world.tiles[chamber_y:chamber_y1, chamber_x:chamber_x1] = Tile.AIR
        fill_y = chamber_y + 3
        world.liquid_kind[fill_y:chamber_y1, chamber_x:chamber_x1] = liquid
        world.liquid_amount[fill_y:chamber_y1, chamber_x:chamber_x1] = 255

    chute_x0, chute_x1 = impact_x - 3, impact_x + 4
    chute_y0 = max(1, impact_y + 5)
    world.tiles[chute_y0:arena_bottom, chute_x0:chute_x1] = Tile.AIR
    world.liquid_amount[chute_y0:arena_bottom, chute_x0:chute_x1] = 0
    world.liquid_kind[chute_y0:arena_bottom, chute_x0:chute_x1] = Liquid.NONE
    channel_levels = (arena_top + 12, arena_top + 33)
    for index, channel_y in enumerate(channel_levels):
        world.tiles[channel_y : channel_y + 2, x0 + 14 : chute_x0] = Tile.AIR
        world.tiles[channel_y : channel_y + 2, chute_x1 : x1 - 14] = Tile.AIR
        gate = Tile.SAND if index == 0 else Tile.SILT
        world.tiles[channel_y, chute_x0:chute_x1] = gate

    contact_y = arena_bottom - 3
    contact_pairs = (
        (Liquid.WATER, Liquid.LAVA),
        (Liquid.WATER, Liquid.HONEY),
        (Liquid.LAVA, Liquid.HONEY),
        (Liquid.SHIMMER, Liquid.WATER),
    )
    pair_x = x0 + 5
    for first, second in contact_pairs:
        world.tiles[contact_y, pair_x : pair_x + 2] = Tile.AIR
        world.liquid_kind[contact_y, pair_x] = first
        world.liquid_kind[contact_y, pair_x + 1] = second
        world.liquid_amount[contact_y, pair_x : pair_x + 2] = 255
        pair_x += 11
    world.tiles[arena_top - 3 : arena_top, impact_x - 11 : impact_x + 12] = Tile.SAND
    world.tiles[arena_top - 5 : arena_top - 3, impact_x - 7 : impact_x + 8] = Tile.SILT


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
