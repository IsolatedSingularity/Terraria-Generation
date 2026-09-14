"""Independent Small-world algorithms informed by the pinned 1.4.5.7 source.

These operate on TerraExplorer state, never imported oracle cells. Source paths,
distributions and departures are recorded in the generation fidelity report.
The compact Preview path remains available for compatibility experiments.
"""

from __future__ import annotations

import math

import numpy as np

from terraexplorer.config import Evil
from terraexplorer.fidelity.living_trees import LivingTreesReplay
from terraexplorer.fidelity.pass_snapshot import RAW_DTYPE
from terraexplorer.fidelity.unified_random import UnifiedRandom, int32
from terraexplorer.model import LayerDepths, StructureMarker
from terraexplorer.tiles import Biome, Liquid, Tile, Wall


def terrain(world, _rng):
    """TerrainPass feature walk, native Small limits and extrema-derived layers."""
    h, w = world.shape
    r = UnifiedRandom(int32(world.config.seed_value)).next
    surface = h * 0.3 * r(90, 110) * 0.005
    rock = (surface + h * 0.2) * r(90, 110) * 0.01
    rocks = np.empty(w, dtype=np.int16)
    feature, remaining = 0, 300
    for x in range(w):
        if remaining <= 0:
            feature, remaining = r(5), r(5, 40)
            if feature == 0:
                remaining *= int(r(5, 30) * 0.2)
        remaining -= 1
        if w * 0.45 < x < w * 0.55 and feature >= 3:
            feature = r(3)
        if w * 0.48 < x < w * 0.52:
            feature = 0
        if feature == 0:
            while r(7) == 0:
                surface += r(-1, 2)
        else:
            direction = -1 if feature in (1, 3) else 1
            while r(4 if feature < 3 else 2) == 0:
                surface += direction
            while r(10 if feature < 3 else (6 if feature == 3 else 5)) == 0:
                surface -= direction
        ceiling = h * (0.23 if x < 300 or x > w - 300 else 0.26)
        bounded = max(h * 0.19, min(ceiling, surface))
        if bounded != surface:
            remaining = 0
        surface = bounded
        while r(3) == 0:
            rock += r(-2, 3)
        rock += int(rock < surface + h * 0.06) - int(rock > surface + h * 0.35)
        world.surface[x] = int(surface)
        rocks[x] = math.ceil(rock)
    # A gradual right beach transition replaces TerrainPass's history retarget.
    edge = w - 300
    excess = max(0, int(world.surface[edge - 1]) - round(h * 0.23))
    if excess:
        start = edge - 500
        world.surface[start:edge] -= np.linspace(0, excess, 500).astype(np.int16)
    surface_boundary = int(world.surface.max()) + 25
    rock_boundary = surface_boundary + int((rocks.max() - surface_boundary) / 6) * 6
    world.layers = LayerDepths(surface_boundary, rock_boundary, h - 200)
    yy = np.arange(h)[:, None]
    world.tiles[:] = np.where(
        yy < world.surface, Tile.AIR, np.where(yy < rocks, Tile.DIRT, Tile.STONE)
    )
    world.biomes[:] = np.where(yy < world.surface, Biome.SKY, Biome.FOREST)
    world.biomes[h - 200 :] = Biome.UNDERWORLD
    water = (rock_boundary + h) // 2 + r(-100, 20)
    world.metadata.update(
        surface_low=int(world.surface.min()),
        surface_high=int(world.surface.max()),
        rock_high=int(rocks.max()),
        water_line=water,
        lava_line=water + r(50, 80),
        generation_model="source-backed Small / Terraria 1.4.5.7",
        terrain_rng="UnifiedRandom; reset from normalized seed",
    )


def reset(world, rng):
    from terraexplorer.generation import reset as legacy_reset

    legacy_reset(world, rng)
    w = world.shape[1]
    left = world.metadata["dungeon_side"] == "left"
    dungeon_x = int(rng.integers(round(w * 0.12), round(w * 0.20)))
    world.metadata["dungeon_x"] = dungeon_x if left else w - dungeon_x
    # Keep opposite-side relationships; vary origins rather than fixed stripes.
    world.metadata["snow_x"] = int(w * (0.27 if left else 0.73) + rng.integers(-120, 121))
    world.metadata["jungle_x"] = int(w * (0.74 if left else 0.26) + rng.integers(-140, 141))


def ocean_sand(world, rng):
    """Coastal shelf and descending sand basin, tied to local beach terrain."""
    h, w = world.shape
    for reverse in (False, True):
        coast = int(rng.integers(250, 330))
        edge = w - coast - 1 if reverse else coast
        sea = int(world.surface[edge]) + 8
        floor = float(sea + 80)
        for distance in range(coast):
            x = w - 1 - distance if reverse else distance
            target = sea + 4 + 80 * (1 - distance / coast) ** 0.7
            floor += (target - floor) * 0.12 + rng.uniform(-0.3, 0.3)
            y = int(floor)
            world.tiles[:y, x] = Tile.AIR
            world.tiles[y : min(h, y + 50), x] = Tile.SAND
            world.walls[:y, x] = Wall.NONE
            world.liquid_amount[sea:y, x] = 255
            world.liquid_kind[sea:y, x] = Liquid.WATER
            world.biomes[: y + 50, x] = Biome.OCEAN
            world.surface[x] = y


def beaches(world, _rng):
    """Keep late beach touch-up close to the surface, above the rock substrate."""
    h, w = world.shape
    for x in (*range(340), *range(w - 340, w)):
        y = int(world.surface[x])
        patch = world.tiles[y : min(h, y + 50), x]
        patch[np.isin(patch, (Tile.DIRT, Tile.GRASS, Tile.STONE))] = Tile.SAND


def hardmode(world, rng):
    """GERunner origin distribution, inward 3:5 slopes and native-scale breadth.

    Vectorized rough edges and reduced material families remain approximations.
    The two bands may meet below the playable region, as in the source.
    """
    h, w = world.shape
    fraction = int(rng.integers(300, 400)) * 0.001
    edge_fraction = int(rng.integers(200, 300)) * 0.001
    origins = [int(w * fraction), int(w * (1 - fraction))]
    if world.metadata["dungeon_side"] == "left":
        origins[0] = int(w * edge_fraction)
    else:
        origins[1] = int(w * (1 - edge_fraction))
    good_left = bool(rng.integers(2))
    evil_tile = Tile.EBONSTONE if world.config.evil is Evil.CORRUPTION else Tile.CRIMSTONE
    evil_grass = Tile.CORRUPT_GRASS if world.config.evil is Evil.CORRUPTION else Tile.CRIMSON_GRASS
    evil_kind = Biome.CORRUPTION if world.config.evil is Evil.CORRUPTION else Biome.CRIMSON
    for side, origin in enumerate(origins):
        good = good_left if side == 0 else not good_left
        radius = int(rng.integers(200, 250)) / 2
        for y in range(h):
            center = origin + (1 if side == 0 else -1) * y * 3 / 5
            left, right = max(0, int(center - radius)), min(w, int(center + radius + 1))
            patch = world.tiles[y, left:right]
            edge = abs(np.arange(left, right) - center) < radius * (
                1 + rng.integers(-10, 11, size=len(patch)) * 0.015
            )
            stone = edge & np.isin(
                patch, (Tile.STONE, Tile.EBONSTONE, Tile.CRIMSTONE, Tile.PEARLSTONE)
            )
            grass = edge & np.isin(
                patch, (Tile.GRASS, Tile.CORRUPT_GRASS, Tile.CRIMSON_GRASS, Tile.HALLOW_GRASS)
            )
            # Registry lacks converted sand/ice variants: retain substrate ID,
            # assign biome membership, rather than replacing sand with stone.
            substrate = edge & np.isin(
                patch, (Tile.SAND, Tile.HARDENED_SAND, Tile.SANDSTONE, Tile.ICE)
            )
            patch[stone] = Tile.PEARLSTONE if good else evil_tile
            patch[grass] = Tile.HALLOW_GRASS if good else evil_grass
            world.biomes[y, left:right][stone | grass | substrate] = (
                Biome.HALLOW if good else evil_kind
            )
    world.metadata["hardmode_origins"] = origins
    world.metadata["hardmode"] = True


def runner(world, rng, x, y, strength, steps, wet=False):
    """TileRunner's clipped taper, jittered L1 footprint and bounded velocity.

    Vectorized edge draws use the pass PCG stream, not native cell RNG order.
    Strength is diameter; it is not the old ellipse radius.
    """
    h, w = world.shape
    vx, vy = rng.integers(-10, 11, size=2) * 0.1
    for step in range(steps):
        radius = strength * (1 - step / steps) * 0.5
        x0, x1 = max(1, int(x - radius)), min(w - 1, int(x + radius))
        y0, y1 = max(1, int(y - radius)), min(h - 1, int(y + radius))
        if x0 < x1 and y0 < y1:
            yy, xx = np.ogrid[y0:y1, x0:x1]
            patch = world.tiles[y0:y1, x0:x1]
            selected = abs(xx - x) + abs(yy - y) < strength * 0.5 * (
                1 + rng.integers(-10, 11, size=patch.shape) * 0.015
            )
            selected &= np.isin(patch, (Tile.DIRT, Tile.STONE, Tile.CLAY, Tile.SILT))
            patch[selected] = Tile.AIR
            if wet:
                wet_mask = selected & (
                    (yy < world.metadata["water_line"]) | (yy > world.metadata["lava_line"])
                )
                world.liquid_amount[y0:y1, x0:x1][wet_mask] = 255
                kinds = np.broadcast_to(
                    np.where(yy > world.metadata["lava_line"], Liquid.LAVA, Liquid.WATER),
                    patch.shape,
                )
                world.liquid_kind[y0:y1, x0:x1][wet_mask] = kinds[wet_mask]
        x, y = x + vx, y + vy
        vx = max(-1.0, min(1.0, vx + rng.integers(-10, 11) * 0.05))
        vy = max(-1.0, min(1.0, vy + rng.integers(-10, 11) * 0.05))


def small_holes(world, rng):
    h, w = world.shape
    # Independent hybrid pipeline retains supplemental cave passes. A measured
    # 0.6 budget avoids double-counting their excavated volume; radii/step
    # distributions remain source-derived. This is calibration, not vanilla.
    for _ in range(int(w * h * 0.0015 * 0.6)):
        wet = rng.integers(5) == 0
        for strength, steps in (((2, 5), (2, 20)), ((8, 15), (7, 30))):
            runner(
                world,
                rng,
                int(rng.integers(1, w - 1)),
                int(rng.integers(world.metadata["surface_high"], h - 1)),
                int(rng.integers(*strength)),
                int(rng.integers(*steps)),
                wet,
            )


def dirt_caves(world, rng):
    h, w = world.shape
    for _ in range(int(w * h * 0.00003)):
        x = int(rng.integers(350, w - 350))
        y = int(rng.integers(world.metadata["surface_low"], world.metadata["rock_high"] + 1))
        if abs(x - w // 2) < w * 0.05 and y < world.layers.world_surface:
            continue
        runner(
            world,
            rng,
            x,
            y,
            int(rng.integers(5, 15)),
            int(rng.integers(30, 200)),
            rng.integers(6) == 0,
        )


def rock_caves(world, rng):
    h, w = world.shape
    for _ in range(int(w * h * 0.00013 * 0.6)):
        runner(
            world,
            rng,
            int(rng.integers(1, w - 1)),
            int(rng.integers(world.metadata["rock_high"], h - 1)),
            int(rng.integers(6, 20)),
            int(rng.integers(50, 300)),
            rng.integers(10) == 0,
        )


def ice_biome(world, rng):
    center = world.metadata["snow_x"]
    left, right = center - int(rng.integers(200, 300)), center + int(rng.integers(200, 300))
    bottom = world.metadata["lava_line"] - 140
    for y in range(bottom):
        drift = -1 if world.metadata["dungeon_side"] == "left" else 1
        left += int(rng.integers(-4, 4) / 2) + (drift if rng.integers(4) == 0 else 0)
        right += int(rng.integers(-3, 5) / 2) + (drift if rng.integers(4) == 0 else 0)
        x0, x1 = max(320, left), min(world.shape[1] - 320, right)
        patch = world.tiles[y, x0:x1]
        patch[np.isin(patch, (Tile.DIRT, Tile.GRASS, Tile.SAND, Tile.CLAY))] = Tile.SNOW
        patch[patch == Tile.STONE] = Tile.ICE
        world.biomes[y, x0:x1][y >= world.surface[x0:x1]] = Biome.SNOW


def jungle(world, rng):
    """Overlapping deep mud masses and an upward mud runner to the surface."""
    h, w = world.shape
    origin = int(world.metadata["jungle_x"])
    center_y = (h + world.layers.rock_layer) // 2
    mask = np.zeros(world.shape, bool)
    centers = [
        (
            origin + int(rng.integers(-150, 151)),
            center_y + int(rng.integers(-100, 101)),
            int(rng.integers(375, 750)),
        )
        for _ in range(3)
    ]
    # Union of the source runner's L1 shapes, preserving caves and other hosts.
    x = float(np.mean([p[0] for p in centers]))
    for y in range(center_y, 100, -12):
        x = float(np.clip(x + rng.integers(-12, 13), origin - 120, origin + 120))
        centers.append((int(x), y, int(rng.integers(600, 900))))
    for x, y, strength in centers:
        radius = strength // 2
        x0, x1 = max(350, x - radius), min(w - 350, x + radius)
        y0, y1 = max(0, y - radius), min(h - 200, y + radius)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask[y0:y1, x0:x1] |= abs(xx - x) + abs(yy - y) < radius
    yy = np.arange(h)[:, None]
    mask &= yy >= world.surface
    hosts = np.isin(world.tiles, (Tile.DIRT, Tile.STONE, Tile.GRASS, Tile.CLAY, Tile.SILT))
    world.tiles[mask & hosts] = Tile.MUD
    world.biomes[mask] = Biome.JUNGLE
    world.walls[mask & (world.walls != Wall.NONE)] = Wall.JUNGLE


def _desert_clusters(blocks, rng):
    """Depth-two discovery and neighboring cluster claim/remove competition."""
    rows, columns = blocks.shape
    groups = []
    labels = np.full(blocks.shape, -1, dtype=int)
    for x in range(columns):
        for y in range(rows):
            if not blocks[y, x] or rng.integers(2):
                continue
            found, stack = [], [(x, y, 2)]
            while stack:
                px, py, depth = stack.pop()
                if not (0 <= px < columns and 0 <= py < rows) or not blocks[py, px]:
                    continue
                blocks[py, px] = False
                found.append((px, py))
                if depth:
                    stack.extend(
                        (px + dx, py + dy, depth - 1)
                        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                    )
            if len(found) > 2:
                index = len(groups)
                groups.append(found)
                for px, py in found:
                    labels[py, px] = index
    for group in groups:
        for x, y in group:
            current = labels[y, x]
            if current < 0:
                break
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                px, py = x + dx, y + dy
                if 0 <= px < columns and 0 <= py < rows:
                    other = labels[py, px]
                    if other >= 0 and other != current:
                        labels[labels == other] = -1 if rng.integers(2) == 0 else current
    return [
        np.flatnonzero(labels == index)
        for index in np.unique(labels)
        if index >= 0 and np.count_nonzero(labels == index) >= 4
    ]


def full_desert(world, rng):
    """Cluster potential field with the source DesertHive material thresholds.

    Cluster growth/merging is simplified to connected occupied blocks; inverse
    distance contributions and strongest-two-cluster thresholds are retained.
    """
    h, w = world.shape
    columns, rows = 80, int((rng.random() * 0.5 + 1.5) * 170)
    width, depth = columns * 4, rows * 2
    x0 = int(np.clip(world.metadata["desert_x"] - width // 2, 350, w - 350 - width))
    terrain = world.surface[x0 : x0 + width]
    y0 = int((terrain.mean() + terrain.max()) / 2) + int(rng.integers(40, 60))
    depth = min(depth, h - 220 - y0)
    rows = depth // 2
    by, bx = np.ogrid[:rows, :columns]
    domain = ((bx - columns / 2) / (columns / 2)) ** 2 + (
        (by - rows / 2) / (rows / 2)
    ) ** 2 < 0.8**2
    blocks = (rng.random((rows, columns)) < 0.5) & domain
    clusters = _desert_clusters(blocks, rng)
    first, second = np.zeros((depth, width)), np.zeros((depth, width))
    for cluster in clusters:
        # Bound work to the same ten-block locality used by DesertHive.
        cy, cx = np.divmod(cluster, columns)
        if len(cx) < 2:
            continue
        lx, rx = max(0, int(cx.min()) - 10) * 4, min(columns, int(cx.max()) + 11) * 4
        ty, by_end = max(0, int(cy.min()) - 10) * 2, min(rows, int(cy.max()) + 11) * 2
        yy, xx = np.ogrid[ty:by_end, lx:rx]
        potential = np.zeros((by_end - ty, rx - lx))
        for px, py in zip(cx, cy, strict=True):
            px, py = px + rng.uniform(-0.25, 0.25), py + rng.uniform(-0.25, 0.25)
            potential += 1 / np.maximum(0.01, (xx / 4 - px) ** 2 + (yy / 2 - py) ** 2)
        a, b = first[ty:by_end, lx:rx], second[ty:by_end, lx:rx]
        b[:] = np.maximum(b, np.minimum(a, potential))
        a[:] = np.maximum(a, potential)
    total = first + second
    region = world.tiles[y0 : y0 + depth, x0 : x0 + width]
    region[total > 0.7] = Tile.HARDENED_SAND
    region[total > 1.8] = Tile.SANDSTONE
    region[total > 3.5] = Tile.AIR
    selected = total > 0.7
    world.biomes[y0 : y0 + depth, x0 : x0 + width][selected] = Biome.DESERT
    world.walls[y0 : y0 + depth, x0 : x0 + width][selected] = Wall.SANDSTONE
    world.liquid_amount[y0 : y0 + depth, x0 : x0 + width][selected] = 0
    world.liquid_kind[y0 : y0 + depth, x0 : x0 + width][selected] = Liquid.NONE
    for x in range(x0, x0 + width):
        top = int(world.surface[x])
        world.tiles[top : y0 + 15, x][
            np.isin(world.tiles[top : y0 + 15, x], (Tile.DIRT, Tile.STONE, Tile.GRASS, Tile.CLAY))
        ] = Tile.SAND
        world.biomes[top : y0 + 15, x] = Biome.DESERT
    world.metadata["underground_desert_bounds"] = [x0, y0, x0 + width, y0 + depth]


def evil_biome(world, rng):
    """Surface conversion plus ChasmRunner/CrimStart-derived connected scars."""
    h, w = world.shape
    candidates = [
        x
        for x in range(450, w - 450, 25)
        if abs(x - w // 2) > 500
        and abs(x - world.metadata["jungle_x"]) > 500
        and abs(x - world.metadata["snow_x"]) > 350
    ]
    center = candidates[int(rng.integers(len(candidates)))]
    world.metadata["evil_x"] = center
    crimson = world.config.evil is Evil.CRIMSON
    stone, grass, biome = (
        (Tile.CRIMSTONE, Tile.CRIMSON_GRASS, Biome.CRIMSON)
        if crimson
        else (Tile.EBONSTONE, Tile.CORRUPT_GRASS, Biome.CORRUPTION)
    )
    hosts = np.isin(
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
    outer, inner = np.zeros(world.shape, bool), np.zeros(world.shape, bool)

    def stamp(x, y, radius, shell=12):
        x0, x1 = max(1, int(x - radius - shell)), min(w - 1, int(x + radius + shell) + 1)
        y0, y1 = max(1, int(y - radius - shell)), min(h - 1, int(y + radius + shell) + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        distance = abs(xx - x) + abs(yy - y)
        outer[y0:y1, x0:x1] |= distance < radius + shell
        inner[y0:y1, x0:x1] |= distance < radius

    for x in range(max(350, center - 240), min(w - 350, center + 240)):
        y = int(world.surface[x])
        outer[max(0, y - 1) : y + int(rng.integers(15, 40)), x] = True
    for origin in (center - 70, center + 80):
        x, y = float(origin), float(world.surface[origin] - 1)
        vx, vy = float(rng.uniform(-1, 1)), float(rng.uniform(0.5, 2.5))
        target = world.layers.world_surface + (100 if crimson else 45)
        radius = float(rng.integers(7, 12))
        for _ in range(700):
            radius = float(np.clip(radius + rng.integers(-1, 2), 7, 12 if crimson else 10))
            stamp(x, y, radius, 15)
            if y >= target:
                break
            vx = float(np.clip(vx + rng.uniform(-0.3, 0.3), -1.5, 1.5))
            vx += 0.1 if x < origin - 50 else -0.1 if x > origin + 50 else 0
            vy = float(np.clip(vy + rng.uniform(-0.2, 0.2), 0.5, 2))
            x, y = x + vx, y + vy
        if crimson:
            for _ in range(50):
                stamp(x + rng.integers(-20, 21), y + rng.integers(-20, 21), rng.integers(10, 14), 8)
            for direction in (-1, 1):
                for finger in range(3):
                    px, py = x, y
                    for step in range(int(rng.integers(35, 70))):
                        stamp(px, py, 4, 9)
                        px += direction
                        py += (finger - 1) * 0.45 + rng.uniform(-0.15, 0.15)
        else:
            for direction in (-1, 1):
                px, py = x, y
                for _ in range(int(rng.integers(20, 40))):
                    stamp(px, py, 5, 14)
                    px += direction * 1.3
                    py += rng.uniform(-0.5, 0.5)
    grassy = np.isin(world.tiles, (Tile.GRASS, Tile.SNOW))
    world.tiles[outer & hosts & ~grassy] = stone
    world.tiles[outer & hosts & grassy] = grass
    world.tiles[inner & (hosts | (world.tiles == Tile.AIR))] = Tile.AIR
    world.walls[inner] = Wall.STONE
    world.biomes[outer] = biome
    world.liquid_amount[outer], world.liquid_kind[outer] = 0, Liquid.NONE
    world.metadata["evil_initial_depth"] = int(np.nonzero(outer)[0].max()) + 1


def underworld(world, rng):
    """Ash ceiling and terrain masses around the lava sea, keeping original buildings."""
    from terraexplorer.generation import underworld as legacy_underworld

    legacy_underworld(world, rng)
    h, w = world.shape
    top, bottom = h - int(rng.integers(150, 190)), h - int(rng.integers(40, 70))
    for x in range(w):
        top = int(np.clip(top + rng.integers(-3, 4), h - 190, h - 160))
        bottom += int(rng.integers(-10, 11))
        bottom = int(np.clip(bottom, h - 75, h - 40))
        # Protect existing building material/interiors while filling ash terrain.
        protected = np.isin(world.walls[:, x], (Wall.OBSIDIAN, Wall.HELLSTONE))
        for a, b in ((h - 200, top), (bottom, h)):
            patch = world.tiles[a:b, x]
            mask = ~protected[a:b] & np.isin(patch, (Tile.AIR, Tile.ASH, Tile.HELLSTONE))
            patch[mask] = Tile.ASH
        world.walls[h - 200 :, x][~protected[h - 200 :]] = Wall.NONE
    # Hellstone occurs in deposits rather than a continuous basal stratum.
    for _ in range(180):
        x, y = int(rng.integers(5, w - 5)), int(rng.integers(h - 190, h - 5))
        for _ in range(int(rng.integers(3, 8))):
            patch = world.tiles[max(h - 200, y - 2) : min(h, y + 3), max(0, x - 3) : min(w, x + 4)]
            patch[patch == Tile.ASH] = Tile.HELLSTONE
            x, y = (
                x + int(rng.integers(-2, 3)),
                int(np.clip(y + rng.integers(-2, 3), h - 195, h - 5)),
            )
    occupied = world.tiles != Tile.AIR
    world.liquid_amount[occupied], world.liquid_kind[occupied] = 0, Liquid.NONE


def _paint_structure(world, outer, inner, brick, wall, biome=None):
    """Paint the union once, so subsequent rooms cannot seal earlier halls."""
    world.tiles[outer] = brick
    world.walls[outer] = wall
    world.tiles[inner] = Tile.AIR
    world.liquid_amount[outer] = 0
    world.liquid_kind[outer] = Liquid.NONE
    if biome is not None:
        world.biomes[outer] = biome


def _box(mask, x, y, rx, ry):
    h, w = mask.shape
    mask[
        max(1, int(y - ry)) : min(h - 1, int(y + ry) + 1),
        max(1, int(x - rx)) : min(w - 1, int(x + rx) + 1),
    ] = True


def _hall(outer, inner, start, end, radius, shell):
    distance = max(abs(end[0] - start[0]), abs(end[1] - start[1]))
    for t in np.linspace(0, 1, max(2, int(distance) + 1)):
        x, y = start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t
        _box(outer, x, y, radius + shell, radius + shell)
        _box(inner, x, y, radius, radius)


def _marker(world, name, mask, symbol):
    yy, xx = np.nonzero(mask)
    if len(xx):
        world.structures.append(
            StructureMarker(
                name,
                int(xx.min()),
                int(yy.min()),
                int(xx.max() - xx.min() + 1),
                int(yy.max() - yy.min() + 1),
                symbol,
            )
        )


def dungeon(world, rng):
    """Legacy layout's advancing trunk and temporary branches with local seeds."""
    h, w = world.shape
    outer, inner = np.zeros(world.shape, bool), np.zeros(world.shape, bool)
    origin = int(world.metadata["dungeon_x"])
    surface = int(world.surface[origin])
    r = UnifiedRandom(int32(world.config.seed_value)).next
    x, y = origin, world.layers.rock_layer + 20
    nodes, edges = [(x, y)], []
    bounds = (max(80, origin - 230), min(w - 80, origin + 230))
    delay = 0

    def room(at):
        local = UnifiedRandom(r()).next
        strength = 15 + local(15)
        px, py = at
        dx, dy = local(-10, 11) * 0.1, local(-10, 11) * 0.1
        for _ in range(local(10, 20)):
            _box(outer, px, py, strength * 0.8 + 5, strength * 0.8 + 5)
            _box(inner, px, py, strength * 0.5, strength * 0.5)
            px, py = px + dx, py + dy
            dx = max(-1, min(1, dx + local(-10, 11) * 0.05))
            dy = max(-1, min(1, dy + local(-10, 11) * 0.05))

    def hall(at):
        local = UnifiedRandom(r()).next
        px, py = at
        length, radius = 35 + local(45), 4 + local(2)
        if local(5) == 0:
            radius, length = radius * 2, length // 2
        direction = local(4)
        dx, dy = ((-1, 0), (1, 0), (local(-4, 5) * 0.1, -1), (local(-4, 5) * 0.1, 1))[direction]
        if y < world.layers.rock_layer + 60 and dy < 0:
            dy = 1
        end = (
            int(np.clip(px + dx * length, *bounds)),
            int(np.clip(py + dy * length, world.layers.world_surface + 60, h - 300)),
        )
        _hall(outer, inner, at, end, radius, 5)
        edges.append((list(at), list(end)))
        nodes.append(end)
        return end

    room((x, y))
    for _ in range(w // 60 + r(w // 180)):
        delay = max(0, delay - 1)
        if delay == 0 and r(3) == 0:
            delay = 5
            if r(2) == 0:
                branch = hall((x, y))
                if r(2) == 0:
                    branch = hall(branch)
                room(branch)
            else:
                room((x, y))
        else:
            x, y = hall((x, y))
    room((x, y))
    # Stair shaft from surface entrance to the initial underground junction.
    entrance = (origin, surface - 10)
    _hall(outer, inner, entrance, nodes[0], 5, 7)
    _box(outer, *entrance, 22, 16)
    _box(inner, *entrance, 15, 10)
    _hall(outer, inner, entrance, (origin + 30, surface - 3), 3, 3)
    _paint_structure(world, outer, inner, Tile.DUNGEON_BRICK, Wall.DUNGEON, Biome.DUNGEON)
    world.tiles[surface - 7 : surface + 1, origin + 28 : origin + 36] = Tile.AIR
    _marker(world, "Dungeon", outer, "D")
    world.metadata["dungeon_graph"] = {"nodes": nodes, "edges": edges}


def jungle_temple(world, rng):
    """Variable non-overlapping rooms, increasing switchback rows, large final chamber."""
    h, w = world.shape
    r = UnifiedRandom(int32(world.config.seed_value)).next
    count, direction, row_count, row_used = r(10, 16), (-1 if r(2) else 1), r(1, 3), 0
    rooms = []
    x, y = 0, 0
    for index in range(count):
        rw, rh = r(25, 50), r(20, 35)
        rh = min(rw, rh)
        if index == count - 1:
            rw, rh = int(r(55, 65) * 1.6), int(r(45, 50) * 1.35)
        previous_w, previous_h = rooms[-1][2:] if rooms else (rw, rh)
        if row_used >= row_count:
            direction *= -1
            y += (previous_h + rh) // 2 + r(5, 12)
            x += r(-5, 6)
            row_count += 1
            row_used = 0
        else:
            x += direction * ((previous_w + rw) // 2 + r(5, 12))
            y += r(-3, 4)
        # The large altar room can touch the row above; lower until disjoint.
        while any(
            abs(x - px) < (rw + pw) / 2 + 3 and abs(y - py) < (rh + ph) / 2 + 3
            for px, py, pw, ph in rooms
        ):
            y += 1
        rooms.append((x, y, rw, rh))
        row_used += 1
    left = min(x - rw // 2 for x, y, rw, rh in rooms) - 10
    top = min(y - rh // 2 for x, y, rw, rh in rooms) - 10
    right = max(x + rw // 2 for x, y, rw, rh in rooms) + 10
    bottom = max(y + rh // 2 for x, y, rw, rh in rooms) + 10
    target_x = int(
        np.clip(
            world.metadata["jungle_x"] + rng.integers(-170, 171),
            350 + (right - left) // 2,
            w - 350 - (right - left) // 2,
        )
    )
    target_y = int(
        rng.integers(
            world.layers.rock_layer, max(world.layers.rock_layer + 1, h - 230 - (bottom - top))
        )
    )
    ox, oy = target_x - (left + right) // 2, target_y - top
    rooms = [(x + ox, y + oy, rw, rh) for x, y, rw, rh in rooms]
    outer, inner = np.zeros(world.shape, bool), np.zeros(world.shape, bool)
    for x, y, rw, rh in rooms:
        _box(outer, x, y, rw // 2 + 6, rh // 2 + 6)
        _box(inner, x, y, rw // 2 - r(3, 8), rh // 2 - r(3, 8))
    for a, b in zip(rooms, rooms[1:]):
        _hall(outer, inner, a[:2], b[:2], 3, 7)
    # Strict-angle enclosing brick mass follows the row envelope, not a rectangle.
    for row in range(max(1, target_y), min(h - 1, bottom + oy)):
        columns = np.flatnonzero(outer[row])
        if columns.size:
            outer[row, columns[0] : columns[-1] + 1] = True
    _paint_structure(world, outer, inner, Tile.LIHZAHRD_BRICK, Wall.LIHZAHRD)
    _marker(world, "Jungle temple", outer, "T")
    world.metadata["temple_rooms"] = rooms
    world.metadata["temple_room_count"] = len(rooms)


class _ProductTree(LivingTreesReplay):
    """Reuse validated crown/root geometry; connected work is a product approximation."""

    def unsupported(self, method, *args):
        if method in ("PlaceTile", "PlaceSmallPile"):
            # Cosmetic piles require canonical object/global state unavailable here.
            return False
        if method != "GrowLivingTree_MakePassage":
            return super().unsupported(method, *args)
        y, width, left, right, _patch = args
        x = (left + right) // 2
        start = y - 8
        end = min(self.height - 240, y + self.rng.next(80, 150))
        for row in range(start, end):
            if row % 14 == 0:
                x += self.rng.next(-1, 2)
            for column in range(x - 4, x + 5):
                self.place_wood(column, row, False)
                self.cells["wall"][column, row] = 244
            self.cells["sTileHeader"][x - 1 : x + 2, row] &= np.uint16(65503)
        direction = -1 if self.rng.next(2) else 1
        end_x = x + direction * self.rng.next(25, 65)
        for column in range(min(x, end_x) - 1, max(x, end_x) + 2):
            for row in range(end - 5, end + 4):
                self.place_wood(column, row, False)
                self.cells["wall"][column, row] = 244
            self.cells["sTileHeader"][column, end - 2 : end + 1] &= np.uint16(65503)
        for column in range(end_x - 12, end_x + 13):
            for row in range(end - 11, end + 4):
                self.place_wood(column, row, False)
                self.cells["wall"][column, row] = 244
            if end_x - 10 <= column <= end_x + 10:
                self.cells["sTileHeader"][column, end - 9 : end + 1] &= np.uint16(65503)
        return None


def living_trees(world, rng):
    """Project simplified state into a temporary tree geometry adapter, no oracle input."""
    h, w = world.shape
    tree = _ProductTree.__new__(_ProductTree)
    tree.width, tree.height, tree.surface = w, h, world.layers.world_surface
    tree.rng = UnifiedRandom(int32(world.config.seed_value))
    tree.cells = np.zeros((w, h), dtype=RAW_DTYPE)
    tree.cells["type"] = 1
    mapping = {
        Tile.DIRT: 0,
        Tile.GRASS: 2,
        Tile.STONE: 1,
        Tile.LIVING_WOOD: 191,
        Tile.LEAF: 192,
        Tile.CLOUD: 189,
        Tile.RAIN_CLOUD: 196,
    }
    for tile, native in mapping.items():
        tree.cells["type"][world.tiles.T == tile] = native
    # All other occupied materials are deliberately invalid tree supports.
    tree.cells["type"][~np.isin(world.tiles.T, tuple(mapping))] = 147
    tree.cells["sTileHeader"] = np.where(world.tiles.T != Tile.AIR, 32, 0)
    tree.cells["wall"] = np.where(
        world.walls.T == Wall.DUNGEON, 7, np.where(world.walls.T != Wall.NONE, 2, 0)
    )
    tree.solid_types = [True] * 800
    tree.solid_top, tree.ores, tree.clouds = [False] * 800, [False] * 800, [False] * 800
    tree.clouds[189] = tree.clouds[196] = True
    tree.dungeon_walls = [False] * 400
    tree.dungeon_walls[7] = True
    tree.trace = []
    count = tree.rng.next(3)
    if count == 0 and tree.rng.next(2) == 0:
        count = 1
    completed = 0
    for _ in range(count):
        for _attempt in range(w // 2):
            x = tree.rng.next(350, w - 350)
            if abs(x - w // 2) < 200:
                continue
            y = int(world.surface[x]) - 1
            if world.tiles[y + 1, x] not in (Tile.DIRT, Tile.GRASS) or y <= 150:
                continue
            if tree.grow(x, y):
                completed += 1
                break
    wood = ((tree.cells["sTileHeader"] & 32) != 0) & (tree.cells["type"] == 191)
    leaves = ((tree.cells["sTileHeader"] & 32) != 0) & (tree.cells["type"] == 192)
    passage = tree.cells["wall"] == 244
    world.tiles[wood.T] = Tile.LIVING_WOOD
    world.tiles[leaves.T] = Tile.LEAF
    world.tiles[(passage & ((tree.cells["sTileHeader"] & 32) == 0)).T] = Tile.AIR
    world.walls[passage.T] = Wall.DIRT
    mask = (wood | leaves | passage).T
    world.liquid_amount[mask], world.liquid_kind[mask] = 0, Liquid.NONE
    from terraexplorer.fidelity.structures import components

    for indices in components(mask, diagonal=True):
        if len(indices) < 100:
            continue
        yy, xx = np.divmod(indices, w)
        world.structures.append(
            StructureMarker(
                "Living tree",
                int(xx.min()),
                int(yy.min()),
                int(xx.max() - xx.min() + 1),
                int(yy.max() - yy.min() + 1),
                "Y",
            )
        )
    world.metadata["living_tree_count"] = completed


SMALL_HANDLERS = {
    "reset": reset,
    "terrain": terrain,
    "ocean_sand": ocean_sand,
    "beaches": beaches,
    "small_holes": small_holes,
    "dirt_caves": dirt_caves,
    "rock_caves": rock_caves,
    "ice_biome": ice_biome,
    "dungeon": dungeon,
    "jungle_temple": jungle_temple,
    "living_trees": living_trees,
    "jungle": jungle,
    "full_desert": full_desert,
    "evil_biome": evil_biome,
    "underworld": underworld,
}
