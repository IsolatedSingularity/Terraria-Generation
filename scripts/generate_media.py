"""Regenerate repository visuals and Windows icon from the tested engine."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from scripts.media_views import crop_world, detail, overview
from terraexplorer.config import Evil, WorldConfig, WorldScale
from terraexplorer.generation import advance_biome_spread, apply_hardmode
from terraexplorer.model import GeneratedWorld, StructureMarker
from terraexplorer.passes import PASS_SPECS, Fidelity
from terraexplorer.pipeline import TerraExplorerPipeline, generate_world
from terraexplorer.render import GENERATION_MILESTONES, add_title_bar, render_world
from terraexplorer.simulations import simulate_catastrophe_chain
from terraexplorer.spawn_opportunity import ground_candidate_mass
from terraexplorer.tiles import TILE_STYLES, Biome, Liquid, Tile, Wall

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "docs" / "media"
ASSETS = ROOT / "terraexplorer" / "assets"
BG = "#0b1220"
PANEL = "#131d2e"
TEXT = "#e7edf7"
MUTED = "#9fb0c9"
ACCENT = "#63d3c1"
GOLD = "#e4b85c"
LARGE_WORLD_SEED = "Ash Compass"
README_EXTRA_MILESTONES = {
    "Mushroom Patches",
    "Marble",
    "Granite",
    "Create Ocean Caves",
    "Spider Caves",
    "Gem Caves",
    "Micro Biomes",
}


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        ("seguisb.ttf", "DejaVuSans-Bold.ttf") if bold else ("segoeui.ttf", "DejaVuSans.ttf")
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def build_icon() -> None:
    logo = Image.open(ASSETS / "terraexplorer_logo.png").convert("RGBA")
    logo.thumbnail((512, 512), Image.Resampling.LANCZOS)
    logo.save(ASSETS / "terraexplorer_logo.png", optimize=True)
    logo.save(MEDIA / "terraexplorer_logo.png", optimize=True)

    pixels = np.asarray(logo).copy()
    height, width = pixels.shape[:2]
    rgb = pixels[:, :, :3].astype(np.int16)
    corner = max(8, min(height, width) // 24)
    background_samples = np.concatenate(
        (
            rgb[:corner, :corner].reshape(-1, 3),
            rgb[:corner, -corner:].reshape(-1, 3),
            rgb[-corner:, :corner].reshape(-1, 3),
            rgb[-corner:, -corner:].reshape(-1, 3),
        )
    )
    background = np.median(background_samples, axis=0)
    distance = np.max(np.abs(rgb - background), axis=2)
    alpha = np.clip((distance - 7) * (255 / 24), 0, 255).astype(np.uint8)
    definite_foreground = Image.fromarray((distance >= 31).astype(np.uint8) * 255)
    near_foreground = np.asarray(definite_foreground.filter(ImageFilter.MaxFilter(5))) > 0
    alpha[near_foreground & (distance > 9)] = np.maximum(
        alpha[near_foreground & (distance > 9)],
        170,
    )

    yy, xx = np.indices((height, width))
    brown = (
        (rgb[:, :, 0] > rgb[:, :, 1] + 8)
        & (rgb[:, :, 1] > rgb[:, :, 2] + 4)
        & (rgb[:, :, 0] > 34)
        & (xx >= width * 0.22)
        & (xx <= width * 0.78)
        & (yy >= height * 0.27)
        & (yy <= height * 0.95)
    )
    pixels[brown, :3] = np.clip(pixels[brown, :3].astype(np.float32) * 0.60, 0, 255).astype(
        np.uint8
    )
    pixels[:, :, 3] = alpha
    readme_logo = Image.fromarray(pixels, mode="RGBA")
    alpha_bounds = readme_logo.getchannel("A").getbbox()
    if alpha_bounds is None:
        readme_logo = logo.copy()
    else:
        padding = 8
        readme_logo = readme_logo.crop(
            (
                max(0, alpha_bounds[0] - padding),
                max(0, alpha_bounds[1] - padding),
                min(readme_logo.width, alpha_bounds[2] + padding),
                min(readme_logo.height, alpha_bounds[3] + padding),
            )
        )
    readme_logo.save(MEDIA / "terraexplorer_readme_logo.png", optimize=True)
    logo.save(
        ASSETS / "terraexplorer.ico",
        sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )
    logo.resize((256, 256), Image.Resampling.LANCZOS).save(
        MEDIA / "terraexplorer_icon.png", optimize=True
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
        metadata=dict(world.metadata),
        structures=list(world.structures),
        pass_results=list(world.pass_results),
    )


@lru_cache(maxsize=1)
def _large_reference_world() -> GeneratedWorld:
    return generate_world(
        WorldConfig(
            seed=LARGE_WORLD_SEED,
            scale=WorldScale.SMALL,
            evil=Evil.CORRUPTION,
            hardmode=True,
        )
    )


def _meteor_site(world: GeneratedWorld) -> int:
    width = world.shape[1]
    border = 150 if width >= 1000 else 12
    spawn = int(world.metadata.get("spawn_x", width // 2))
    exclusion = round(width * 0.08)
    protected = [
        marker
        for marker in world.structures
        if marker.kind in {"Dungeon", "Jungle temple", "Floating island", "Aether"}
    ]
    candidates: list[int] = []
    for x in range(border, width - border):
        if abs(x - spawn) <= exclusion:
            continue
        surface_y = int(world.surface[x])
        if any(
            marker.x - 36 <= x <= marker.x + marker.width + 36
            and marker.y - 36 <= surface_y <= marker.y + marker.height + 36
            for marker in protected
        ):
            continue
        y0 = max(0, surface_y - 15)
        y1 = min(world.shape[0], surface_y + 15)
        x0 = max(0, x - 15)
        x1 = min(width, x + 15)
        liquid_tiles = np.count_nonzero(world.liquid_amount[y0:y1, x0:x1])
        cloud_tiles = np.count_nonzero(
            np.isin(world.tiles[y0:y1, x0:x1], (Tile.CLOUD, Tile.RAIN_CLOUD))
        )
        if liquid_tiles < 25 and cloud_tiles == 0:
            candidates.append(x)
    if not candidates:
        return int(np.clip(width * 0.28, border, width - border - 1))
    rng = np.random.default_rng(world.config.seed_value ^ 0x4D455445)
    return candidates[int(rng.integers(0, len(candidates)))]


def _with_media_meteorite(source: GeneratedWorld) -> GeneratedWorld:
    world = _clone_world(source)
    x = _meteor_site(world)
    y = int(world.surface[x])
    radius_x, radius_y = (9, 6) if world.config.scale is WorldScale.PREVIEW else (70, 36)
    y0 = max(0, y - radius_y)
    y1 = min(world.shape[0], y + radius_y + 1)
    x0 = max(0, x - radius_x)
    x1 = min(world.shape[1], x + radius_x + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    outer = ((xx - x) / radius_x) ** 2 + ((yy - y) / radius_y) ** 2 <= 1.0
    inner = ((xx - x) / max(1, radius_x * 0.58)) ** 2 + (
        (yy - (y - radius_y * 0.12)) / max(1, radius_y * 0.55)
    ) ** 2 <= 1.0
    local_tiles = world.tiles[y0:y1, x0:x1]
    local_liquid = world.liquid_amount[y0:y1, x0:x1]
    local_kind = world.liquid_kind[y0:y1, x0:x1]
    local_tiles[outer] = Tile.METEORITE
    local_tiles[inner] = Tile.AIR
    local_liquid[outer] = 0
    local_kind[outer] = Liquid.NONE
    world.structures.append(StructureMarker("Meteorite", x0, y0, x1 - x0, y1 - y0, "M"))
    world.metadata["media_meteorite_x"] = x
    return world


def _save_readme_generation_gif(config: WorldConfig, path: Path) -> None:
    milestones = set(GENERATION_MILESTONES) | README_EXTRA_MILESTONES
    if not config.hardmode:
        milestones.discard("Hardmode V Transformation")
    frames: list[Image.Image] = []

    def capture(spec, world: GeneratedWorld) -> None:
        if spec.name not in milestones:
            return
        frames.append(
            add_title_bar(
                overview(world),
                spec.name,
                spec.phase.value,
            )
        )

    final_world = TerraExplorerPipeline().generate(config, snapshot=capture)
    meteor_world = _with_media_meteorite(final_world)
    frames.append(
        add_title_bar(
            overview(meteor_world),
            "METEORITE IMPACT",
            "post-generation world event",
        )
    )
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=[420] * (len(frames) - 1) + [2000],
        loop=0,
        optimize=True,
        disposal=2,
    )


def _marker_anchor(world: GeneratedWorld, kind: str) -> tuple[int, int]:
    markers = [marker for marker in world.structures if marker.kind == kind]
    marker = max(markers, key=lambda item: item.width * item.height)
    return marker.x + marker.width // 2, marker.y + marker.height // 2


def _mask_anchor(mask: np.ndarray) -> tuple[int, int]:
    component = _largest_connected_component(mask)
    if not len(component):
        return mask.shape[1] // 2, mask.shape[0] // 2
    return round(float(component[:, 1].mean())), round(float(component[:, 0].mean()))


def _draw_label_group(
    draw: ImageDraw.ImageDraw,
    items: list[tuple[str, tuple[int, int]]],
    *,
    world_shape: tuple[int, int],
    image_box: tuple[int, int, int, int],
    side: str,
) -> None:
    image_x, image_y, image_width, image_height = image_box
    world_height, world_width = world_shape
    label_font = font(13, bold=True)
    sorted_items = sorted(items, key=lambda item: item[1][0])
    rows = (sorted_items[::2], sorted_items[1::2])
    for row_index, row in enumerate(rows):
        if not row:
            continue
        spacing = image_width / len(row)
        for index, (label, (world_x, world_y)) in enumerate(row):
            anchor_x = image_x + round(world_x / world_width * image_width)
            anchor_y = image_y + round(world_y / world_height * image_height)
            label_x = image_x + round((index + 0.5) * spacing)
            if side == "top":
                text_y = 48 + row_index * 27
                line_y = image_y - 7
            else:
                text_y = image_y + image_height + 25 + row_index * 27
                line_y = image_y + image_height + 7
            draw.line((anchor_x, anchor_y, label_x, line_y), fill=GOLD, width=1)
            draw.rectangle(
                (anchor_x - 4, anchor_y - 4, anchor_x + 4, anchor_y + 4),
                outline=GOLD,
                width=2,
            )
            bounds = draw.textbbox((0, 0), label, font=label_font)
            draw.text(
                (label_x - (bounds[2] - bounds[0]) / 2, text_y),
                label,
                fill=TEXT,
                font=label_font,
            )


def _build_feature_distribution(world: GeneratedWorld) -> Image.Image:
    rendered = render_world(
        world,
        scale=1,
        biome_overlay=True,
        markers=False,
        material_texture=False,
    ).resize((1400, 400), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (1560, 650), BG)
    canvas.paste(rendered, (80, 120))
    draw = ImageDraw.Draw(canvas)
    title_font = font(24, bold=True)
    title = f"FEATURE DISTRIBUTION | {LARGE_WORLD_SEED.upper()}"
    title_width = draw.textlength(title, font=title_font)
    draw.text(((canvas.width - title_width) / 2, 8), title, fill=TEXT, font=title_font)

    bounded_kinds = {
        "Aether",
        "Dungeon",
        "Floating island",
        "Gem cave",
        "Glowing mushroom",
        "Granite biome",
        "Hive",
        "Jungle temple",
        "Living tree",
        "Meteorite",
        "Minecart track",
        "Pyramid",
        "Ruined house",
        "Spider cave",
        "Underground cabin",
        "Underground ocean",
    }
    for marker in world.structures:
        if marker.kind not in bounded_kinds:
            continue
        x0 = 80 + round(marker.x / world.shape[1] * 1400)
        x1 = 80 + round((marker.x + marker.width) / world.shape[1] * 1400)
        y0 = 120 + round(marker.y / world.shape[0] * 400)
        y1 = 120 + round((marker.y + marker.height) / world.shape[0] * 400)
        draw.rectangle((x0, y0, max(x0 + 2, x1), max(y0 + 2, y1)), outline=GOLD, width=1)

    spawn_x = int(world.metadata["spawn_x"])
    snow_x = int(world.metadata["snow_x"])
    desert_x = int(world.metadata["desert_x"])
    jungle_x = int(world.metadata["jungle_x"])
    meteor_x = int(world.metadata["media_meteorite_x"])
    top_items = [
        ("WEST OCEAN", (140, int(world.surface[140]))),
        ("FLOATING ISLANDS", _marker_anchor(world, "Floating island")),
        ("LIVING TREES", _marker_anchor(world, "Living tree")),
        ("SNOW", (snow_x, int(world.surface[snow_x]) + 40)),
        ("DUNGEON", _marker_anchor(world, "Dungeon")),
        ("METEORITE", (meteor_x, int(world.surface[meteor_x]))),
        ("SPAWN FOREST", (spawn_x, int(world.surface[spawn_x]))),
        ("PYRAMID", _marker_anchor(world, "Pyramid")),
        ("DESERT", (desert_x, int(world.surface[desert_x]) + 60)),
        ("JUNGLE", (jungle_x, int(world.surface[jungle_x]) + 80)),
        ("EAST OCEAN", (world.shape[1] - 140, int(world.surface[-140]))),
    ]
    west = slice(0, 420)
    water_rows = np.arange(world.shape[0])[:, None] > world.layers.world_surface
    ocean_cave = (
        (world.liquid_kind[:, west] == Liquid.WATER)
        & (world.liquid_amount[:, west] > 0)
        & water_rows
    )
    ocean_x, ocean_y = (
        _marker_anchor(world, "Underground ocean")
        if any(marker.kind == "Underground ocean" for marker in world.structures)
        else _mask_anchor(ocean_cave)
    )
    bottom_items = [
        ("UNDERGROUND OCEAN", (ocean_x, ocean_y)),
        ("GRANITE", _mask_anchor(world.tiles == Tile.GRANITE)),
        ("GLOWING MUSHROOM", _mask_anchor(world.biomes == Biome.MUSHROOM)),
        ("SPIDER NEST", _marker_anchor(world, "Spider cave")),
        ("AETHER", _marker_anchor(world, "Aether")),
        ("HIVES", _mask_anchor(world.walls == Wall.HIVE)),
        ("MINECART TRACKS", _marker_anchor(world, "Minecart track")),
        ("CORRUPTION", _mask_anchor(world.biomes == Biome.CORRUPTION)),
        ("HALLOW", _mask_anchor(world.biomes == Biome.HALLOW)),
        ("JUNGLE TEMPLE", _marker_anchor(world, "Jungle temple")),
        ("GEM CAVES", _marker_anchor(world, "Gem cave")),
        ("UNDERGROUND CABINS", _marker_anchor(world, "Underground cabin")),
        ("RUINED HOUSES", _marker_anchor(world, "Ruined house")),
    ]
    image_box = (80, 120, 1400, 400)
    _draw_label_group(
        draw,
        top_items,
        world_shape=world.shape,
        image_box=image_box,
        side="top",
    )
    _draw_label_group(
        draw,
        bottom_items,
        world_shape=world.shape,
        image_box=image_box,
        side="bottom",
    )
    return canvas


def build_world_media() -> None:
    config = WorldConfig(
        seed="TerraExplorer", scale=WorldScale.SMALL, evil=Evil.CORRUPTION, hardmode=True
    )
    _save_readme_generation_gif(config, MEDIA / "terraexplorer_generation.gif")
    reference_world = _with_media_meteorite(_large_reference_world())
    _build_feature_distribution(reference_world).save(
        MEDIA / "terraexplorer_world.png",
        optimize=True,
    )


def build_idle_world_evolution() -> None:
    scenarios = (
        (
            "idle_corruption.gif",
            "Chasm Crown",
            Evil.CORRUPTION,
            False,
            "CORRUPTION UNCHECKED",
        ),
        (
            "idle_crimson.gif",
            "The Red Burrow",
            Evil.CRIMSON,
            False,
            "CRIMSON UNCHECKED",
        ),
        (
            "idle_opposition.gif",
            "Two Fronts at Dusk 20",
            Evil.CORRUPTION,
            True,
            "CORRUPTION AGAINST HALLOW",
        ),
    )
    for filename, seed, evil, include_hallow, title in scenarios:
        config = WorldConfig(seed=seed, evil=evil, scale=WorldScale.SMALL)
        world = generate_world(config)
        rng = np.random.default_rng(config.seed_value ^ 0x1D1E5EED)
        if include_hallow:
            apply_hardmode(world, rng)
        else:
            world.metadata["hardmode"] = True

        # A native-tile detail window keeps front motion legible. Include a
        # 32-tile halo while evolving; only the interior is rendered.
        center_x = int(world.metadata["evil_x"])
        center_y = world.layers.world_surface + 35
        if include_hallow:
            center_y = world.layers.underworld - 40
            # Fixed seed chosen for two native conversion legs close enough
            # to share a detail view. No bands are painted into the crop.
            center_x = round(sum(world.metadata["hardmode_origins"]) / 2)
        world = crop_world(world, center_x - 152, center_y - 102, 304, 204)

        frames: list[Image.Image] = []
        captures = 9
        for capture_index in range(captures):
            evil_tiles = (
                (Tile.EBONSTONE, Tile.CORRUPT_GRASS)
                if evil is Evil.CORRUPTION
                else (Tile.CRIMSTONE, Tile.CRIMSON_GRASS)
            )
            evil_count = int(np.count_nonzero(np.isin(world.tiles, evil_tiles)))
            hallow_count = int(
                np.count_nonzero(np.isin(world.tiles, (Tile.PEARLSTONE, Tile.HALLOW_GRASS)))
            )
            subtitle = f"batch {capture_index:02d} | native Small-world crop | evil {evil_count:,}"
            if include_hallow:
                subtitle += f" | Hallow {hallow_count:,} tiles"
            frames.append(
                add_title_bar(
                    detail(world, 32, 32, 240, 140, 4),
                    title,
                    subtitle,
                )
            )
            if capture_index < captures - 1:
                advance_biome_spread(world, rng, iterations=12)

        frames[0].save(
            MEDIA / filename,
            save_all=True,
            append_images=frames[1:],
            duration=[620] * (len(frames) - 1) + [1800],
            loop=0,
            optimize=True,
            disposal=2,
        )


_HAZARD_TILES = {
    Biome.CORRUPTION: (Tile.EBONSTONE, Tile.CORRUPT_GRASS),
    Biome.CRIMSON: (Tile.CRIMSTONE, Tile.CRIMSON_GRASS),
    Biome.HALLOW: (Tile.PEARLSTONE, Tile.HALLOW_GRASS),
}
_HAZARD_HOSTS = (
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


def _clear_media_hazards(world: GeneratedWorld) -> None:
    stone_hazards = np.isin(world.tiles, (Tile.EBONSTONE, Tile.CRIMSTONE, Tile.PEARLSTONE))
    grass_hazards = np.isin(
        world.tiles,
        (Tile.CORRUPT_GRASS, Tile.CRIMSON_GRASS, Tile.HALLOW_GRASS),
    )
    world.tiles[stone_hazards] = Tile.STONE
    world.tiles[grass_hazards] = Tile.GRASS
    world.biomes[np.isin(world.biomes, tuple(_HAZARD_TILES))] = Biome.FOREST


def _paint_hazard_ellipse(
    world: GeneratedWorld,
    biome: Biome,
    center_x: int,
    center_y: int,
    radius_x: int,
    radius_y: int,
) -> None:
    y0 = max(0, center_y - radius_y)
    y1 = min(world.shape[0], center_y + radius_y + 1)
    x0 = max(0, center_x - radius_x)
    x1 = min(world.shape[1], center_x + radius_x + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    ellipse = ((xx - center_x) / radius_x) ** 2 + ((yy - center_y) / radius_y) ** 2 <= 1.0
    local = world.tiles[y0:y1, x0:x1]
    vulnerable = ellipse & np.isin(local, _HAZARD_HOSTS)
    stone, grass = _HAZARD_TILES[biome]
    grassy = vulnerable & np.isin(local, (Tile.GRASS, Tile.JUNGLE_GRASS))
    local[vulnerable & ~grassy] = stone
    local[grassy] = grass
    world.biomes[y0:y1, x0:x1][vulnerable] = biome


def _seed_media_hazards(world: GeneratedWorld, scenario: str) -> tuple[Biome, ...]:
    _clear_media_hazards(world)
    world.metadata["hardmode"] = True
    evil = Biome.CORRUPTION if world.config.evil is Evil.CORRUPTION else Biome.CRIMSON
    hazards = (evil, Biome.HALLOW) if scenario == "all" else (evil,)
    for index, biome in enumerate(hazards):
        x0, x1 = (105, 111) if index == 0 else (155, 161)
        patch = world.tiles[:, x0:x1]
        valid = np.isin(patch, (Tile.STONE, Tile.GRASS, Tile.JUNGLE_GRASS, Tile.SAND, Tile.ICE))
        stone, grass = _HAZARD_TILES[biome]
        grassy = valid & np.isin(patch, (Tile.GRASS, Tile.JUNGLE_GRASS))
        patch[valid & ~grassy] = stone
        patch[grassy] = grass
        world.biomes[:, x0:x1][valid] = biome
    return hazards


def _install_media_containment(world: GeneratedWorld, technique: str) -> tuple[np.ndarray, int]:
    barrier_x = 120
    if technique == "trench":
        world.tiles[:, barrier_x - 1 : barrier_x + 2] = Tile.AIR
        world.walls[:, barrier_x - 1 : barrier_x + 2] = Wall.NONE
        world.liquid_amount[:, barrier_x - 1 : barrier_x + 2] = 0
        world.liquid_kind[:, barrier_x - 1 : barrier_x + 2] = Liquid.NONE
    elif technique == "sunflowers":
        for x in range(barrier_x - 8, barrier_x + 9, 2):
            y = int(np.clip(world.surface[x], 2, world.shape[0] - 2))
            world.tiles[y - 1, x] = Tile.SUNFLOWER
    elif technique == "chlorophyte":
        world.tiles[95:98, barrier_x - 1 : barrier_x + 2] = Tile.CHLOROPHYTE
    else:
        x0, x1, top, bottom = 117, 151, 30, 120
        world.tiles[top:bottom, x0 : x0 + 3] = Tile.OBSIDIAN_BRICK
        world.tiles[top:bottom, x1 - 3 : x1] = Tile.OBSIDIAN_BRICK
        world.tiles[top : top + 3, x0:x1] = Tile.OBSIDIAN_BRICK
        world.tiles[bottom - 3 : bottom, x0:x1] = Tile.OBSIDIAN_BRICK
    return np.zeros(world.shape, dtype=bool), barrier_x


def _advance_media_hazards(
    world: GeneratedWorld,
    hazards: tuple[Biome, ...],
    protected: np.ndarray,
    rng: np.random.Generator,
    *,
    iterations: int,
) -> None:
    del hazards, protected
    advance_biome_spread(world, rng, iterations=iterations)


def build_hazard_containment_animation() -> None:
    scenarios = (
        ("corruption", "Violet Quarantine", Evil.CORRUPTION, "trench", "THREE-TILE TRENCH"),
        ("crimson", "Red Garden", Evil.CRIMSON, "sunflowers", "SUNFLOWER CORDON"),
        (
            "corruption",
            "Verdant Defense",
            Evil.CORRUPTION,
            "chlorophyte",
            "CHLOROPHYTE DEFENSE",
        ),
        ("all", "Three Front Siege", Evil.CRIMSON, "bastion", "BRICK BASTION"),
    )
    states: list[
        tuple[str, str, GeneratedWorld, tuple[Biome, ...], np.ndarray, int, np.random.Generator]
    ] = []
    for scenario, seed, evil, technique, title in scenarios:
        source = generate_world(WorldConfig(seed=seed, evil=evil, scale=WorldScale.SMALL))
        x = int(source.metadata["spawn_x"])
        world = crop_world(source, x - 120, int(source.surface[x]) - 24, 240, 140)
        hazards = _seed_media_hazards(world, scenario)
        protected, barrier_x = _install_media_containment(world, technique)
        rng = np.random.default_rng(world.config.seed_value ^ 0x48415A)
        states.append((title, seed, world, hazards, protected, barrier_x, rng))

    frames: list[Image.Image] = []
    for cycle in range(8):
        cards: list[Image.Image] = []
        for title, seed, world, hazards, protected, barrier_x, rng in states:
            counts = [int(np.count_nonzero(world.biomes == biome)) for biome in hazards]
            count_text = "/".join(f"{count:,}" for count in counts)
            image = render_world(world, scale=2, markers=False, biome_overlay=True)
            draw = ImageDraw.Draw(image)
            draw.line((barrier_x * 2, 0, barrier_x * 2, image.height - 1), fill=GOLD, width=2)
            cards.append(
                add_title_bar(
                    image,
                    title,
                    f"controlled crop | batch {cycle:02d} | infected {count_text}",
                )
            )
            if cycle < 7:
                _advance_media_hazards(
                    world,
                    hazards,
                    protected,
                    rng,
                    iterations=4,
                )
        frames.append(_card_grid(cards))
    frames[0].save(
        MEDIA / "containment_lab.gif",
        save_all=True,
        append_images=frames[1:],
        duration=[700] * (len(frames) - 1) + [1800],
        loop=0,
        optimize=True,
        disposal=2,
    )


def build_biome_study() -> None:
    studies = (
        ("FOREST", Evil.CORRUPTION),
        ("SNOW", Evil.CORRUPTION),
        ("DESERT", Evil.CRIMSON),
        ("JUNGLE", Evil.CRIMSON),
        ("CORRUPTION", Evil.CORRUPTION),
        ("CRIMSON", Evil.CRIMSON),
        ("GLOWING MUSHROOM", Evil.CORRUPTION),
        ("METEORITE", Evil.CRIMSON),
        ("UNDERGROUND OCEAN", Evil.CORRUPTION),
        ("SPIDER NEST", Evil.CRIMSON),
    )
    fixed_seeds = dict(
        zip(
            (s[0] for s in studies),
            (
                "Violet Scar",
                "Salt Cathedral",
                "World Below",
                "Red Descent",
                "Glass Horizon",
                "Cinder Archive",
                "Emerald Hunger",
                "Iron Orchard",
                "Ash Compass",
                "Deep Lantern",
            ),
            strict=True,
        )
    )
    cards: list[Image.Image] = []
    selected_seeds: dict[str, str] = {}
    used_seeds: set[str] = set()
    scale = 6
    crop_width, crop_height = 48, 72
    for label, evil in studies:
        best: tuple[float, str, GeneratedWorld, int, int, int, str] | None = None
        for seed in (fixed_seeds[label],):
            if seed in used_seeds:
                continue
            if label != "UNDERGROUND OCEAN" and seed in {"Ash Compass", "Frost Engine"}:
                continue
            base_world = _biome_candidate_world(seed, evil)
            if label == "UNDERGROUND OCEAN" and not base_world.metadata.get("ocean_cave_generated"):
                continue
            world, mask, center, surface_aligned, descriptor = _biome_subject(base_world, label)
            left, top, score, subject_count = _best_subject_crop(
                world,
                mask,
                center,
                crop_width,
                crop_height,
                surface_aligned=surface_aligned,
            )
            candidate = (score, seed, world, left, top, subject_count, descriptor)
            if best is None or candidate[0] > best[0]:
                best = candidate
        if best is None:
            raise RuntimeError(f"No unused biome seed available for {label}")
        score, seed, world, left, top, subject_count, descriptor = best
        del score
        used_seeds.add(seed)
        selected_seeds[label] = seed
        image = detail(world, left, top, crop_width, crop_height, scale)
        cards.append(
            add_title_bar(
                image,
                label,
                f"seed {seed} | {descriptor.format(count=subject_count)}",
            )
        )

    columns = 2
    gap = 8
    rows = math.ceil(len(cards) / columns)
    card_width, card_height = cards[0].size
    canvas = Image.new(
        "RGB",
        (columns * card_width + (columns - 1) * gap, rows * card_height + (rows - 1) * gap),
        BG,
    )
    for index, card in enumerate(cards):
        x = (index % columns) * (card_width + gap)
        y = (index // columns) * (card_height + gap)
        canvas.paste(card, (x, y))
    canvas.save(MEDIA / "biome_atlas.png", optimize=True)
    (MEDIA / "biome_seeds.json").write_text(
        json.dumps(selected_seeds, indent=2),
        encoding="utf-8",
    )


@lru_cache(maxsize=2)
def _biome_candidate_world(seed: str, evil: Evil) -> GeneratedWorld:
    return generate_world(WorldConfig(seed=seed, evil=evil, scale=WorldScale.SMALL))


def _biome_subject(
    world: GeneratedWorld,
    label: str,
) -> tuple[GeneratedWorld, np.ndarray, tuple[int, int], bool, str]:
    if label == "FOREST":
        x = int(world.metadata["spawn_x"])
        return (
            world,
            world.biomes == Biome.FOREST,
            (x, int(world.surface[x])),
            True,
            "{count} tiles",
        )
    if label == "SNOW":
        x = int(world.metadata["snow_x"])
        return world, world.biomes == Biome.SNOW, (x, int(world.surface[x])), True, "{count} tiles"
    if label == "DESERT":
        x = int(world.metadata["desert_x"])
        return (
            world,
            world.biomes == Biome.DESERT,
            (x, int(world.surface[x])),
            True,
            "{count} tiles",
        )
    if label == "JUNGLE":
        x = int(world.metadata["jungle_x"])
        return (
            world,
            world.biomes == Biome.JUNGLE,
            (x, int(world.surface[x])),
            True,
            "{count} tiles",
        )
    if label == "CORRUPTION":
        x = int(world.metadata["evil_x"])
        return (
            world,
            np.isin(world.tiles, (Tile.EBONSTONE, Tile.CORRUPT_GRASS)),
            (x, int(world.surface[x])),
            True,
            "{count} tiles",
        )
    if label == "CRIMSON":
        x = int(world.metadata["evil_x"])
        return (
            world,
            np.isin(world.tiles, (Tile.CRIMSTONE, Tile.CRIMSON_GRASS)),
            (x, int(world.surface[x])),
            True,
            "{count} tiles",
        )
    if label == "GLOWING MUSHROOM":
        mask = (world.walls == Wall.MUSHROOM) | (world.tiles == Tile.MUSHROOM_GRASS)
        return world, mask, _mask_anchor(mask), False, "{count} Mushroom tiles"
    if label == "METEORITE":
        meteor_world = _with_media_meteorite(world)
        mask = meteor_world.tiles == Tile.METEORITE
        return meteor_world, mask, _mask_anchor(mask), True, "{count} Meteorite tiles"
    if label == "UNDERGROUND OCEAN":
        marker = next(marker for marker in world.structures if marker.kind == "Underground ocean")
        mask = np.zeros(world.shape, dtype=bool)
        y0, y1 = marker.y, marker.y + marker.height
        x0, x1 = marker.x, marker.x + marker.width
        mask[y0:y1, x0:x1] = (world.liquid_kind[y0:y1, x0:x1] == Liquid.WATER) & (
            world.liquid_amount[y0:y1, x0:x1] > 0
        )
        return world, mask, _marker_anchor(world, "Underground ocean"), False, "{count} water tiles"
    if label == "SPIDER NEST":
        mask = world.walls == Wall.SPIDER
        return world, mask, _mask_anchor(mask), False, "{count} Spider Wall tiles"
    raise ValueError(f"Unknown biome study {label}")


def _best_subject_crop(
    world: GeneratedWorld,
    subject_mask: np.ndarray,
    center: tuple[int, int],
    crop_width: int,
    crop_height: int,
    *,
    surface_aligned: bool,
) -> tuple[int, int, float, int]:
    best_score = -1.0
    best_position = (0, 0, 0)
    center_x, center_y = center
    minimum_center = max(crop_width // 2, center_x - 18)
    maximum_center = min(world.shape[1] - crop_width // 2, center_x + 18)
    for candidate_center in range(minimum_center, maximum_center + 1):
        left = int(
            np.clip(
                candidate_center - crop_width // 2,
                0,
                world.shape[1] - crop_width,
            )
        )
        if surface_aligned:
            top_candidates = [int(world.surface[candidate_center]) - 8]
        else:
            base_top = center_y - crop_height // 2
            top_candidates = range(base_top - 8, base_top + 9, 4)
        for top_candidate in top_candidates:
            top = int(np.clip(top_candidate, 0, world.shape[0] - crop_height))
            local_subject = subject_mask[top : top + crop_height, left : left + crop_width]
            subject_count = int(np.count_nonzero(local_subject))
            local_tiles = world.tiles[top : top + crop_height, left : left + crop_width]
            visible = local_tiles != Tile.AIR
            visible |= world.walls[top : top + crop_height, left : left + crop_width] != Wall.NONE
            visible |= world.liquid_amount[top : top + crop_height, left : left + crop_width] > 0
            share = subject_count / max(1, np.count_nonzero(visible))
            score = (
                share
                + min(subject_count, crop_width * crop_height * 0.35) / (crop_width * crop_height)
                - abs(candidate_center - center_x) * 0.001
                - abs((top + crop_height // 2) - center_y) * 0.0005
            )
            if score > best_score:
                best_score = score
                best_position = (left, top, subject_count)
    return best_position[0], best_position[1], best_score, best_position[2]


def build_spread_animation() -> None:
    config = WorldConfig(seed="The World Breathes", evil=Evil.CORRUPTION)
    world = generate_world(config)
    frames: list[Image.Image] = []

    def capture(title: str, subtitle: str) -> None:
        frames.append(
            add_title_bar(
                render_world(world, scale=4, markers=True),
                title,
                subtitle,
            )
        )

    capture("Pre-Hardmode", "evil pockets established during world creation")
    rng = np.random.default_rng(config.seed_value ^ 0x5EED5EED)
    for cycle in range(1, 4):
        advance_biome_spread(world, rng, iterations=3)
        capture("Natural spread", f"growth cycle {cycle}")
    apply_hardmode(world, rng)
    capture("Hardmode V", "Hallow and evil break through the Caverns")
    for cycle in range(1, 5):
        advance_biome_spread(world, rng, iterations=4)
        capture("Hardmode spread", f"growth cycle {cycle}")

    frames[0].save(
        MEDIA / "biome_spread.gif",
        save_all=True,
        append_images=frames[1:],
        duration=[700] * (len(frames) - 1) + [1800],
        loop=0,
        optimize=True,
        disposal=2,
    )


def build_catastrophe_animation() -> None:
    world = generate_world(WorldConfig(seed="Catastrophe Laboratory", evil=Evil.CRIMSON))
    simulation_steps = 30
    result = simulate_catastrophe_chain(world, seed=20260802, steps=simulation_steps)
    crop_width, crop_height, scale = 104, 92, 4
    left = int(np.clip(result.impact_x - crop_width // 2, 0, world.shape[1] - crop_width))
    top = int(np.clip(result.impact_y - 8, 0, world.shape[0] - crop_height))
    frames: list[Image.Image] = []
    for index, state in enumerate(result.frames):
        image = render_world(
            state,
            scale=scale,
            markers=False,
        ).crop(
            (
                left * scale,
                top * scale,
                (left + crop_width) * scale,
                (top + crop_height) * scale,
            )
        )
        if index == 0:
            title = "NATURAL CAVERN CROSS-SECTION"
            subtitle = "generated Preview geology | four connected cave pools"
        elif index == 1:
            title = "METEOR IMPACT"
            subtitle = "protected-site selection, crater, and Meteorite rim"
        else:
            title = "CHAIN REACTION"
            captured_step = (index - 1) * max(1, simulation_steps // 6)
            products = sum(
                np.count_nonzero(state.tiles == tile)
                for tile in (
                    Tile.OBSIDIAN,
                    Tile.HONEY_BLOCK,
                    Tile.CRISPY_HONEY_BLOCK,
                    Tile.AETHERIUM,
                )
            )
            subtitle = f"liquid step {captured_step} | {products:,} contact tiles"
        frames.append(add_title_bar(image, title, subtitle))
    frames[0].save(
        MEDIA / "catastrophe_chain.gif",
        save_all=True,
        append_images=frames[1:],
        duration=[520] * (len(frames) - 1) + [1800],
        loop=0,
        optimize=True,
        disposal=2,
    )


def _card_grid(cards: list[Image.Image], columns: int = 2) -> Image.Image:
    gap = 8
    rows = math.ceil(len(cards) / columns)
    card_width, card_height = cards[0].size
    canvas = Image.new(
        "RGB",
        (card_width * columns + gap * (columns - 1), card_height * rows + gap * (rows - 1)),
        BG,
    )
    for index, card in enumerate(cards):
        canvas.paste(
            card,
            ((index % columns) * (card_width + gap), (index // columns) * (card_height + gap)),
        )
    return canvas


def _preview_crop_card(
    world: GeneratedWorld,
    center_x: int,
    center_y: int,
    title: str,
    subtitle: str,
    *,
    crop_width: int = 108,
    crop_height: int = 55,
    scale: int = 4,
) -> Image.Image:
    left = int(np.clip(center_x - crop_width // 2, 0, world.shape[1] - crop_width))
    top = int(np.clip(center_y - crop_height // 2, 0, world.shape[0] - crop_height))
    image = render_world(world, scale=scale, markers=False).crop(
        (
            left * scale,
            top * scale,
            (left + crop_width) * scale,
            (top + crop_height) * scale,
        )
    )
    return add_title_bar(image, title, f"Preview world | {subtitle}")


def build_surface_diagnostic() -> None:
    cards = []
    studies = (
        ("Caldera Crown", Evil.CORRUPTION, "Corruption | Snow east, Jungle west"),
        ("Amber Ruin", Evil.CRIMSON, "Crimson | Snow west, Jungle east"),
        ("Lava Psalm", Evil.CRIMSON, "Crimson | low central basin and east Snow"),
        ("Emerald Hunger", Evil.CORRUPTION, "Corruption | high west and east Jungle"),
    )
    for seed, evil, subtitle in studies:
        world = generate_world(WorldConfig(seed=seed, evil=evil))
        cards.append(
            add_title_bar(
                render_world(world, scale=2, markers=False),
                seed.upper(),
                subtitle,
            )
        )
    _card_grid(cards).save(MEDIA / "surface_profiles.png", optimize=True)


def build_cave_diagnostic() -> None:
    world = generate_world(WorldConfig(seed="Cave Landscape Study", evil=Evil.CORRUPTION))
    underground_y = (world.layers.world_surface + world.layers.rock_layer) // 2
    cavern_y = (world.layers.rock_layer + world.layers.underworld) // 2
    studies = (
        (
            int(world.metadata["spawn_x"]),
            int(world.surface[int(world.metadata["spawn_x"])]) + 15,
            "SURFACE CAVES",
            "straight and zig-zag openings",
        ),
        (
            int(world.metadata["snow_x"]),
            underground_y,
            "SNOW TRAPEZOID",
            "Snow above Ice with smaller tunnels",
        ),
        (
            int(world.metadata["desert_x"]),
            underground_y + 8,
            "UNDERGROUND DESERT",
            "oval hardened-sand ant-hive caves",
        ),
        (
            int(world.metadata["jungle_x"]),
            cavern_y,
            "JUNGLE CAVERNS",
            "Mud, Jungle Grass, and larger rock-layer voids",
        ),
    )
    cards = [
        _preview_crop_card(world, center_x, center_y, title, subtitle)
        for center_x, center_y, title, subtitle in studies
    ]
    _card_grid(cards).save(MEDIA / "cave_density.png", optimize=True)


def build_ore_diagnostic() -> None:
    world = generate_world(WorldConfig(seed="Ore Landscape Study"))
    scale = 8
    world_image = render_world(world, scale=scale, markers=False)
    cards: list[Image.Image] = []
    crop_width, crop_height = 48, 34
    for ore in world.metadata["selected_ore_ids"]:
        tile = Tile(int(ore))
        positions = _largest_connected_component(world.tiles == tile)
        if not len(positions):
            continue
        target_y = round(float(positions[:, 0].mean()))
        target_x = round(float(positions[:, 1].mean()))
        left = int(np.clip(target_x - crop_width // 2, 0, world.shape[1] - crop_width))
        top = int(np.clip(target_y - crop_height // 2, 0, world.shape[0] - crop_height))
        crop = world_image.crop(
            (
                left * scale,
                top * scale,
                (left + crop_width) * scale,
                (top + crop_height) * scale,
            )
        )
        draw = ImageDraw.Draw(crop)
        min_y, min_x = positions.min(axis=0)
        max_y, max_x = positions.max(axis=0)
        draw.rectangle(
            (
                (int(min_x) - left) * scale - 2,
                (int(min_y) - top) * scale - 2,
                (int(max_x) - left + 1) * scale + 1,
                (int(max_y) - top + 1) * scale + 1,
            ),
            outline=GOLD,
            width=2,
        )
        cards.append(
            add_title_bar(
                crop,
                f"{TILE_STYLES[tile].name.upper()} VEIN",
                f"largest cluster {len(positions)} tiles | depth {int(target_y):03d}",
            )
        )
    _card_grid(cards).save(MEDIA / "ore_depth.png", optimize=True)


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    visited = np.zeros(mask.shape, dtype=bool)
    largest: list[tuple[int, int]] = []
    height, width = mask.shape
    for start_y, start_x in np.argwhere(mask):
        y = int(start_y)
        x = int(start_x)
        if visited[y, x]:
            continue
        visited[y, x] = True
        component: list[tuple[int, int]] = []
        pending = [(y, x)]
        while pending:
            current_y, current_x = pending.pop()
            component.append((current_y, current_x))
            for delta_y in (-1, 0, 1):
                for delta_x in (-1, 0, 1):
                    if delta_y == 0 and delta_x == 0:
                        continue
                    next_y = current_y + delta_y
                    next_x = current_x + delta_x
                    if not (0 <= next_y < height and 0 <= next_x < width):
                        continue
                    if mask[next_y, next_x] and not visited[next_y, next_x]:
                        visited[next_y, next_x] = True
                        pending.append((next_y, next_x))
        if len(component) > len(largest):
            largest = component
    return np.asarray(largest, dtype=np.int16).reshape(-1, 2)


def build_active_diagnostics() -> None:
    build_surface_diagnostic()
    build_cave_diagnostic()
    build_ore_diagnostic()


def _depth_name(world: GeneratedWorld, tile_y: int) -> str:
    space_boundary = round(world.layers.world_surface * 0.67)
    if tile_y <= space_boundary:
        return "SPACE"
    if tile_y < world.layers.world_surface:
        return "SURFACE"
    if tile_y < world.layers.rock_layer:
        return "UNDERGROUND"
    if tile_y < world.layers.underworld:
        return "CAVERNS"
    return "UNDERWORLD"


def build_depth_descent() -> None:
    world = _with_media_meteorite(
        generate_world(
            WorldConfig(
                seed="The Long Way Down",
                scale=WorldScale.SMALL,
                evil=Evil.CRIMSON,
                hardmode=True,
            )
        )
    )
    scale = 960 / world.shape[1]
    vertical_exaggeration = 1.7
    base_image = overview(world)
    scale_y = base_image.height / world.shape[0]
    world_image = base_image.resize(
        (base_image.width, round(base_image.height * vertical_exaggeration)),
        Image.Resampling.NEAREST,
    )
    grid = ImageDraw.Draw(world_image, "RGBA")
    for tile_x in range(0, world.shape[1], 500):
        screen_x = tile_x * scale
        grid.line((screen_x, 0, screen_x, world_image.height), fill=(231, 237, 247, 34), width=1)
    for tile_y in range(0, world.shape[0], 100):
        screen_y = round(tile_y * scale_y * vertical_exaggeration)
        grid.line((0, screen_y, world_image.width, screen_y), fill=(231, 237, 247, 38), width=1)
    for tile_y, color in (
        (round(world.layers.world_surface * 0.67), (99, 211, 193, 220)),
        (world.layers.world_surface, (110, 215, 232, 230)),
        (world.layers.rock_layer, (228, 184, 92, 230)),
        (world.layers.underworld, (239, 98, 98, 230)),
    ):
        screen_y = round(tile_y * scale_y * vertical_exaggeration)
        grid.line((0, screen_y, world_image.width, screen_y), fill=color, width=2)

    viewport_width, viewport_height = world_image.width, 238
    maximum_y = world_image.height - viewport_height
    down = [round(maximum_y * (0.5 - 0.5 * math.cos(math.pi * step / 27))) for step in range(28)]
    positions = [down[0]] * 4 + down + [down[-1]] * 7 + list(reversed(down[1:-1]))
    frames: list[Image.Image] = []

    for top in positions:
        tile_y = round((top + viewport_height // 2) / (scale_y * vertical_exaggeration))
        layer = _depth_name(world, tile_y)
        crop = world_image.crop((0, top, viewport_width, top + viewport_height))
        frames.append(
            add_title_bar(
                crop,
                layer,
                f"Depth {tile_y:03d}",
            )
        )

    frames[0].save(
        MEDIA / "depth_descent.gif",
        save_all=True,
        append_images=frames[1:],
        duration=220,
        loop=0,
        optimize=True,
        disposal=2,
    )


def _spawn_heat_scores(world: GeneratedWorld) -> np.ndarray:
    """Normalized candidate-column mass; the plot is conditional on player eligibility."""
    return ground_candidate_mass(world).astype(np.float32) / 103.0


def _viridis(values: np.ndarray) -> np.ndarray:
    stops = np.array(
        (
            (0.00, 68, 1, 84),
            (0.25, 59, 82, 139),
            (0.50, 33, 145, 140),
            (0.75, 94, 201, 98),
            (1.00, 253, 231, 37),
        ),
        dtype=np.float32,
    )
    flat = np.clip(values, 0.0, 1.0).ravel()
    rgb = np.column_stack(
        tuple(np.interp(flat, stops[:, 0], stops[:, channel]) for channel in (1, 2, 3))
    )
    return rgb.reshape((*values.shape, 3)).astype(np.uint8)


def _max_pool(values: np.ndarray, factor: int) -> np.ndarray:
    height = values.shape[0] // factor * factor
    width = values.shape[1] // factor * factor
    return (
        values[:height, :width]
        .reshape(
            height // factor,
            factor,
            width // factor,
            factor,
        )
        .max(axis=(1, 3))
    )


def build_spawn_heatmap() -> None:
    world = _with_media_meteorite(_large_reference_world())
    scores = _spawn_heat_scores(world)
    factor = 3
    pooled = _max_pool(scores, factor)
    heat = Image.fromarray(_viridis(pooled), mode="RGB").resize(
        (1400, 400),
        Image.Resampling.NEAREST,
    )
    base = render_world(
        world,
        scale=1,
        biome_overlay=True,
        markers=False,
        material_texture=False,
    ).resize((1400, 400), Image.Resampling.NEAREST)
    alpha = Image.fromarray(np.uint8(np.clip(pooled * 0.90, 0.10, 0.82) * 255)).resize(
        (1400, 400),
        Image.Resampling.NEAREST,
    )
    blended = Image.composite(heat, base, alpha)
    canvas = Image.new("RGB", (1560, 610), BG)
    canvas.paste(blended, (80, 94))
    draw = ImageDraw.Draw(canvas)
    title = f"SPAWN HEAT MAP | {LARGE_WORLD_SEED.upper()}"
    title_font = font(24, bold=True)
    draw.text(
        ((canvas.width - draw.textlength(title, font=title_font)) / 2, 8),
        title,
        fill=TEXT,
        font=title_font,
    )
    subtitle = (
        "eligible ground-candidate column mass | source-backed search geometry"
        " | no biome/light weights"
    )
    subtitle_font = font(13)
    draw.text(
        ((canvas.width - draw.textlength(subtitle, font=subtitle_font)) / 2, 43),
        subtitle,
        fill=MUTED,
        font=subtitle_font,
    )

    legend_y = 70
    gradient = np.linspace(0, 1, 260, dtype=np.float32)[None, :]
    gradient_image = Image.fromarray(_viridis(gradient), mode="RGB").resize((260, 12))
    canvas.paste(gradient_image, (650, legend_y))
    draw.text((603, legend_y - 2), "LOW", fill=MUTED, font=font(11, bold=True))
    draw.text((921, legend_y - 2), "HIGH", fill=MUTED, font=font(11, bold=True))

    lines = (
        "GROUND CANDIDATE MASS | 2 x 3 clearance | downward search up to 103 tiles",
        "Hypothetical off-screen player; eligible floor in range; no town, buffs or events",
        "Conditional geometry only: no NPC rate, population cap, enemy selection or sky spawns",
    )
    for index, line in enumerate(lines):
        draw.text(
            ((canvas.width - draw.textlength(line, font=font(12))) / 2, 516 + index * 23),
            line,
            fill=TEXT,
            font=font(12),
        )
    canvas.save(MEDIA / "spawn_heatmap.png", optimize=True)


def benchmark(scale: WorldScale, iterations: int) -> list[float]:
    timings = []
    for index in range(iterations):
        started = time.perf_counter()
        generate_world(WorldConfig(seed=f"media-benchmark-{scale.value}-{index}", scale=scale))
        timings.append(time.perf_counter() - started)
    return timings


def build_performance_chart() -> None:
    preview = benchmark(WorldScale.PREVIEW, 7)
    small = benchmark(WorldScale.SMALL, 3)
    medians = (statistics.median(preview), statistics.median(small))
    preview_text = f"{medians[0] * 1000:.0f} ms"
    small_text = f"{medians[1]:.2f} s"
    world = generate_world(WorldConfig(seed="Generation Benchmark"))
    canvas = add_title_bar(
        render_world(world, scale=4, markers=False),
        "GENERATION BENCHMARK",
        (
            f"Preview {preview_text} | Small {small_text} | "
            f"{platform.system()} Python {platform.python_version()}"
        ),
    )
    canvas.save(MEDIA / "performance.png", optimize=True)

    payload = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "preview_seconds": preview,
        "small_seconds": small,
        "preview_median_seconds": medians[0],
        "small_median_seconds": medians[1],
    }
    (MEDIA / "benchmarks.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_fidelity_chart() -> None:
    counts = {
        fidelity: sum(spec.fidelity is fidelity for spec in PASS_SPECS) for fidelity in Fidelity
    }
    world = generate_world(WorldConfig(seed="Pass Fidelity"))
    count_text = " | ".join(f"{fidelity.value.title()} {counts[fidelity]}" for fidelity in Fidelity)
    canvas = add_title_bar(
        render_world(world, scale=4, markers=False),
        "107-PASS FIDELITY",
        f"generated Preview world | {count_text}",
    )
    canvas.save(MEDIA / "fidelity.png", optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=MEDIA)
    parser.add_argument("--branding", action="store_true")
    parser.add_argument(
        "--only", nargs="+", choices=("world", "idle", "biomes", "containment", "depth", "spawn")
    )
    args = parser.parse_args()
    globals()["MEDIA"] = args.output.resolve()
    MEDIA.mkdir(parents=True, exist_ok=True)
    if args.branding:
        build_icon()
    builders = {
        "world": build_world_media,
        "idle": build_idle_world_evolution,
        "biomes": build_biome_study,
        "containment": build_hazard_containment_animation,
        "depth": build_depth_descent,
        "spawn": build_spawn_heatmap,
    }
    for name, builder in builders.items():
        if args.only is None or name in args.only:
            print(f"Building {name}", flush=True)
            builder()
    print(f"Wrote TerraExplorer media to {MEDIA}", flush=True)


if __name__ == "__main__":
    main()
