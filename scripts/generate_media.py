"""Regenerate repository visuals and Windows icon from the tested engine."""

from __future__ import annotations

import json
import math
import platform
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from terraexplorer.config import Evil, WorldConfig, WorldScale
from terraexplorer.generation import advance_biome_spread, apply_hardmode
from terraexplorer.model import GeneratedWorld
from terraexplorer.passes import PASS_SPECS, Fidelity
from terraexplorer.pipeline import generate_world
from terraexplorer.render import add_title_bar, render_world, save_generation_gif
from terraexplorer.simulations import (
    ContainmentStrategy,
    simulate_biome_containment,
    simulate_catastrophe_chain,
)
from terraexplorer.tiles import TILE_STYLES, Biome, Tile

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "docs" / "media"
ASSETS = ROOT / "terraexplorer" / "assets"
BG = "#0b1220"
PANEL = "#131d2e"
TEXT = "#e7edf7"
MUTED = "#9fb0c9"
ACCENT = "#63d3c1"
GOLD = "#e4b85c"


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
    alpha_bounds = logo.getchannel("A").getbbox()
    if alpha_bounds is None:
        readme_logo = logo
    else:
        padding = 8
        readme_logo = logo.crop(
            (
                max(0, alpha_bounds[0] - padding),
                max(0, alpha_bounds[1] - padding),
                min(logo.width, alpha_bounds[2] + padding),
                min(logo.height, alpha_bounds[3] + padding),
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


def build_world_media() -> None:
    config = WorldConfig(seed="TerraExplorer", evil=Evil.CORRUPTION, hardmode=True)
    save_generation_gif(config, MEDIA / "terraexplorer_generation.gif", scale=4)

    small_world = generate_world(
        WorldConfig(seed="TerraExplorer Atlas", scale=WorldScale.SMALL, evil=Evil.CORRUPTION)
    )
    overview = render_world(
        small_world,
        scale=1,
        biome_overlay=True,
        layer_lines=True,
        markers=True,
        material_texture=False,
    )
    overview = overview.resize((1400, 400), Image.Resampling.NEAREST)
    add_title_bar(
        overview,
        "One world, from Space to the Underworld",
        "Biome tint, layer guides, and original landmark symbols",
    ).save(MEDIA / "biome_overview.png", optimize=True)

    studies = (
        (
            "Floating island",
            "FLOATING ISLAND",
            "Cloud base, forest cap, and sky house",
            "Landmark Study 02",
        ),
        (
            "Dungeon",
            "DUNGEON",
            "Weathered entrance and branching chambers",
            "Landmark Study 08",
        ),
        (
            "Pyramid",
            "PYRAMID",
            "Buried tip, zigzag passage, and treasure room",
            "Landmark Study 07",
        ),
        (
            "Aether",
            "AETHER",
            "Outer-fifth cavern, Gem Trees, and Shimmer",
            "Landmark Study 27",
        ),
        (
            "Jungle temple",
            "JUNGLE TEMPLE",
            "Irregular brick rooms, passages, traps, and altar",
            "Landmark Study 27",
        ),
        (
            "Ruined house",
            "RUINED HOUSE",
            "Multi-floor Underworld tower and Hellforge",
            "Landmark Study 24",
        ),
    )
    cards: list[Image.Image] = []
    scale = 2
    for kind, label, subtitle, seed in studies:
        landmark_world = generate_world(WorldConfig(seed=seed, evil=Evil.CORRUPTION))
        image = render_world(
            landmark_world,
            scale=scale,
            markers=False,
        )
        matching = [item for item in landmark_world.structures if item.kind == kind]
        if kind == "Floating island":
            marker = matching[0]
        else:
            marker = max(matching, key=lambda item: item.width * item.height)
        draw = ImageDraw.Draw(image)
        bounds = (
            marker.x * scale,
            marker.y * scale,
            (marker.x + marker.width) * scale - 1,
            (marker.y + marker.height) * scale - 1,
        )
        draw.rectangle(bounds, outline=GOLD, width=2)
        cards.append(add_title_bar(image, label, f"Preview world | {subtitle}"))

    gap = 8
    card_width, card_height = cards[0].size
    atlas = Image.new(
        "RGB",
        (card_width * 2 + gap, card_height * 3 + gap * 2),
        BG,
    )
    for index, card in enumerate(cards):
        atlas.paste(
            card,
            (
                (index % 2) * (card_width + gap),
                (index // 2) * (card_height + gap),
            ),
        )
    atlas.save(MEDIA / "terraexplorer_world.png", optimize=True)


def build_seed_comparison() -> None:
    configs = (
        WorldConfig(seed="One Seed, Three Futures", evil=Evil.CORRUPTION),
        WorldConfig(seed="One Seed, Three Futures", evil=Evil.CRIMSON),
        WorldConfig(
            seed="One Seed, Three Futures",
            evil=Evil.CORRUPTION,
            hardmode=True,
        ),
    )
    labels = ("CORRUPTION", "CRIMSON", "HARDMODE V")
    subtitles = (
        "same seed | violet chasms",
        "same seed | crimson chambers",
        "same seed | evil and Hallow",
    )
    cards: list[Image.Image] = []
    for config, label, subtitle in zip(configs, labels, subtitles, strict=True):
        world = generate_world(config)
        image = render_world(world, scale=2, markers=True)
        cards.append(add_title_bar(image, label, subtitle))

    gap = 8
    canvas = Image.new(
        "RGB",
        (sum(card.width for card in cards) + gap * (len(cards) - 1), max(c.height for c in cards)),
        BG,
    )
    x = 0
    for card in cards:
        canvas.paste(card, (x, 0))
        x += card.width + gap
    canvas.save(MEDIA / "seed_comparison.png", optimize=True)


def build_biome_study() -> None:
    studies = (
        ("FOREST", "Mapmaker's Home", Evil.CORRUPTION, "spawn_x", "trees and open caves"),
        ("SNOW", "Frost Lens", Evil.CORRUPTION, "snow_x", "ice shelves and frozen chambers"),
        ("DESERT", "Buried Gold", Evil.CRIMSON, "desert_x", "sandstone and hardened sand"),
        ("JUNGLE", "Green Depths", Evil.CRIMSON, "jungle_x", "mud, vines, and dense caves"),
        ("CORRUPTION", "Violet Scar", Evil.CORRUPTION, "evil_x", "branching ebonstone chasms"),
        ("CRIMSON", "Red Descent", Evil.CRIMSON, "evil_x", "linked crimstone chambers"),
    )
    cards: list[Image.Image] = []
    scale = 4
    crop_width, crop_height = 108, 78
    for label, seed, evil, center_key, subtitle in studies:
        world = generate_world(WorldConfig(seed=seed, evil=evil))
        center = int(world.metadata[center_key])
        surface_y = int(world.surface[center])
        left = int(np.clip(center - crop_width // 2, 0, world.shape[1] - crop_width))
        top = int(np.clip(surface_y - 12, 0, world.shape[0] - crop_height))
        image = render_world(world, scale=scale, markers=False).crop(
            (
                left * scale,
                top * scale,
                (left + crop_width) * scale,
                (top + crop_height) * scale,
            )
        )
        cards.append(add_title_bar(image, label, subtitle))

    columns = 3
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


def build_containment_animation() -> None:
    strategies = tuple(ContainmentStrategy)
    results = {strategy: simulate_biome_containment(strategy, seed=42) for strategy in strategies}
    titles = {
        ContainmentStrategy.OPEN: "OPEN GROUND",
        ContainmentStrategy.TRENCH: "THREE-TILE TRENCH",
        ContainmentStrategy.SUNFLOWERS: "SUNFLOWERS",
        ContainmentStrategy.CHLOROPHYTE: "CHLOROPHYTE CLUSTER",
    }
    frames: list[Image.Image] = []
    gap = 8
    for frame_index in range(len(results[strategies[0]].frames)):
        cards = []
        for strategy in strategies:
            result = results[strategy]
            state = result.frames[frame_index]
            infected = result.infected_counts[frame_index]
            if result.spread_direction > 0:
                safe_region = state.biomes[:, result.barrier_x + 3 :]
            else:
                safe_region = state.biomes[:, : result.barrier_x - 2]
            crossed = int(np.count_nonzero(safe_region == Biome.CORRUPTION))
            image = render_world(state, scale=2, markers=False)
            draw = ImageDraw.Draw(image)
            barrier = result.barrier_x * 2
            draw.line((barrier, 0, barrier, image.height - 1), fill=GOLD, width=2)
            card = add_title_bar(
                image,
                titles[strategy],
                f"generated Preview world | infected {infected:,} | safe side {crossed:,}",
            )
            cards.append(card)
        card_width, card_height = cards[0].size
        canvas = Image.new(
            "RGB",
            (card_width * 2 + gap, card_height * 2 + gap),
            BG,
        )
        for index, card in enumerate(cards):
            canvas.paste(
                card,
                ((index % 2) * (card_width + gap), (index // 2) * (card_height + gap)),
            )
        frames.append(canvas)
    frames[0].save(
        MEDIA / "containment_lab.gif",
        save_all=True,
        append_images=frames[1:],
        duration=[620] * (len(frames) - 1) + [1800],
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
        ("Surface Landscape 1", Evil.CORRUPTION, "rolling hills, coasts, and surface mouths"),
        ("Surface Landscape 2", Evil.CORRUPTION, "Snow opposite the Jungle and main Desert"),
        ("Surface Landscape 3", Evil.CORRUPTION, "dunes above an oval Underground Desert"),
        ("Surface Landscape 4", Evil.CRIMSON, "slanted Crimson entries and linked chambers"),
    )
    for index, (seed, evil, subtitle) in enumerate(studies, start=1):
        world = generate_world(WorldConfig(seed=seed, evil=evil))
        cards.append(
            add_title_bar(
                render_world(world, scale=2, markers=False),
                f"LANDSCAPE STUDY {index}",
                f"generated Preview world | {subtitle}",
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
    world_image = render_world(world, scale=4, markers=False)
    cards: list[Image.Image] = []
    crop_width, crop_height, scale = 108, 55, 4
    for ore in world.metadata["selected_ore_ids"]:
        tile = Tile(int(ore))
        positions = np.argwhere(world.tiles == tile)
        if not len(positions):
            continue
        target_y, target_x = positions[len(positions) // 2]
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
        marker_x = (int(target_x) - left) * scale + scale // 2
        marker_y = (int(target_y) - top) * scale + scale // 2
        draw.ellipse(
            (marker_x - 8, marker_y - 8, marker_x + 8, marker_y + 8),
            outline=GOLD,
            width=2,
        )
        cards.append(
            add_title_bar(
                crop,
                f"{TILE_STYLES[tile].name.upper()} VEIN",
                f"actual Preview tiles | depth {int(target_y):03d}",
            )
        )
    _card_grid(cards).save(MEDIA / "ore_depth.png", optimize=True)


def build_active_diagnostics() -> None:
    build_surface_diagnostic()
    build_cave_diagnostic()
    build_ore_diagnostic()


def _depth_name(world: GeneratedWorld, tile_y: int) -> str:
    if tile_y < world.layers.world_surface:
        return "SKY AND SURFACE"
    if tile_y < world.layers.rock_layer:
        return "UNDERGROUND"
    if tile_y < world.layers.underworld:
        return "CAVERNS"
    return "UNDERWORLD"


def build_depth_descent() -> None:
    world = generate_world(
        WorldConfig(
            seed="The Long Way Down",
            evil=Evil.CRIMSON,
        )
    )
    scale = 4
    world_image = render_world(world, scale=scale, markers=False)
    viewport_width, viewport_height = world_image.width, 176
    maximum_y = world_image.height - viewport_height
    down = [round(maximum_y * (0.5 - 0.5 * math.cos(math.pi * step / 21))) for step in range(22)]
    positions = [down[0]] * 3 + down + [down[-1]] * 5 + list(reversed(down[1:-1]))
    frames: list[Image.Image] = []

    for top in positions:
        tile_y = (top + viewport_height // 2) // scale
        layer = _depth_name(world, tile_y)
        crop = world_image.crop((0, top, viewport_width, top + viewport_height))
        frames.append(
            add_title_bar(
                crop,
                layer,
                f"generated Preview world | depth {tile_y:03d} of {world.shape[0] - 1}",
            )
        )

    frames[0].save(
        MEDIA / "depth_descent.gif",
        save_all=True,
        append_images=frames[1:],
        duration=150,
        loop=0,
        optimize=True,
        disposal=2,
    )


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
    MEDIA.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    build_icon()
    build_world_media()
    build_seed_comparison()
    build_biome_study()
    build_spread_animation()
    build_containment_animation()
    build_catastrophe_animation()
    build_depth_descent()
    build_active_diagnostics()
    build_performance_chart()
    build_fidelity_chart()
    print(f"Wrote TerraExplorer media to {MEDIA}")


if __name__ == "__main__":
    main()
