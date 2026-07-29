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

    deep_world = generate_world(
        WorldConfig(seed="TerraExplorer Atlas", scale=WorldScale.SMALL, evil=Evil.CORRUPTION)
    )
    overview = render_world(
        deep_world,
        scale=1,
        biome_overlay=True,
        layer_lines=True,
        markers=True,
    )
    overview = overview.resize((1400, 400), Image.Resampling.NEAREST)
    add_title_bar(
        overview,
        "One world, from Space to the Underworld",
        "Biome tint, layer guides, and original landmark symbols",
    ).save(MEDIA / "biome_overview.png", optimize=True)

    image = render_world(deep_world, scale=1, markers=True)
    studies = (
        (
            "Floating island",
            "SKY LAKE & HOUSE",
            "sky brick, reservoir, and waterfall",
            220,
            110,
            28,
        ),
        (
            "Dungeon",
            "DUNGEON KEEP",
            "surface tower and descending rooms",
            170,
            95,
            0,
        ),
        (
            "Pyramid",
            "BURIED PYRAMID",
            "shaft, chamber, and desert shell",
            260,
            130,
            0,
        ),
        (
            "Aether",
            "AETHER",
            "gem-ringed cavern and Shimmer pool",
            260,
            130,
            0,
        ),
        (
            "Jungle temple",
            "JUNGLE TEMPLE",
            "sealed brick maze and trap corridors",
            320,
            160,
            0,
        ),
        (
            "Underworld city",
            "UNDERWORLD CITY",
            "obsidian ruins, bridges, and Hellforge",
            190,
            105,
            0,
        ),
    )
    cards = []
    for kind, label, subtitle, crop_width, crop_height, offset_y in studies:
        matching = [item for item in deep_world.structures if item.kind == kind]
        marker = (
            max(matching, key=lambda item: item.width * item.height)
            if kind == "Underworld city"
            else matching[0]
        )
        center_x = marker.x + marker.width // 2
        center_y = marker.y + marker.height // 2 + offset_y
        if kind == "Dungeon":
            center_x = int(deep_world.metadata["dungeon_x"])
            center_y = int(deep_world.surface[center_x])
        left = int(np.clip(center_x - crop_width // 2, 0, image.width - crop_width))
        top = int(np.clip(center_y - crop_height // 2, 0, image.height - crop_height))
        crop = image.crop((left, top, left + crop_width, top + crop_height))
        panel = crop.resize((440, 220), Image.Resampling.NEAREST)
        cards.append(add_title_bar(panel, label, subtitle))

    gap = 8
    card_width, card_height = cards[0].size
    atlas = Image.new(
        "RGB",
        (card_width * 3 + gap * 2, card_height * 2 + gap),
        BG,
    )
    for index, card in enumerate(cards):
        atlas.paste(
            card,
            (
                (index % 3) * (card_width + gap),
                (index // 3) * (card_height + gap),
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
            scale=WorldScale.SMALL,
            evil=Evil.CRIMSON,
        )
    )
    scale = 1
    world_image = render_world(world, scale=scale, markers=True)
    viewport_width, viewport_height = 1080, 540
    focus_x = round((int(world.metadata["snow_x"]) + int(world.metadata["dungeon_x"])) / 2)
    left = int(np.clip(focus_x - viewport_width // 2, 0, world_image.width - viewport_width))
    maximum_y = world_image.height - viewport_height
    down = [round(maximum_y * (0.5 - 0.5 * math.cos(math.pi * step / 21))) for step in range(22)]
    positions = [down[0]] * 3 + down + [down[-1]] * 5 + list(reversed(down[1:-1]))
    frames: list[Image.Image] = []

    for top in positions:
        tile_y = (top + viewport_height // 2) // scale
        layer = _depth_name(world, tile_y)
        frame = Image.new("RGB", (1280, 590), BG)
        crop = world_image.crop((left, top, left + viewport_width, top + viewport_height))
        frame.paste(crop, (24, 24))
        draw = ImageDraw.Draw(frame)
        draw.rectangle((22, 22, 1106, 568), outline=GOLD, width=2)
        label = f"{layer}  |  depth {tile_y:04d}"
        label_width = round(draw.textlength(label, font=font(15, True)))
        draw.rounded_rectangle((38, 36, 58 + label_width, 64), radius=7, fill=BG, outline=GOLD)
        draw.text((48, 41), label, fill=ACCENT, font=font(15, True))
        gauge_x = 1140
        gauge_top, gauge_bottom = 34, 556
        draw.line((gauge_x, gauge_top, gauge_x, gauge_bottom), fill=MUTED, width=3)
        depths = (
            (0, "SPACE"),
            (world.layers.world_surface, "SURFACE"),
            (
                (world.layers.world_surface + world.layers.rock_layer) // 2,
                "UNDERGROUND",
            ),
            (world.layers.rock_layer, "ROCK"),
            (
                (world.layers.rock_layer + world.layers.underworld) // 2,
                "DEEP CAVERNS",
            ),
            (world.layers.underworld, "UNDERWORLD"),
            (world.shape[0] - 1, "BOTTOM"),
        )
        for depth, label in depths:
            y = gauge_top + round((gauge_bottom - gauge_top) * depth / (world.shape[0] - 1))
            draw.line((gauge_x - 8, y, gauge_x + 7, y), fill=GOLD, width=2)
            draw.text((1154, y - 7), label, fill=MUTED, font=font(10, True))
        for depth in range(100, world.shape[0] - 1, 100):
            y = gauge_top + round((gauge_bottom - gauge_top) * depth / (world.shape[0] - 1))
            draw.line((gauge_x - 4, y, gauge_x + 4, y), fill=MUTED, width=1)
        pointer_y = gauge_top + round((gauge_bottom - gauge_top) * tile_y / (world.shape[0] - 1))
        draw.polygon(
            ((gauge_x - 15, pointer_y), (gauge_x - 4, pointer_y - 7), (gauge_x - 4, pointer_y + 7)),
            fill=ACCENT,
        )
        frames.append(frame)

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
    labels = ("Preview 240 x 140", "Small 4200 x 1200")
    canvas = Image.new("RGB", (1000, 420), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((48, 32), "Generation benchmark", fill=TEXT, font=font(28, True))
    draw.text(
        (48, 72),
        "Median wall time; deterministic 107-pass pipeline; lower is better",
        fill=MUTED,
        font=font(16),
    )
    maximum = max(medians)
    for index, (label, value) in enumerate(zip(labels, medians, strict=True)):
        y = 145 + index * 115
        draw.text((48, y), label, fill=TEXT, font=font(18, True))
        draw.rounded_rectangle((285, y, 920, y + 42), radius=8, fill=PANEL)
        width = max(8, round(635 * value / maximum))
        draw.rounded_rectangle((285, y, 285 + width, y + 42), radius=8, fill=ACCENT)
        value_text = f"{value * 1000:.0f} ms" if value < 1 else f"{value:.2f} s"
        value_x = 300 if width > 130 else 285 + width + 12
        value_color = BG if width > 130 else TEXT
        draw.text((value_x, y + 9), value_text, fill=value_color, font=font(17, True))
    draw.text(
        (48, 375),
        f"Measured on {platform.system()} | Python {platform.python_version()}",
        fill=MUTED,
        font=font(14),
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
    canvas = Image.new("RGB", (1000, 310), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((48, 30), "Pass fidelity is explicit", fill=TEXT, font=font(28, True))
    draw.text(
        (48, 70),
        "Every public pass is labeled; TerraExplorer does not claim source parity.",
        fill=MUTED,
        font=font(16),
    )
    colors = {
        Fidelity.MODELED: ACCENT,
        Fidelity.APPROXIMATED: GOLD,
        Fidelity.DOCUMENTED: MUTED,
    }
    x0, y0, total_width, height = 48, 125, 904, 62
    cursor = x0
    for fidelity in Fidelity:
        width = round(total_width * counts[fidelity] / len(PASS_SPECS))
        draw.rectangle((cursor, y0, cursor + width, y0 + height), fill=colors[fidelity])
        cursor += width
    x = 48
    for fidelity in Fidelity:
        draw.rectangle((x, 225, x + 20, 245), fill=colors[fidelity])
        draw.text(
            (x + 30, 222),
            f"{fidelity.value.title()}  {counts[fidelity]}",
            fill=TEXT,
            font=font(17, True),
        )
        x += 280
    canvas.save(MEDIA / "fidelity.png", optimize=True)


def main() -> None:
    MEDIA.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    build_icon()
    build_world_media()
    build_seed_comparison()
    build_biome_study()
    build_spread_animation()
    build_depth_descent()
    build_performance_chart()
    build_fidelity_chart()
    print(f"Wrote TerraExplorer media to {MEDIA}")


if __name__ == "__main__":
    main()
