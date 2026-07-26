"""Regenerate repository visuals and Windows icon from the tested engine."""

from __future__ import annotations

import json
import math
import platform
import statistics
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from terraforge.config import Evil, WorldConfig, WorldScale
from terraforge.model import GeneratedWorld
from terraforge.passes import PASS_SPECS, Fidelity
from terraforge.pipeline import generate_world
from terraforge.render import add_title_bar, render_world, save_generation_gif, save_png
from terraforge.tiles import Biome, Tile

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "docs" / "media"
ASSETS = ROOT / "terraforge" / "assets"
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
    logo = Image.open(ASSETS / "terraforge_logo.png").convert("RGBA")
    logo.thumbnail((512, 512), Image.Resampling.LANCZOS)
    logo.save(ASSETS / "terraforge_logo.png", optimize=True)
    logo.save(MEDIA / "terraforge_logo.png", optimize=True)
    logo.save(
        ASSETS / "terraforge.ico",
        sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )
    logo.resize((256, 256), Image.Resampling.LANCZOS).save(
        MEDIA / "terraforge_icon.png", optimize=True
    )


def build_world_media() -> None:
    config = WorldConfig(seed="TerraForge", evil=Evil.CORRUPTION, hardmode=True)
    world = generate_world(config)
    save_png(world, MEDIA / "terraforge_world.png", scale=4, markers=True)
    save_generation_gif(config, MEDIA / "terraforge_generation.gif", scale=4)

    overview = render_world(
        world,
        scale=4,
        biome_overlay=True,
        layer_lines=True,
        markers=True,
    )
    add_title_bar(
        overview,
        "Biome, layer, and structure overview",
        "Biome tint + layer guides + original geometric map symbols",
    ).save(MEDIA / "biome_overview.png", optimize=True)


def build_seed_comparison() -> None:
    configs = (
        WorldConfig(seed="Clockwork Dawn", evil=Evil.CORRUPTION),
        WorldConfig(seed="Copper Canopy", evil=Evil.CRIMSON),
        WorldConfig(seed="Hallowed Circuit", evil=Evil.CORRUPTION, hardmode=True),
    )
    labels = ("CORRUPTION", "CRIMSON", "HARDMODE V")
    cards: list[Image.Image] = []
    for config, label in zip(configs, labels, strict=True):
        world = generate_world(config)
        image = render_world(world, scale=2, markers=True)
        cards.append(add_title_bar(image, label, f"seed: {config.seed}"))

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


def _biome_variant(base: GeneratedWorld, biome: Biome, mapping: dict[Tile, Tile]) -> GeneratedWorld:
    variant = GeneratedWorld(
        config=base.config,
        tiles=base.tiles.copy(),
        walls=base.walls.copy(),
        liquid_amount=base.liquid_amount.copy(),
        liquid_kind=base.liquid_kind.copy(),
        biomes=base.biomes.copy(),
        surface=base.surface.copy(),
        layers=base.layers,
        metadata=base.metadata.copy(),
        structures=[],
        pass_results=base.pass_results.copy(),
    )
    original = variant.tiles.copy()
    for source, target in mapping.items():
        variant.tiles[original == source] = target
    variant.biomes[variant.tiles != Tile.AIR] = biome
    return variant


def build_biome_study() -> None:
    base = generate_world(WorldConfig(seed="One Patch of Earth"))
    studies = (
        (
            "FOREST",
            Biome.FOREST,
            {Tile.DIRT: Tile.DIRT, Tile.STONE: Tile.STONE, Tile.GRASS: Tile.GRASS},
        ),
        ("SNOW", Biome.SNOW, {Tile.DIRT: Tile.SNOW, Tile.STONE: Tile.ICE, Tile.GRASS: Tile.SNOW}),
        (
            "DESERT",
            Biome.DESERT,
            {Tile.DIRT: Tile.SAND, Tile.STONE: Tile.SANDSTONE, Tile.GRASS: Tile.SAND},
        ),
        (
            "JUNGLE",
            Biome.JUNGLE,
            {Tile.DIRT: Tile.MUD, Tile.STONE: Tile.MUD, Tile.GRASS: Tile.JUNGLE_GRASS},
        ),
        (
            "CORRUPTION",
            Biome.CORRUPTION,
            {
                Tile.DIRT: Tile.EBONSTONE,
                Tile.STONE: Tile.EBONSTONE,
                Tile.GRASS: Tile.CORRUPT_GRASS,
                Tile.SAND: Tile.EBONSTONE,
            },
        ),
        (
            "CRIMSON",
            Biome.CRIMSON,
            {
                Tile.DIRT: Tile.CRIMSTONE,
                Tile.STONE: Tile.CRIMSTONE,
                Tile.GRASS: Tile.CRIMSON_GRASS,
                Tile.SAND: Tile.CRIMSTONE,
            },
        ),
    )
    cards: list[Image.Image] = []
    scale = 3
    crop = (42 * scale, 10 * scale, 198 * scale, 118 * scale)
    for label, biome, mapping in studies:
        world = _biome_variant(base, biome, mapping)
        image = render_world(world, scale=scale, markers=False).crop(crop)
        cards.append(add_title_bar(image, label, "same seed | same terrain | different material"))

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
    canvas.save(MEDIA / "biome_variants.png", optimize=True)


def _depth_name(world: GeneratedWorld, tile_y: int) -> str:
    if tile_y < world.layers.world_surface:
        return "SKY AND SURFACE"
    if tile_y < world.layers.rock_layer:
        return "UNDERGROUND"
    if tile_y < world.layers.underworld:
        return "CAVERNS"
    return "UNDERWORLD"


def build_depth_descent() -> None:
    world = generate_world(WorldConfig(seed="The Long Way Down", evil=Evil.CRIMSON))
    scale = 4
    world_image = render_world(world, scale=scale, markers=True)
    viewport_width, viewport_height = 850, 290
    left = (world_image.width - viewport_width) // 2
    maximum_y = world_image.height - viewport_height
    down = [round(maximum_y * (0.5 - 0.5 * math.cos(math.pi * step / 17))) for step in range(18)]
    positions = [down[0]] * 3 + down + [down[-1]] * 4 + list(reversed(down[1:-1]))
    frames: list[Image.Image] = []

    for top in positions:
        tile_y = (top + viewport_height // 2) // scale
        layer = _depth_name(world, tile_y)
        frame = Image.new("RGB", (960, 390), BG)
        crop = world_image.crop((left, top, left + viewport_width, top + viewport_height))
        frame.paste(crop, (24, 74))
        draw = ImageDraw.Draw(frame)
        draw.rectangle((22, 72, 876, 366), outline=GOLD, width=2)
        draw.text((24, 18), "DESCENT OF A NEW WORLD", fill=TEXT, font=font(23, True))
        draw.text((24, 47), f"{layer} | depth {tile_y:03d}", fill=ACCENT, font=font(15, True))
        gauge_x = 920
        gauge_top, gauge_bottom = 82, 356
        draw.line((gauge_x, gauge_top, gauge_x, gauge_bottom), fill=MUTED, width=3)
        depths = (
            (0, "0"),
            (world.layers.world_surface, "SURFACE"),
            (world.layers.rock_layer, "ROCK"),
            (world.layers.underworld, "HELL"),
            (world.shape[0] - 1, "BOTTOM"),
        )
        for depth, label in depths:
            y = gauge_top + round((gauge_bottom - gauge_top) * depth / (world.shape[0] - 1))
            draw.line((gauge_x - 8, y, gauge_x + 7, y), fill=GOLD, width=2)
            if label not in {"0", "BOTTOM"}:
                draw.text((882, y - 7), label, fill=MUTED, font=font(9, True))
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
        duration=160,
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
        "Every public pass is labeled; TerraForge does not claim source parity.",
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
    build_depth_descent()
    build_performance_chart()
    build_fidelity_chart()
    print(f"Wrote TerraForge media to {MEDIA}")


if __name__ == "__main__":
    main()
