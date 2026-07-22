"""Regenerate repository visuals and Windows icon from the tested engine."""

from __future__ import annotations

import json
import platform
import statistics
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from terraforge.config import Evil, WorldConfig, WorldScale
from terraforge.passes import PASS_SPECS, Fidelity
from terraforge.pipeline import generate_world
from terraforge.render import add_title_bar, render_world, save_generation_gif, save_png

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
    logo = Image.open(ASSETS / "terraforge_logo.png").convert("RGB")
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
    build_performance_chart()
    build_fidelity_chart()
    print(f"Wrote TerraForge media to {MEDIA}")


if __name__ == "__main__":
    main()
