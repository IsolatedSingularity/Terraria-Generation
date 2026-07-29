"""Original pixel renderer and media exporters for TerraExplorer worlds."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from terraexplorer.config import WorldConfig, WorldScale
from terraexplorer.model import GeneratedWorld, StructureMarker
from terraexplorer.pipeline import TerraExplorerPipeline
from terraexplorer.tiles import TILE_STYLES, WALL_COLORS, Biome, Liquid, Tile, Wall

_LIQUID_COLORS = {
    Liquid.WATER: "#2e7dba",
    Liquid.LAVA: "#f05a2a",
    Liquid.HONEY: "#d9a62e",
    Liquid.SHIMMER: "#b38de5",
}
_BIOME_COLORS = {
    Biome.FOREST: "#4d9f54",
    Biome.SNOW: "#a9d9e8",
    Biome.JUNGLE: "#2f8e4a",
    Biome.DESERT: "#cfad63",
    Biome.CORRUPTION: "#7e4ca5",
    Biome.CRIMSON: "#ae3f52",
    Biome.HALLOW: "#72d5ce",
    Biome.OCEAN: "#397eb4",
    Biome.UNDERWORLD: "#a83d2d",
    Biome.MUSHROOM: "#5f67cc",
    Biome.DUNGEON: "#405d88",
    Biome.SHIMMER: "#b58be0",
}

GENERATION_MILESTONES = (
    "Reset",
    "Terrain",
    "Tunnels",
    "Rock Layer Caves",
    "Generate Ice Biome",
    "Jungle",
    "Full Desert",
    "Floating Islands",
    "Shinies",
    "Underworld",
    "Corruption",
    "Dungeon",
    "Shimmer",
    "Pyramids",
    "Living Trees",
    "Jungle Temple",
    "Hives",
    "Settle Liquids",
    "Waterfalls",
    "Life Crystals",
    "Temple",
    "Floating Island Houses",
    "Hellforge",
    "Planting Trees",
    "Final Cleanup",
    "Hardmode V Transformation",
)


def _rgb(hex_color: str) -> tuple[int, int, int]:
    color = hex_color.lstrip("#")
    red, green, blue = (int(color[index : index + 2], 16) for index in (0, 2, 4))
    return red, green, blue


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = ("seguisb.ttf", "DejaVuSans-Bold.ttf") if bold else ("segoeui.ttf", "DejaVuSans.ttf")
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _palette(enum_values: Iterable, colors: dict) -> np.ndarray:
    maximum = max(int(value) for value in enum_values)
    palette = np.zeros((maximum + 1, 3), dtype=np.uint8)
    for value in enum_values:
        palette[int(value)] = _rgb(colors[value])
    return palette


_TILE_PALETTE = _palette(Tile, {tile: style.color for tile, style in TILE_STYLES.items()})
_WALL_PALETTE = _palette(Wall, WALL_COLORS)


def render_world(
    world: GeneratedWorld,
    scale: int = 4,
    *,
    biome_overlay: bool = False,
    layer_lines: bool = False,
    markers: bool = True,
) -> Image.Image:
    """Render an original sprite-free pixel map."""

    height, width = world.shape
    y = np.arange(height, dtype=np.float32)[:, None]
    depth = np.clip(y / max(1, height - 1), 0.0, 1.0)

    # Blue-black sky with a slight horizon glow.
    sky_top = np.array(_rgb("#0b1220"), dtype=np.float32)
    sky_bottom = np.array(_rgb("#263752"), dtype=np.float32)
    sky_mix = np.clip(y / max(1, world.layers.world_surface * 1.8), 0.0, 1.0)
    sky_column = sky_top * (1.0 - sky_mix[..., None]) + sky_bottom * sky_mix[..., None]
    pixels = np.repeat(sky_column, width, axis=1)

    wall_mask = (world.tiles == Tile.AIR) & (world.walls != Wall.NONE)
    wall_pixels = _WALL_PALETTE[world.walls]
    pixels[wall_mask] = wall_pixels[wall_mask] * 0.72

    solid = world.tiles != Tile.AIR
    tile_pixels = _TILE_PALETTE[world.tiles].astype(np.float32)

    # Coordinate hash adds subtle, deterministic material texture.
    yy, xx = np.indices(world.shape, dtype=np.uint32)
    noise = ((xx * 73856093) ^ (yy * 19349663) ^ world.config.seed_value) & 15
    texture = 0.91 + noise.astype(np.float32) / 110.0
    lighting = 1.0 - depth * 0.22
    lighting[world.layers.underworld :] = 0.92
    shaded = tile_pixels * texture[..., None] * lighting[..., None]
    pixels[solid] = shaded[solid]

    # Crisp top-edge light and bottom-edge shade suggest connected tiles without
    # reproducing Terraria's proprietary spritesheet framing.
    air = ~solid
    top_edge = solid & np.roll(air, 1, axis=0)
    bottom_edge = solid & np.roll(air, -1, axis=0)
    pixels[top_edge] = np.minimum(255, pixels[top_edge] * 1.16 + 6)
    pixels[bottom_edge] *= 0.78

    liquid_mask = (world.liquid_amount > 0) & (world.tiles == Tile.AIR)
    for liquid, color in _LIQUID_COLORS.items():
        mask = liquid_mask & (world.liquid_kind == liquid)
        if not np.any(mask):
            continue
        alpha = (world.liquid_amount[mask].astype(np.float32) / 255.0)[:, None] * 0.78
        liquid_rgb = np.array(_rgb(color), dtype=np.float32)
        pixels[mask] = pixels[mask] * (1.0 - alpha) + liquid_rgb * alpha

    if biome_overlay:
        overlay = np.zeros_like(pixels)
        overlay_mask = np.zeros(world.shape, dtype=bool)
        for biome, color in _BIOME_COLORS.items():
            mask = world.biomes == biome
            overlay[mask] = _rgb(color)
            overlay_mask |= mask
        pixels[overlay_mask] = pixels[overlay_mask] * 0.72 + overlay[overlay_mask] * 0.28

    image = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), mode="RGB")
    if scale != 1:
        image = image.resize((width * scale, height * scale), Image.Resampling.NEAREST)

    draw = ImageDraw.Draw(image)
    if layer_lines:
        for layer_y, color in (
            (world.layers.world_surface, "#6ed7e8"),
            (world.layers.rock_layer, "#e4b85c"),
            (world.layers.underworld, "#ef6262"),
        ):
            sy = int(layer_y * scale)
            for start in range(0, image.width, max(4, scale * 3)):
                draw.line(
                    (start, sy, start + max(1, scale), sy), fill=color, width=max(1, scale // 2)
                )

    if markers:
        for marker in world.structures:
            _draw_marker(draw, marker, scale)
    return image


def _draw_marker(draw: ImageDraw.ImageDraw, marker: StructureMarker, scale: int) -> None:
    center_x = round((marker.x + marker.width / 2) * scale)
    center_y = round((marker.y + marker.height / 2) * scale)
    radius = max(3, scale * 2)
    colors = {
        "Dungeon": "#6ea3e2",
        "Living tree": "#71c16b",
        "Spawn": "#f4d06f",
        "Aether": "#c9a0ef",
        "Jungle temple": "#e0a85a",
        "Floating island": "#9bc6e5",
        "Pyramid": "#e1ba6e",
        "Underworld city": "#f08a4b",
        "Spider cave": "#cf7f9c",
        "Gem cave": "#67e0d2",
    }
    color = colors.get(marker.kind, "#f0f4ff")
    outline = "#111827"
    draw.ellipse(
        (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
        fill=outline,
        outline=color,
        width=max(1, scale // 2),
    )
    # Each symbol is a small original geometric glyph, readable without fonts.
    if marker.kind == "Living tree":
        draw.line(
            (center_x, center_y - radius + 2, center_x, center_y + radius - 2),
            fill=color,
            width=max(1, scale // 2),
        )
        draw.line(
            (center_x, center_y, center_x - radius + 2, center_y - radius // 2), fill=color, width=1
        )
        draw.line(
            (center_x, center_y, center_x + radius - 2, center_y - radius // 2), fill=color, width=1
        )
        draw.line(
            (center_x, center_y + radius // 2, center_x - radius + 2, center_y + radius - 1),
            fill=color,
            width=1,
        )
        draw.line(
            (center_x, center_y + radius // 2, center_x + radius - 2, center_y + radius - 1),
            fill=color,
            width=1,
        )
    elif marker.kind == "Dungeon":
        draw.rectangle(
            (
                center_x - radius // 2,
                center_y - radius // 2,
                center_x + radius // 2,
                center_y + radius // 2,
            ),
            outline=color,
            width=1,
        )
        draw.line((center_x, center_y - radius // 2, center_x, center_y + radius // 2), fill=color)
        draw.line((center_x - radius // 2, center_y, center_x + radius // 2, center_y), fill=color)
    elif marker.kind == "Spawn":
        draw.polygon(
            (
                (center_x - radius + 2, center_y),
                (center_x, center_y - radius + 2),
                (center_x + radius - 2, center_y),
            ),
            outline=color,
        )
        draw.rectangle(
            (center_x - radius // 2, center_y, center_x + radius // 2, center_y + radius - 2),
            outline=color,
        )
    else:
        draw.line((center_x - radius + 2, center_y, center_x + radius - 2, center_y), fill=color)
        draw.line((center_x, center_y - radius + 2, center_x, center_y + radius - 2), fill=color)


def add_title_bar(image: Image.Image, title: str, subtitle: str = "") -> Image.Image:
    height = 56 if subtitle else 38
    canvas = Image.new("RGB", (image.width, image.height + height), _rgb("#0b1220"))
    canvas.paste(image, (0, height))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(18, bold=True)
    title_width = draw.textlength(title, font=title_font)
    draw.text(((image.width - title_width) / 2, 6), title, fill="#e7edf7", font=title_font)
    if subtitle:
        subtitle_font = _font(12)
        subtitle_width = draw.textlength(subtitle, font=subtitle_font)
        draw.text(
            ((image.width - subtitle_width) / 2, 30),
            subtitle,
            fill="#9fb0c9",
            font=subtitle_font,
        )
    return canvas


def save_png(world: GeneratedWorld, path: str | Path, scale: int = 4, **render_options) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    render_world(world, scale, **render_options).save(destination, optimize=True)
    return destination


def save_generation_gif(
    config: WorldConfig,
    path: str | Path,
    *,
    scale: int = 4,
    duration_ms: int = 420,
) -> Path:
    """Run the pipeline once and capture visually meaningful milestones."""

    if config.scale is not WorldScale.PREVIEW:
        raise ValueError("Generation GIFs intentionally use the responsive Preview scale")
    milestones = set(GENERATION_MILESTONES)
    if not config.hardmode:
        milestones.remove("Hardmode V Transformation")
    frames: list[Image.Image] = []

    def capture(spec, world: GeneratedWorld) -> None:
        if spec.name not in milestones:
            return
        image = render_world(world, scale, markers=True)
        frames.append(
            add_title_bar(
                image,
                spec.name,
                spec.phase.value,
            )
        )

    TerraExplorerPipeline().generate(config, snapshot=capture)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        destination,
        save_all=True,
        append_images=frames[1:],
        duration=[duration_ms] * (len(frames) - 1) + [2000],
        loop=0,
        optimize=True,
        disposal=2,
    )
    return destination


def save_npz(world: GeneratedWorld, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        tiles=world.tiles,
        walls=world.walls,
        liquid_amount=world.liquid_amount,
        liquid_kind=world.liquid_kind,
        biomes=world.biomes,
        surface=world.surface,
        metadata=json.dumps(world.metadata, default=str),
    )
    return destination


def summary_json(world: GeneratedWorld) -> str:
    payload = {
        "size": {"width": world.shape[1], "height": world.shape[0]},
        "memory_bytes": world.memory_bytes,
        "metadata": world.metadata,
        "structures": [
            {
                "kind": marker.kind,
                "x": marker.x,
                "y": marker.y,
                "width": marker.width,
                "height": marker.height,
                "symbol": marker.symbol,
            }
            for marker in world.structures
        ],
        "passes": [
            {
                "index": result.index,
                "name": result.name,
                "phase": result.phase,
                "fidelity": result.fidelity,
                "elapsed_ms": round(result.elapsed_ms, 3),
                "note": result.note,
            }
            for result in world.pass_results
        ],
    }
    return json.dumps(payload, indent=2, default=str)
