"""Render native saved geometry in TerraExplorer's original sprite-free style.

Only color interpretation is approximate. Native cells never become simulation
cells; unknown IDs are colored deterministically without discarding their state.
"""

from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from terraexplorer.fidelity.structures import DUNGEON_WALLS, TREE_WALLS
from terraexplorer.fidelity.world import CanonicalWorld
from terraexplorer.tiles import TILE_STYLES, WALL_COLORS, Tile, Wall

# Native IDs verified against pinned Terraria.ID.TileID constants. These mappings
# select existing original art colors; they do not translate or overwrite state.
STYLE_GROUPS = {
    Tile.DIRT: (0,),
    Tile.GRASS: (2,),
    Tile.FLOWER: (3, 73, 82, 83, 84, 227),
    Tile.COPPER: (7,),
    Tile.GOLD: (8,),
    Tile.SILVER: (9,),
    Tile.LIFE_CRYSTAL: (12,),
    Tile.PLATFORM: (19,),
    Tile.CHEST: (21, 467),
    Tile.CORRUPT_GRASS: (23,),
    Tile.EBONSTONE: (25,),
    Tile.ALTAR: (26, 237),
    Tile.POT: (28,),
    Tile.CLAY: (40,),
    Tile.DUNGEON_BRICK: (41, 43, 44, 481, 482, 483),
    Tile.COBWEB: (51,),
    Tile.VINE: (52, 62, 382, 528),
    Tile.SAND: (53,),
    Tile.MUD: (59,),
    Tile.JUNGLE_GRASS: (60,),
    Tile.LEAF: (61, 192, 384),
    Tile.MUSHROOM_GRASS: (70, 71),
    Tile.OBSIDIAN: (56,),
    Tile.ASH: (57,),
    Tile.HELLSTONE: (58,),
    Tile.GEM: (63, 64, 65, 66, 67, 68, 178),
    Tile.SILT: (123,),
    Tile.TRAP: (48, 135, 136, 137, 138, 141, 232, 411, 714),
    Tile.STALACTITE: (165,),
    Tile.TIN: (166,),
    Tile.LEAD: (167,),
    Tile.TUNGSTEN: (168,),
    Tile.PLATINUM: (169,),
    Tile.STONE: (1, 179, 180, 181, 182, 183, 185, 186, 187, 381, 536),
    Tile.CLOUD: (189,),
    Tile.LIVING_WOOD: (191, 383),
    Tile.RAIN_CLOUD: (196,),
    Tile.LIHZAHRD_BRICK: (226,),
    Tile.MARBLE: (367,),
    Tile.GRANITE: (368,),
    Tile.HARDENED_SAND: (397,),
    Tile.SANDSTONE: (396,),
    Tile.SNOW: (147,),
    Tile.ICE: (161,),
    Tile.TREE: (5, 10, 14, 15, 18, 30, 87, 88, 94, 101, 104, 124, 304),
}
NATIVE_STYLE = {native: style for style, ids in STYLE_GROUPS.items() for native in ids}
WALL_STYLE = {
    **dict.fromkeys(DUNGEON_WALLS, Wall.DUNGEON),
    **dict.fromkeys(TREE_WALLS, Wall.DIRT),
    87: Wall.LIHZAHRD,
    1: Wall.STONE,
    2: Wall.DIRT,
    15: Wall.DIRT,
    16: Wall.DIRT,
    64: Wall.JUNGLE,
    3: Wall.DIRT,
    4: Wall.DIRT,
}
LIQUID_RGB = {1: (46, 125, 186), 2: (240, 90, 42), 3: (217, 166, 46), 4: (179, 141, 229)}


def rgb(color):
    return tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))


def fallback_color(native_id: int, wall=False):
    # Muted original material colors, with an ID-specific color and texture hash.
    value = (native_id * 2654435761) & 0xFFFFFFFF
    base = np.asarray((65 + (value & 63), 55 + ((value >> 8) & 63), 65 + ((value >> 16) & 63)))
    return tuple(int(x) for x in (base * (0.55 if wall else 1)))


def colors(ids, wall=False):
    result = np.empty((*ids.shape, 3), dtype=np.float32)
    mapping = WALL_STYLE if wall else NATIVE_STYLE
    for native in np.unique(ids):
        native = int(native)
        if native in mapping:
            color = (
                rgb(WALL_COLORS[mapping[native]])
                if wall
                else rgb(TILE_STYLES[mapping[native]].color)
            )
        else:
            color = fallback_color(native, wall)
        result[ids == native] = color
    return result


def font(size, bold=False):
    for name in (
        ("seguisb.ttf", "DejaVuSans-Bold.ttf") if bold else ("segoeui.ttf", "DejaVuSans.ttf")
    ):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def render_crop(world: CanonicalWorld, bbox, *, scale=4, title="Reference structure"):
    if not isinstance(world, CanonicalWorld):
        raise TypeError("Reference renders require imported canonical world data")
    left, top, right, bottom = bbox
    height, width = world.cells.shape
    if not (
        0 <= left < right <= width
        and 0 <= top < bottom <= height
        and isinstance(scale, int)
        and 2 <= scale <= 8
    ):
        raise ValueError("Invalid native crop or scale")
    c = world.cells[top:bottom, left:right]
    y, x = np.indices(c.shape, dtype=np.uint32)
    y += top
    x += left
    depth = y.astype(np.float32) / max(1, height - 1)
    mix = np.clip(y / (world.metadata["world_surface"] * 1.8), 0, 1)[..., None]
    background = np.array((11, 18, 32)) * (1 - mix) + np.array((38, 55, 82)) * mix
    wall = c["wall"] != 0
    background[wall] = colors(c["wall"], wall=True)[wall] * 0.72
    seed = int(world.metadata["seed"]) if world.metadata["seed"].isdigit() else 0
    noise = ((x * 73856093) ^ (y * 19349663) ^ seed) & 15
    texture = 0.91 + noise.astype(np.float32) / 110
    foreground = colors(c["tile"]) * texture[..., None] * (1 - depth * 0.22)[..., None]
    active = c["active"] != 0
    # Fetch the real neighboring cells, so crop edges do not become false highlights.
    above = world.cells["active"][np.maximum(y.astype(int) - 1, 0), x] == 0
    below = world.cells["active"][np.minimum(y + 1, height - 1), x] == 0
    foreground[active & above] = np.minimum(255, foreground[active & above] * 1.16 + 6)
    foreground[active & below] *= 0.78

    # Subcell masks preserve serialized half-block/slope geometry at display scale.
    def expand(a):
        return np.repeat(np.repeat(a, scale, axis=0), scale, axis=1)

    pixels = expand(background)
    foreground = expand(foreground)
    occupied = expand(active)
    sy, sx = np.indices(occupied.shape)
    sy, sx = sy % scale, sx % scale
    shape = expand(c["shape"])
    occupied &= ~((shape == 1) & (sy < scale / 2))
    occupied &= ~((shape == 2) & (sy < sx))
    occupied &= ~((shape == 3) & (sy < scale - 1 - sx))
    occupied &= ~((shape == 4) & (sy > scale - 1 - sx))
    occupied &= ~((shape == 5) & (sy > sx))
    # Inactive/invisible blocks remain visible as ghosted reference geometry.
    ghost = expand(((c["wires_actuator_inactive"] & 32) != 0) | ((c["coatings"] & 2) != 0))
    foreground[ghost] = foreground[ghost] * 0.45 + pixels[ghost] * 0.55
    pixels[occupied] = foreground[occupied]
    for kind, color in LIQUID_RGB.items():
        liquid = expand((c["liquid_kind"] == kind) & (c["liquid_amount"] > 0)) & ~occupied
        alpha = expand(c["liquid_amount"].astype(np.float32) / 255 * 0.78)[liquid, None]
        pixels[liquid] = pixels[liquid] * (1 - alpha) + np.array(color) * alpha
    image = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
    # Labels are outside the imported pixel region.
    canvas = Image.new("RGB", (max(620, image.width + 32), image.height + 108), (11, 18, 32))
    canvas.paste(image, ((canvas.width - image.width) // 2, 90))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (16, 10),
        "IMPORTED VANILLA REFERENCE  |  Terraria 1.4.5.7",
        font=font(15, True),
        fill="#88bde8",
    )
    draw.text((16, 32), title, font=font(24, True), fill="#eff4fb")
    draw.text(
        (16, 65),
        f"Seed {world.metadata['seed']}  |  {left},{top} - {right},{bottom}  |  {scale}px/tile",
        font=font(13),
        fill="#a6b3c8",
    )
    image_hash = hashlib.sha256(image.tobytes()).hexdigest()
    provenance = {
        "source_wld_sha256": world.report["sha256"],
        "source_cells_sha256": world.semantic_sha256,
        "bbox": list(bbox),
        "scale": scale,
        "crop_cell_sha256": hashlib.sha256(c.T.copy().tobytes()).hexdigest(),
        "unlabelled_rgb_sha256": image_hash,
        "unsupported_active_tile_ids": sorted(
            set(c["tile"][active].tolist()) - NATIVE_STYLE.keys()
        ),
        "unsupported_wall_ids": sorted(set(c["wall"][wall].tolist()) - WALL_STYLE.keys()),
        "geometry": "Imported v325 saved cells; no procedural generation",
        "style": "Existing TerraExplorer palette, coordinate texture and edge lighting",
        "limits": "Paint, wall visibility and fullbright preserved, not recolored; objects "
        "are occupied cells, not sprites; wires/actuators are not overlaid.",
    }
    return canvas, provenance
