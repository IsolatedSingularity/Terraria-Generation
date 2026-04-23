"""Unified visual theme for Terraria-Generation plots.

Tokyo Night Storm palette for chrome (axes, panels, legends, gridlines, text),
combined with game-semantic palettes that remain Terraria-accurate
(`BIOME_COLORS`, `TILE_COLORS`, `ORE_COLORS`).

Usage::

    from Engine.theme import applyTokyoNight, COLORS, TILE_COLORS
    applyTokyoNight()
"""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


# ---------------------------------------------------------------------------
# Tokyo Night Storm palette (chrome only)
# ---------------------------------------------------------------------------
PALETTE = {
    "bg": "#1a1b26",
    "panel": "#24283b",
    "surface": "#292e42",
    "fg": "#c0caf5",
    "muted": "#a9b1d6",
    "subtle": "#565f89",
    "blue": "#7aa2f7",
    "cyan": "#7dcfff",
    "purple": "#bb9af7",
    "red": "#f7768e",
    "green": "#9ece6a",
    "yellow": "#e0af68",
    "orange": "#ff9e64",
}

# Backward-compat alias dict (existing code references COLORS["bg"], etc.).
# Keys mirror the previous dark-theme dict so callers do not need rewriting.
COLORS = {
    "bg": PALETTE["bg"],
    "axes": PALETTE["bg"],
    "panel": PALETTE["panel"],
    "surface": PALETTE["surface"],
    "text": PALETTE["fg"],
    "fg": PALETTE["fg"],
    "muted": PALETTE["muted"],
    "subtitle": PALETTE["muted"],
    "subtle": PALETTE["subtle"],
    "edge": PALETTE["subtle"],
    "grid": PALETTE["subtle"],
    "legend_bg": PALETTE["panel"],
    "accent": PALETTE["cyan"],
}

# Multi-series cycle order: blue, cyan, purple, red, green, yellow, orange.
CYCLE = [PALETTE[k] for k in
         ("blue", "cyan", "purple", "red", "green", "yellow", "orange")]


# ---------------------------------------------------------------------------
# Game-semantic biome colors (Terraria-accurate, NOT Tokyo Night)
# ---------------------------------------------------------------------------
BIOME_COLORS = {
    "forest": "#228B22",
    "corruption": "#7B2D8B",
    "crimson": "#C41E3A",
    "hallow": "#FFD700",
    "jungle": "#2E8B57",
    "snow": "#B0E0E6",
    "desert": "#EDC9AF",
    "ocean": "#1E90FF",
    "mushroom": "#2020CC",
    "dungeon": "#4A2060",
    "underworld": "#8B0000",
    "marble": "#E8E8E8",
    "granite": "#383850",
    "spider": "#3D2B1F",
    "shimmer": "#C8A2C8",
    "gem": "#9B59B6",
    "underground_desert": "#D2B48C",
}


# ---------------------------------------------------------------------------
# Tile-level rendering colors (single source of truth)
# Indexed by tile ID. All scripts MUST import from here, never duplicate.
# ---------------------------------------------------------------------------
TILE_COLORS = {
    # --- Basic terrain ---
    0: PALETTE["bg"],   # AIR (matches plot bg so empty space is invisible)
    1: "#8B6914",       # DIRT
    2: "#808080",       # STONE
    3: "#90EE90",       # GRASS
    4: "#EDC9AF",       # SAND
    5: "#4A4A4A",       # ASH
    6: "#FF4500",       # HELLSTONE
    7: "#6B4423",       # MUD
    8: "#E8E8F0",       # SNOW
    9: "#ADD8E6",       # ICE
    10: "#C4A882",      # CLAY
    11: "#5A4B3C",      # SILT

    # --- Liquids ---
    50: "#1E90FF",      # WATER
    51: "#FF2400",      # LAVA
    52: "#FFD700",      # HONEY
    53: "#484848",      # OBSIDIAN
    54: "#FFA500",      # CRISPY_HONEY_BLOCK

    # --- Evil/holy biome conversions ---
    60: "#5C3D6E",      # CORRUPT_DIRT
    61: "#7B2D8B",      # EBONSTONE
    62: "#8B1A1A",      # CRIMSON_DIRT
    63: "#C41E3A",      # CRIMSTONE
    64: "#FFB6C1",      # PEARLSTONE
    65: "#FFE4B5",      # PEARLSAND
    66: "#FFD700",      # HALLOW_DIRT
    67: "#8B6FAC",      # CORRUPT_ICE
    68: "#CD5C5C",      # CRIMSON_ICE
    69: "#FFB6C1",      # HALLOW_ICE

    # --- Pre-hardmode ores ---
    100: "#B87333",     # COPPER
    101: "#D3D3D3",     # TIN
    102: "#A9A9A9",     # IRON
    103: "#4A6670",     # LEAD
    104: "#C0C0C0",     # SILVER
    105: "#4F7942",     # TUNGSTEN
    106: "#FFD700",     # GOLD
    107: "#E5E4E2",     # PLATINUM

    # --- Hardmode ores ---
    110: "#0047AB",     # COBALT
    111: "#FF6600",     # PALLADIUM
    112: "#008080",     # MYTHRIL
    113: "#FF1493",     # ORICHALCUM
    114: "#8B0000",     # ADAMANTITE
    115: "#483D8B",     # TITANIUM
    116: "#00FF00",     # CHLOROPHYTE

    # --- Structure tiles ---
    120: "#4A2060",     # DUNGEON_BRICK
    121: "#8B7355",     # LIHZAHRD_BRICK
    122: "#E8E8E8",     # MARBLE
    123: "#383850",     # GRANITE
    124: "#D2B48C",     # HARDENED_SAND
    125: "#C4A882",     # SANDSTONE
    126: "#3D2B1F",     # SPIDER_WALL (cobweb)
    127: "#C8A2C8",     # SHIMMER

    # --- Decorative / special ---
    130: "#FF69B4",     # LIFE_CRYSTAL
    131: "#8B0000",     # ALTAR
    132: "#5C3317",     # WOOD (cabin/floating-island walls)
    133: "#3F2A1A",     # WOOD_PLATFORM
    134: "#FFB347",     # TORCH
    135: "#FFD700",     # CHEST
    136: "#5C3317",     # DOOR
    137: "#558B2F",     # LIVING_WOOD
    138: "#7CB342",     # LIVING_LEAF
    139: "#C97B5A",     # POT
    140: "#A0522D",     # CACTUS
}

DEFAULT_TILE_COLOR = PALETTE["subtle"]


def buildTileColormap(maxTileId: int = 200) -> ListedColormap:
    """Construct a `ListedColormap` indexed by tile ID up to ``maxTileId``.

    Missing IDs fall back to ``DEFAULT_TILE_COLOR`` so unmapped tiles read
    as a soft Tokyo Night subtle grey instead of jet black.
    """
    palette = [TILE_COLORS.get(i, DEFAULT_TILE_COLOR) for i in range(maxTileId)]
    return ListedColormap(palette, name="terrariaTiles")


# ---------------------------------------------------------------------------
# Ore palette for distribution plots (game-semantic)
# ---------------------------------------------------------------------------
ORE_COLORS = {
    "copper": "#B87333",
    "tin": "#D3D3D3",
    "iron": "#A9A9A9",
    "lead": "#4A6670",
    "silver": "#C0C0C0",
    "tungsten": "#4F7942",
    "gold": "#FFD700",
    "platinum": "#E5E4E2",
    "cobalt": "#0047AB",
    "palladium": "#FF6600",
    "mythril": "#008080",
    "orichalcum": "#FF1493",
    "adamantite": "#8B0000",
    "titanium": "#483D8B",
    "chlorophyte": "#00FF00",
    "hellstone": "#FF4500",
}


# ---------------------------------------------------------------------------
# Tokyo Night colormaps (replace mako/cubehelix defaults)
# ---------------------------------------------------------------------------
seqCmap = LinearSegmentedColormap.from_list(
    "tokyoSeq",
    [PALETTE["bg"], PALETTE["subtle"], PALETTE["blue"], PALETTE["cyan"]],
    N=256,
)

divCmap = LinearSegmentedColormap.from_list(
    "tokyoDiv",
    [PALETTE["red"], PALETTE["bg"], PALETTE["cyan"]],
    N=256,
)

lightCmap = LinearSegmentedColormap.from_list(
    "tokyoLight",
    [PALETTE["bg"], PALETTE["purple"], PALETTE["yellow"]],
    N=256,
)


# ---------------------------------------------------------------------------
# rcParams entry point
# ---------------------------------------------------------------------------
def applyTokyoNight() -> None:
    """Apply Tokyo Night Storm rcParams globally.

    Call once at module import in any visualization entry point. Sets
    chrome (figure/axes bg, text, ticks, gridlines, legend, color cycle)
    and locks pixel-tile rendering rcParams so `imshow` does not antialias.
    """
    mpl.rcParams.update({
        # Figure / axes surfaces
        "figure.facecolor": PALETTE["bg"],
        "axes.facecolor": PALETTE["bg"],
        "savefig.facecolor": PALETTE["bg"],
        "savefig.edgecolor": "none",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",

        # Text + axes chrome
        "text.color": PALETTE["fg"],
        "axes.labelcolor": PALETTE["fg"],
        "axes.titlecolor": PALETTE["fg"],
        "axes.edgecolor": PALETTE["subtle"],
        "xtick.color": PALETTE["muted"],
        "ytick.color": PALETTE["muted"],

        # Gridlines
        "grid.color": PALETTE["subtle"],
        "grid.linestyle": "--",
        "grid.alpha": 0.4,

        # Legend
        "legend.facecolor": PALETTE["panel"],
        "legend.edgecolor": PALETTE["subtle"],
        "legend.labelcolor": PALETTE["fg"],

        # Multi-series color cycle
        "axes.prop_cycle": cycler(color=CYCLE),

        # Pixel-tile rendering: NEAREST, no resample (sprites must be crisp)
        "image.interpolation": "nearest",
        "image.resample": False,

        # Typography
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 11,
        "figure.titlesize": 18,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "lines.linewidth": 2.0,
    })


__all__ = [
    "PALETTE",
    "COLORS",
    "CYCLE",
    "BIOME_COLORS",
    "TILE_COLORS",
    "DEFAULT_TILE_COLOR",
    "ORE_COLORS",
    "seqCmap",
    "divCmap",
    "lightCmap",
    "applyTokyoNight",
    "buildTileColormap",
]
