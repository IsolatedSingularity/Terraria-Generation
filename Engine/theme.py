"""
Unified dark theme for Terraria-Generation plots.

Matches the TQNN/QLDPC visual identity: #1a1a1a background, mako/cubehelix
colormaps, 300 DPI, white text, publication-quality output.

Usage:
    from Engine.theme import applyDarkTheme, COLORS, seqCmap, divCmap, lightCmap
    applyDarkTheme()
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Core palette
# ---------------------------------------------------------------------------
COLORS = {
    "bg": "#1a1a1a",
    "axes": "#111111",
    "text": "#ffffff",
    "accent": "#00ff88",
    "grid": "#2d2d2d",
    "edge": "#444444",
    "subtitle": "#aaaaaa",
    "legend_bg": "#1e1e1e",
}


# ---------------------------------------------------------------------------
# Terraria-specific biome colors (game-accurate hex values)
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

# Tile-level rendering colors (for world grid visualization)
TILE_COLORS = {
    0: "#111111",       # AIR
    1: "#8B6914",       # DIRT
    2: "#808080",       # STONE
    3: "#228B22",       # GRASS
    4: "#EDC9AF",       # SAND
    5: "#4A4A4A",       # ASH
    6: "#C41E3A",       # HELLSTONE
    7: "#6B4423",       # MUD
    8: "#B0E0E6",       # SNOW
    9: "#ADD8E6",       # ICE
    50: "#1E90FF",      # WATER
    51: "#FF4500",      # LAVA
    52: "#FFD700",      # HONEY
    53: "#484848",      # OBSIDIAN
    54: "#FFA500",      # CRISPY_HONEY_BLOCK
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
    100: "#B87333",     # COPPER
    101: "#D3D3D3",     # TIN
    102: "#8B7355",     # IRON
    103: "#4A6670",     # LEAD
    104: "#C0C0C0",     # SILVER
    105: "#4F7942",     # TUNGSTEN
    106: "#FFD700",     # GOLD
    107: "#E5E4E2",     # PLATINUM
    110: "#0047AB",     # COBALT
    111: "#FF6600",     # PALLADIUM
    112: "#008080",     # MYTHRIL
    113: "#FF1493",     # ORICHALCUM
    114: "#8B0000",     # ADAMANTITE
    115: "#483D8B",     # TITANIUM
    116: "#00FF00",     # CHLOROPHYTE
    # Structure tiles
    120: "#4A2060",     # DUNGEON_BRICK
    121: "#8B7355",     # LIHZAHRD_BRICK
    122: "#E8E8E8",     # MARBLE
    123: "#383850",     # GRANITE
    124: "#D2B48C",     # HARDENED_SAND
    125: "#C4A882",     # SANDSTONE
    126: "#3D2B1F",     # SPIDER_WALL (cobweb)
    127: "#C8A2C8",     # SHIMMER
}

# Ore-specific palette for distribution plots
ORE_COLORS = {
    "copper": "#B87333",
    "tin": "#D3D3D3",
    "iron": "#8B7355",
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
    "hellstone": "#C41E3A",
}


# ---------------------------------------------------------------------------
# Colormaps (matching TQNN/QLDPC)
# ---------------------------------------------------------------------------
seqCmap = sns.color_palette("mako", as_cmap=True)
divCmap = sns.cubehelix_palette(start=0.5, rot=-0.5, as_cmap=True)
lightCmap = sns.cubehelix_palette(
    start=2, rot=0, dark=0.05, light=0.45, reverse=True, as_cmap=True
)


# ---------------------------------------------------------------------------
# Apply rcParams
# ---------------------------------------------------------------------------
def applyDarkTheme() -> None:
    """Set matplotlib rcParams to the unified dark theme."""
    params = {
        "figure.facecolor": COLORS["bg"],
        "axes.facecolor": COLORS["axes"],
        "savefig.facecolor": COLORS["bg"],
        "savefig.edgecolor": "none",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "text.color": COLORS["text"],
        "axes.labelcolor": COLORS["text"],
        "axes.edgecolor": COLORS["edge"],
        "xtick.color": COLORS["text"],
        "ytick.color": COLORS["text"],
        "grid.color": COLORS["grid"],
        "grid.alpha": 0.3,
        "legend.facecolor": COLORS["legend_bg"],
        "legend.edgecolor": COLORS["edge"],
        "legend.labelcolor": COLORS["text"],
        "figure.titlesize": 18,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "font.size": 12,
        "lines.linewidth": 2.5,
    }
    mpl.rcParams.update(params)
