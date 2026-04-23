"""
Terraria Biome Distribution Analysis

Visualizes the spatial distribution and mathematical relationships
of biomes in large Terraria worlds, following the actual game's
103-pass GenPass pipeline and placement constraints.

Biome placement rules (from decompiled WorldGen.cs):
- dungeonX polarity determines Jungle vs Dungeon/Snow hemisphere
- Evil biome (Corruption/Crimson) placed independently of dungeon side
- 1 surface desert + 1 Underground Desert per world
- 6 floating islands for large worlds
- Marble caves (16-32), Granite caves, surface Mushroom biome
- 45-tile border buffer from all edges
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Ellipse, Rectangle

from Engine.algorithms import (
    AIR, DIRT, EBONSTONE, GRASS, MUD, SAND, STONE,
)
from Engine.constants import DETAIL_PLOT, LARGE, LayerDepths, StructureQuotas
from Engine.spriteRenderer import drawTileGrid
from Engine.theme import applyTokyoNight, COLORS, PALETTE

applyTokyoNight()


def generateWorldLayout(seed: int = 12345) -> dict:
    """Generate a large-world biome layout following actual game placement rules.

    Placement constraints (WorldGen.cs):
    - dungeonSide chosen randomly; Jungle on opposite side; Snow on dungeon side
    - Evil biome placed independently (either hemisphere)
    - 1 surface desert, 1 Underground Desert (circular ant-hive)
    - 6 floating islands (StructureQuotas.forLarge)
    - Marble caves 16-32, Granite caves ~16-32, 1 surface Mushroom biome
    - 45-tile border buffer enforced on all edges

    Args:
        seed: RNG seed for reproducible layout.

    Returns:
        Dict of biome positions, types, and world metadata.
    """
    rng = np.random.default_rng(seed)

    width = LARGE.width
    height = LARGE.height
    buffer = LARGE.borderBuffer  # 45 tiles
    layers = LayerDepths.forLarge()
    quotas = StructureQuotas.forLarge()

    # Spawn near world center
    spawnX = width // 2 + int(rng.integers(-100, 100))

    # Dungeon side (50/50)
    dungeonSide = str(rng.choice(["left", "right"]))

    # Dungeon, Jungle, Snow placement based on dungeonX polarity
    if dungeonSide == "left":
        dungeonX = int(rng.integers(buffer, width // 4))
        jungleX = int(rng.integers(3 * width // 4, width - buffer))
        snowX = int(rng.integers(buffer, width // 3))
    else:
        dungeonX = int(rng.integers(3 * width // 4, width - buffer))
        jungleX = int(rng.integers(buffer, width // 4))
        snowX = int(rng.integers(2 * width // 3, width - buffer))

    # Evil biome: placed independently of dungeon side
    evilType = str(rng.choice(["corruption", "crimson"]))
    evilX = int(rng.integers(buffer + 200, width - buffer - 200))

    # 1 surface desert (not 3)
    desertX = int(rng.integers(buffer + 300, width - buffer - 300))
    while abs(desertX - spawnX) < 400:
        desertX = int(rng.integers(buffer + 300, width - buffer - 300))

    # 1 Underground Desert (circular ant-hive beneath surface desert)
    undergroundDesertCenter = (desertX, int(layers.rockLayer + 100))
    undergroundDesertRadius = int(rng.integers(150, 250))

    # Oceans at edges
    oceanLeftX = 0
    oceanRightX = width

    # Floating islands: exactly 6 for large world
    numIslands = quotas.floatingIslands
    islandPositions = []
    usedIslandX: set[int] = set()
    for _ in range(numIslands):
        attempts = 0
        while attempts < 100:
            ix = int(rng.integers(buffer + 200, width - buffer - 200))
            tooClose = any(abs(ix - ux) < 300 for ux in usedIslandX)
            if not tooClose:
                break
            attempts += 1
        iy = int(rng.integers(100, int(layers.worldSurface * 0.4)))
        islandPositions.append((ix, iy))
        usedIslandX.add(ix)

    # Marble caves: 16-32 for large world
    numMarble = int(rng.integers(quotas.marbleCavesMin, quotas.marbleCavesMax + 1))
    marblePositions = []
    for _ in range(numMarble):
        mx = int(rng.integers(buffer, width - buffer))
        my = int(rng.integers(int(layers.rockLayer), int(layers.hellLayer - 50)))
        marblePositions.append((mx, my))

    # Granite caves: similar count to marble
    numGranite = int(rng.integers(quotas.marbleCavesMin, quotas.marbleCavesMax + 1))
    granitePositions = []
    for _ in range(numGranite):
        gx = int(rng.integers(buffer, width - buffer))
        gy = int(rng.integers(int(layers.rockLayer), int(layers.hellLayer - 50)))
        granitePositions.append((gx, gy))

    # Surface Mushroom biome
    mushroomX = int(rng.integers(buffer + 500, width - buffer - 500))
    while abs(mushroomX - jungleX) < 600:
        mushroomX = int(rng.integers(buffer + 500, width - buffer - 500))
    mushroomY = int(layers.rockLayer + int(rng.integers(50, 200)))

    return {
        "dimensions": (width, height),
        "borderBuffer": buffer,
        "layers": {
            "worldSurface": layers.worldSurface,
            "rockLayer": layers.rockLayer,
            "hellLayer": layers.hellLayer,
        },
        "spawn": (spawnX, int(layers.worldSurface)),
        "dungeon": (dungeonX, int(layers.worldSurface - 50)),
        "dungeonSide": dungeonSide,
        "jungle": (jungleX, height // 2),
        "snow": (snowX, height // 2),
        "evil": (evilX, height // 2),
        "evilType": evilType,
        "desert": desertX,
        "undergroundDesert": {
            "center": undergroundDesertCenter,
            "radius": undergroundDesertRadius,
        },
        "oceans": [(oceanLeftX, height // 2), (oceanRightX, height // 2)],
        "floatingIslands": islandPositions,
        "marbleCaves": marblePositions,
        "graniteCaves": granitePositions,
        "mushroom": mushroomX,
        "mushroomY": mushroomY,
        "numMarble": numMarble,
        "numGranite": numGranite,
    }


def createBiomeLayoutVisualization(savePath: str) -> None:
    """Create visualization showing 3 large world biome layouts with correct rules."""
    print("Creating biome layout visualization for large worlds...")

    layouts = [
        generateWorldLayout(seed=111),
        generateWorldLayout(seed=222),
        generateWorldLayout(seed=333),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(20, 18))
    fig.suptitle(
        "Terraria Large World Biome Layouts -- Placement Analysis\n"
        "World Generation Rules & Spatial Distribution Patterns",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    colors = {
        "forest": "#2E8B57",
        "jungle": "#228B22",
        "desert": "#DEB887",
        "undergroundDesert": "#C4A35A",
        "snow": "#E0F6FF",
        "corruption": "#9370DB",
        "crimson": "#DC143C",
        "ocean": "#4682B4",
        "dungeon": "#696969",
        "spawn": "#FFD700",
        "floatingIsland": "#87CEEB",
        "marble": "#F5F5F5",
        "granite": "#2F4F4F",
        "mushroom": "#4169E1",
        "hell": "#8B0000",
    }

    for i, layout in enumerate(layouts):
        ax = axes[i]
        width, height = layout["dimensions"]
        buf = layout["borderBuffer"]
        worldSurface = layout["layers"]["worldSurface"]
        rockLayer = layout["layers"]["rockLayer"]
        hellLayer = layout["layers"]["hellLayer"]

        # Background forest
        ax.add_patch(
            Rectangle((0, 0), width, height, facecolor=colors["forest"], alpha=0.2)
        )

        # Border buffer zones (shaded)
        ax.add_patch(
            Rectangle((0, 0), buf, height, facecolor="gray", alpha=0.15)
        )
        ax.add_patch(
            Rectangle((width - buf, 0), buf, height, facecolor="gray", alpha=0.15)
        )

        # Hell layer
        ax.add_patch(
            Rectangle(
                (0, hellLayer), width, height - hellLayer,
                facecolor=colors["hell"], alpha=0.4,
            )
        )
        ax.text(
            width // 2, hellLayer + (height - hellLayer) // 2,
            "Underworld", ha="center", va="center",
            fontweight="bold", color="white", fontsize=9, alpha=0.8,
        )

        # Layer depth lines
        ax.axhline(y=worldSurface, color="green", linestyle="--", alpha=0.5, lw=1)
        ax.axhline(y=rockLayer, color="brown", linestyle="--", alpha=0.5, lw=1)
        ax.axhline(y=hellLayer, color="red", linestyle="--", alpha=0.5, lw=1)

        # Oceans
        oceanWidth = 280
        ax.add_patch(
            Rectangle(
                (0, 0), oceanWidth, height * 0.6,
                facecolor=colors["ocean"], alpha=0.7,
            )
        )
        ax.add_patch(
            Rectangle(
                (width - oceanWidth, 0), oceanWidth, height * 0.6,
                facecolor=colors["ocean"], alpha=0.7,
            )
        )

        # Biome widths
        biomeW = 400

        # Jungle (extends deep)
        jx = layout["jungle"][0]
        ax.add_patch(
            Rectangle(
                (jx - biomeW // 2, 0), biomeW, hellLayer,
                facecolor=colors["jungle"], alpha=0.7,
            )
        )
        ax.text(
            jx, worldSurface + 100, "Jungle",
            ha="center", va="center", fontweight="bold", color="white", fontsize=10,
        )

        # Snow biome
        sx = layout["snow"][0]
        ax.add_patch(
            Rectangle(
                (sx - biomeW // 2, 0), biomeW, rockLayer,
                facecolor=colors["snow"], alpha=0.7,
            )
        )
        ax.text(
            sx, worldSurface + 80, "Snow",
            ha="center", va="center", fontweight="bold", color="black", fontsize=10,
        )

        # Evil biome (independent placement)
        ex = layout["evil"][0]
        evilColor = colors[layout["evilType"]]
        evilW = int(biomeW * 0.65)
        ax.add_patch(
            Rectangle(
                (ex - evilW // 2, 0), evilW, int(rockLayer * 0.9),
                facecolor=evilColor, alpha=0.75,
            )
        )
        evilName = layout["evilType"].capitalize()
        ax.text(
            ex, worldSurface + 60, evilName,
            ha="center", va="center", fontweight="bold", color="white", fontsize=10,
        )

        # 1 surface desert
        dx = layout["desert"]
        desertW = 350
        ax.add_patch(
            Rectangle(
                (dx - desertW // 2, 0), desertW, int(worldSurface + 200),
                facecolor=colors["desert"], alpha=0.8,
            )
        )
        ax.text(
            dx, worldSurface * 0.6, "Desert",
            ha="center", va="center", fontweight="bold", color="black", fontsize=9,
        )

        # Underground Desert (circular ant-hive)
        udCenter = layout["undergroundDesert"]["center"]
        udRadius = layout["undergroundDesert"]["radius"]
        ax.add_patch(
            Circle(
                udCenter, udRadius,
                facecolor=colors["undergroundDesert"], alpha=0.6,
                edgecolor="#8B7355", linewidth=1.5, linestyle="--",
            )
        )
        ax.text(
            udCenter[0], udCenter[1], "UG Desert",
            ha="center", va="center", fontweight="bold", color="black", fontsize=7,
        )

        # Mushroom biome (underground, cavern layer)
        mx = layout["mushroom"]
        my = layout["mushroomY"]
        mushW = 200
        mushH = 120
        ax.add_patch(
            Ellipse(
                (mx, my), mushW, mushH,
                facecolor=colors["mushroom"], alpha=0.7,
            )
        )
        ax.text(
            mx, my, "Mushroom",
            ha="center", va="center", fontweight="bold", color="white", fontsize=7,
        )

        # Marble caves (show a sample -- too many to show all)
        for mc in layout["marbleCaves"][:6]:
            ax.add_patch(
                Ellipse(
                    mc, 40, 20,
                    facecolor=colors["marble"], alpha=0.6,
                    edgecolor="gray", linewidth=0.5,
                )
            )

        # Granite caves (show a sample)
        for gc in layout["graniteCaves"][:6]:
            ax.add_patch(
                Ellipse(
                    gc, 40, 20,
                    facecolor=colors["granite"], alpha=0.6,
                    edgecolor="gray", linewidth=0.5,
                )
            )

        # Dungeon
        dungeonX, dungeonY = layout["dungeon"]
        ax.add_patch(
            Rectangle(
                (dungeonX - 50, dungeonY - 50), 100, 300,
                facecolor=colors["dungeon"], alpha=0.9,
            )
        )
        ax.text(
            dungeonX, dungeonY + 50, "D",
            ha="center", va="center", fontweight="bold", color="white", fontsize=12,
        )

        # Spawn point
        spawnX, spawnY = layout["spawn"]
        ax.add_patch(
            Circle(
                (spawnX, spawnY), 30,
                facecolor=colors["spawn"], edgecolor="black", linewidth=2,
            )
        )
        ax.text(
            spawnX, spawnY, "S",
            ha="center", va="center", fontweight="bold", color="black", fontsize=10,
        )

        # Floating islands (exactly 6)
        for isX, isY in layout["floatingIslands"]:
            ax.add_patch(
                Ellipse(
                    (isX, isY), 70, 22,
                    facecolor=colors["floatingIsland"], alpha=0.8,
                    edgecolor="white", linewidth=0.5,
                )
            )

        # Axes config
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.invert_yaxis()
        ax.set_title(
            f"Large World #{i + 1} ({width} x {height}) | "
            f"Dungeon: {layout['dungeonSide'].title()}, "
            f"Evil: {layout['evilType'].title()} "
            f"(independent placement)",
            fontsize=13, fontweight="bold", pad=12,
        )
        ax.set_xlabel("X (tiles)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Y (tiles)", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.set_facecolor(COLORS["axes"])

        # Rules text on first subplot only
        if i == 0:
            rulesText = (
                "Large World Generation Rules:\n"
                "  Jungle always opposite Dungeon side\n"
                "  Snow biome on same side as Dungeon\n"
                "  Evil biome placed independently\n"
                "  Spawn near world center\n"
                "  Oceans at both edges\n"
                "  1 surface Desert + 1 Underground Desert\n"
                "  6 Floating Islands in sky layer\n"
                "  Marble caves: 16-32, Granite: similar\n"
                "  Underground Mushroom biome (cavern)\n"
                "  45-tile border buffer on all edges"
            )
            ax.text(
                0.02, 0.98, rulesText, transform=ax.transAxes,
                fontsize=9, va="top", ha="left", fontweight="bold",
                fontfamily="monospace",
                bbox=dict(
                    boxstyle="round,pad=0.5", facecolor=COLORS["legend_bg"],
                    alpha=0.92, edgecolor="navy", linewidth=2,
                ),
            )

    # Legend
    legendElements = [
        patches.Patch(color=colors["forest"], alpha=0.5, label="Forest"),
        patches.Patch(color=colors["jungle"], alpha=0.7, label="Jungle"),
        patches.Patch(color=colors["desert"], alpha=0.7, label="Surface Desert"),
        patches.Patch(color=colors["undergroundDesert"], alpha=0.6, label="UG Desert"),
        patches.Patch(color=colors["snow"], alpha=0.7, label="Snow"),
        patches.Patch(color=colors["corruption"], alpha=0.7, label="Corruption"),
        patches.Patch(color=colors["crimson"], alpha=0.7, label="Crimson"),
        patches.Patch(color=colors["ocean"], alpha=0.7, label="Ocean"),
        patches.Patch(color=colors["dungeon"], alpha=0.7, label="Dungeon"),
        patches.Patch(color=colors["spawn"], alpha=0.7, label="Spawn"),
        patches.Patch(color=colors["floatingIsland"], alpha=0.7, label="Sky Island"),
        patches.Patch(color=colors["marble"], alpha=0.7, label="Marble"),
        patches.Patch(color=colors["granite"], alpha=0.7, label="Granite"),
        patches.Patch(color=colors["mushroom"], alpha=0.7, label="Mushroom"),
        patches.Patch(color=colors["hell"], alpha=0.5, label="Underworld"),
    ]
    fig.legend(
        handles=legendElements, loc="center right",
        bbox_to_anchor=(0.99, 0.5), ncol=1, fontsize=9,
    )

    plt.tight_layout()
    plt.subplots_adjust(right=0.84)
    plt.savefig(savePath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Biome layout visualization saved to {savePath}")


# ===================================================================
# Detail panel: sprite-rendered biome transition (Forest -> Jungle -> Desert)
# ===================================================================
def createBiomeTransitionDetail(savePath: str) -> None:
    """DETAIL_PLOT (600x400) sprite render of a 3-biome surface transition.

    Replaces the cut 200-world statistics figure with a tile-scale view of
    how Terraria biome converters swap base materials (Dirt -> Mud / Sand)
    while preserving topography across hard boundaries.
    """
    print("Creating biome transition detail (sprite render)...")
    width = DETAIL_PLOT.width   # 600
    height = DETAIL_PLOT.height  # 400
    rng = np.random.default_rng(seed=20250423)

    grid = np.full((height, width), STONE, dtype=np.int32)
    surfaceY = int(height * 0.18)
    dirtBottom = int(height * 0.42)

    # Wavy surface for visual interest.
    xs = np.arange(width)
    surfaceWave = surfaceY + (4 * np.sin(xs / 18.0) + 2 * np.cos(xs / 7.0)).astype(int)
    for x in range(width):
        ys = int(surfaceWave[x])
        grid[:ys, x] = AIR
        grid[ys, x] = GRASS
        grid[ys + 1:dirtBottom, x] = DIRT

    # A handful of caves so the conversion is visible underground.
    from Engine.algorithms import tileRunner
    for _ in range(40):
        sx = int(rng.integers(10, width - 10))
        sy = int(rng.integers(surfaceY + 12, height - 20))
        tileRunner(grid, sx, sy, float(rng.uniform(4.0, 9.0)),
                   int(rng.integers(20, 60)), tileType=-1)

    # Three biome bands with hard tile-type conversion boundaries.
    b1, b2 = width // 3, 2 * width // 3
    # Jungle middle band: Dirt -> Mud, Grass -> Mud.
    for j in range(height):
        for i in range(b1, b2):
            if grid[j, i] == DIRT or grid[j, i] == GRASS:
                grid[j, i] = MUD
    # Desert right band: Dirt -> Sand, Grass -> Sand, Stone -> EBONSTONE for contrast.
    for j in range(height):
        for i in range(b2, width):
            if grid[j, i] == DIRT or grid[j, i] == GRASS:
                grid[j, i] = SAND
            elif grid[j, i] == STONE and j > dirtBottom + 20:
                grid[j, i] = EBONSTONE

    fig, ax = plt.subplots(figsize=(12, 6))
    drawTileGrid(ax, grid)

    for bx in (b1, b2):
        ax.axvline(bx, color=PALETTE["fg"], linestyle="--",
                   linewidth=1.0, alpha=0.7)

    for mid, label, color in [
        (b1 // 2, "Forest", PALETTE["green"]),
        ((b1 + b2) // 2, "Jungle", "#73daca"),
        ((b2 + width) // 2, "Desert", PALETTE["yellow"]),
    ]:
        ax.text(mid, surfaceY * 0.45, label, ha="center", va="center",
                fontweight="bold", fontsize=10, color=PALETTE["bg"],
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor=color, alpha=0.9))

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_xlabel("X (tiles)", fontweight="bold")
    ax.set_ylabel("Depth (tiles)", fontweight="bold")
    ax.set_title("Biome Transition Detail (DETAIL_PLOT sprite render)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(savePath, dpi=200, bbox_inches="tight",
                facecolor=COLORS["bg"])
    plt.close()
    print(f"Biome transition detail saved to {savePath}")


if __name__ == "__main__":
    print("Starting Terraria biome distribution analysis")

    outputDir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Plots"
    )
    os.makedirs(outputDir, exist_ok=True)

    try:
        createBiomeLayoutVisualization(
            os.path.join(outputDir, "terraria_biome_layouts.png")
        )
        createBiomeTransitionDetail(
            os.path.join(outputDir, "terraria_biome_transition_detail.png")
        )
        print("All biome analysis visualizations complete.")
    except Exception as e:
        print(f"Error: {e}")
        raise
