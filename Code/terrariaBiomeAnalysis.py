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

from Engine.constants import LARGE, LayerDepths, StructureQuotas
from Engine.theme import applyDarkTheme, COLORS, BIOME_COLORS

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle, Ellipse
import seaborn as sns

applyDarkTheme()


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
    rng = np.random.RandomState(seed)

    width = LARGE.width
    height = LARGE.height
    buffer = LARGE.borderBuffer  # 45 tiles
    layers = LayerDepths.forLarge()
    quotas = StructureQuotas.forLarge()

    # Spawn near world center
    spawnX = width // 2 + rng.randint(-100, 100)

    # Dungeon side (50/50)
    dungeonSide = rng.choice(["left", "right"])

    # Dungeon, Jungle, Snow placement based on dungeonX polarity
    if dungeonSide == "left":
        dungeonX = rng.randint(buffer, width // 4)
        jungleX = rng.randint(3 * width // 4, width - buffer)
        snowX = rng.randint(buffer, width // 3)
    else:
        dungeonX = rng.randint(3 * width // 4, width - buffer)
        jungleX = rng.randint(buffer, width // 4)
        snowX = rng.randint(2 * width // 3, width - buffer)

    # Evil biome: placed independently of dungeon side
    evilType = rng.choice(["corruption", "crimson"])
    evilX = rng.randint(buffer + 200, width - buffer - 200)

    # 1 surface desert (not 3)
    desertX = rng.randint(buffer + 300, width - buffer - 300)
    # Avoid overlapping with spawn area
    while abs(desertX - spawnX) < 400:
        desertX = rng.randint(buffer + 300, width - buffer - 300)

    # 1 Underground Desert (circular ant-hive beneath surface desert)
    undergroundDesertCenter = (desertX, int(layers.rockLayer + 100))
    undergroundDesertRadius = rng.randint(150, 250)

    # Oceans at edges (within buffer region)
    oceanLeftX = 0
    oceanRightX = width

    # Floating islands: exactly 6 for large world
    numIslands = quotas.floatingIslands  # 6
    islandPositions = []
    usedIslandX = set()
    for _ in range(numIslands):
        attempts = 0
        while attempts < 100:
            ix = rng.randint(buffer + 200, width - buffer - 200)
            # Ensure minimum spacing between islands
            tooClose = any(abs(ix - ux) < 300 for ux in usedIslandX)
            if not tooClose:
                break
            attempts += 1
        iy = rng.randint(100, int(layers.worldSurface * 0.4))
        islandPositions.append((ix, iy))
        usedIslandX.add(ix)

    # Marble caves: 16-32 for large world
    numMarble = rng.randint(quotas.marbleCavesMin, quotas.marbleCavesMax + 1)
    marblePositions = []
    for _ in range(numMarble):
        mx = rng.randint(buffer, width - buffer)
        my = rng.randint(int(layers.rockLayer), int(layers.hellLayer - 50))
        marblePositions.append((mx, my))

    # Granite caves: similar count to marble (game uses comparable formula)
    numGranite = rng.randint(quotas.marbleCavesMin, quotas.marbleCavesMax + 1)
    granitePositions = []
    for _ in range(numGranite):
        gx = rng.randint(buffer, width - buffer)
        gy = rng.randint(int(layers.rockLayer), int(layers.hellLayer - 50))
        granitePositions.append((gx, gy))

    # Surface Mushroom biome (placed in cavern layer, surfaces at mud patches)
    mushroomX = rng.randint(buffer + 500, width - buffer - 500)
    # Avoid overlap with jungle
    while abs(mushroomX - jungleX) < 600:
        mushroomX = rng.randint(buffer + 500, width - buffer - 500)
    mushroomY = int(layers.rockLayer + rng.randint(50, 200))

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


def createBiomeStatisticsVisualization(savePath: str) -> None:
    """Create statistical analysis of biome distributions from many large-world samples."""
    print("Creating biome statistics visualization for large worlds...")

    numSamples = 200
    quotas = StructureQuotas.forLarge()

    stats = {
        "jungleDistances": [],
        "dungeonDistances": [],
        "evilDistances": [],
        "snowDistances": [],
        "desertCounts": [],
        "islandCounts": [],
        "marbleCounts": [],
        "graniteCounts": [],
        "dungeonSides": {"left": 0, "right": 0},
        "evilTypes": {"corruption": 0, "crimson": 0},
        "biomeSpacings": [],
        "worldBalanceScores": [],
        "evilOnDungeonSide": 0,
    }

    for i in range(numSamples):
        layout = generateWorldLayout(seed=i * 37)
        width = layout["dimensions"][0]
        spawnX = layout["spawn"][0]
        center = width // 2

        # Normalized distances from spawn
        stats["jungleDistances"].append(abs(layout["jungle"][0] - spawnX) / width)
        stats["dungeonDistances"].append(abs(layout["dungeon"][0] - spawnX) / width)
        stats["evilDistances"].append(abs(layout["evil"][0] - spawnX) / width)
        stats["snowDistances"].append(abs(layout["snow"][0] - spawnX) / width)

        # Counts (fixed: 1 desert, 6 islands)
        stats["desertCounts"].append(1)
        stats["islandCounts"].append(len(layout["floatingIslands"]))
        stats["marbleCounts"].append(layout["numMarble"])
        stats["graniteCounts"].append(layout["numGranite"])

        stats["dungeonSides"][layout["dungeonSide"]] += 1
        stats["evilTypes"][layout["evilType"]] += 1

        # Track whether evil ended up on dungeon side (should be ~50/50)
        evilSide = "left" if layout["evil"][0] < center else "right"
        if evilSide == layout["dungeonSide"]:
            stats["evilOnDungeonSide"] += 1

        # Average spacing between major biomes
        majorBiomes = [
            layout["jungle"][0], layout["dungeon"][0],
            layout["evil"][0], layout["snow"][0],
        ]
        spacings = []
        for j in range(len(majorBiomes)):
            for k in range(j + 1, len(majorBiomes)):
                spacings.append(abs(majorBiomes[j] - majorBiomes[k]) / width)
        stats["biomeSpacings"].append(np.mean(spacings))

        # Balance score
        positions = sorted(majorBiomes)
        expectedSpacing = width / 5
        actualSpacings = [positions[m + 1] - positions[m] for m in range(len(positions) - 1)]
        balanceScore = 1 - np.std(actualSpacings) / expectedSpacing
        stats["worldBalanceScores"].append(max(0.0, balanceScore))

    # ---- Visualization ----
    fig = plt.figure(figsize=(20, 18))
    fig.suptitle(
        "Terraria Large World Biome Statistics\n"
        f"Distribution Patterns from {numSamples} Generated Worlds",
        fontsize=18, fontweight="bold", y=0.98,
    )

    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)
    palette = sns.color_palette("husl", 8)

    # Row 0: Violin plot of biome distances
    ax1 = fig.add_subplot(gs[0, :])
    distances = [
        stats["jungleDistances"], stats["dungeonDistances"],
        stats["evilDistances"], stats["snowDistances"],
    ]
    labels = ["Jungle", "Dungeon", "Evil Biome", "Snow"]
    violinColors = [palette[0], palette[1], palette[2], palette[3]]

    parts = ax1.violinplot(distances, positions=range(1, 5), showmeans=True, showmedians=True)
    for idx, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(violinColors[idx])
        pc.set_alpha(0.7)
    ax1.set_xlabel("Biome Type", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Distance from Spawn (normalized)", fontsize=12, fontweight="bold")
    ax1.set_title("Biome Distance Distributions from Spawn Point", fontsize=14, fontweight="bold")
    ax1.set_xticks(range(1, 5))
    ax1.set_xticklabels(labels)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor(COLORS["axes"])

    # Row 1 col 0: Island count (should be all 6)
    ax2 = fig.add_subplot(gs[1, 0])
    islandArr = np.array(stats["islandCounts"])
    islandBins = np.bincount(islandArr, minlength=8)
    ax2.bar(range(len(islandBins)), islandBins, color=palette[5], alpha=0.8)
    ax2.set_title(f"Floating Island Count (always {quotas.floatingIslands})", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Number of Islands")
    ax2.set_ylabel("Frequency")
    ax2.grid(True, alpha=0.3)

    # Row 1 col 1: Marble cave count distribution
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(stats["marbleCounts"], bins=range(
        quotas.marbleCavesMin, quotas.marbleCavesMax + 2),
        color="#F5F5F5", edgecolor="gray", alpha=0.9,
    )
    ax3.set_title(f"Marble Cave Count ({quotas.marbleCavesMin}-{quotas.marbleCavesMax})", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Count")
    ax3.set_ylabel("Frequency")
    ax3.grid(True, alpha=0.3)

    # Row 1 col 2: Granite cave count distribution
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.hist(stats["graniteCounts"], bins=range(
        quotas.marbleCavesMin, quotas.marbleCavesMax + 2),
        color="#2F4F4F", edgecolor="gray", alpha=0.7,
    )
    ax4.set_title("Granite Cave Count", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Count")
    ax4.set_ylabel("Frequency")
    ax4.grid(True, alpha=0.3)

    # Row 2 col 0: Dungeon side distribution
    ax5 = fig.add_subplot(gs[2, 0])
    dungeonData = list(stats["dungeonSides"].values())
    ax5.pie(
        dungeonData, labels=["Left", "Right"], autopct="%1.1f%%",
        colors=[palette[6], palette[7]], startangle=90,
    )
    ax5.set_title("Dungeon Side Distribution", fontsize=12, fontweight="bold")

    # Row 2 col 1: Evil biome type
    ax6 = fig.add_subplot(gs[2, 1])
    evilData = list(stats["evilTypes"].values())
    ax6.pie(
        evilData, labels=["Corruption", "Crimson"], autopct="%1.1f%%",
        colors=["#9370DB", "#DC143C"], startangle=90,
    )
    ax6.set_title("Evil Biome Type Distribution", fontsize=12, fontweight="bold")

    # Row 2 col 2: Evil biome side independence
    ax7 = fig.add_subplot(gs[2, 2])
    evilDungeonPct = stats["evilOnDungeonSide"] / numSamples * 100
    evilOppositePct = 100 - evilDungeonPct
    ax7.bar(
        ["Same as Dungeon", "Opposite"],
        [evilDungeonPct, evilOppositePct],
        color=[palette[2], palette[4]], alpha=0.8,
    )
    ax7.set_title("Evil vs Dungeon Side (independent)", fontsize=12, fontweight="bold")
    ax7.set_ylabel("Percentage")
    ax7.set_ylim(0, 100)
    ax7.axhline(y=50, color="red", linestyle="--", alpha=0.5, label="50% (expected)")
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # Row 3: Correlation matrix
    ax8 = fig.add_subplot(gs[3, :])
    corrData = np.array([
        stats["jungleDistances"],
        stats["dungeonDistances"],
        stats["evilDistances"],
        stats["snowDistances"],
        stats["biomeSpacings"],
        stats["worldBalanceScores"],
    ])
    corrMatrix = np.corrcoef(corrData)
    corrLabels = [
        "Jungle Dist", "Dungeon Dist", "Evil Dist",
        "Snow Dist", "Avg Spacing", "Balance",
    ]

    im = ax8.imshow(corrMatrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax8.set_xticks(range(len(corrLabels)))
    ax8.set_yticks(range(len(corrLabels)))
    ax8.set_xticklabels(corrLabels, rotation=45, ha="right")
    ax8.set_yticklabels(corrLabels)
    ax8.set_title("Biome Parameter Correlation Matrix", fontsize=14, fontweight="bold")

    for ci in range(len(corrLabels)):
        for cj in range(len(corrLabels)):
            textColor = "black" if abs(corrMatrix[ci, cj]) < 0.5 else "white"
            ax8.text(
                cj, ci, f"{corrMatrix[ci, cj]:.2f}",
                ha="center", va="center", color=textColor,
            )

    cbar = fig.colorbar(im, ax=ax8, orientation="horizontal", pad=0.12, shrink=0.8)
    cbar.set_label("Correlation Coefficient", fontsize=12)

    # Summary text
    summaryText = (
        f"Statistical Summary (n={numSamples}):\n"
        f"  Jungle dist: {np.mean(stats['jungleDistances']):.3f} +/- {np.std(stats['jungleDistances']):.3f}\n"
        f"  Dungeon dist: {np.mean(stats['dungeonDistances']):.3f} +/- {np.std(stats['dungeonDistances']):.3f}\n"
        f"  Desert count: always 1 (+ 1 Underground Desert)\n"
        f"  Island count: always {quotas.floatingIslands}\n"
        f"  Marble caves: {np.mean(stats['marbleCounts']):.1f} +/- {np.std(stats['marbleCounts']):.1f}\n"
        f"  Evil on dungeon side: {evilDungeonPct:.1f}% (independent)\n"
        f"  Dungeon side: L={stats['dungeonSides']['left']}, R={stats['dungeonSides']['right']}"
    )
    fig.text(
        0.02, 0.13, summaryText, fontsize=10, ha="left", va="top",
        fontfamily="monospace",
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor=COLORS["legend_bg"],
            alpha=0.8, edgecolor="navy", linewidth=2,
        ),
    )

    plt.tight_layout(rect=[0, 0.18, 1, 0.96])
    plt.savefig(savePath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Biome statistics visualization saved to {savePath}")


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
        createBiomeStatisticsVisualization(
            os.path.join(outputDir, "terraria_biome_statistics.png")
        )
        print("All biome analysis visualizations complete.")
    except Exception as e:
        print(f"Error: {e}")
        raise
