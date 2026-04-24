# Terraria World Generation

[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.2-013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/scipy-1.15-8CAAE6.svg?style=for-the-badge&logo=scipy&logoColor=white)](https://scipy.org/)
[![Matplotlib](https://img.shields.io/badge/matplotlib-3.10-11557C.svg?style=for-the-badge&logo=plotly&logoColor=white)](https://matplotlib.org/)

![Master Evolution](Plots/Advanced/terraria_master_evolution.gif)

## Overview

A faithful Python reimplementation of Terraria's **103-pass world generation pipeline**, derived from decompiled `WorldGen.cs` source analysis. The project provides an `Engine/` core library of game-accurate algorithms and constants, plus visualization scripts that produce dark-themed publication-quality plots and animations.

Key algorithms reproduced:

- **TileRunner** diamond-brush random walk (cave carving, ore veins, biome conversion)
- **Cellular automata smoothing** for organic cave edges
- **Gravity-based liquid settling** with water/lava/honey interaction rules
- **Tile update cycle infection spread** with surface vs underground rates and air gap blocking
- **StructureMap exclusion zones** for conflict-free placement
- **Dungeon eating algorithm** with interlocking rooms

## Visualizations

### Surface Terrain

![Surface Terrain](Plots/terraria_surface_terrain.png)

Multi-octave sine noise terrain generation showing individual frequency components, composite waveform, and tile-grid rendering. Each biome type uses distinct noise parameters.

### Cave Systems

![Cave Systems](Plots/terraria_cave_systems.png)

TileRunner-based cave carving with depth-dependent density. Surface caves use smaller strength (4-8) and fewer steps; rock-layer caves use strength 10-22 with up to 100 steps. Cellular automata smoothing rounds cave edges.

### Biome Distribution

![Biome Layouts](Plots/terraria_biome_layouts.png)

Large world biome placement following game rules:

- Jungle always opposite the Dungeon side
- Snow biome on the same side as the Dungeon
- Evil biome placed independently (either hemisphere)
- 1 surface Desert + 1 Underground Desert (circular ant-hive)
- 6 Floating Islands, 16-32 Marble caves, similar Granite caves
- Underground Mushroom biome in the cavern layer
- 45-tile border buffer on all edges

![Biome Transition Detail](Plots/terraria_biome_transition_detail.png)

Native 240x140 TINY world rendered at full resolution. Shows the five surface biomes (Forest, Snow, Jungle, Desert, Corruption) with caves carved through every band so the converters' tile swaps (Dirt to Mud or Sand, Stone to Ebonstone) are clearly visible across hard boundaries.

### Biome Tile Conversion

![Tile Conversion](Plots/terraria_biome_tile_conversion.png)

Side-by-side tile-grid visualization of biome conversion rules: Snow (dirt to snow/ice), Jungle (dirt to mud), Corruption (stone to ebonstone, dirt to corrupt dirt).

### Ore Distribution

![Ore Distribution](Plots/ore_distribution.png)

Three TINY worlds rendered at full resolution. Pre-Hardmode picks alternating pairs (Copper or Tin, Iron or Lead, Silver or Tungsten, Gold or Platinum). Hardmode Tier 1 (Cobalt, Palladium) appears after smashing the first three altars. Hardmode Tier 3 (Adamantite, Titanium, Chlorophyte) settles in the deep cavern layer. Each tier panel highlights its ores while dimming the rest of the world.

![Ore Density](Plots/ore_density.png)

Depth-binned heatmap showing ore counts across 10 depth slices and 15 ore types from a SMALL world sample. Cells use a log-scaled Tokyo Night colormap with non-zero counts annotated.


## Advanced Simulations

### Generation Pipeline

![World Generation Animation](Plots/Advanced/world_generation_animation.gif)

Native TINY world (240x140) replayed pass by pass. Each frame is the full world rendered at ~6 px/tile: bare stone shell, surface and strata, cave carving, CA smoothing, biome painting (Snow, Jungle, Desert, Corruption), pre-Hardmode ores, V-pattern, and three altar tiers. The frame title names the active pass.

### Corruption/Crimson/Hallow Evolution

![Corruption Evolution](Plots/Advanced/corruption_evolution.png)

Four TINY world snapshots showing the corruption lifecycle: Pre-Hardmode evil pocket, V-pattern diagonal carving from the Wall of Flesh event, early infection spread, and late-stage saturation. Each panel is a full 240x140 render so the diagonals and infection halos are unambiguous.

![Corruption Spread](Plots/Advanced/corruption_spread.gif)

![Crimson Evolution](Plots/Advanced/crimson_evolution.png)

Parallel simulation for the Crimson biome variant. Identical TileRunner V-pattern and `INFECTION_SPREAD_RADIUS` rules, but the converter swaps to Crimson tile IDs (Crimstone, Crimson grass, flesh blocks).

### Hardmode Transformation

![Hardmode Transformation](Plots/Advanced/terraria_hardmode_transformation.png)

Detailed hardmode mechanics: 3-cycle ore generation (Cobalt/Palladium tier 1, Mythril/Orichalcum tier 2, Adamantite/Titanium tier 3), Chlorophyte growth in jungle cavern, and biome conversion visualization.

![Hardmode Animation](Plots/Advanced/terraria_hardmode_animation.gif)

### World Evolution

![Master Evolution](Plots/Advanced/terraria_master_evolution.gif)

Complete TINY-world lifecycle hero animation. Frames trace the full progression: bare stone, surface and caves, biome painting, pre-Hardmode ores, V-pattern reveal, three altar tiers, and late-stage infection spread. Every frame is the full 240x140 world.

## Architecture

```
Engine/                              # Core library
    __init__.py
    algorithms.py                    # tileRunner, digTunnel, cavinator (all accept seed=), cellularAutomataSmooth, settleLiquids
    constants.py                     # WorldSize (TINY/SMALL/MEDIUM/LARGE/FEATURE_PLOT/DETAIL_PLOT), LayerDepths.forTiny/forSmall/forLarge, StructureQuotas, OreConfig, tile/wall IDs
    structures.py                    # 12 structure generators + 8 placement passes, vectorized spreadGrass
    structureMap.py                  # Rectangle + StructureMap exclusion zones
    spriteRenderer.py                # Crisp pixel-tile rendering + structure composers (drawDungeon/drawCabin/drawFloatingIsland/drawPyramid/...)
    worldgen.py                      # generateSmallWorld + generateMiniWorld (TINY 240x140), renderMiniWorld
    theme.py                         # Tokyo Night Storm PALETTE, COLORS, BIOME_COLORS, TILE_COLORS, ORE_COLORS, buildTileColormap, applyTokyoNight, saveTinyGif
Code/                                # Static visualizations
    terrariaBiomeAnalysis.py         # Two TINY-world biome figures
    terrariaNoiseSystems.py          # Surface terrain noise theory + cave systems + tile conversion
    terrariaOreDistribution.py       # 3 TINY-world ore tier panels + SMALL-world heatmap
Advanced/                            # Animations and multi-frame simulations
    __init__.py
    terrariaWorldGeneration.py       # TINY pipeline pass-by-pass GIF
    terrariaCorruptionEvolution.py   # Evolution figure + spread GIF (corruption + crimson core)
    terrariaCrimsonEvolution.py      # Crimson wrapper
    terrariaHardmodeStructures.py    # 3-panel hardmode transformation figure
    terrariaHardmodeDetailedAnimation.py # Hardmode transition GIF
    terrariaMasterEvolution.py       # 25-frame hero lifecycle GIF
Plots/                               # Generated output (tracked so README renders on GitHub)
    Advanced/
References/                          # Research documentation (gitignored)
```

## Setup

```bash
git clone https://github.com/IsolatedSingularity/Terraria-Generation
cd Terraria-Generation
pip install -e .          # editable install (Engine becomes importable)
```

> `pip install -e .` uses the `[build-system]` in `pyproject.toml` to expose the `Engine/` package.

## Theory

### Surface Terrain as Fractional Brownian Motion

Terraria surface heights are synthesized as a sum of sinusoidal octaves, a
discrete approximation of fractional Brownian motion (fBm):

$$h(x) = \sum_{i=0}^{N-1} A_0 \, p^i \sin\!\bigl(2\pi f_0 \, l^i \, x + \phi_i\bigr)$$

where $A_0$ is the base amplitude, $p \in (0, 1)$ is the persistence, $f_0$ is
the base frequency, $l > 1$ is the lacunarity, and $\phi_i$ are independent
uniform random phases. The resulting power spectrum satisfies
$S(f) \propto f^{-\beta}$ with $\beta = 2 + 2\log_2 p$, giving the pink-noise
texture characteristic of natural terrain. In the engine implementation,
$p \approx 0.5$, $l = 2$, and $N = 4$ octaves produce $\beta \approx 2$, a
well-known geomorphology exponent.

### Infection Spread as a Stochastic Cellular Automaton

Each tile update cycle, every infected cell $\mathbf{r}$ attempts to convert
one neighbor $\mathbf{r}' \in B_3(\mathbf{r})$ drawn uniformly at random.
The macroscopic dynamics approximate a stochastic reaction-diffusion equation

$$\partial_t \rho = D\,\nabla^2\rho + f(\rho)$$

where $\rho$ is the infected-tile density, $D$ is an effective diffusivity set
by `INFECTION_SPREAD_RADIUS`, and $f(\rho)$ encodes logistic saturation.
Air-gap blocking maps onto percolation: a contiguous void of width
$w \geq 4$ tiles (one `INFECTION_GAP_TILES` quantum) constitutes a barrier
because the path integral of $\mathbf{1}[\text{air}]$ along any geodesic
exceeds the conduction threshold. In 2-D site percolation, the critical
occupation probability is $p_c \approx 0.593$; a 4-tile air trench forces
the local percolation below $p_c$ and halts spread deterministically.

### Cellular Automata Cave Smoothing

After cavinator carves raw voids, several majority-rule CA passes shape the
caverns into the lacy organic look characteristic of Terraria. Each cell
$s_i \in \{0, 1\}$ (solid / air) updates from its 8-neighbor Moore
neighborhood $\mathcal{N}_8(i)$ via separate birth and death thresholds.

A solid cell survives when it has at least $d = 4$ solid neighbors:

$$s_i^{(t+1)} = 1 \quad \text{if} \quad s_i^{(t)} = 1 \text{ and } n_i^{(t)} \geq 4$$

An air cell becomes solid when more than $b = 5$ neighbors are solid:

$$s_i^{(t+1)} = 1 \quad \text{if} \quad s_i^{(t)} = 0 \text{ and } n_i^{(t)} > 5$$

Otherwise $s_i^{(t+1)} = 0$ (air). Here $n_i^{(t)} = \sum_{j \in \mathcal{N}_8(i)} s_j^{(t)}$ is the solid
neighbor count.

---

## License

MIT License. All Terraria-related content and mechanics are owned by Re-Logic.
