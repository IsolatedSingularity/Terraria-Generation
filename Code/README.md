# Code

> **Legacy research archive.** New generation, rendering, tests, and CLI/GUI
> work lives in `terraexplorer/`. These plots remain useful teaching aids, but
> their older Terraria-fidelity wording should not be read as source parity.

Static visualizations of Terraria worldgen primitives. Each script is a
standalone entry point; outputs land in `../Plots/`.

## Modules

### terrariaBiomeAnalysis.py

Two TINY-world figures.

- `createBiomeLayoutVisualization(savePath)`. Full 240x140 render of a
  TINY world showing all five surface biomes (Forest, Snow, Jungle,
  Desert, Corruption) plus layer markers (worldSurface, rockLayer,
  hellLayer). Title: "Biome Distribution".
- `createBiomeTransitionDetail(savePath)`. Second TINY render with a
  different seed emphasizing biome transitions and converter behavior at
  the boundaries. Title: "Biome Transitions".

Outputs: `Plots/terraria_biome_layouts.png`,
`Plots/terraria_biome_transition_detail.png`.

### terrariaNoiseSystems.py

Three theory plots that explain the underlying algorithms in isolation.
These are intentionally not TINY world renders; they exist to teach the
math.

- `createSurfaceTerrainVisualization(savePath)`. Multi-octave 1D
  fractional Brownian noise: shows per-octave waves, the composite
  height, and the resulting tile grid for four surface biomes.
- `createCaveSystemVisualization(savePath)`. TileRunner cave carving
  with depth-dependent strength, plus a before/after CA smoothing pair.
- `createBiomeTileConversionVisualization(savePath)`. Side-by-side
  before/after grids showing how Snow, Jungle, Corruption, and Crimson
  converters swap base tiles at hard boundaries.

Outputs: `Plots/terraria_surface_terrain.png`,
`Plots/terraria_cave_systems.png`,
`Plots/terraria_biome_tile_conversion.png`.

### terrariaOreDistribution.py

Two figures.

- `createOreDistributionFigure(savePath)`. Three TINY worlds rendered
  with `oreScale=10.0` so vein placements are clearly visible. Each
  panel highlights one tier (Pre-Hardmode, Hardmode Tier 1, Hardmode
  Tier 3) and dims the rest of the world.
- `createOreDensityFigure(savePath)`. Heatmap from a SMALL-world ore
  count sample: 10 depth bins x 15 ore types, log1p scaled, annotated
  non-zero cells, Tokyo Night colormap.

Outputs: `Plots/ore_distribution.png`, `Plots/ore_density.png`.

## Usage

```bash
python Code/terrariaBiomeAnalysis.py
python Code/terrariaNoiseSystems.py
python Code/terrariaOreDistribution.py
```

All scripts use fixed seeds defined at the top of each file. Re-running
produces byte-identical PNGs.

## Conventions

- camelCase for all user-authored functions and variables.
- Tokyo Night Storm chrome (axes, panels, text) via
  `Engine.theme.applyTokyoNight()`.
- Game-semantic colors stay Terraria-accurate (`Engine.theme.TILE_COLORS`,
  `Engine.theme.BIOME_COLORS`).
- No tile-dimension suffixes or "(crop)" markers in titles.
