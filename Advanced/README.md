# Advanced

> **Legacy research archive.** The supported 107-pass implementation and its
> real milestone GIF now live in `terraforge/` and `docs/media/`. These scripts
> are retained for historical simulation experiments only.

Long-running simulations and animated visualizations of Terraria world
evolution. Every script renders the full TINY world (240x140 tiles) at
native resolution; no cropping, no downsampling.

## Modules

### terrariaWorldGeneration.py

Replays the world generation pipeline pass by pass and saves a
palette-quantized GIF.

Passes captured: Reset, Surface and Strata, Stone and Hellstone Shell,
Caves Carved, CA Smoothing, Snow and Jungle, Desert, Corruption,
Pre-Hardmode Ores, three Hardmode V-Pattern frames, three Altar Tier
frames, Final World.

Output: `Plots/Advanced/world_generation_animation.gif`

### terrariaCorruptionEvolution.py

Module-level helpers and two outputs:

- `carveVPattern(grid, evilType, seed)`. Diagonal V carving from surface to
  hell layer with 30 TileRunner passes. Evil + Hallow sides are randomly
  swapped per call.
- `spreadInfection(grid, cycles, seed)`. Stochastic CA spread bounded by
  `INFECTION_SPREAD_RADIUS` and `INFECTION_GAP_TILES` from
  `Engine.constants`.
- `buildEvolutionSnapshots(evilType, seed)`. Returns four progressive grids:
  Pre-Hardmode, V-Pattern, Early Spread, Late Spread.
- `createEvolutionFigure(...)`. 4-panel TINY-world figure.
- `createSpreadAnimation(...)`. 30-frame spread GIF.

Outputs: `Plots/Advanced/corruption_evolution.png`,
`Plots/Advanced/corruption_spread.gif`.

### terrariaCrimsonEvolution.py

Thin wrapper that invokes `createEvolutionFigure` and
`createSpreadAnimation` with `evilType="crimson"`. Identical mechanics,
distinct tile palette.

Outputs: `Plots/Advanced/crimson_evolution.png`,
`Plots/Advanced/crimson_spread.gif`.

### terrariaHardmodeStructures.py

Three-panel TINY figure showing pre-hardmode, V-pattern, and
post-altar-x9 ore distribution. Suptitle: "Hardmode Transformation".

Output: `Plots/Advanced/terraria_hardmode_transformation.png`.

### terrariaHardmodeDetailedAnimation.py

~30-frame GIF of the Hardmode transition: baseline holds, six V-carve
increments, six altar-tier overlays, four final holds.

Output: `Plots/Advanced/terraria_hardmode_animation.gif`.

### terrariaMasterEvolution.py

25-frame hero lifecycle GIF. Bare stone shell, surface and strata, caves,
biomes, pre-Hardmode ores, V-pattern reveal, three altar tiers, late
infection spread.

Output: `Plots/Advanced/terraria_master_evolution.gif`.

## GIF rendering pipeline

All animations use `Engine.theme.saveTinyGif(frames, savePath, fps, scale,
title)`, which writes palette-quantized GIFs via PIL. Frames are upscaled
nearest-neighbor (preserves the pixel-art look). Per-frame titles are
supported by passing a list to `title`.

Typical sizes after this pipeline:

| GIF | Frames | Size |
|---|---|---|
| `world_generation_animation.gif` | 16 | ~500 KB |
| `terraria_hardmode_animation.gif` | ~30 | ~450 KB |
| `terraria_master_evolution.gif` | 25 | ~650 KB |
| `corruption_spread.gif` | 31 | ~1.3 MB |
| `crimson_spread.gif` | 31 | ~1.3 MB |

## Usage

```bash
python Advanced/terrariaWorldGeneration.py
python Advanced/terrariaCorruptionEvolution.py
python Advanced/terrariaCrimsonEvolution.py
python Advanced/terrariaHardmodeStructures.py
python Advanced/terrariaHardmodeDetailedAnimation.py
python Advanced/terrariaMasterEvolution.py
```

Each script is a standalone entry point. All are reproducible at fixed
seeds (seeds live next to each `__main__` block).

## Theory references

The supported package design is documented in
[`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md), and implementation fidelity
is tracked in [`docs/FIDELITY.md`](../docs/FIDELITY.md). Legacy TINY-world
layer depths remain defined in [`Engine/constants.py`](../Engine/constants.py)
as `LayerDepths.forTiny()` (`worldSurface=28`, `rockLayer=70`,
`hellLayer=125`, `maxTilesY=140`).
