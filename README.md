<p align="center">
  <img src="docs/media/terraexplorer_readme_logo.png" width="180" alt="TerraExplorer mechanical tree">
</p>

<h1 align="center">TerraExplorer</h1>

<p align="center">
  <strong>A deterministic, explorable 2D world-generation laboratory.</strong><br>
  One seed enters. Stone, jungle, ruin, and hunger answer.
</p>

<p align="center">
  <a href="https://github.com/IsolatedSingularity/Terraria-Generation/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/IsolatedSingularity/Terraria-Generation/ci.yml?branch=main&label=CI&logo=github" alt="CI status"></a>
  <img src="https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-3776AB?logo=python&logoColor=white" alt="Python 3.11 through 3.13">
  <img src="https://img.shields.io/badge/worlds-deterministic-d09a45" alt="Deterministic worlds">
</p>

TerraExplorer turns a text seed into a complete tile world that can be watched,
scrubbed through, inspected, rendered, and exported. It is a laboratory built
for the moment before the first torch is placed, when the map is still becoming
itself. Terrain rises. Caverns split. The Dungeon drives downward. The Jungle
closes over its Temple. Far below, lava waits beneath ruined towers.

This is original Python code and original art inspired by public Terraria world
rules. It does not read or write `.wld` files, copy private game code, or ship
Re-Logic sprites.

## A world from one seed

> The machine does not guess twice.

Every named pass receives its own random stream. For text seed (s) and pass
label (p), TerraExplorer derives

$$
s_{32}=\operatorname{CRC32}(s), \qquad
r_p=\operatorname{PCG64}\!\left(\operatorname{uint64}
\left(\operatorname{BLAKE2s}(s_{32}\mathbin{:}p)\right)\right).
$$

The important consequence is practical: editing `Dungeon` cannot silently
reshuffle `Living Trees`, ore veins, or every later pass.

```python
from terraexplorer import Evil, WorldConfig, generate_world

config = WorldConfig(seed="mechanical-tree", evil=Evil.CRIMSON)
world = generate_world(config)
```

The animation records real pipeline snapshots, not painted transitions. Each
frame is a state produced by the same handlers used by the API, CLI, and GUI.

![A TerraExplorer world taking shape](docs/media/terraexplorer_generation.gif)

The final frame pauses so the world can be read: Floating Islands above the
surface, biome wedges below it, minecart tracks cutting through the dark, and
the Underworld holding its line at the bottom. The evolution rail exposes 26
meaningful milestones from the 107-pass run.

## Getting started

> Open the forge. Mind the lava.

Python 3.11 or newer is required. Tkinter is included with standard Windows
Python; some Linux distributions provide it separately as `python3-tk`.

```bash
git clone https://github.com/IsolatedSingularity/Terraria-Generation.git
cd Terraria-Generation
python -m pip install -e .
python -m terraexplorer gui
```

![TerraExplorer desktop workshop](docs/media/gui.png)

The desktop app keeps the whole forge visible. The light-brown native Windows
caption separates the application from the dark map table, while the seed and
world controls remain on the left, the generated world stays central, and the
107-pass log remains readable on the right.

Generation runs outside the Tk event loop and can be cancelled between passes.
The map supports pan, zoom, tile inspection, biome and depth overlays,
previous-world comparison, and PNG, GIF, NPZ, or JSON export. Playback reverses
at either end of the evolution rail so the world can grow and un-grow without a
new generation run.

For a headless run:

```bash
terraexplorer generate --seed "mechanical-tree" --evil crimson --hardmode \
  --png world.png --npz world.npz --json world.json --gif generation.gif
```

Preview worlds are fast `240 x 140` experiments. Small worlds use the much
deeper `4200 x 1200` grid. Windows packaging produces
`dist/TerraExplorer.exe`.

## Idle World Evolution

> Leave the world alone long enough and the borders begin to move.

Hardmode infection checks the square neighborhood within three tiles of a
source. TerraExplorer models that reach as

$$
\mathcal N_3(S)=\{(y+\Delta y,x+\Delta x):
(y,x)\in S,\ |\Delta x|\le3,\ |\Delta y|\le3\}.
$$

The batch probability is weighted to explain the visible difference between
surface and underground advance:

$$
P(\text{conversion})=
\begin{cases}
0.18, & y\le s(x)+4,\\
0.03, & y>s(x)+4.
\end{cases}
$$

That six-to-one ratio represents the separate surface and underground update
rates described by the public biome-spread rules. This remains an educational
batch scheduler, not a claim of frame-perfect game timing.

```python
world.metadata["hardmode"] = True
for _ in range(12):
    advance_biome_spread(world, rng)
```

Each study uses a different seed and a complete generated Preview world. The
panels are now separate, full-width animations, which makes structures, biome
fronts, and deep paths easier to inspect. Natural materials can convert;
Corruption, Crimson, and Hallow cannot overwrite one another, so opposed fronts
meet and harden into a boundary.

![Corruption spreading unchecked](docs/media/idle_corruption.gif)

Corruption begins in narrow violet scars. It moves quickly along exposed soil,
then presses more slowly through Stone, Ice, Sand, and the interrupted geometry
of caves and structures.

![Crimson spreading unchecked](docs/media/idle_crimson.gif)

Crimson begins from a slanted descent and linked deep chambers. The same update
law produces a different history because the seed, starting geometry, and evil
structure are different.

![Corruption and Hallow spreading against each other](docs/media/idle_opposition.gif)

The last world begins after the Hardmode V cuts two diagonal bands through the
Caverns. Corruption and Hallow consume pure hosts until they collide. Neither is
allowed to convert the other's established material.

## Landmark generation

> Every ruin is a decision written into stone.

Many structures begin with clipped geometric stamps. An elliptical mask, for
example, includes a tile when

$$
\left(\frac{x-c_x}{r_x}\right)^2+
\left(\frac{y-c_y}{r_y}\right)^2\le1.
$$

The mask is only a starting primitive. Connected rooms, directed walks,
material replacement rules, walls, liquids, and structure-specific polish turn
it into a Dungeon, Hive, Aether, or chamber rather than a bare oval.

```python
stamp_ellipse(world.tiles, x, y, rx, ry, Tile.HIVE, replace=_CARVABLE)
_place_marker(world, "Jungle temple", x, y, width, height, "T")
```

The gold frames below identify structures while keeping their coast, biome,
depth, and surrounding caves visible.

![TerraExplorer landmark atlas](docs/media/terraexplorer_world.png)

| Above and below | What is modeled |
|---|---|
| Floating Island | Cloud and Rain Cloud foundation, forest cap, and compact sky-brick house |
| Dungeon | Surface entrance connected to branching rooms, corridors, and platforms |
| Pyramid | Mostly buried sandstone shell with a zigzag passage and treasure chamber |
| Aether | Stone cavern in the Jungle-side outer fifth with Shimmer and Gem Trees |
| Jungle Temple | Irregular Lihzahrd-brick shell, connected rooms, traps, and deep altar chamber |
| Ruined House | Individual multi-floor obsidian or Hellstone-brick tower, sometimes flooded by lava |

Layer placement can be read with a normalized depth coordinate. If (s(x)) is
the local surface and (u) is the Underworld boundary, then

$$
d(y,x)=\frac{y-s(x)}{u-s(x)}.
$$

Values near zero sit at the surface; values approaching one descend toward the
Underworld. The overview uses that coordinate only for guides and preserves the
actual generated materials beneath them.

```python
overview = render_world(
    world,
    biome_overlay=True,
    layer_lines=True,
    markers=True,
)
```

![Biome, layer, and landmark overview](docs/media/biome_overview.png)

This is the README's current Small-world overview. The new authoritative visual
references are preserved in
[`docs/references/information`](docs/references/information/README.md). They now
inform the active Corruption, Crimson, Granite, Mushroom, Aether, Hive, Living
Tree, Floating Island, Temple, Dungeon, Underworld, and minecart-track rules.

## Biome generation

> A biome is not a color. It is terrain with a history.

Large biome regions begin from clipped horizontal bands around a seeded center
(c) with half-width (h):

$$
x_0=\max(0,c-h), \qquad x_1=\min(W,c+h).
$$

Inside that band, each biome applies its own vertical profile and material
rules. Snow narrows underground, Jungle replaces deep terrain with Mud, Desert
caps an oval sandstone system with dunes, and the two evils carve different
entrances and chamber networks.

```python
x0, x1 = _band(world, center, half_width)
region = world.tiles[:, x0:x1]
region[np.isin(region, (Tile.STONE, Tile.DIRT))] = evil_stone
```

![Six independently generated biome studies](docs/media/biome_atlas.png)

Each crop comes from a generated world rather than a painted biome swatch.
Forest shows open surface caves; Snow forms a narrowing wedge opposite the
Jungle; Desert places dunes over hardened Sand and Sandstone; Jungle packs Mud
and vines around larger Cavern openings. Corruption cuts steep chasms that join
below, while Crimson descends toward rounded, linked chambers.

## Biome containment simulation

> The Dryad asked for a boundary. The world tested every weakness in it.

Containment uses the same three-tile reach, but samples source tiles with a
surface weight of six:

$$
w(y,x)=
\begin{cases}
6, & y\le s(x)+4,\\
1, & y>s(x)+4,
\end{cases}
\qquad
P(i)=\frac{w_i}{\sum_j w_j}.
$$

```python
results = {
    strategy: simulate_biome_containment(strategy, seed=42)
    for strategy in ContainmentStrategy
}
```

All four panels start from the same generated terrain and deterministic random
stream. The intervention is the independent variable, so infected counts and
protected-side crossings remain comparable.

![Four biome-containment strategies under the same starting conditions](docs/media/containment_lab.gif)

The gold line marks the protected-side boundary. Open ground provides a
baseline; the trench removes convertible tiles; Sunflowers protect the surface;
and Chlorophyte protects a local radius. This is a controlled model of selected
mechanics, not a complete in-game tick scheduler.

## World layers

> Depth is not distance. Depth is what the world permits to survive.

For world height (H), TerraExplorer's default layer boundaries are

$$
y_{surface}=\operatorname{round}(0.19H), \qquad
y_{rock}=\operatorname{round}(0.46H), \qquad
y_{underworld}=H-H_u,
$$

where (H_u=200) for a Small world and approximately (H/6) for a Preview
world.

```python
layers = WorldLayers.for_height(config.height)
print(layers.world_surface, layers.rock_layer, layers.underworld)
```

The descent keeps the full Preview-world width visible while the camera moves
through each depth interval.

![Animated descent from the surface to the Underworld](docs/media/depth_descent.gif)

The horizontal guides are diagnostic boundaries. They do not flatten the local
surface profile or replace the generated terrain. Floating Islands remain above
the surface, structures retain their true placement, and Ruined Houses stand in
the lava-cut Underworld instead of a schematic layer box.

## Generated-world studies

> The plot is the world. The statistic only teaches us where to look.

For a tile region (R), cave density and material frequency are simple counts
over the generated arrays:

$$
D_{cave}(R)=\frac{|\{(y,x)\in R:T(y,x)=\mathrm{AIR}\}|}{|R|},
$$

$$
F_t(R)=\frac{|\{(y,x)\in R:T(y,x)=t\}|}{|R|}.
$$

```python
cave_density = np.count_nonzero(world.tiles[region] == Tile.AIR) / region.size
ore_tiles = np.isin(world.tiles, world.metadata["selected_ore_ids"])
```

The figures keep terrain visible instead of replacing it with bars or a noisy
heat map. The landscape studies expose coast-to-coast relationships; cave crops
show where the density statistic came from; ore studies mark veins at their
actual depths.

![Four generated Preview-world landscape studies](docs/media/surface_profiles.png)

The same layout reveals how Snow, Desert, Jungle, and evil placement alter the
surface profile without reducing the world to one line.

![Generated cave and biome cross-sections](docs/media/cave_density.png)

These exact tile crops make large Jungle openings, narrow surface mouths, and
the denser fractured Cavern layer directly comparable.

![Actual generated ore veins at their world depths](docs/media/ore_depth.png)

Ore alternatives are selected once per seed. Veins then use depth-specific
ranges, so the highlighted material remains embedded in the geometry that
constrained its placement.

## Data model

> The world survives because it remembers each kind of state separately.

```python
from terraexplorer import Evil, WorldConfig, generate_world

world = generate_world(
    WorldConfig(seed="TerraExplorer", evil=Evil.CRIMSON, hardmode=True)
)

print(world.tiles.shape)                 # (140, 240) in Preview mode
print(world.metadata["selected_ores"])

tiles = world.tiles                      # uint8
walls = world.walls                      # uint8
liquid_amount = world.liquid_amount      # uint8
liquid_kind = world.liquid_kind          # uint8
biomes = world.biomes                    # uint8
surface_height = world.surface           # int16
```

Tiles, walls, liquid amount, liquid kind, biomes, and surface height use
independent NumPy buffers. A wall or liquid can therefore coexist with an air
tile, and experiments can copy one state without aliasing another.

```text
WorldConfig
    -> TerraExplorerPipeline
        -> isolated RNG stream for each named pass
            -> tiles + walls + liquids + biomes + metadata + landmarks
                -> desktop app | CLI | PNG/GIF | NPZ/JSON
```

The supported package is `terraexplorer`. The former `terraforge` imports and
console commands remain compatibility aliases, but new code should use the new
name.

## Fidelity and scope

> Accuracy begins by naming the distance between model and world.

TerraExplorer follows the public 107-step Terraria 1.4.4.9 generation order as
a reference, with its own IDs, algorithms, random streams, and art. The current
inventory contains 68 modeled passes, 38 approximated passes, and one documented
pass. `Micro Biomes` is now modeled because it adds long underground minecart
tracks as well as compact gem caves.

Modeled means a distinct operation changes world state or metadata. It does not
mean byte-for-byte compatibility. The
[fidelity inventory](docs/FIDELITY.md) states those boundaries; the
[architecture guide](docs/ARCHITECTURE.md) explains data ownership and extension
rules.

The landscape rules are checked against public Terraria Wiki descriptions of
[world generation](https://terraria.wiki.gg/wiki/World_generation),
[biome spread](https://terraria.wiki.gg/wiki/Biome_spread),
[the Aether](https://terraria.wiki.gg/wiki/The_Aether),
[minecart tracks](https://terraria.wiki.gg/wiki/Minecart_Track), and
[Floating Islands](https://terraria.wiki.gg/wiki/Floating_Island). The supplied
[reference atlas](docs/references/information/README.md) remains authoritative
for this project's current visual correction cycle.

### Open problems

Liquid transfer is conservative but does not model Terraria's complete settling
cadence or pressure behavior. Biome spread is deterministic and mechanically
bounded, but it is still a batch approximation. Secret-seed branches, richer
structure variants, biome-transition microterrain, and independent
high-resolution validation of the Small-world generator remain useful next
experiments.

`Engine/`, `Code/`, and `Advanced/` are labeled research archives. They are not
part of the runtime package and must not be imported into it.

## Development

```bash
python -m pip install -e ".[dev,build]"
ruff check terraexplorer terraforge tests scripts packaging
ruff format --check terraexplorer terraforge tests scripts packaging
mypy terraexplorer
pytest --cov=terraexplorer
python -m build
```

Rebuild the tracked original media with:

```bash
python -m scripts.generate_media
python -m scripts.capture_gui  # requires a visible Windows desktop
```

CI tests Python 3.11, 3.12, and 3.13 on Windows and Ubuntu. Windows release
packaging builds `TerraExplorer.exe`. See [CONTRIBUTING.md](CONTRIBUTING.md) and
[CHANGELOG.md](CHANGELOG.md) before sending a patch.

## Legal

TerraExplorer is not affiliated with, endorsed by, or sponsored by Re-Logic.
Terraria and its related names and assets belong to their respective owners.
No license is granted for this repository. All rights are reserved by the
project copyright holder.
