<h1 align="center">
  <img src="docs/media/terraexplorer_readme_logo.png" width="160" alt="TerraExplorer mechanical tree"><br>
  TerraExplorer
</h1>

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
s_{32}=\mathrm{CRC32}(s), \qquad
r_p=\mathrm{PCG64}\!\left(\mathrm{uint64}
\left(\mathrm{BLAKE2s}(s_{32}\mathbin{:}p)\right)\right).
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

The README animation adds Mushroom, Marble, Granite, Ocean Cave, Spider Cave,
Gem Cave, and Micro Biome snapshots to the principal generation milestones,
then pauses on a visualization-only Meteorite impact. The desktop evolution
rail remains a focused 26-stop view of the 107-pass run.

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

## Feature Distribution

> Every ruin is a decision written into stone.

Every generated structure records a bounding box

$$
B_i=(x_i,y_i,w_i,h_i), \qquad
q_i=\left(x_i+\frac{w_i}{2},y_i+\frac{h_i}{2}\right).
$$

The figure projects those bounds and biome masks into one `4200 x 1200` world.
Repeated instances all remain visible and retain their full generated bounds;
one external label names each feature class so the annotation layer stays
readable.

```python
world = generate_world(WorldConfig(seed="Ash Compass", scale=WorldScale.SMALL))
for marker in world.structures:
    draw_bounds(marker.x, marker.y, marker.width, marker.height)
```

![Feature distribution across the Ash Compass world](docs/media/terraexplorer_world.png)

The map includes both oceans, surface biomes, Floating Islands, Living Trees,
the Dungeon, Pyramid, Aether, Jungle Temple, Hives, Granite, Glowing Mushroom
and Spider pockets, Gem Caves, minecart tracks, Ruined Houses, the Underground
Ocean, Corruption, Hallow, and every repeated generated instance. The Meteorite
is a deterministic visualization-only post-generation event. Its crater obeys
the documented border, spawn, liquid, cloud, and protected-structure exclusions
without changing the core 107-pass generator.

## Biomes

> A biome is not a color. It is terrain with a history.

Biome regions start from clipped seeded bands, but the atlas crop is selected
by a separate presentation objective. For candidate crop (C), target mask (M),
and crop center (c_C), the optimizer maximizes

$$
J(C)=\frac{|C\cap M|}{|C|}
-\lambda\frac{\lVert c_C-c_M\rVert_2}{\sqrt{48^2+72^2}}.
$$

This favors biome purity first and centering second. Relevant structures and
caves remain part of the crop rather than being painted into it.

```python
crop, score = _best_subject_crop(world, subject_mask, width=48, height=72)
selected = max(candidate_seeds, key=lambda seed: score_for(seed))
```

![Ten independently optimized biome studies](docs/media/biome_atlas.png)

Every panel represents the same `48 x 72` tiles at the same six-pixel scale.
The seeds are printed in the figure and recorded here:

| Biome | Seed | Biome | Seed |
|---|---|---|---|
| Forest | `Violet Scar` | Snow | `Salt Cathedral` |
| Desert | `World Below` | Jungle | `Red Descent` |
| Corruption | `Glass Horizon` | Crimson | `Cinder Archive` |
| Glowing Mushroom | `Emerald Hunger` | Meteorite | `Iron Orchard` |
| Underground Ocean | `Ash Compass` | Spider Nest | `Deep Lantern` |

Corruption is selected around its branching chasms and Crimson around its
linked chambers. The four additional panels expose Mushroom walls and grass, a
media-only Meteorite crater, the water-filled Dungeon-side Ocean tunnel, and a
Cobweb-filled Spider pocket. The exact optimizer output is also tracked in
[`docs/media/biome_seeds.json`](docs/media/biome_seeds.json).

## Hazard Containment Strategies

> A wall is an argument with time.

The media study advances hostile biome fronts through valid host materials
within a three-tile neighborhood. A containment mask (K) rejects otherwise
valid targets:

$$
S_{t+1}=S_t\cup
\{z:z\in\mathcal N_3(S_t),\ z\in V,\ z\notin K\}.
$$

Surface source tiles receive six times the sampling weight of underground
sources. Established Corruption, Crimson, and Hallow cannot overwrite one
another.

```python
accepted = vulnerable & ~occupied & ~protected[target_y, target_x]
world.biomes[target_y[accepted], target_x[accepted]] = biome
```

![Four hazard containment strategies](docs/media/containment_lab.gif)

`Violet Quarantine` uses a three-tile trench against Corruption. `Red Garden`
uses a Sunflower cordon against Crimson. `Pearl Ward` places a Chlorophyte
cluster against Hallow. `Three Front Siege` protects a brick bastion while all
three hazards advance from different regions. These are explanatory generated
world studies, not frame-exact game timing, and the public containment API is
unchanged.

## World Layers

> Depth is not distance. Depth is what the world permits to survive.

For world height (H), the generator stores three primary boundaries. The
animation adds a visualization-only Space guide and stretches vertical render
coordinates:

$$
y_{surface}=\mathrm{round}(0.19H),\quad
y_{rock}=\mathrm{round}(0.46H),\quad
y_{space}=\mathrm{round}(0.67y_{surface}),\quad
y'=1.7y.
$$

The stretch changes pixels only. Tile coordinates, structure placement, and
simulation boundaries remain untouched.

```python
world_image = base_image.resize(
    (base_image.width, round(base_image.height * 1.7)),
    Image.Resampling.NEAREST,
)
```

![Animated descent through all five world layers](docs/media/depth_descent.gif)

The extended camera pass names Space, Surface, Underground, Caverns, and
Underworld. Its subtitle is only `Depth XXX`. A light coordinate grid and
colored boundary guides expose the transitions while the complete generated
biomes, structures, Meteorite, and Underworld ruins stay visible.

## Spawn Heat Map

> The world does not choose an enemy until it finds somewhere to stand.

The heat score combines valid three-tile standing space (I), a depth factor
(D), biome factor (B), moderate-to-low light factor (L), and suppression factor
(S):

$$
H(y,x)=I(y,x)D(y)B(y,x)L(y,x)S(y,x).
$$

The score is zero inside the modeled spawn-safe and occupied housing zone,
multiplied by `0.77` inside Peace Candle range, and multiplied by `0.83` near a
Sunflower. A local Gaussian aggregation turns valid individual spawn tiles into
a readable regional likelihood while preserving those mechanical inputs.

```python
valid = three_air_tiles & solid_floor & ~lava
heat = valid * depth_weight * biome_weight * darkness_weight
heat[safe_zone | npc_housing] = 0.0
```

![Relative hostile spawn heat across the Ash Compass world](docs/media/spawn_heatmap.png)

This uses the same `Ash Compass` `4200 x 1200` world and identical
post-generation Meteorite overlay as Feature Distribution. The `viridis`
overlay represents relative hostile spawn opportunity, not an exact per-tick
enemy simulator. The legend names common or mechanically relevant enemies for
the major regions, while the marked control zones make town, Peace Candle, and
Sunflower suppression inspectable.

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
[Floating Islands](https://terraria.wiki.gg/wiki/Floating_Island),
[world layers](https://terraria.wiki.gg/wiki/Layers),
[Ocean caves](https://terraria.wiki.gg/wiki/Ocean_cave), and
[Spider Nests](https://terraria.wiki.gg/wiki/Spider_Cavern). Public wiki
mechanics take precedence when a supplied visual reference conflicts with a
documented rule. The
[reference atlas](docs/references/information/README.md) guides composition,
silhouette, and visual inspection within those rules.

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
