# Terraria World Generation Passes

Source: `Terraria.WorldGen.cs` method `GenerateWorld()` (Desktop 1.4.4.9/1.4.5.5), Gemini deep research (2026-04-16)

## Overview

Terraria uses a deterministic 103-pass sequential generation pipeline managed by `List<GenPass>` in `Terraria.World.Generation`. Each pass runs in order, building on the output of previous passes. The world seed (CRC-32 of input string, or raw integer) determines all RNG. Seed format: `[size].[difficulty].[evil].[special].[identifier]`.

**Critical**: Tile framing is suppressed during all 103 passes. The engine runs a global asynchronous sweep to frame the entire tile array only after all passes complete. Running framing mid-generation would cause catastrophic CPU bottlenecks.

Later passes unconditionally overwrite earlier passes unless `TileID.Sets.CanBeClearedDuringGeneration` is checked (protects Lihzahrd Brick, Granite, Hardened Sand, Sandstone from late-stage cave carvers).

## Five Macro-Phases

### Phase 1: Terrain Baseline (Passes 0-7)
1D height array via fractal noise/wave superposition across X-axis. Below = Dirt, above = Air. Stone pockets, Sand for oceans, Clay injected.

### Phase 2: Carving Caves (Passes 8-11)
Random Holes, Small Caves, Large Caves, Surface Caves. Uses directional tunneling (TileRunner, digTunnel) + cellular automata smoothing. **NOT 2D Perlin noise thresholds** (common mistake in clones).

### Phase 3: Macro-Biomes (Passes 12-27)
Snow, Jungle, Underworld carved. Dungeon and Evil biomes override standard caves. Shinies pass distributes pre-Hardmode ores.

### Phase 4: Micro-Biomes and Structures (Passes 28-57)
Floating Islands, Living Trees, Pyramids, Jungle Temple, Spider Caves. Shape-based logic (ellipses, rectangles) via `WorldUtils.Gen` with `Shape` + `Action` pairs, not noise.

### Phase 5: Simulation and Polish (Passes 58-78+)
Gravity simulation (floating sand), liquid settling (water/lava pooling), grass spreading. Final: `Finding Tile Frames` (bitmask neighbor calculation for sprite borders on every tile).

## Complete Pass List (Ordered)

| # | Pass Name | Description |
|---|-----------|-------------|
| 1 | Reset | Clear tile array, set world dimensions |
| 2 | Terrain | Perlin noise surface height map generation |
| 3 | Dunes | Sand dune formations in desert regions |
| 4 | Ocean Sand | Sand placement at ocean boundaries |
| 5 | Sand Patches | Random sand deposits throughout surface |
| 6 | Tunnels | Long horizontal/diagonal tunnel carving |
| 7 | Mount Caves | Caves inside mountain peaks |
| 8 | Dirt Walls | Background dirt wall placement |
| 9 | Rocks In Dirt | Stone block patches in dirt layer |
| 10 | Dirt In Rocks | Dirt block patches in stone layer |
| 11 | Clay | Clay deposit generation |
| 12 | Small Holes | Tiny cave pockets |
| 13 | Dirt Layer Caves | Medium caves in dirt/underground layer |
| 14 | Rock Layer Caves | Large caves in rock/cavern layer |
| 15 | Surface Caves | Caves opening to surface |
| 16 | Generate Ice Biome | Snow/ice biome terrain |
| 17 | Grass | Grass block placement on surface dirt |
| 18 | Jungle | Jungle biome terrain and mud placement |
| 19 | Mud Caves To Grass | Convert mud cave walls to jungle grass |
| 20 | Full Desert | Underground Desert structure (ant-hive shape, Hardened Sand, Sandstone) |
| 21 | Mushroom Patches | Glowing mushroom biome pockets |
| 22 | Marble | Marble cave generation (Small: 4-8, Med: 9-18, Large: 16-32) |
| 23 | Granite | Granite cave generation (Small: 4-8, Med: 6-12, Large: 8-16) |
| 24 | Floating Islands | Sky islands (Small: 3, Med: 5, Large: 6) |
| 25 | Shinies | Ore vein generation (Copper/Tin, Iron/Lead, Silver/Tungsten, Gold/Platinum) |
| 26 | Webs | Spider web placement in caves |
| 27 | Underworld | Hell layer: ash, lava lakes, obsidian buildings, hellstone |
| 28 | Corruption / Crimson | Evil biome chasms or crimson caves |
| 29 | Lakes | Surface lake generation |
| 30 | Dungeon | Full dungeon structure with rooms and corridors |
| 31 | Mountain Caves | Additional mountain cavity systems |
| 32 | Beaches | Ocean biome sand shaping |
| 33 | Gems | Gem deposit placement (all gem types) |
| 34 | Shimmer | Aether/Shimmer biome placement |
| 35 | Pyramids | Desert pyramid structures (chance-based) |
| 36 | Living Trees | Living tree generation with root systems |
| 37 | Altars | Demon/Crimson altar placement |
| 38 | Jungle Temple | Lihzahrd Temple structure |
| 39 | Hives | Bee hive structures (Small: 6-8, Med: 8-12, Large: 11-16) |
| 40 | Smooth World | Terrain smoothing pass |
| 41 | Life Crystals | Crystal heart placement (Small: 100, Med: 230, Large: 403) |
| 42 | Statues | Random statue placement |
| 43 | Buried Chests | Underground chest placement |
| 44 | Surface Chests | Surface-level chest placement |
| 45 | Jungle Chests | Jungle-specific chest placement |
| 46 | Water Chests | Underwater chest placement |
| 47 | Spider Caves | Spider nest mini-biome generation |
| 48 | Gem Caves | Gem-themed cave rooms |
| 49 | Moss | Glowing moss placement on stone |
| 50 | Cave Walls | Background wall placement in caves |
| 51 | Traps | Dart traps, boulders, explosives |
| 52 | Pots | Clay pot decoration placement |
| 53 | Hellforge | Hellforge placement in underworld buildings |
| 54 | Spawn Point | Calculate valid spawn location |
| 55 | Guide | Place Guide NPC at spawn |
| 56 | Sunflowers | Sunflower placement on surface grass |
| 57 | Trees | Tree generation on grass |
| 58 | Herbs | Herb plant placement |
| 59 | Weeds | Small plant/weed decoration |
| 60 | Vines | Vine growth from grass/jungle blocks |
| 61 | Flowers | Flower decoration placement |
| 62 | Micro Biomes | Enchanted sword shrines, underground cabins, treasure rooms |
| 63 | Final Cleanup | Last validation and correction pass |

Note: Passes 43-63 are approximate groupings. The actual source has additional sub-passes and conditional logic within many of these steps. The full decompiled source contains roughly 103 discrete generation steps.

## Key Algorithms

### Terrain Generation (Pass 2)
- Multi-octave Perlin noise (or simplified 1D Midpoint Displacement) for surface height
- Height varies by world size (surface level differs per size)
- Biome boundaries determined by horizontal position

### TileRunner (Most Critical Function)
Exact C# signature:
```
WorldGen.TileRunner(int x, int y, double strength, int steps, int type,
    bool addTile = false, double speedX = 0f, double speedY = 0f,
    bool noYChange = false, bool overRide = true)
```
- **strength**: radius/thickness of splotch. Decays each step (organic taper).
- **steps**: iteration count. Small strength + large steps = narrow tunnel. Large strength + small steps = bulbous cavern.
- **type**: TileID to place. type == -1 destroys tiles (cave carver). type == -2 replaces with lava below lava threshold.
- **speedX/speedY**: directional drift vector. Zero = true drunkard's walk with random vectors.
- **noYChange**: clamps vertical mutation for flat horizontal tunnels.
- **overRide**: when true, forces replacement regardless of existing tile.
- Creates diamond-shaped brush with noisy edges at each step point.
- **WARNING**: Do not run with overRide after FrameImportant tiles are placed (corrupts multi-tile objects).

### digTunnel (Grand Void Carving)
```
WorldGen.digTunnel(double X, double Y, double xDir, double yDir, int Steps, int Size, bool Wet)
```
- Cleaner geometric sphere-cutter interpolation (vs TileRunner's noisy diamond brush).
- Traverses precise vector line, clearing tiles along radius.
- Creates smooth-bored vertical drops or diagonal shafts.

### Cavinator (Macro Cave Generation)
- Primary destructive method for macro-cave passes.
- Self-checking: evaluates `TileID.Sets.CanBeClearedDuringGeneration` before destroying.
- Aborts destruction if tile type is protected (Lihzahrd Brick, Granite, Hardened Sand, Sandstone).
- Ensures primary biome shells survive late-stage random cave carving.

### Cellular Automata (Cave Smoothing)
- After TileRunner/digTunnel carve caves, a cellular automata pass smooths edges.
- **Smoothing rule**: If tile has fewer than N solid neighbors, destroy it. If empty space has too many solid neighbors, fill it.
- Creates organic, rounded cave walls characteristic of Terraria's underground.

### Cellular Automata (Corruption/Crimson)
- Used for spreading evil biome during generation
- Corruption: vertical chasms with branching
- Crimson: round cave systems with connecting tunnels

### Liquid Settling (SettleLiquids Pass)
- Liquids spawned as solid blocks during generation.
- Engine scans grid bottom-up: liquid with empty space below moves down.
- If space below is solid, liquid spreads left and right.
- Runs recursively until equilibrium. Often the longest visible generation step.

### Ore Density Scaling
Proportional formula ensures consistent resource density across world sizes:
```
iterations = (int)((double)(Main.maxTilesX * Main.maxTilesY) * 6E-05)
```
- Small (5,040,000 tiles): 302 TileRunner calls per ore type
- Large (20,160,000 tiles): 1,209 TileRunner calls per ore type

### Structure Placement (StructureMap)
- `WorldBuilding.StructureMap.AddProtectedStructure(Rectangle, int)` reserves grid areas.
- Subsequent passes call `CanPlace()` against StructureMap; abort if overlap detected.
- Used for Jungle Temple, Dungeon, and other exclusive structures.

### Tile Framing
- Each block checks 8 neighbors to determine spritesheet coordinates (frameX, frameY).
- Suppressed during all 103 passes; global sweep runs after generation completes.
- **FrameImportant tiles** (Chests, Trees, Anvils, Workbenches): rely on static frameX/frameY values. Any alteration corrupts the multi-tile object.

### Underground Desert (Pass 20)
- Circular/oval structure beneath main surface desert
- Ant-hive pattern of interconnected small caves
- Composed of Hardened Sand Blocks + Sandstone Blocks
- Min size increased to 75% of max in 1.4.4

## Biome Placement Rules

- Snow biome + Dungeon: always same side of world
- Jungle + Underground Desert: opposite side from Dungeon
- Evil biome: placed after major biomes, carves into existing terrain
- Ocean: both world edges, within 338 tiles of edge
- Floating Islands: above surface, count varies by world size

## Seed Mechanics

- Seed format: `[size].[difficulty].[evil].[special_seeds_bitmask].[identifier]`
- Size: 1=Small, 2=Medium, 3=Large
- Difficulty: 1=Classic, 2=Expert, 3=Master, 4=Journey
- Evil: 1=Corruption, 2=Crimson
- Special seeds: bitmask (1=Drunk, 2=Bees, 4=FTW, 8=Celebration, 16=Constant, 32=Remix, 64=NoTraps, 128=Zenith, 256=Skyblock)
- Text identifiers converted via CRC-32
- 35 secret world seeds (1.4.5.0+) can modify generation

## Structure Quotas by World Size

| Structure | Small | Medium | Large |
|-----------|-------|--------|-------|
| Floating Islands | 3 | 5 | 6 |
| Underground Cabins | 35-40 | 80-91 | 140-160 |
| Living Mahogany Trees | 6-11 | 9-16 | 12-22 |
| Marble Caves | 4-8 | 9-18 | 16-32 |
| Granite Caves | 4-8 | 6-12 | 8-16 |
| Standard Minecart Tracks | 4-7 | 9-15 | 16-28 |
| Surface Chests | 21 | 32 | 42 |
| Life Crystals (max) | 100 | 230 | 403 |
| Bee Hives | 6-8 | 8-12 | 11-16 |

## Meteorite (Post-Generation Event)
- Triggered by `dropMeteor()` after defeating Eater of Worlds/Brain of Cthulhu.
- Scans X-axis, projects vector down to find valid surface.
- Max meteorite tiles: `400 * (Main.maxTilesX / 4200)` (Large world = 800 tiles max).
- Aborts if count exceeds threshold or impact zone covers Dungeon/Temple.
- Uses large-radius TileRunner to convert terrain to meteorite.

## Secret Seeds (Key Overrides)

### Drunk World (05162020)
- Both Corruption AND Crimson (mutual exclusivity disabled)
- All ore variants placed simultaneously
- Dungeon entrance under enlarged Living Tree

### Zenith / Get Fixed Boi (Main.zenithWorld)
- Master compilation: toggles ALL other secret seed flags
- Spawn overridden to Underworld (Main.maxTilesY - 200)
- NPC Happiness disabled
- Tombstones replaced with bouncing boulders

### Celebrationmk10
- Spawn at Ocean boundary instead of center
- All chest loot gets highest-tier modifiers (Legendary, Warding, Mythical)

## Modding Insertion Point
Custom passes injected via tModLoader:
```csharp
int genIndex = tasks.FindIndex(genpass => genpass.Name.Equals("Shinies"));
if (genIndex != -1) {
    tasks.Insert(genIndex + 1, new PassLegacy("Custom Ores", CustomOreMethod));
}
```
Insert after "Shinies" ensures macro-terrain is finalized before custom placements.
