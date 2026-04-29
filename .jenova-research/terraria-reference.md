# Terraria World Reference

Source: terraria.wiki.gg, Desktop 1.4.4.9/1.4.5.5 decompiled source, Gemini deep research (2026-04-16)
Consolidated from biomes.md + world-sizes.md on 2026-04-20 (Phase 16).

---

## World Dimensions

| Size | Width (tiles) | Height (tiles) | Width (feet) | Height (feet) | Total Tiles |
|------|--------------|----------------|-------------|--------------|-------------|
| Tiny (legacy 3DS/Mobile) | 1750 | 900 | 3500 | 1800 | 1,575,000 |
| Small | 4200 | 1200 | 8400 | 2400 | 5,040,000 |
| Medium | 6400 | 1800 | 12800 | 3600 | 11,520,000 |
| Large | 8400 | 2400 | 16800 | 4800 | 20,160,000 |

Each tile = 2 feet. Each tile renders as 16x16 pixels.

### Border Buffer Zone
- Fixed 40-50 tile border between playable area and true array edges.
- `RandomWorldPoint(int top, int right, int bottom, int left)` generates coordinates within safe bounds.
- Accessible area is smaller than total array (e.g., Small: 4120x1120 accessible vs 4200x1200 total).

### Note on Medium World Parsing
Server `autocreate` in v1.2+ occasionally generated Medium worlds at 6300 tiles wide. Legacy .wld parsers must account for this.

## Memory and Serialization

- Each `Tile` object: 13 bytes (type, wall, liquid, wiring, paint, actuator, active state, frameX, frameY).
- Large world uncompressed: ~262 MB.
- .wld format: RLE + GZIP compression. Reduces ~250 MB to ~10-12 MB on disk.

## Generation Limits by World Size

| Feature | Small | Medium | Large |
|---------|-------|--------|-------|
| Floating Islands | 3 | 5 | 6 |
| Life Crystals | 100 | 230 | 403 |
| Underground Chests | 35-40 | 80-91 | 140-160 |
| Bee Hives | 6-8 | 8-12 | 11-16 |
| Marble Caves | 4-8 | 9-18 | 16-32 |
| Granite Caves | 4-8 | 6-12 | 8-16 |

## Vertical Strata Constants (from source)

| Variable | Description |
|----------|-------------|
| `0` | Absolute top of world array |
| `Main.worldSurface` | Surface/Underground boundary |
| `Main.rockLayer` | Underground/Cavern boundary |
| `Main.maxTilesY - 200` | Underworld ceiling |
| `Main.maxTilesY` | Absolute bottom |

### Approximate Layer Heights (Medium World, 1800 tiles tall)
- Space: top ~60-80 tiles
- Surface: Space boundary down to ~0 feet
- Underground: 0 feet to roughly 1/3 world height
- Cavern: Underground boundary to Underworld
- Underworld: bottom ~150 tiles (300 feet)

### Vertical Distribution (approximate)
```
Space         (top ~4-7% of world)
Surface       (above 0 feet)
Underground   (0 feet to ~33% depth)
Cavern        (33% to ~85% depth)
Underworld    (bottom ~15%, specifically last ~150 tiles)
```

## Horizontal Distribution Variables

| Variable | Description |
|----------|-------------|
| `Main.maxTilesX / 2` | Geometric center |
| `Main.dungeonX` | Dynamic integer rolled early in generation |
| `Main.spawnTileX` | Near center (300-400 tile radius) |
| `WorldGen.UndergroundDesertLocation` | Rectangle storing UG Desert bounds |

If `Main.dungeonX < (Main.maxTilesX / 2)`: Snow + Dungeon on left, Jungle + UG Desert on right. Vice versa.

### Horizontal Layout (left to right)
```
[Ocean] [Snow/Dungeon side] [Spawn/Forest] [Jungle/Desert side] [Ocean]
  OR
[Ocean] [Jungle/Desert side] [Spawn/Forest] [Snow/Dungeon side] [Ocean]
```

- Snow + Dungeon: always same side
- Jungle + Underground Desert: always same side, opposite from Dungeon
- Evil biome: either side
- Oceans: both edges, within 338 tiles of world boundary

---

## World Layers

| Layer | Description | Depth |
|-------|-------------|-------|
| Space | Reduced gravity, Harpies, Floating Islands | Above surface threshold |
| Surface | Forest, Desert, Snow, Jungle, Ocean, Evil biomes | 0 feet and above |
| Underground | Same biomes as surface, dirt backdrop | 0 feet to cavern threshold |
| Cavern | Ice, UG Jungle, UG Desert, Mushroom, Granite, Marble | Below underground threshold |
| Underworld | Lava, ash, hellstone, obsidian buildings | Bottom 300 feet |

## Biome Hierarchy

### Surface Biomes
- **Forest** (default): grass, trees, normal enemies
- **Snow/Ice**: snow blocks, ice blocks; same side as Dungeon
- **Desert**: sand blocks, cacti; opposite side from Dungeon
- **Corruption**: ebonstone, shadow orbs, chasms (evil option 1)
- **Crimson**: crimstone, crimson hearts, cave networks (evil option 2)
- **Jungle**: mud, jungle grass, beehives; opposite side from Dungeon
- **Ocean**: water body at world edges, within 338 tiles of edge
- **Dungeon**: brick structure, same side as Snow
- **Glowing Mushroom**: never natural on surface (player-created only)

### Underground Biomes (Cavern Layer)
- **Ice biome**: beneath Snow, ends at bottom of upper Cavern
- **Underground Jungle**: beneath Jungle, extends through Cavern
- **Underground Desert**: beneath main Desert, circular/oval ant-hive structure
- **Underground Mushroom**: scattered pockets
- **Dungeon**: extends deep through Cavern

### Hardmode Biomes
- **Hallow**: V-shape from center, converts stone/sand/ice/grass
- **Underground Hallow**: subsurface Hallow
- **Underground Corruption/Crimson**: evil spreads underground

### Mini-Biomes / Structures
- Oasis (Desert water body), Granite Cave (4-32 per world), Marble Cave (4-32 per world)
- Spider Nest, Bee Hive (6-16 per world), Glowing Moss (6 variants)
- Jungle Temple (Lihzahrd), Meteorite (post-orb), Aether/Shimmer (1.4.4+)
- Enchanted Sword Shrine, Underground Cabin, Treasure Room, Living Tree, Pyramid, Floating Island

## Biome Detection (Tile Thresholds)

Tile rectangle around player: 84 tiles each side, 61 below, 62 above center.

| Biome | Required Tiles | Tile Types |
|-------|---------------|------------|
| Corruption | 300 | Ebonstone, Corrupt grass, Ebonsand, Purple Ice |
| Crimson | 300 | Crimstone, Crimson grass, Crimsand, Red Ice |
| Hallow | 125 | Pearlstone, Hallowed grass, Pearlsand, Pink Ice |
| Jungle | 140 | Jungle grass, Lihzahrd Brick |
| Glowing Mushroom | 100 | Mushroom grass |
| Desert (surface) | 1500 | Sand, Hardened Sand, Sandstone |
| Snow | 1500 | Snow, Ice |
| Ocean | 1000 water tiles | Within 338 tiles of world edge |
| Dungeon | 250 | Dungeon Brick |
| Graveyard | 5 | Tombstones (within range) |
| Meteorite | 75 | Meteorite Ore |

## Biome Spread Mechanics

### Pre-Hardmode
- Only grass-type spread (Corrupt/Crimson grass to adjacent blocks). Very slow.

### Hardmode
- Evil biomes and Hallow spread to stone, sand, ice, hardened sand, sandstone, mud (evil only), jungle grass (evil only)
- Spread range: 3 tiles from any spreading tile (7x7 square)
- Surface tiles update ~6x faster than underground
- Defeating Plantera halves spread speed
- Sunflowers block conversion within 2 tiles
- 3-tile air gap stops spread (thorny bushes can bridge gaps)

### Conversion Table (Hardmode)
| Pure | Corrupt | Crimson | Hallow |
|------|---------|---------|--------|
| Stone | Ebonstone | Crimstone | Pearlstone |
| Ice | Purple Ice | Red Ice | Pink Ice |
| Sand | Ebonsand | Crimsand | Pearlsand |
| Hardened Sand | Hardened Ebonsand | Hardened Crimsand | Hardened Pearlsand |
| Sandstone | Ebonsandstone | Crimsandstone | Pearlsandstone |
| Mud | Dirt (evil only) | Dirt (evil only) | n/a |
| Jungle grass | Corrupt grass | Crimson grass | n/a |

### Wall Infection
- Walls spread within 2-tile range (5x5 square)
- Walls cannot infect blocks, but blocks can infect walls
- Hallow cannot infect Jungle walls
- Chlorophyte Ore limits evil spread (5+ tiles in 10-tile radius blocks conversion)

## WallDungeon Logic Array

`Main.wallDungeon` Boolean array tracks Dungeon territory. Set during Dungeon generation pass.

- **Pots pass**: if `wallDungeon[tile.wall]` is true, pot style overrides to index 10-13 (Dungeon-themed)
- **Chest loot**: if WallDungeon true, 1-in-8 RNG roll for Dungeon-exclusive items (e.g., Golden Key, ID 2192)
- **Room validation**: used by NPC housing checks and structure overlap prevention
