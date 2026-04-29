# Existing Repo Analysis: Terraria-Generation

Repository: github.com/IsolatedSingularity/Terraria-Generation
Language: Python 100% | Dependencies: numpy, matplotlib, seaborn
Updated: 2026-04-16 (merged Gemini deep research findings)

## Repository Structure

```
Code/           Main implementation files
Code+/          Extended/alternative implementations
Plots/          Generated visualization outputs
References/     Reference materials and images
```

## Current Implementation

### Techniques Used
1. **Multi-octave Perlin noise**: Surface terrain height generation
2. **TileRunner (random walks)**: Cave system carving, ore vein placement
3. **Cellular automata**: Corruption/Crimson biome spreading
4. **Poisson point processes**: Structure placement (temples, shrines, cabins)

### Claimed Accuracy
- 95% accuracy in biome placement
- 87% correlation with actual in-game patterns
- README self-note: "not rigorous and not super accurate"

## Gap Analysis (Wiki Data vs Current Implementation)

### Critical Gaps
1. **Pass ordering**: The actual game uses 103 sequential passes; current implementation likely uses a simplified subset
2. **Underground Desert**: Complex ant-hive circular structure with Hardened Sand/Sandstone; likely simplified or missing
3. **Dungeon generation**: Full room+corridor "eating algorithm" with rectangular rooms, WallDungeon array validation, colored background walls
4. **Layer depth thresholds**: Randomized per-world layer boundaries not captured. Key variables: `Main.worldSurface`, `Main.rockLayer`, `Main.maxTilesY - 200`
5. **Biome tile detection**: 169x124 tile rectangle counting system not implemented
6. **Seed mechanics**: CRC-32 conversion, full seed format (size.difficulty.evil.special.id) not modeled
7. **Cave carving algorithm**: Repo may use 2D Perlin noise thresholds (common clone mistake). Actual game uses TileRunner/digTunnel directional random walks + cellular automata smoothing
8. **Liquid settling**: Bottom-up recursive scan with gravity/spread. Not noise-based. Runs as dedicated pass after geometry finalized.
9. **Tile framing**: frameX/frameY bitmask neighbor calculation, FrameImportant protection for multi-tile objects. Not relevant for simulation but important for accuracy context.
10. **Border buffer zone**: 40-50 tile unplayable border around array edges. Coordinate generation must use safe bounds.
11. **StructureMap protection**: Protected rectangles prevent structure overlap. Jungle Temple, Dungeon, etc. register exclusion zones.

### Moderate Gaps
12. **Marble/Granite caves**: Count varies dramatically by world size (4-32)
13. **Floating Islands**: Height and count constraints per world size
14. **Bee Hives**: Complex honey-filled structures in Jungle
15. **Shimmer/Aether biome**: Newer feature (1.4.4+), likely not present
16. **Secret world seeds**: 35 generation modifiers (1.4.5.0+). Drunk World allows both evils. Zenith overrides spawn to Underworld.
17. **Biome spread wall infection**: Separate grassy/sandy wall conversion algorithms
18. **Ore density scaling**: Proportional formula `(maxTilesX * maxTilesY) * 6E-05` ensures consistent density across world sizes
19. **Meteorite collision**: Post-gen event with max tile quota `400 * (maxTilesX / 4200)`, exclusion zones for Dungeon/Temple
20. **WallDungeon Boolean array**: Controls Dungeon-themed pot styles, chest loot tables, room validation

### Strengths of Current Approach
- Perlin noise for terrain is correct foundational approach (but actual game may use simpler 1D wave superposition)
- TileRunner random walks match the actual algorithm concept (Gemini confirms this is THE most critical function)
- Cellular automata for corruption spreading is mechanically sound
- Visualization pipeline (matplotlib/seaborn) produces clear output
- Poisson processes for structure placement is reasonable approximation (actual game uses Shape+Action system with `WorldUtils.Gen`)

### Gemini Research Confirmations
- TileRunner IS the most frequently called method in the source code
- Caves use directional tunneling, NOT 2D Perlin noise thresholds
- Cellular automata is used for smoothing cave edges (separate from biome spreading)
- Ore distribution scales proportionally with world area (6E-05 multiplier)
- Full research file available at: `References/worldgen-research.md` in target repo (gitignored)

## Improvement Priorities

### Phase 1: Core Accuracy
- Implement correct pass ordering (at least major passes in sequence)
- Add proper world size dimensions (4200x1200, 6400x1800, 8400x2400)
- Implement layer depth calculation with proper thresholds
- Add biome placement constraints (Snow+Dungeon same side, etc.)

### Phase 2: Biome Detail
- Underground Desert: circular/oval structure with correct block types
- Proper Corruption chasm vs Crimson cave network distinction
- Marble/Granite cave count scaling by world size
- Floating Island height and count constraints

### Phase 3: Advanced Features
- Biome spread simulation (Hardmode conversion)
- Seed input/output matching actual format
- Structure generation (Dungeon, Temple, Pyramids)
- Tile threshold biome detection simulation
