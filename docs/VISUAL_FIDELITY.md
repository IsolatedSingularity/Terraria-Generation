# Visual and structural fidelity audit

This audit records what the README figures demonstrate and where they remain
educational approximations. The stable generator order is the public
Terraria 1.4.4.9-inspired 107-step sequence. Mechanical comparisons below use
the optional local Terraria 1.4.5.6 corpus as a version-locked research target.

## Provenance and interpretation

- `VERIFIED_LOCAL` means a behavior was checked against a narrow symbol range in
  the local 1.4.5.6 corpus.
- `PROJECT_MODEL` means the visual is generated from TerraExplorer state and is
  deterministic, but its numerical or structural rules remain original.
- No figure is evidence of seed, RNG, tile-ID, or `.wld` compatibility.

## README figure audit

| Figure | Status | What it establishes | Important boundary |
|---|---|---|---|
| README logo | Unchanged | Original project branding | No mechanical claim |
| Generation animation | Regenerated | Frames are real pipeline snapshots | Pass handlers remain approximations |
| GUI capture | UI-only | Controls, threading surface, map inspection, and pass log | Captured world predates this audit and is not fidelity evidence |
| Three idle-evolution animations | Regenerated | Three-tile reach, source-sampled batches, and opposed fronts | Iterations do not map to elapsed game time |
| Feature Distribution | Regenerated | Bounds from actual generated structures, including underground cabins | Bounds do not prove vanilla frequency or topology |
| Biome atlas | Regenerated | Equal-size crops selected from generated worlds | Crop selection is a presentation optimizer, not biome validation |
| Containment laboratory | Regenerated | Trench, two-tile Sunflower, count-dependent Chlorophyte, and bastion interventions | Chlorophyte defense applies to evil conversion, not Hallow |
| Depth descent | Regenerated | TerraExplorer's stored layer boundaries and a presentation-only camera stretch | Layer ratios are project coordinates, not recovered vanilla formulas |
| Spawn heat map | Regenerated | Standing space plus depth, biome, darkness, and attempt-frequency suppression | It is not `NPC.SpawnNPC`, a spawn-pool table, or a per-tick probability map |

## Structure findings

| Feature | Local reference evidence | TerraExplorer state | Priority gap |
|---|---|---|---|
| Structure placement | `StructureMap.CanPlace` checks bounds, protected intersections, and valid tiles; `AddProtectedStructure` stores padded bounds | New cabins use bounds and overlap rejection; minecart tracks respect marker padding | Generalize the guard and retry telemetry to every structure pass |
| Underground cabins | `CaveHouseBiome.Place` rejects wires/chests and delegates validated room placement; room validation checks lava and `StructureMap` | One- or two-floor wooden cabins now contain a chest and avoid existing markers/liquid | Add biome-specific shells, furniture, multi-room layouts, and placement statistics |
| Hives | `HiveBiome.Place` requires a valid 100-by-100 region, Jungle material share, tunnel chains, honey falls, a larva stand, and protected padding | Connected shell/cavity ellipses with honey and walls | Replace isolated ellipses with branching tunnels and explicit protected bounds |
| Pyramid | `WorldGen.Pyramid` creates a deep tapered body, alternating descent, treasure room, and continuing tunnel | Tapered body, zigzag passage, and treasure chamber | Add the long continuation tunnel, retry exclusions, decoration variants, and frequency validation |
| Aether | `ShimmerMakeBiome` validates a large region, builds asymmetric shells/openings, places partial/full Shimmer levels, and attempts Gem Trees | Stone shell, cavity, Shimmer pool, Gem Trees, and Jungle-side placement | Improve asymmetric variants, opening cleanup, partial liquid surface, and exclusion retries |
| Floating Islands | The Floating Islands pass enforces horizontal separation and chooses Cloud Island versus Cloud Lake before a later house pass | Three separated Cloud/Rain Cloud islands with later houses | Scale count by world width and introduce a true Cloud Lake variant |
| Dungeon | `LegacyDungeonLayout` alternates seeded halls and rooms with a delayed branching cadence | Connected entrance, rooms, corridors, platforms, and a bounded marker | Add style variants, branch/topology metrics, locked progression spaces, and protected bounds |
| Jungle Temple | Vanilla generation is split across body, part-two, and altar passes | Connected room rows, corridors, traps, and deep altar chamber | Audit exact room-growth and entrance rules before calling its topology representative |
| Underworld Ruined Houses | Generated independently from the new underground cabins | Multi-floor obsidian/Hellstone-brick towers with supports, lava flooding, and Hellforges | Add spacing guards and variant-frequency validation |

## Mechanics corrected in this audit

- Desert and Jungle origins retain the intended same-side relation but now leave
  a visible transition instead of substantially overwriting each other.
- Runtime infection samples sources and offsets rather than converting every
  eligible edge cell in one vectorized sweep.
- Surface sources receive two modeled attempts per deeper source, derived from
  the 1.4.5.6 overground and underground dispatcher rates.
- Sunflowers block infection targets within two tiles.
- Chlorophyte scans a five-tile square; three nearby Chlorophyte tiles guarantee
  defense, while one or two use the count-dependent probability. It blocks
  Corruption/Crimson conversion, not Hallow.
- The post-Plantera metadata flag can halve sampled evil/Hallow attempts, matching
  the reference branch while remaining opt-in to experiments.

## Structural smoke evidence

- Eight fixed Preview seeds each generated two underground cabins and two Living
  Trees. No cabin bounds intersected any other recorded structure bounds.
- The `Ash Compass` Small reference world generated five Living Trees, three
  Floating Islands, one Pyramid, one Dungeon, one Jungle Temple, ten Gem Caves,
  24 underground cabins, and 18 Underworld Ruined Houses.
- These checks establish deterministic presence and non-overlap for the sampled
  worlds; they do not establish vanilla frequency distributions.

## Next validation work

1. Expand the fixed Preview smoke corpus into a multi-seed Small corpus.
2. Record biome occupancy, cave connectivity, structure counts, failed retries,
   overlaps, entrance reachability, and liquid volumes.
3. Generalize the local structure guard before increasing visual detail.
4. Audit Hive, Floating Island/Cloud Lake, Temple, Dungeon, and Aether methods in
   that order because they dominate the README silhouettes.
5. Keep the GUI screenshot as UI evidence unless a visible desktop capture is
   intentionally refreshed.
