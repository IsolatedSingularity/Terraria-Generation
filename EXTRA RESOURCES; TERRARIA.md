# Terraria World Generation Reconstruction: Agent Resource and Acquisition Guide

**Target:** Reconstruct Terraria world generation only.

**Explicit exclusions:** Loot generation, Terraria UI, sound effects, unrelated gameplay systems, and cosmetic imitation that does not help world generation.

**Version rule:** Pin a target Terraria/tModLoader version before implementing parity. Prefer the current PC generation model, but use older implementations when they are substantially clearer and label them by version.

**Acquisition rule:** Prefer online repositories, official documentation, public examples, and public recreations. The user has local Terraria files, but local decompilation or inspection is a fallback rather than the starting point.

**Verification date:** 2026-07-27.

## Verdict

Terraria is the most reconstructable of the three projects, but the compact research report overstates what is already available.

The correct hierarchy is:

1. **Reference architecture:** tModLoader's official world-generation APIs and current source tree.
2. **Behavioral experiments:** a minimal tModLoader mod that logs, disables, inserts, and snapshots generation passes.
3. **Output validation:** TEdit and TerraMap.
4. **Independent prototype:** TerraForge, useful for studying pass-oriented procedural generation and visualization.
5. **Historical fallback:** old decompiled code, explicitly version-pinned.

TerraForge is not a source-faithful Python port of Terraria. Its own README says it is an independent educational project inspired by public generation concepts, uses original Python code and art, does not read or write `.wld`, and contains no private game code. Do not use it as the parity oracle.

Do not hard-code “103 passes” as a universal fact. The pass list and behavior vary with Terraria/tModLoader version, world settings, and special seeds. Enumerate the pass queue at runtime for the pinned target.

## Priority resource ledger

| Priority | Resource | Direct link | What to obtain | Best use | Caveat |
|---|---|---|---|---|---|
| 1 | tModLoader `ModSystem` documentation | https://docs.tmodloader.net/docs/stable/class_mod_system.html | Official `ModifyWorldGenTasks`, `PreWorldGen`, and related hooks | Learn how passes are exposed, ordered, inserted, and disabled | API reference, not a complete vanilla algorithm specification |
| 2 | tModLoader `Terraria.WorldBuilding` namespace | https://docs.tmodloader.net/docs/1.4-stable/namespace_terraria_1_1_world_building.html | `GenPass`, `WorldGenerator`, `WorldUtils`, Shapes, Modifiers, Actions | Reproduce the public conceptual architecture | Exact target branch/version must match the implementation being studied |
| 3 | tModLoader source | https://github.com/tModLoader/tModLoader | Current source, ExampleMod, tests, and wiki links | Build the instrumentation mod and inspect public/decompiled interfaces | Repository branches move; pin a commit and record it |
| 4 | TEdit | https://github.com/TEdit/Terraria-Map-Editor | Map editor, world parser, current format handling | Visual validation, structure inspection, tile/wall/liquid comparison | Editor behavior is not a generator specification |
| 5 | TerraMap Native | https://terramap.github.io/native.html | Read-only map viewer | Fast, low-risk inspection of generated worlds | Viewer only |
| 6 | TerraForge / Terraria-Generation | https://github.com/IsolatedSingularity/Terraria-Generation | Independent Python generator, visualizer, exports, tests | Procedural-generation laboratory and prototype reference | Explicitly not Terraria source parity and does not read/write `.wld` |
| 7 | Terraria 1.4.0.5 decompiled/refactored source | https://github.com/AliceSavard/Terarria1405 | Historical source mapping | Fallback for understanding older `WorldGen` organization and algorithms | Old version, decompiled, modified, and not authoritative for current behavior |

## Required first decision: target ruleset

Before coding, write `target_ruleset.yaml`:

```yaml
game: Terraria
platform: PC
terraria_version: "REQUIRED"
tmodloader_version: "REQUIRED"
tmodloader_commit: "REQUIRED"
world_sizes:
  - small
  - medium
  - large
difficulties:
  - classic
evil_types:
  - corruption
  - crimson
special_seeds:
  - none
hardmode_generation: included
output_goal:
  exact_wld_compatibility: false
  structural_parity: true
  seed_reproducibility: true
```

Do not compare seeds across different rulesets as though a mismatch were an implementation bug.

## Online-first setup order

### Phase 1: Clone public sources

```bash
git clone https://github.com/tModLoader/tModLoader.git public-code/tModLoader
git clone https://github.com/TEdit/Terraria-Map-Editor.git public-code/TEdit
git clone https://github.com/IsolatedSingularity/Terraria-Generation.git public-code/TerraForge
git clone https://github.com/AliceSavard/Terarria1405.git historical-reference/Terarria1405
```

Immediately record commits:

```bash
git -C public-code/tModLoader rev-parse HEAD
git -C public-code/TEdit rev-parse HEAD
git -C public-code/TerraForge rev-parse HEAD
git -C historical-reference/Terarria1405 rev-parse HEAD
```

### Phase 2: Run TerraForge only as a laboratory

```bash
cd public-code/TerraForge
python -m pip install -e .
python -m terraforge gui
```

Use it to study:

- Deterministic seed handling.
- Named generation phases.
- Tile/wall/liquid separation.
- Intermediate-frame visualization.
- PNG, GIF, JSON, and NumPy export patterns.
- Regression-test organization.

Do not copy its parameters into a Terraria-parity implementation without independent validation.

### Phase 3: Build the minimal tModLoader instrumentation mod

The mod should:

1. Hook `PreWorldGen`.
2. Hook `ModifyWorldGenTasks`.
3. Log every pass index, name, weight, and enabled state.
4. Write the pass list before any modification.
5. Allow one no-op custom pass to be inserted before/after a named vanilla pass.
6. Optionally disable one harmless pass for an A/B validation experiment.
7. Snapshot aggregate world state after selected passes when technically feasible.
8. Record seed, target version, world size, evil type, difficulty, and special-seed flags.

Suggested log schema:

```csv
run_id,terraria_version,tmodloader_commit,seed,world_size,special_seed,pass_index,pass_name,weight,enabled,start_ms,end_ms,tile_hash,wall_hash,liquid_hash
```

### Phase 4: Establish validation tools

For every generated test world:

- Open it in TerraMap for read-only inspection.
- Open a copy in TEdit for detailed tile/structure inspection.
- Export screenshots at fixed coordinates and zoom levels.
- Record world metadata and tile/wall/liquid counts.
- Keep the original `.wld` unchanged.

### Phase 5: Reimplement as named passes

Use a mutable world-state model:

```text
WorldState
  dimensions
  seed_state
  tiles[]
  walls[]
  liquids[]
  wires_and_actuators[]
  structures
  protected_regions
  generation_metadata

GenerationPipeline
  SeedNormalization
  Terrain
  Caves
  Biomes
  Structures
  Ores
  Liquids
  Traps
  Decoration
  SpawnValidation
  FinalCleanup
  HardmodeTransformation
```

The exact subpasses must come from runtime enumeration and experiments, not from a report's fixed count.

## Differential validation plan

Use three levels of comparison.

### Level 1: Determinism

- Same implementation + same version + same seed must produce identical hashes.
- Different seeds must produce measurably different outputs.

### Level 2: Structural behavior

Measure:

- Surface-height distribution.
- Dirt/stone transition depth.
- Cave connected components and density by depth.
- Biome positions and extents.
- Dungeon/jungle/snow/evil side relationships.
- Structure counts, overlap exclusions, and depth ranges.
- Ore distribution by depth.
- Liquid volume and settling outcomes.
- Spawn safety and access constraints.
- Hardmode diagonal biome transformation.

### Level 3: Seed parity

Only attempt exact or near-exact seed parity after:

- RNG type and call order are verified.
- Pass ordering is verified.
- Version-specific constants are mapped.
- Conditional branches and secret-seed modifiers are mapped.
- Serialization and tile IDs are pinned.

## Local fallback workflow

Use the local installation only when public APIs and experiments cannot answer a specific question.

1. Record the exact executable and assembly hashes.
2. Keep the original files read-only.
3. Inspect only the minimum required classes/methods.
4. Record version and symbol names in a provenance log.
5. Never blend an old decompiled implementation into a current-version claim.
6. Convert observations into independent tests/specifications before implementing.

Suggested questions for fallback inspection:

- What RNG object is used by a specific pass?
- What exact condition gates a structure placement?
- Which tile IDs are converted by a biome pass?
- What is the retry or exclusion policy?
- How are protected structures registered?
- Which passes are conditional on world size or special seed?

## Claims the agent must not repeat as facts without verification

- “Terraria always has exactly 103 world-generation passes.”
- “TerraForge is a source-code-faithful Python translation.”
- “The surface is primarily generated by a particular fBm formula.”
- “All cave smoothing uses one majority-rule cellular automaton.”
- “Dungeon generation is a greedy rectangle-packing algorithm.”
- “A given biome is always on a fixed side under every special seed.”
- “The 1.4.0.5 decompilation represents current 1.4.5 behavior.”

## Agent execution prompt

```text
Build an online-first technical corpus and experimental plan for reconstructing Terraria world generation only. Exclude loot, UI, sound, and unrelated gameplay.

First pin a Terraria/tModLoader version. Treat official tModLoader APIs and runtime pass enumeration as the reference architecture. Use TEdit and TerraMap for validation. Use TerraForge only as an independent procedural-generation laboratory, not as a parity implementation. Use the 1.4.0.5 decompiled repository only as a labeled historical fallback.

Produce:
1. target_ruleset.yaml
2. source_manifest.csv
3. runtime_pass_inventory.csv
4. generation_dependency_graph.md
5. experiment_matrix.md
6. world_metrics_spec.md
7. unresolved_version_differences.md
8. implementation_roadmap.md

Mark each algorithmic claim VERIFIED, EXPERIMENTALLY INFERRED, HISTORICAL, or UNVERIFIED. Never use a fixed pass count without enumerating the pinned runtime.
```

## Acceptance criteria

The agent is finished only when:

- The exact Terraria and tModLoader versions and commits are pinned.
- The actual pass queue is logged from the target runtime.
- TerraForge is correctly labeled independent rather than source-faithful.
- TEdit and TerraMap are incorporated into a reproducible validation loop.
- World metrics are defined before parity claims are made.
- All historical/decompiled observations carry version labels.
- The implementation can begin without loot, UI, or audio scope creep.
