# Terraria fidelity: implementation plan and first local agent task

Prepared for Jeff, 12 September 2026.
Repository: `Documents/Github/Terraria-Generation` (resolve the actual local checkout, including redirected Documents folders).

## Product decision

Preserve TerraExplorer's existing artistic renderer and builder UI. Replace invented terrain, structure geometry, placement, and eventually spawning rules with implementations grounded in the version-locked game and executable comparisons.

The intended destination is an independent generator: no installed Terraria, Terraria executable/assemblies, or live game process required to generate new supported worlds. The owned game and local extracted corpus may be development and validation tools. Importing an existing vanilla world or pasting captured structures is a useful intermediate capability, but is not an independent generator.

Target same-seed, same-settings parity for a pinned vanilla build. Establish this incrementally; never advertise whole-world parity because a few local fixtures pass. A routine can match recorded inputs exactly before earlier world-generation passes have been reimplemented. Unsupported configurations must be explicit, not silently approximated.

Priority structures: Dungeon, Jungle Temple, living trees. Then pyramids, floating islands/lakes, Aether, hives, relevant biome structures, Underworld buildings, and underground cabins. Broader coverage remains on the backlog; this first task is not authorization for an all-at-once rewrite.

Items, inventory, combat, crafting, loot presentation, audio, and UI redesign are out of scope. Preserve physical objects or internal computations only when they affect terrain, structure geometry, placement, spawn rules, or random-state evolution needed for parity. Do not implement a loot system for its own sake.

## What Nova actually inspected

Read-only review of GitHub main's active Python implementation, selected tests, and previously saved extraction logs. No local game execution or local index-query tests were performed. The complete ignored `Game Reference/` corpus was not directly accessible through GitHub.

Observed source artifacts:

- `pyproject.toml`: current entry points target `terraexplorer`, not the legacy `Engine/` scripts. File blob `15361eeb059250f102e8fd1e2b33c528b4579d83`.
- `terraexplorer/generation.py`: custom terrain noise; procedural approximations for the three priority structures. Blob `cb22b9400c1841800497791d6732b53d7c89e75b`.
- `terraexplorer/pipeline.py`: `_rng_for` hashes seed plus pass label and constructs a separate NumPy generator for every pass. Blob `4dcbb2aa40f604f6de4433b51100400b99d3ada8`.
- `terraexplorer/config.py`: project-specific text-seed conversion; Preview is 240 x 140; Small is 4200 x 1200. Blob `26b7f5024f44c492032449b34d60cccb2f3aaaf7`.
- `terraexplorer/model.py`: tile and wall buffers use `uint8`; the explicit model lacks tile-presence flags, slopes/half-blocks, wires/actuation, and object frame coordinates. Blob `539b31bb61225f62b6b9dcbb7266e5ef88fc3087`.
- `terraexplorer/passes.py`: hardcoded 107-entry sequence with project fidelity labels. Blob `8d717432f5061d0139ec40a0b9c46bf2cce91879`.
- `tests/test_pipeline.py`, reviewed lines 1-190: self-repeatability, buffer types, feature presence, and some structural checks, not independent vanilla-output comparison. Blob `61a33962b7dd7d7570e46993c4394df2fae8094b`.
- `docs/VISUAL_FIDELITY.md`: existing local-reference research and explicitly limited fidelity claims. Blob `da85e323f34e805f8d3784a802b69340700b70ef`.

Specific implementation observations, to reconfirm locally:

- `terrain` combines smooth noise with a sinusoidal component.
- `dungeon` chains custom rectangular rooms with L-shaped corridors; a Small run starts with a requested count of 32 rooms but may terminate earlier.
- `jungle_temple` uses a fixed 190 x 120 Small envelope and rows of rooms, with a separate final chamber.
- `living_trees` stamps a solid trunk, an elliptical canopy, and root walks in the reviewed handler. Audit later passes before claiming no other tree mutations exist.
- The current code is a useful project model, not a failed attempt at executing the vanilla algorithms exactly. The new product target changes its compatibility contract.

Saved extraction evidence:

- `BUILD-20260823-145312.log` records 1,549 decompiled client C# files, 15,997 assets, a 301.4 MiB SQLite index, an offline query executable, and a successful final corpus health check. It still records failed optional golden-world validation.
- The earlier `BUILD-20260823-144723.log` gives the runtime cause: `Microsoft.Xna.Framework, Version=4.0.0.0` could not be loaded while invoking TerrariaServer. That earlier build also had a later PowerShell validation error; do not confuse it with the successful later corpus health check.
- The previous corpus location was under the user's profile as `Terraria-Reference`. The current repo may contain a copy, junction, or other layout. Discover it; do not assume an old absolute path is authoritative.

The README is not the new requirements document. Jeff identifies its narrative as stale. Preserve its art and plots during the first task, but do not treat plots or equations as validation of vanilla behavior.

## Roadmap and acceptance gates

### Stage A: make the existing reference usable and establish the baseline

Resolve the local corpus, inspect its target lock, and verify hashes and query tools. Expected research target is 1.4.5.6, but the actual manifest and binary must decide. Freeze one build, platform/runtime, world size, difficulty, evil choice, and ordinary numeric seed for the first fixture. Do not switch to a freshly downloaded game version without explicitly changing the target and fixtures.

Repair retrieval only where tests show it fails. Keep the raw corpus and working database in place. Prefer a small routing layer over moving thousands of files, rebuilding all indexes, or launching a new embedding system.

Recover the golden-world path and prove that the reference executable can generate a world. Run the same configuration twice in fresh processes; compare a normalized semantic world state, not just file bytes. Record nondeterministic fields explicitly. A runtime failure must not be reported as a successful fidelity check.

Gate: version and hashes recorded; topic queries demonstrated; at least one real world available or a precisely reproduced blocker; no production-generator rewrite.

### Stage B: actual world data, existing artwork

Implement a version-aware local importer or game-side exporter producing normalized tile state. Verify any chosen third-party reader against the pinned format; do not assume current tModLoader or TEdit implies exact support for the target build.

Keep a canonical data representation separate from the project's artistic tile palette. Preserve tile presence, full tile/wall IDs, liquids, slopes and half-blocks, relevant object frames, wires/actuation, and required world flags. Unknown IDs must survive in canonical state. The renderer may use an explicit visual fallback, but the importer must not turn unknown data into air or collapse distinctions used by mechanics.

Render real Dungeon, Jungle Temple, and living-tree crops in TerraExplorer's art style. This supplies immediate faithful geometry without claiming it was independently generated. Use a neutral diagnostic render and field comparisons to validate the importer as well as attractive views for humans.

Gate: lossless storage of the declared canonical fields; crops traced to a recorded seed/build/world; unsupported fields disclosed; imported data clearly labeled as imported.

### Stage C: deterministic compatibility foundations

Introduce a separate compatibility path while keeping the existing generator as an explicitly labeled legacy/project model. Validate seed conversion, the actual generator(s) and their complete states, random-call sequencing, numeric semantics, tile operations, placement checks, and generation pass ordering against the pinned implementation.

Do not assume every pass shares one stream or every call uses only `WorldGen.genRand`; discover the actual stream routing and state resets. Do not retain independent NumPy pass streams in compatibility mode simply because they are convenient. Likewise, do not impose a fixed pass count borrowed from another game version.

For ignored subsystems, inspect whether their calls affect later random state or geometry. Preserve proven dependencies, including branch-dependent random draws, without exposing an unnecessary gameplay feature. A fixed dummy draw count is not valid unless it is demonstrated to match.

Gate: primitive and random-sequence fixtures pass; expected and resulting RNG states are compared; no fallback to an approximate implementation in a claimed-compatible path.

### Stage D: exact structure replays, then full integration

For Dungeon, Jungle Temple, and living trees, capture the actual pre-call world and relevant global state, method inputs, all used RNG states, and expected post-call state. Include any later passes that finalize the structure. A crop is sufficient only when its bounds plus dependencies provably contain all reads and writes needed for replay; otherwise capture a larger region or full world.

Prefer non-mutating instrumentation. Verify that instrumentation itself does not change final output compared with the uninstrumented baseline. Trace only a failing region or routine when full traces are too large.

Independently implement and compare each routine. Compare changed-cell masks, every declared state field, writes outside expected bounds, placements/rejections, and final RNG states. A pasted captured output is not a passing implementation.

Then integrate compatible terrain, cave, biome, and placement passes in actual dependency order. Debug the first divergent pass rather than visually retuning the final world. Passing a structure replay with a vanilla pre-state does not imply the standalone pipeline can yet construct that pre-state.

Gate: per-routine replay agreement on fixed and held-out cases; then end-to-end world-state agreement for declared configurations. Say “matched these fixtures” rather than “all seeds are exact.”

### Stage E: complete structure coverage and scenario-conditioned spawn maps

Extend the same method to the remaining major structures and eventually cabins. Record legitimate absence and failed placement attempts; do not force every structure into every seed for screenshots.

Build future spawning analysis as three separate layers: spatial eligibility, conditional enemy selection, and attempt rate/population suppression. Declare player position or hypothetical player-position interpretation, time, progression, events, difficulty, and relevant world/tile/wall conditions. Never label an arbitrary darkness/depth color field as exact per-enemy probability.

### Stage F: demonstrate game independence

Run compatible generation with the game and raw reference corpus unavailable. This should generate new supported worlds, not merely render stored ones. Keep developer-only oracle and extraction tooling separate from distributable runtime code and data. Do not commit the proprietary corpus, game executables, or ripped textures. Review the provenance and redistribution constraints of any reused third-party code before including it.

## First local agent task: execute only this bounded bootstrap

You are the implementing agent in Jeff's local `Terraria-Generation` checkout. The roadmap above is context, not authorization to complete every stage in one run.

### 1. Establish a safe working baseline

Read any existing local agent instructions. Inspect `git status`, the active branch, and `git rev-parse HEAD`; preserve uncommitted work. Confirm the active package and the source observations above. Do not reset, delete, move the raw corpus, perform broad refactors, push, or regenerate README media. Use an isolated worktree or additive files where practical, without hiding the corpus from the task.

Find the actual `Game Reference/` root and its backing corpus. Inspect the target manifest, source/index hashes, local query-tool help, agent entry point, source map, and validation reports. Record exact observed paths locally; do not publish machine-specific absolute paths as repository-wide configuration.

### 2. Run a retrieval acceptance check

Use the existing tool's actual documented syntax. Do not invent flags for `Query-TerrariaLibrary.ps1`, `Query-Terraria.ps1`, or `TerrariaIndexer.exe`.

Demonstrate answers to these ten questions with bounded source excerpts:

1. Where is seed parsing and generator initialization implemented?
2. Where is the actual generation pass list constructed for this target and configuration?
3. What RNG streams and global state do the Dungeon routines read and update?
4. Where are Dungeon layout, room, corridor, entrance, and finishing routines?
5. Which routines construct and later finish the Jungle Temple?
6. Which routines construct living trees and any connected underground structures?
7. Which checks reject or protect proposed structure placements?
8. Where are tile presence, shape, object-frame, wall, and liquid semantics defined?
9. How are world generation and world saving invoked in the matching runtime?
10. Where are the spawning dispatcher and relevant biome/wall/player checks for later work?

For each answer, record version/hash, a relative source path, exact symbol/signature and line range, directly relevant dependencies, and unresolved questions. Inspect callers and callees where needed. Do not load the whole of `WorldGen.cs` into context. Missing evidence is a failed retrieval result, not permission to guess.

Create or repair only the small routing layer needed for these tasks. Reuse existing topic guides. Produce substantive packets for Dungeon, Temple, living trees, RNG, and tile state. Each packet must connect source evidence to future replay inputs/outputs and identify what remains unverified. A symbol list or placeholder heading is insufficient.

### 3. Restore one authoritative runtime test

Inspect the recorded golden-world failure and the current environment. Diagnose the missing XNA assembly or its dependencies using the existing install/runtime rather than assuming the old error still reproduces. Use matching client/server artifacts from the pinned build.

Prefer available local runtime dependencies and reversible per-process resolution. Do not install arbitrary DLLs, silently upgrade the target, or make privileged system changes. Where an actual installer/permission action is required, report the precise dependency and official source. Continue independent read-only work instead of inventing success.

Attempt one fixed numeric-seed, ordinary Small world with fully recorded settings; difficulty must be explicit. Repeat in a fresh process if successful. Verify that the generated file is parseable and matches the intended version and dimensions. Compare normalized semantic outputs when a reliable reader/exporter is available. Raw hashes alone are an intermediate integrity check, not proof of semantic determinism.

If the bootstrap is blocked, stop the runtime attempt with exact command, exit code, stderr, resolved artifact paths/hashes, and the smallest necessary next action. Do not start a different full reimplementation to work around a missing dependency.

### 4. Produce a reviewable handoff and stop

Produce:

- `docs/fidelity/BOOTSTRAP_REPORT.md`: observed baseline, corpus status, ten retrieval results, runtime result, commands executed, changed files, limitations, and next task.
- `docs/fidelity/TARGET_LOCK.json`: actual pinned game/build/hash/runtime/settings information, excluding credentials and unnecessary machine identifiers.
- `docs/fidelity/COMPATIBILITY_GAPS.md`: evidence-backed gaps in current generation/state/RNG/tests, including the local implementation paths.
- A compact routing entry, extending existing instructions where appropriate, plus the five source packets. Keep excerpts and proprietary corpus details local when needed; public docs can carry symbol locators and hashes instead.
- A local-only golden-world manifest and resulting file paths/hashes if generation succeeds, or a precise failure record otherwise.

Run relevant lightweight tests for any scripts you changed. Distinguish tests actually executed from static inspection. Report PASS, FAIL, BLOCKED, or NOT RUN separately; do not turn missing golden fixtures into passing fidelity tests.

Do not implement the three replacement structure generators in this first task. Do not create fake vanilla fixtures from TerraExplorer output. The next task is a canonical importer/exporter plus verified renders of the three real structure types, followed by deterministic routine replay work.

## Evidence trail

Repository source observations were obtained through the connected GitHub tool. Reconfirm them against the local commit because the checkout may differ from the inspected snapshot. Saved build logs are historical evidence, not proof of current runtime health.

Public primary references used as orientation, not substitutes for the pinned vanilla corpus:

```text
https://docs.tmodloader.net/docs/stable/class_world_gen.html
https://docs.tmodloader.net/docs/stable/struct_tile.html
https://docs.tmodloader.net/docs/stable/struct_n_p_c_spawn_info.html
https://github.com/TEdit/Terraria-Map-Editor
```

The WorldGen API documents world-generation randomness and procedural structure methods. Tile documentation distinguishes presence, tile/wall IDs, slopes, liquids, actuation, and frames. NPCSpawnInfo illustrates context dependence of spawning. These are current tModLoader interfaces; exact 1.4.5.6 vanilla behavior must still be established locally.
