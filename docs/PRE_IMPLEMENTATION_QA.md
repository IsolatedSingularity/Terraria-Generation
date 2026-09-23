# Pre-implementation adversarial QA

Audit date: 2026-09-23  
Code baseline reviewed: `main` at `3f49181c08554fd88fa0e3e35d1cdefc2572c933`  
Scope: adversarial audit of the supported `terraexplorer/` product, fidelity/replay tooling, tests, CI/release workflows, legacy surfaces, generated-artifact contracts, and agent handoff. No implementation code was changed by this audit.

## Read this first

Do not treat a green test run as proof of Terraria fidelity or desktop correctness.

Two boundaries must remain explicit:

1. TerraExplorer Small generation is an independent source-informed model, not seed-compatible Terraria generation.
2. The current Small product path directly reuses code under `terraexplorer/fidelity/`. That directory is therefore not purely disposable audit code despite older comments saying otherwise.

Address the P0 ownership/test-gate items before broad generation work, then the P1 state/UX correctness items. Preserve pinned oracle evidence and historical reports.

## Baseline and evidence actually checked

Reviewed repository tree, latest commits, README, `OPEN_PROBLEMS.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, architecture/fidelity docs, `docs/fidelity/*` routing and reports, `pyproject.toml`, both GitHub Actions workflows, packaging scripts/spec, configuration/model/pipeline/pass catalogue, generation/source-backed generation, renderer/exporters, spawn study, simulations, GUI, canonical v325 importer, replay/object-placement code, legacy `Engine/`, `Code/`, `Advanced/`, and all tracked tests.

At the reviewed HEAD, GitHub Actions CI run 44 completed successfully on Ubuntu and Windows for Python 3.11/3.12/3.13 plus lint, format, mypy and Python package build. The Ubuntu 3.11 log reports **106 passed, 11 skipped, 88.82% coverage**. The skipped cases are the genuine-oracle/canonical-import tests whose fixture is under ignored `audit/`; clean CI therefore does not exercise that oracle path.

The September 14 local high-fidelity report records stronger local evidence, including 117 tests, local oracle worlds, generation validation, media repeat checks and package builds. Those are valuable historical evidence, not a substitute for a current clean-checkout gate.

No local checkout was available to this browser audit, so no new pytest process, live Tk desktop, PyInstaller executable, memory profiler or native Terraria process was executed. Static/control-flow findings below are labeled accordingly.

## P0: resolve before substantial further implementation

### P0.1 Production Small generation and fidelity replay code have a contradictory ownership boundary

**Status:** statically proven architectural/governance risk.

`terraexplorer/source_generation.py` is in the default GUI's Small generation path. It imports and uses `LivingTreesReplay` from `terraexplorer.fidelity.living_trees`, `RAW_DTYPE` from `terraexplorer.fidelity.pass_snapshot`, and `components` from `terraexplorer.fidelity.structures`. The Small Living Tree handler subclasses the replay implementation.

However, `terraexplorer/fidelity/living_trees.py` still states that it is "never used by the approximate generator", and `docs/ARCHITECTURE.md` does not describe `source_generation.py` or this product-to-fidelity dependency. An agent can reasonably "clean up audit-only fidelity code" and break the GUI-default Small generator.

A second contradiction compounds it. `CONTRIBUTING.md` tells generation handlers to consume only the RNG passed by the pipeline. The source-backed Small Terrain, Dungeon, Temple and Living Tree handlers intentionally create/reset their own `UnifiedRandom` streams from the normalized seed. The README documents that behavior. An agent following CONTRIBUTING literally could silently change reproducibility and generation behavior.

**Acceptance criterion:** establish one explicit ownership contract before touching these modules. At minimum, architecture/contributor guidance must say which fidelity primitives are product dependencies and which Small handlers intentionally own/reset native-style RNG streams. Add a focused test that fails if the product Small Living Tree dependency or intended RNG boundary changes accidentally. A refactor is not required merely to close this gate.

**Timing:** now, before broad fidelity/generation changes.

### P0.2 Clean CI does not gate the genuine canonical/oracle path

**Status:** directly evidenced from current HEAD CI.

The clean CI run is green with **11 skipped tests**. `tests/test_canonical_import.py` gates its module fixture on `audit/runtime-oracle-20260914/run-1/bootstrap-small.wld`, which is intentionally ignored and absent in clean CI. This skips not only genuine-world assertions but several malformed/import-regression tests built by mutating that fixture.

A regression in the canonical importer can therefore coexist with a fully green required CI matrix.

**Acceptance criterion:** split always-runnable synthetic format/parser tests from genuine-oracle tests so clean CI exercises corruption/truncation/pointer/importance/unknown-ID behavior without proprietary fixtures. Keep genuine oracle validation as an explicit local fidelity gate, and make fidelity-changing handoffs report whether that gate was run rather than allowing "green CI" to imply it.

**Timing:** now, before extending importer/replay compatibility work.

## P1: concrete bugs and high-value adversarial tests

### P1.1 Phase controls do not mean what their labels imply

**Status:** statically proven control-flow bug / UX contract ambiguity.

Pass phases are assigned only by index ranges: 1-16 Terrain, 17-32 Caves/Biomes, 33-50 Structures, 51-69 Simulation, 70-107 Polish. The GUI exposes those as user-selectable groups.

Several structure-producing passes are outside the Structures phase:

- `Buried Chests` (61) can create `Underground cabin` markers;
- `Spider Caves` (65) and `Gem Caves` (66) create structure markers;
- `Temple` (68) mutates Temple contents;
- `Floating Island Houses` (71) creates houses;
- `Hellforge` (74) mutates Underworld houses;
- `Micro Biomes` (103) creates minecart-track markers.

The existing phase-control test only proves that disabling Structures suppresses a Dungeon and marks disabled passes. It does not prove that "Structures off" yields no later structure creation.

**Acceptance criterion:** decide whether phase controls are semantic feature groups or contiguous timeline bands, then test that contract across all 107 passes. If semantic, disabling Structures must not create structure markers or structure-only mutations later. If timeline bands, rename/document them so users and agents do not infer feature isolation.

**Timing:** before relying on phase controls for experiments or comparisons.

### P1.2 GIF export can silently describe a different world from the one requested or displayed

**Status:** statically proven user-visible bug.

CLI generation accepts `--scale small --gif ...`, but `terraexplorer.cli._generate` always constructs a new **Preview** config for the GIF. The README's headless Small example includes `--gif generation.gif`, which makes this easy to misread as a Small-generation animation.

The GUI has a stronger state-sync problem: `export_gif()` reconstructs a fresh Preview config from the **current controls**, not `current_world`. A user can generate seed A, change seed/evil/phase controls to B without generating, and export a GIF for B while the screen still shows A.

**Acceptance criterion:** a GIF export must either be derived from the displayed/generated world's config and clearly identified as a Preview re-generation, or Small GIF must be explicitly unsupported/rejected. Add a regression test where controls are changed after generation and verify export cannot silently switch world identity.

**Timing:** fix before treating exported GIFs as evidence.

### P1.3 Split-comparison tile inspection addresses the wrong world

**Status:** statically proven GUI coordinate/state bug.

In split comparison, the image is `previous | divider | current`, but `_world_for_view()` returns the current world for "Split comparison". `_inspect_tile()` converts the whole combined-canvas x coordinate as though it belonged to that one world.

Consequences: clicks on the left image report current-world state rather than displayed previous-world state; clicks on the right can map beyond the current world's width and be rejected; different Preview/Small dimensions worsen the mismatch.

Current GUI tests cover only the single Current Small fit/probe transform.

**Acceptance criterion:** tests must click both halves and map to the correct world and local coordinate, including previous/current worlds with different dimensions/scales. Divider clicks should fail cleanly.

**Timing:** fix before using split comparison for manual QA.

### P1.4 Difficulty is presented as a generation option but changes only metadata

**Status:** statically proven product-contract ambiguity.

Search of the active package shows `config.difficulty` is read into metadata and transported through CLI/GUI/GIF configs, but no generation handler branches on it. Classic, Expert and Master worlds with otherwise identical configs therefore have identical generated arrays under current code.

`docs/FIDELITY.md` says difficulty "does not yet reproduce all" difficulty-specific branches, which understates that the active generator currently reproduces none.

**Acceptance criterion:** either label Difficulty as metadata-only in GUI/CLI/current docs, or add a test-backed behavioral contract when implementation begins. Until then, do not use difficulty selection as evidence of distinct world generation.

### P1.5 Current fidelity/version documentation has multiple competing truths

**Status:** stale documentation/governance problem.

The pinned target and current README/high-fidelity reports are Terraria **1.4.5.7 / BuildID 24825745**. Current-looking files still say 1.4.5.6, including `docs/FIDELITY.md`, `docs/VISUAL_FIDELITY.md`, and biome-spread comments/docstrings.

More seriously, the prior `docs/fidelity/START_HERE.md` still described runtime confirmation as blocked and framed the bootstrap as if no importer existed, while later tracked reports record successful runtime oracle generation and a canonical importer.

This audit updates the routing file without rewriting historical bootstrap evidence.

**Acceptance criterion:** one current routing page identifies 1.4.5.7 as the live target and links newer runtime/import/generation reports; historical 1.4.5.6 expectations remain clearly historical, not current instructions.

### P1.6 Spawn-heat documentation contains superseded semantics

**Status:** stale documentation, statically verified against code.

Current `spawn_opportunity.py`, README and high-fidelity report correctly define the heat map as conditional ground-candidate geometry only. However, `CHANGELOG.md` still says it accounts for depth, biome, light, spawn safety, housing, Peace Candle and Sunflower suppression, while `docs/VISUAL_FIDELITY.md` still describes depth/biome/darkness/attempt-frequency suppression.

Those are older semantics that the latest implementation deliberately removed.

**Acceptance criterion:** current release/fidelity descriptions must match `ground_candidate_mass()`: supported-floor/clearance geometry plus stated liquid/height exclusions, not NPC spawn rate or environmental weighting.

### P1.7 Main CI does not prove the packaged Windows desktop

**Status:** credible failure hypothesis / CI blind spot.

Main CI tests Python imports/tests on Windows and builds Python distributions on Ubuntu. `release.yml` builds `TerraExplorer.exe` only on tag/manual release and verifies only that the file exists. It does not launch the executable or exercise a packaged GUI path.

GUI coverage is excluded from coverage accounting and current tests are mostly transform/unit shims rather than live Tk behavior.

**Acceptance criterion:** before release, build the EXE from the exact release commit and perform a harmless packaged smoke launch/close plus one generation/export path on Windows. Record that separately from normal CI.

### P1.8 Small-world peak memory and cancellation latency are unmeasured

**Status:** credible performance hypothesis supported by allocation structure and existing report.

The existing report already notes 19-23 second Small runs and unmeasured peak memory. Current source-backed Living Trees allocate a full `4200 x 1200` 14-byte RAW_DTYPE grid (about 67 MiB before additional masks/arrays). `render_world` creates multiple full-grid RGB/float intermediates. GUI generation captures many milestone renders, and split comparison renders two worlds before thumbnailing.

Cancellation is checked only between passes, so one expensive handler is an uncancellable interval.

**Acceptance criterion:** measure peak RSS and longest single-pass/cancel latency for at least one Small generation and split comparison. Set a regression ceiling before adding more full-grid native projections or cached state.

## P2: cleanup, ambiguity and deferred risks

### P2.1 Export formats are not versioned persistence formats

**Status:** architectural risk, not a current defect if exports remain analysis-only.

NPZ stores six arrays plus metadata JSON, but not a schema version, full config object, layers, structures or pass results; there is no supported loader. Summary JSON stores metadata/structures/pass telemetry but no arrays. Neither is a round-trip world-save contract.

**Acceptance criterion:** if future work starts loading old exports or caching generated worlds, define a schema/version/migration policy first. Until then label NPZ/JSON as analysis exports, not durable saves.

### P2.2 Historical code remains executable and sometimes sounds authoritative

**Status:** agent-navigation risk.

`OPEN_PROBLEMS.md`, `Advanced/README.md`, and `Code/README.md` correctly mark `Engine/`, `Code/`, and `Advanced/` as historical archives. But `Engine/__init__.py` still advertises "accurate implementations", and the scripts remain directly runnable. `.jenova-research` also retains older 103-pass language.

They are not shipped in the wheel, but a local agent can still import/run them and revive obsolete assumptions.

**Acceptance criterion:** preserve history, but keep unmistakable archive banners at executable entry points/agent routing. Do not copy algorithms from these directories into active code without re-verification.

### P2.3 Pass telemetry contains an unused `changed_tiles` field

**Status:** stale/dead telemetry.

`PassResult.changed_tiles` exists but the pipeline never fills it. Future tools must not interpret null as "zero changes" or proof a pass is annotation-only.

### P2.4 Package/version identity is weak for the amount of post-1.0 change

**Status:** release hygiene risk.

`pyproject.toml` and `terraexplorer.__version__` remain 1.0.0 while substantial work is accumulated under `[Unreleased]`. This is not a runtime bug, but distributed artifacts can become hard to distinguish without commit metadata.

**Acceptance criterion:** release artifacts identify a release version and exact source commit; do not infer feature set from `1.0.0` alone.

### P2.5 Cross-repository chunk work is a useful boundary reference, not a Terraria generation algorithm

**Status:** future-design guidance, not a defect.

Bloc Fantome now cleanly separates approximate/source-informed terrain from exact imported native Java chunks, and uses chunk-indexed storage/culling for large editable scenes. That separation is a good lesson for TerraExplorer.

Do **not** port Minecraft-style chunk-local generation semantics into Terraria world generation. Terraria's audited passes and RNG/global-state dependencies are world-level. Chunking is safe as a storage/render/cache partition or for exact imported reference regions; chunk-local procedural generation would require proven dependency halos/global state and otherwise becomes another approximation.

Also do not treat the currently approximate Minecraft-Generation chunk path as a fidelity oracle merely because it is chunk-shaped.

## Adversarial test matrix for the next agent

| Area | Minimum adversarial check | Expected contract |
|---|---|---|
| Clean CI / importer | Synthetic v325 malformed/truncated/pointer/RLE/high-ID tests with no `audit/` fixture | Must execute, not skip |
| Genuine oracle | Declared local oracle fixture gate when touching importer/replay | Report RUN/PASS or explicitly NOT RUN |
| Small product dependency | Generate Small Living Trees after fidelity-module edits | Default Small path remains deterministic/operational |
| RNG ownership | Repeat Small seed through Terrain/Dungeon/Temple/Trees | Intentional reset/stream rules unchanged unless migrated |
| Phase controls | Disable each phase and inventory mutations/markers | Matches documented phase meaning |
| Difficulty | Same seed Classic vs Master | Intentionally identical and labeled, or tested differences |
| GIF CLI | `--scale small --gif` | No silent scale/world identity mismatch |
| GIF GUI | Generate A, change controls to B, export | Cannot silently export B while displaying A |
| Split compare | Click left/right/divider, including Preview vs Small | Correct source world/coordinate |
| Cancel | Cancel during slowest Small pass | Bounded stop latency recorded |
| Repeated GUI | generate/cancel/generate; compare/evolution changes | No stale events/frames/wrong-world probe |
| Export paths | existing file, unwritable dir, long/non-ASCII path | Clean error; no false success |
| NPZ/JSON | inspect declared contents | No round-trip claim without schema |
| Renderer | Small current/previous/split with overlays/fit/zoom | Bounded memory and correct transforms |
| Release EXE | build exact release commit, launch/generate/export/close | Packaged behavior verified |
| Multi-seed | held-out seeds beyond 1/2/314159265/17/99 | Bounds/connectivity/liquid/spawn invariants hold |
| Docs/version | search active docs for 1.4.5.6, blocked runtime, old spawn model | Historical vs current state unambiguous |
| Legacy navigation | inspect/import `Engine/`, `Advanced/`, `Code/` | Archives cannot be mistaken for active product |

## What current green tests actually prove

They provide useful regression evidence for the active approximate generator: deterministic same-process outputs, many array/material invariants, selected structure topology, spread rules, CLI output creation, compatibility aliases, source-backed Small checks on a few seeds, RNG vectors, and replay/object-placement behavior that does not require the missing genuine-world fixture.

They do **not** prove whole-world Terraria seed compatibility, clean-checkout genuine `.wld` import, real GUI thread/state correctness, split-comparison probing, current-world GIF identity, packaged EXE behavior, peak memory/long-session stability, cancellation latency inside a pass, broad held-out structure distributions, or semantic correctness of every "modeled" pass.

## Stop conditions for implementation agents

Do not "fix" source-backed RNG resets merely because they violate the old contributor rule. First decide and document the intended stream contract.

Do not delete/move `terraexplorer/fidelity/` as audit-only while `source_generation.py` imports it.

Do not convert the Terraria generator into independent chunk-local generation based on Minecraft work.

Do not use the 117-test local historical report as evidence that current clean CI exercises the oracle fixture.

Do not broaden fidelity claims from a passing invariant, attractive render, or green CI job.

## Audit modifications

This audit intentionally changes documentation only:

- adds this file;
- updates the current fidelity routing page to point here and distinguish historical bootstrap state from current runtime/import evidence;
- adds a small root README agent-route pointer.

No implementation, tests, media, historical evidence, generated worlds, or pinned target data are changed.
