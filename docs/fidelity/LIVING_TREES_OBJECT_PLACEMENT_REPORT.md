# LivingTrees object placement and framing

2026-09-14. **370/370 captured object calls replay exactly. All three independent LivingTrees replays now stop exactly before `GrowLivingTree_MakePassage`. LivingTrees is still incomplete.**

The target remains Terraria **1.4.5.7 / Steam BuildID 24825745 / world format 325**. All 29 pins pass. Fixtures remain Small 4200×1200, Classic/game mode 0, Corruption, numeric seeds **314159265, 1, 2**, all secret options disabled, pre-Hardmode.

## Runtime oracle actually executed

Fresh controls and instrumented generations were run for all three seeds under ignored `audit/living-trees-objects-20260914/`. Previous attempts and captures were preserved. The local host loads the unchanged pinned server; it wraps the original runtime IL with synchronous observations on the generation thread. It does not execute a recompiled decompilation or patch the game binary on disk.

Instrumentation now covers every direct tree-path `PlaceTile(187)` and `PlaceSmallPile(185)` call, including calls later in the complete vanilla pass. For each call it records:

- Full arguments, return value, ordered nested entry/exit events, and RNG before/after every traced helper. `WorldGen.genRand` aliases `Main.rand` in this target (`WorldGen.cs:4597`); the complete `inext` and 56-word state is retained.
- A scan of **all 5,040,000 native cells before and after**. `changes.bin` stores the absolute x-major index and both complete 14-byte native records for every changed cell. All nine fields are compared, including latent IDs, frames and bookkeeping headers.
- A local region padded by 32 cells on every side, expanded to enclose **every observed changed cell plus padding**, and full-grid pre/post SHA-256. Python rejects reads outside its captured region. Region changes are checked against every full-grid change record.
- Primitive/static fields and one-dimensional primitive/enum arrays from Main, WorldGen, GenVars and TileID.Sets; chest slots and primitive item state; active NPC/player fields and positions; map queue counts. These state snapshots are content-addressed and compared exactly.

This records net field changes, not the temporal order of every individual field assignment. Arbitrary engine object graphs are not serialized. Lower-level framework calls are not all intercepted: for example, the dust allocation suppression branch was source-inspected beneath the traced `KillTile_MakeTileDust`. No broader engine-state fidelity is claimed.

| Seed | Control save/load seconds | Instrumented save/load seconds | Instrumented bytes | Saved semantic equality |
|---|---:|---:|---:|---|
| 314159265 | 18.938 | 137.250 | 3,011,324 | Exact |
| 1 | 18.156 | 130.391 | 2,950,093 | Exact |
| 2 | 14.468 | 140.656 | 3,017,756 | Exact |

All six raw world hashes differ between paired processes. The existing trusted version-325 reader confirms equality of every canonical cell and saved section after **only the established normalization** of UUID, creation/save timestamps, manifest duration and physical offsets. Seeded world IDs and manifest RNG results are retained. Differing raw hashes are not evidence of generation nondeterminism.

All six children saved and loaded successfully, but redirected `exit-nosave` did not stop them. The launcher killed its own child after the recorded 30-second grace period: **child exit 1, not a clean exit**. Each saved hash remained unchanged across shutdown; stderr was empty. The earlier primary instrumentation revision also passed its semantic control comparison and remains separately preserved.

## Independent compatibility slice and per-call results

`terraexplorer/fidelity/object_placement.py` implements only the demonstrated dedicated-generation cases: type 187 with styles **47–51**, `mute=true`, `forced=false`, `plr=-1`; type 185 with **size 0/style 72** or **size 1/styles 59–61**. It derives placements from input cells, tables and arguments. It uses no captured output, coordinate patch, approximate framing or assumed placement result.

| Seed | PlaceTile calls: true / false | PlaceSmallPile calls: true / false | Exact independent calls | Exact nested entry/exit events |
|---|---:|---:|---:|---:|
| 314159265 | 83: 17 / 66 | 11: 9 / 2 | 94 / 94 | 9,270 |
| 1 | 103: 16 / 87 | 10: 8 / 2 | 113 / 113 | 10,866 |
| 2 | 149: 36 / 113 | 14: 10 / 4 | 163 / 163 | 17,366 |

Every call starts independently from its vanilla **pre-call** native region and state. For all 370 calls: **zero differing native fields, zero missing/extra net field changes, exact return value, exact nested arguments/results/order, exact RNG at every nested boundary, and exact declared globals/tables/chests/entities/queue state**. None of these calls consumes RNG; the surrounding tree's style/selection draws still do. No per-call state discrepancy or first divergence remains.

The observed helper sequence includes `EmptyTile`, `Place3x2`, `SquareTileFrame`, `TileFrame`, `TileFrameImportant`, `Check3x2`, `CheckPile`, `Check2x1`, `SolidTile2`, `SolidTileAllowBottomSlope`, and `InvalidTileForPilesOrSpeleothems`. Destruction also reaches `KillTile`, `CheckTileBreakability`, `KillTile_GetTileDustAmount`, `KillTile_MakeTileDust`, `AttemptFossilShattering`, `CheckTileBreakability2_ShouldTileSurvive`, and `CheckExploitDestroyQueue`. `TileFrame` reaches `MapUpdateQueue.Add`, whose dedicated-generation guard returns immediately. The dust sentinel and other early returns are implemented from their source conditions, not substituted from recorded results.

| Nested helper | 314159265 | 1 | 2 |
|---|---:|---:|---:|
| Place3x2 | 83 | 103 | 149 |
| SquareTileFrame | 105 | 130 | 188 |
| TileFrame / MapUpdateQueue.Add, each | 970 | 1,245 | 1,717 |
| TileFrameImportant | 224 | 219 | 456 |
| Check3x2 | 222 | 218 | 456 |
| CheckPile / Check2x1 | 2 / 0 | 1 / 1 | 0 / 0 |
| KillTile | 5 | 11 | 3 |

Five placement invocations mutate outside their nominal footprint. For example, seed 314159265 call 5526 attempts `(602,164)` but also changes `(604,164)`, `(604,165)`, `(605,164)`, `(605,165)` while framing destroys an already damaged neighboring object. These mutations match independently. Failed placement can also clear latent origin data; rejection is not implemented as a no-op. Synthetic tests cover both behaviors.

Only neighboring important types 185 and 187 were reached beneath the selected object calls. `SlopeTile` and `TileFrameCosmetic` were not reached there. General object styles/types/flags, other important-tile handlers, unsupported platform framing and pile destruction remain explicit unsupported dependencies. Null tile allocation and general gameplay placement are outside this native, fully allocated generation context.

Source inspected, rather than executed as source, under the unchanged `Game Reference/02_static/decompiled/client/Terraria/`:

- `WorldGen.cs`: `PlaceSmallPile:48136–48194`, `CheckPile:48196–48292`, `Check2x1:48294` onward, `Check3x2:50549` onward, `Place3x2:53487–53617`, `PlaceTile:60925` onward.
- Framing/support: `SquareTileFrame:82780`, `TileFrame:83933`, `TileFrameImportant:87582`, `SolidTile2:72231`, `SolidTileAllowBottomSlope:71896`, invalid pile supports `40255`.
- Destruction: `KillTile:65162–65567`, breakability `63999/64111`, fossil guard `65129`, dust `68267/68357`, exploit queue `65569`.
- `Tile.cs:749` and `909`: clearing/paint/coating masks; `Collision.cs:1684–1709`: entity rectangles; `Map/MapUpdateQueue.cs:Add`; `Dust.cs:86–132`: allocation suppression.

## New exact replay boundary

The existing replay uses the validated helpers and preserves the source's `continue` after the lower-canopy PlaceTile branch. It advances through ordinary canopy/object work and stops before any passage implementation.

| Seed | Previous exact changed cells, rechecked | New exact changed cells | Completed object calls in prefix | Next unsupported call |
|---|---:|---:|---:|---|
| 314159265 | 651 | 2,419 | 47 | `GrowLivingTree_MakePassage(217,4,573,577,false)` |
| 1 | 613 | 2,324 | 54 | `GrowLivingTree_MakePassage(228,4,3782,3786,false)` |
| 2 | 567 | 2,129 | 46 | `GrowLivingTree_MakePassage(231,4,1543,1547,false)` |

At each new checkpoint the **entire native world grid**, call arguments, RNG, relevant globals, solidity table and chest state match. All completed object invocation/results and their nested event sequences match: **4,534 / 5,224 / 5,098 events**. Previously accepted first-object checkpoints were rerun explicitly and remain exact. Seed 2's two earlier rejected tree calls also retain exact results and RNG states.

The 370 independent object fixtures span the complete vanilla pass; only the **147 calls before the first passage** have been reached by integrated Python replay. Later vanilla pre-call fixtures were not used to bypass passage generation.

**Full-pass success is not claimed.** Compared with the complete vanilla post-pass world, stopped native states still differ in **4,338 / 4,793 / 9,272 cells**. Full-pass final RNG, complete invocation/result sequence and final chest state remain unmatched. No divergence occurs within the verified prefixes; the remaining difference starts with the deliberately unsupported passage system.

## Evidence and verification

Detailed local evidence is under `audit/living-trees-objects-20260914/`:

- `seed-<seed>-objects.json`: every call ID/argument/result, native comparison, RNG/state and nested-sequence comparison; hashes of all capture inputs. All three commands exited **0**.
- `seed-<seed>-pass-verified.json`: old and new checkpoint proofs, full-pass differences, pins, manifest checks and exports. All three commands exited **2**, meaning an exact but incomplete boundary; divergence/unverified boundaries now exit 1.
- `seed-<seed>-control/` and `seed-<seed>-capture-final/`: exact configurations, child command lines, PIDs, timings, exit codes, stdout/stderr and file hashes in `result.json`; native/object captures in `capture/`.
- `CaptureHost.cs`, `ObjectCapture.cs`, compiled host, `run_fixture.py`, and preserved `v1-*` host sources/executable: local instrumentation. Original method IL hashes/tokens/module identity are in each capture's `instrumentation.json`.
- `verification.json`, `pass-final-verification.json`, `checks-final.json`, `verified-commands.json`, associated stdout/stderr, and `preservation-verified.json`: command and preservation evidence. Earlier verification outputs remain preserved.

Actual generation commands used `python audit/living-trees-objects-20260914/run_fixture.py --seed <seed> --output <fresh-directory> --timeout 1800`, adding `--capture` for the instrumented process. The host was compiled with the installed .NET Framework x86 C# compiler, `/unsafe /platform:x86 /r:System.Web.Extensions.dll /r:System.Core.dll`, against the two local host source files. No dependencies were installed.

Replay commands used `python -m scripts.fidelity_living_trees_objects` and `python -m scripts.fidelity_living_trees_replay`, each with `--capture`, `--control-world`, `--captured-world`, and a fresh `--output` under audit. The exact absolute arguments are retained in the verification records; reruns require new output filenames.

Actually run: **47 focused tests; 106 full tests; 93.54% coverage**, with the unchanged 75% gate; repository-wide Ruff lint/format; `mypy terraexplorer` (**22 files**). A first synthetic trace test exposed shared test-fixture RNG dictionaries; fixing that fixture resolved its single failure. A later format check caught mixed newlines in the updated replay docstring; formatting was corrected and checks/replay rerun. No runtime replay mismatch was hidden or normalized away.

The initial checkout was clean at `694c0e9fbb3b3ab1d333828aa0a49494f9fe4f52`. Only three existing tracked files changed: the isolated LivingTrees fidelity module, its replay script and its synthetic prefix test. Additions are the object module, object validator, synthetic object tests and this report. The generator, structures, art, GUI, README, public entry points, target lock and raw corpus remain untouched. Changes are unstaged and uncommitted.

**Stop here.** The next bounded dependency is `GrowLivingTree_MakePassage` and its passage/tunnel/room work; none was implemented in this milestone.
