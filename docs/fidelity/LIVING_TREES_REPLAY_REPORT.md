# LivingTrees runtime oracle and independent replay

2026-09-14. **Oracle verified for three seeds; independent full-pass replay remains incomplete.** The Python implementation matches the entire native tile grid, RNG state, solidity table, and chest state at the first object-placement boundary in each fixture. It stops explicitly before that unsupported dependency. None of these fixtures is a full replay success.

Target remains **Terraria 1.4.5.7, Steam BuildID 24825745, world format 325**. All 29 existing pins pass. All 146 files in the clean `80a7e9522c114cdd2388a435cecfcc15d977310c` baseline remain byte-identical. Changes are additive, unstaged, and uncommitted. The approximate generator, canonical importer, GUI, README, art, public entry points, and reference corpus were preserved.

## Actual runtime evidence

The primary seed is **314159265**; additional seeds **1 and 2** were fixed before generation, without inspecting their trees. Each uses Small 4200×1200, Classic/game mode 0, Corruption, no secret options, and pre-Hardmode. The final captures explicitly record every `SecretSeed` option as disabled. No genuine zero-tree fixture occurred; failed placements were retained.

The ignored local host loads the unchanged pinned `TerrariaServer.exe` into a fresh .NET Framework x86 process. It replaces the LivingTrees pass delegate before application with `Reflection.Emit` copies of the original runtime IL. Four tree/passage/tunnel/room call targets are wrapped; direct `PlaceTile` and `PlaceSmallPile` calls from those bodies are forwarded to their original implementations with observation around them. Original IL hashes, tokens and module identity are recorded. No game binary was patched on disk, and no dependency was installed.

Snapshots are taken synchronously on the generation thread, after the normal per-pass RNG reset and before the controller's manifest draw. They contain:

- The complete 5,040,000-cell grid: all nine native `Tile` fields, **14 bytes per cell**, x-major then y. This preserves latent tile IDs, unsaved frames, and liquid/framing bookkeeping.
- The existing **16-byte canonical saved-cell projection**, exported separately without invoking game saving or framing. Native snapshots are the stronger comparison.
- Main/WorldGen/GenVars primitive and array state, TileID set tables, explicit chest slots and primitive item fields, seed/world metadata, secret flags, and the actual empty pass configuration `{}`.
- Initial and final RNG state (`inext` and all 56 words), the manifest result, every ordered tree invocation/result, nested passage/tunnel/room calls and mutable arguments, and a full extra snapshot immediately before the first object-placement call.
- Exact native and canonical net-change masks, including masks for each field. These describe pre/post changes, not a temporal log of every individual write.

The capture does **not** serialize arbitrary runtime object graphs or reconstruct complete `.wld` sections at pass boundaries. Unsupported nested values are explicitly marked `unexpanded_type`; progress/status strings are observed but not replayed. Chest primitive inventory state and relevant tile/global tables are retained. This is a limitation beyond the verified replay prefix, not a claim of unrestricted engine-state replay.

Each final instrumented world was compared with its own fresh uninstrumented control. Raw hashes differ. After the already-established normalization of UUID, creation/save timestamps, manifest `DurationMs`, and physical section offsets, **all canonical cells, normalized headers, file metadata, and every other saved section match exactly**. Seeded world IDs and manifest RNG results are preserved. No differing opaque section was ignored.

| Seed | Control save/load time | Final capture save/load time | Final `.wld` bytes | Saved semantic comparison |
|---|---:|---:|---:|---|
| 314159265 | 14.250 s | 56.765 s | 3,011,320 | Exact |
| 1 | 12.391 s | 53.532 s | 2,950,084 | Exact |
| 2 | 12.703 s | 39.922 s | 3,017,756 | Exact |

All six listed server children saved and loaded their worlds. Redirected `exit-nosave` did not stop them; the launcher killed its owned child after the recorded 30-second grace period. **Child exit codes are 1, not clean exits.** Saved hashes were unchanged across shutdown and stderr was empty. Earlier successful capture revisions remain preserved as separate evidence; final comparisons use `capture-final` directories.

## RNG and source grounding

`terraexplorer/fidelity/unified_random.py` implements unchecked Int32 arithmetic, initialization, state restoration, `Next` overloads, `NextDouble`, and non-advancing `Peek`. It uses neither NumPy RNG nor Python `random`.

The actual pinned assembly produced vectors for seeds `314159265, 0, 1, 2, -1, Int32.MinValue, Int32.MaxValue`. Python matched each initial state, **84 explicit returned values and subsequent states**, exact IEEE-754 double bits, and the final states after **1,000 further draws per seed**. Those 7,000 intervening return values were not separately recorded. A fresh runtime rerun reproduced the original oracle JSON byte-for-byte. Compact numeric vectors and state hashes are in `tests/fixtures/terraria1457_rng.json`.

The inspected direct ordinary tree/passage routines use integer `Next` operations; the range implementation's double arithmetic is validated. No direct `NextFloat`, `NextDouble`, or `NextBytes` call was found in `WorldGen.cs:28934–30314`. This does not claim a completed audit of every transitive placement/framing helper.

Pinned source locators, inspected rather than executed as source:

- `Terraria/Utilities/UnifiedRandom.cs:28–178`: initialization, range draws, internal state and `Peek`.
- `Terraria/WorldBuilding/WorldGenerator.cs:488–517`: pass RNG reset and the extra manifest `RandNext` draw.
- `Terraria/WorldGen.cs:16023–16248`: main candidate selection, rejection and patch calls.
- `WorldGen.cs:28934–29592`: ordinary tree geometry and leaf predicate; `29594–30314`: connected work.
- `WorldGen.cs:71840–71864`: support solidity; `9153–9175`: nearby-tile predicate.
- `WorldGen.cs:60925` onward: `PlaceTile`, including type 187 dispatch to `Place3x2` and `SquareTileFrame`; helper entry points `53487`, `82780`, and `PlaceSmallPile:48136`.

These paths are relative to `Game Reference/02_static/decompiled/client/`. Existing source hashes and packets remain unchanged.

## Exact matches and first unsupported operations

The independent path includes candidate rejection, initial tree rejection, trunk/branch/top growth, roots, leaf predicates, canopy traversal and the patch-call control flow. Tests exercise patch rejection; genuine replay stops before reaching a successful patch tree. Object placement and connected passages/tunnels/rooms are **captured but not implemented in Python**. No captured coordinates or outputs are used to generate Python results.

| Seed | Exact changed cells at checkpoint | First unsupported call | Vanilla return |
|---|---:|---|---|
| 314159265 | 651 | `PlaceTile(585,219,187,true,false,-1,49)` | false |
| 1 | 613 | `PlaceSmallPile(3765,227,59,1,185)` | true |
| 2 | 567 | `PlaceSmallPile(1550,231,72,0,185)` | true |

For every checkpoint: **zero missing/extra changed cells, zero differences in all nine native fields, identical call arguments and RNG state, identical solidity table and unchanged chest state**. Completed tree invocations before the checkpoint also match their results and RNG states. Seed 2 reproduces failures at `(1770,254,false)` and `(2798,272,false)` before entering `(1545,231,false)`.

The primary unsupported call returns false in vanilla. The replay deliberately reports an unsupported operation rather than assuming that return value or skipping the helper. The checkpoint is the first unsupported semantic operation; it is not an observed earlier geometry/RNG mismatch. The first final-post differences below use x-major coordinate order, not temporal write order.

## Full-pass differences

Comparing the stopped Python state with the complete vanilla post-pass state:

| Seed | Native changed expected / actual | Missing / extra changed cells | Native differing cells | Canonical differing cells | First differing coordinate |
|---|---:|---:|---:|---:|---|
| 314159265 | 6,664 / 651 | 6,013 / 0 | 6,100 | 6,071 | (549,258), tile 191 expected, 0 actual |
| 1 | 7,065 / 613 | 6,452 / 0 | 6,504 | 6,466 | (3701,191), active leaf 192 expected, empty actual |
| 2 | 11,331 / 567 | 10,764 / 0 | 10,834 | 10,780 | (1521,273), tile 191 expected, 0 actual |

| Native field differences | 314159265 | 1 | 2 |
|---|---:|---:|---:|
| type | 5,211 | 5,668 | 9,754 |
| wall | 1,665 | 1,877 | 2,262 |
| liquid | 5 | 0 | 0 |
| sTileHeader | 4,255 | 4,550 | 8,157 |
| frameX | 251 | 245 | 442 |
| frameY | 251 | 245 | 442 |
| bTileHeader / bTileHeader2 / bTileHeader3 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |

The fixture manifest also records every canonical field count and missing/extra **field** write count. Canonical expected changed-cell counts are 6,635, 7,027 and 11,277; native counts include unsaved state.

**Full-pass final RNG, invocation/result sequence, solidity table and chest-state equality are false for all three stopped replays.** Vanilla chest counts change `18→19`, `26→27`, `21→23`. The actual complete vanilla manifest draws are respectively **97760788, 102480169, 1335059873**; each was independently reproduced from the captured post-pass RNG state.

The runtime recorded 4, 4 and 12 tree attempts, with 2, 2 and 4 successes. Connected calls numbered 2/2/4 downward passages, 2/2/1 horizontal tunnels and 1/1/2 rooms. Complete ordered invocation arguments, results and RNG hashes are retained in the fixture manifest and local trace. These are generation-time observations, not ownership inferred from a saved world.

## Reproduction and verification

Versionable additions: the three fidelity modules, `scripts/fidelity_living_trees_replay.py`, `tests/test_living_trees_replay.py`, numeric RNG vectors, [compact fixture manifest](fixtures/LIVING_TREES_1457.json), and this report. The fixture manifest contains full world/snapshot hashes, timing and exit results, field differences and evidence paths.

All runtime instrumentation, proprietary snapshots, canonical exports and masks remain ignored under `audit/living-trees-replay-20260914/`. `CaptureHost.cs`, `RngOracle.cs`, their compiled hosts and `run_fixture.py` are local-only. For each fixture, `result.json` records exact child arguments, settings/config hash, timestamps, timings, exit code and stdout/stderr hashes. `verification-commands.json`, `ci-commands.json`, and `rng-revalidation.json` record the exact validation commands and output files.

Executed replay command pattern (each seed ran; **exit 2 is the declared incomplete replay**, not a match):

```powershell
python -m scripts.fidelity_living_trees_replay `
  --capture audit/living-trees-replay-20260914/seed-314159265-capture-final/capture `
  --control-world audit/living-trees-replay-20260914/seed-314159265-control/bootstrap-small.wld `
  --captured-world audit/living-trees-replay-20260914/seed-314159265-capture-final/bootstrap-small.wld `
  --output audit/living-trees-replay-20260914/replay-314159265.json
```

The output must be a fresh file under `audit/`; use a new filename when repeating the command. The harness refuses a changed pin, unequal control world, incomplete trace, or mismatched manifest draw.

Actually run:

- Narrow tests: **29 passed** with a fresh workspace `--basetemp`.
- Full CI test command with coverage: **79 passed, 93.34% coverage**, exceeding the unchanged 75% gate.
- Repository-wide Ruff lint and format checks: passed.
- `mypy terraexplorer`: passed, 21 source files.
- All three final runtime control comparisons and independent replays; fresh actual-runtime RNG validation; baseline/pin preservation and ignore checks.

The first test invocation had 27 passes and two setup errors because the default Windows temporary directory was inaccessible; rerunning with a fresh workspace temporary directory resolved both. An initial C# compile using relative slash paths failed before execution; recompiling with absolute Windows paths succeeded. These were tooling failures, not runtime-oracle mismatches. No GUI session, distribution build, Temple/Dungeon implementation or whole-world compatibility work was performed.

The next bounded task is to implement and independently validate the observed `PlaceTile(187)` / `PlaceSmallPile(185)` placement and framing dependencies, then advance this same replay until the next exact mismatch or unsupported dependency. Do not treat the successful prefix as approval to replace the existing generator.
