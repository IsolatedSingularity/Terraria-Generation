# Runtime oracle validation

**PASS: two genuine Terraria 1.4.5.7 Small worlds were generated, saved, loaded by the server, and semantically compared.** Observed 14 September 2026 UTC (13 September locally). The former XNA startup blocker is resolved. This validates this configuration on this host, not universal seed determinism or TerraExplorer fidelity.

The original [TARGET_LOCK.json](TARGET_LOCK.json), bootstrap report, reference corpus and failed attempt remain unchanged. Their historical `BLOCKED` status describes that earlier run; this report records the subsequent successful execution. All **29 pinned hash checks** and **15 preservation checks** passed. The server SHA-256 remains `6cf1100bcfab5733fdb4da745b34551362b24fcebe05ac153f5b59445082ed49`, Terraria **1.4.5.7 / Steam BuildID 24825745**. XNA assemblies were found in the Windows GAC; no installation was performed in this task.

## Executed worlds

Both fresh processes used Small **4200 × 1200**, **Classic / game mode 0**, **Corruption**, numeric seed **314159265**, no secret options, pre-Hardmode. The same bootstrap copied-seed transport, `1.1.1.0.314159265`, explicitly selects these options. Only output/save/banlist paths changed between configurations. World name: `FidelityBootstrap`. The earlier failed attempt was not reused.

| Observation | Run 1 | Run 2 |
|---|---|---|
| Fresh directory under `audit/runtime-oracle-20260914/` | `run-1/` | `run-2/` |
| Process ID | 19224 | 15924 |
| Start UTC | 03:03:52.779332 | 03:11:14.541895 |
| Generation and load observed after | 13.453 s | 20.750 s |
| Manifest pass durations, sum | 11.106 s | 18.511 s |
| Total child lifetime, including shutdown timeout | 43.609 s | 50.968 s |
| Saved `.wld` size | 3,011,311 bytes | 3,011,325 bytes |
| Child exit code | 1 (`0x1`), launcher termination | 1 (`0x1`), launcher termination |
| stderr | Empty | Empty |

Files and SHA-256:

- [Run 1 world](../../audit/runtime-oracle-20260914/run-1/bootstrap-small.wld): `cae384ddc3466ca80f398912797b3dbbe241972653067c306b96142719551a8e`
- [Run 2 world](../../audit/runtime-oracle-20260914/run-2/bootstrap-small.wld): `caea7534eb437b8e9b6e11838e93e2ad2c7ee6ae491e1669f2480ab24f8efcdc`

Both files parse as **Re-Logic world format 325**, revision **1**, generator version **1395864371201**. Recovered metadata agrees: seed text `314159265`, dimensions 4200 × 1200, game mode 0, `crimson=false`, `hardmode=false`, world ID **1896665969**, spawn **(2099, 231)**, world surface **337**, rock layer **469**, Dungeon entrance **(741, 208)**. The nine early secret-world booleans are false. Full recovered metadata, UUIDs, raw .NET creation/save timestamps and embedded manifests are in the comparison JSON.

**Generation succeeded; graceful shutdown did not.** Both logs reached final cleanup, world saving, Terraria Server v1.4.5.7 and `Listening on port 17789`. After this completed save/load boundary, the launcher sent `exit-nosave` through redirected stdin. Neither child exited within 30 seconds, so the launcher killed and waited for its own child. The file hash before and after termination was identical in each run. No server processes remained in the successful `Get-Process` check. An earlier CIM enumeration was denied and is explicitly marked inconclusive in the corrected evidence.

Run 1's original wrapper returned 1 and labeled its result `BLOCKED` because it conflated shutdown and generation. That original JSON is preserved. The wrapper was corrected before Run 2 to record `SAVED_AND_LOADED` separately from `FORCED_OR_FAILED_EXIT`; Run 2's wrapper returned 0. Neither child exit is represented as a normal server shutdown.

## Semantic comparison

Raw hashes differ, but **all 5,040,000 saved cells match exactly after decoding**, including tile IDs, walls, saved frames, paints, liquid kind/amount, shape, wires, actuators, inactive state and coatings. Each stream contains 889,606 encoded records. Their compressed tile sections also match byte for byte.

Canonical expanded cell SHA-256: `97d590a10019a682b04bdbf076c70b18f8f557fc9c6c0ce3964158994eac9b4c`.

Chest, sign, NPC, tile-entity, pressure-plate, town-manager, bestiary, creative-power and footer sections are byte-identical. After normalizing **only UUID, creation/save timestamps and manifest pass `DurationMs`**, every remaining header byte/manifest value matches. Physical section offsets differ because the timing text has different lengths. The seeded world ID is deliberately **not** normalized.

The manifests contain 106 pass results. **99 pass durations differ**; all names, `RandNext`, `Skipped`, hashes and other manifest values match. Both report Terraria's final world hash **3656490740**. Normalized header SHA-256: `674c14996b2f96f3a5447c606bd7f679b861223a6f452a291243c37799e4e537`. No tile/wall/liquid or other saved world-state difference was found.

## Minimum reader and source evidence

No established local `.wld` reader was found in the bootstrap or targeted current filename search. Added [fidelity_wld325.py](../../scripts/fidelity_wld325.py), a read-only audit validator restricted to version 325 and Small dimensions. It validates magic, section boundaries, the embedded frame-importance table, cell header/RLE encoding and footer identity. It reads identifying header fields and locates the unique final manifest using its .NET length prefix, exact header endpoint and pinned version. Undecoded header fields and non-cell sections are retained for exact comparison; unequal opaque bytes remain unresolved rather than being ignored.

This is a locally tested validator, not a general importer or complete Terraria loader. Unsaved frames use an explicit sentinel; no frame reconstruction, tile migration or gameplay simulation is attempted. Equality concerns serialized state, not every transient in-memory field.

Inspected pinned source (paths relative to `Game Reference/02_static/decompiled/`):

- `client/Terraria/IO/WorldFile.cs:1186-1272,1274-1466,1468-1641,1790-1797,2559-2764`: section order, header/flags, cell serialization, footer and matching loader.
- `client/Terraria/IO/FileMetadata.cs:8-33`: format metadata and revision; `WorldFile.cs:594-620`: GUID/time creation.
- `client/Terraria/WorldBuilding/WorldManifest.cs:10-67`, `GenPassResult.cs:3-40`, `WorldGenerator.cs:487-552`: manifest serialization, stopwatch-derived durations and game world hash. `client/Terraria/WorldGen.cs:11767`: seeded world ID.
- `server/Terraria/Main.cs:5789-5859,5980-6000,6047-6065,6292-6301`: load completion, input thread and shutdown commands.

Source hashes and bounded excerpts are saved in `verification.json`. These C# methods were **inspected**, not separately compiled or invoked by reflection. The pinned server executable itself was **actually run twice**.

## Reproduction and evidence

Executed from repository root:

```powershell
python scripts/fidelity_oracle_run.py --output audit/runtime-oracle-20260914/run-1 --timeout 1200
python scripts/fidelity_oracle_run.py --output audit/runtime-oracle-20260914/run-2 --timeout 1200
python audit/runtime-oracle-20260914/verify.py
```

Each launcher invoked the absolute pinned `Game Reference/01_canonical/client/TerrariaServer.exe` with `-config <fresh-directory>/serverconfig.txt -savedirectory <fresh-directory>/saves`, using the fresh directory as cwd. Exact argument arrays, config text, UTC start, elapsed times, child exit codes, stdout/stderr and their hashes are preserved in each run's `launch.json`, `result.json`, `serverconfig.txt`, `stdout.txt`, `stderr.txt`. Hash-verified copies of both executed launcher revisions are retained as `launcher-run-1.py` and `launcher-run-2.py`.

- [Final comparison](../../audit/runtime-oracle-20260914/acceptance-comparison.json): metadata, section/cell hashes, full manifests and classified differences.
- [Verification transcript](../../audit/runtime-oracle-20260914/verification.json): exact commands, stdout/stderr, exit codes/timings, source excerpts and preservation checks.
- [Process check](../../audit/runtime-oracle-20260914/process-check-corrected.json): both owned PIDs absent; records the inconclusive CIM probe.

Actually passed: **9 targeted pytest tests**, Ruff lint and format checks on the three added Python files, real-world parsing/comparison, and three sensitivity probes using clearly labeled synthetic copies (UUID-only difference accepted; seeded world-ID and opaque chest-byte changes rejected as equal). Both tracked and staged Git diffs remain empty. No application test suite, GUI capture or TerraExplorer generation was run.

Additions are this report, [launcher](../../scripts/fidelity_oracle_run.py), [validator](../../scripts/fidelity_wld325.py), [tests](../../tests/test_fidelity_wld325.py), and ignored audit evidence/world files. Production algorithms, structures, art, GUI, README and public APIs were untouched; existing uncommitted work is preserved. Nothing was staged, committed or published. Next logical task, only when separately authorized: use these fixtures in bounded fidelity comparisons. No reimplementation began.
