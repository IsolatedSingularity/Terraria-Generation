# Terraria fidelity bootstrap report

Observed 13 September 2026 (America/Toronto); integrity timestamp 14 September 2026 UTC. Scope: only **“First local agent task: execute only this bounded bootstrap”** in the root fidelity brief.

**Bootstrap handoff complete; reference-world generation BLOCKED.** The existing corpus and executable identify Terraria **1.4.5.7 / Steam BuildID 24825745**, not the brief's anticipated 1.4.5.6. No version was downloaded or upgraded. Ten bounded source-retrieval answers are documented below; none is a passing runtime fidelity test. The server failed before generation because XNA 4.0 could not be loaded.

## Baseline and preservation

- Branch: `main`. HEAD: `2ff1d25a0d70c8e343d8e15db138cf7f30a29c55`.
- Initial `git status --short`: only `?? Terraria_Fidelity_Plan_and_First_Agent_Task.md`. It remains untouched and untracked. No tracked application changes were present at baseline.
- Read project AGENTS.md, the root brief and corpus retrieval instructions. Inspected active entry points in `pyproject.toml`: `terraexplorer.cli:main` and `terraexplorer.gui:main`; terraforge aliases still point to the active package.
- All eight local blob hashes match those listed in the brief: manifest, generation, pipeline, config, model, pass catalogue, pipeline tests and VISUAL_FIDELITY.md. Exact values are in local `audit/fidelity-bootstrap/integrity.json`.
- Confirmed by inspection: custom smooth-noise/sine terrain; project Dungeon room chain with requested Small count 32 and an early stop; 190 by 120 Small Temple envelope; solid living-tree trunk, ellipse canopy and root walks; uint8 tile/wall arrays; per-label hashed NumPy RNG; 107 hardcoded project passes. See [COMPATIBILITY_GAPS.md](COMPATIBILITY_GAPS.md).
- Work is additive in `docs/fidelity/` and ignored `audit/fidelity-bootstrap/`. Production packages, tests, art, UI, README/media, raw corpus, query tools and existing database were not edited. No reset, move, installation, public API change, commit, push or publishing occurred.

## Corpus and target lock

The actual root is the checkout's `Game Reference/`. It is a directory with no LinkType/Target reported, not a discovered junction to the historical profile-root location. The historical profile-root directory no longer exists at the probed path. Client **and server** executables reside in `Game Reference/01_canonical/client/`; there is no `01_canonical/server/` directory. Exact resolved paths and probe results are kept only in ignored `environment.json` and `integrity.json`.

| Artifact | Fresh SHA-256 | Verification |
|---|---|---|
| `01_canonical/client/Terraria.exe` | `840f158abf2e0b699e939630e6f0a7a360588242c1e54598268f5cb4b4e131f9` | PASS: matches canonical inventory and target manifest; PE file version 1.4.5.7 |
| `01_canonical/client/TerrariaServer.exe` | `6cf1100bcfab5733fdb4da745b34551362b24fcebe05ac153f5b59445082ed49` | PASS: matches inventory; PE file version 1.4.5.7 |
| `07_index/library.sqlite` | `9f49263649b1a6e3ca14062b4acae62f842da0e111383a999aa1d70b38d68949` | PASS: read-only quick_check returns `ok`; target metadata agrees |
| `01_canonical/client/xnafx40_redist.msi` | `e6c41d692ebcba854dad4b1c52bb7ddd05926bad3105595d6596b8bab01c25e7` | PASS: matches inventory; Authenticode Valid, Microsoft Corporation |

[TARGET_LOCK.json](TARGET_LOCK.json) also records installscript, manifest, query launcher/executable and selected source hashes. Eighteen source files are pinned; sixteen selected client/server source pairs are byte-identical. This verifies local identity, not decompiler correctness. There is no historical source-file hash field in the inspected index schema, so fresh source/index hashing must not be described as an independently verified rebuild from the binaries. The complete 16,017-file canonical hash scan from the historical report was **not rerun**; four selected canonical artifacts were compared in this run.

The existing `08_agent_library/START_HERE.md`, `SYSTEM_MAP.md`, topic guides and SQLite AST spans provide the source map. No separate SOURCE_MAP file was discovered in the filename search. The SQLite `files.full_path` sample still refers to the old root, but the tested source printer resolves `relative_path` against the supplied current root correctly. Stats executed successfully: 3,517 files, 98,052 symbols, 1,933,954 relations, 31,184 registries, 15,997 assets, **zero indexed worldgen passes**.

Historical `09_validation/reports/FINAL_HEALTHCHECK.json` reports successful corpus validation on 23 August; `golden-world-failure.json` separately reports the failed XNA launch. These were inspected, not rerun as if they proved present runtime health. Runtime/worldgen and format-note directories contain no fixture/exporter files, and the validation worlds directory is empty. No active-package .wld importer was found by the targeted search.

## Ten retrieval acceptance results

**Status definitions:** PASS here means the answer's stated locators and bounded source evidence were retrieved and inspected. It does not mean an exhaustive dependency audit or an algorithm was executed. Missing broad query results are FAIL even when the process exits zero. Runtime comparison is separately BLOCKED.

For every answer below, the version is **1.4.5.7 / 24825745**, and the binary hash is the client hash above; matching server behavior was checked through selected source-file identity, not game execution. `C/` abbreviates `02_static/decompiled/client/Terraria/`, `D/` abbreviates `C/GameContent/Generation/Dungeon/`, and `S/` abbreviates `02_static/decompiled/server/Terraria/`. Every cited file's full SHA-256, exact indexed signatures/AST ranges and inspected excerpt ranges are in [SOURCE_LOCATORS.json](SOURCE_LOCATORS.json) and TARGET_LOCK.json. Original bounded excerpts remain local in `audit/fidelity-bootstrap/excerpts.jsonl` (53 distinct ranges). This shared provenance applies explicitly to all ten results.

### 1. Seed parsing and generator initialization — PASS

`C/IO/WorldFileData.cs:393-410`: `public void SetSeed(string seedText)` and `public static int TranslateSeed(string seedText)` parse numeric values, apply the signed Int32 rules and call Crc32 for other text. `TryApplyingCopiedSeed(...):232-286` applies copied-seed settings. `C/WorldGen.cs:11496-11532`, `public static bool GenerateWorld(GenerationProgress customProgressObject = null, WorldGenerator.Controller customController = null)`, constructs WorldGenerator, clears/resets the world, constructs/disables passes and executes them. `C/WorldBuilding/WorldGenerator.cs:408-414` stores the seed/configuration; `488-517` resets the RNG per enabled pass. Dependencies: WorldFileData, generation options, embedded configuration, Main, GenVars and UnifiedRandom. Unresolved: exact text Crc32 equivalence and primitive runtime fixtures. See [RNG packet](packets/RNG.md).

### 2. Actual generation pass construction — PASS for static retrieval; runtime list BLOCKED

`C/WorldGen.cs:11939-23296`, `public static void AddPasses()`, calls AddGenerationPass; overloads at `10476-10489` append to WorldGenerator. Bounded branches inspected at `11939-11988`, `12914-12926`, `15273-15343` and `23298-23314`; `3127` confirms denyAllGeneration aliases skyblockWorldGen. The static scan found 109 registration call sites. Excluding two Skyblock registrations and DualDungeonsDitherSnake while including ordinary Jungle gives **106 registrations for the intended ordinary configuration**. [PASS_REGISTRATION.json](PASS_REGISTRATION.json) records all sites in order and the selection conditions. This is not a runtime trace, and does not prove that every registered pass mutates the world. Dependencies: Skyblock/secret-seed flags, GenPassNameID, TerrainPass/JunglePass, configuration and DisablePassesForSpecialSeeds. Unresolved: observed enabled-pass manifest and runtime ordering.

### 3. Dungeon RNG streams and global state — PASS for bounded evidence

`D/DungeonCrawler.cs:344-420,465-534`, `public static void MakeDungeon(int x, int y, GenerationProgress progress = null)`, reads the aliased global RNG, CurrentDungeonData, dimensions/style and writes generating position, bounds, entrance strengths, flags and solidity. `C/WorldGen.cs:4597` shows genRand is Main.rand, reset in WorldGenerator.RunPass. `D/LayoutProviders/LegacyDungeonLayoutProvider.cs:42-58,69-93` draws settings seeds; `D/Rooms/LegacyDungeonRoom.cs:169-183`, `D/Halls/LegacyDungeonHall.cs:192-208`, `D/Entrances/LegacyDungeonEntrance.cs:89-108` construct local UnifiedRandom instances. Dependencies: caller setup, DungeonGenVars/Data, style/settings, tile tables and physical feature callees. Unresolved: complete transitive global-state and RNG census, dynamic read/write set and pre/post-state capture. See [Dungeon](packets/DUNGEON.md) and [RNG](packets/RNG.md).

### 4. Dungeon layout, rooms, corridors, entrance and finishing — PASS

Exact entry is DungeonCrawler.MakeDungeon, not WorldGen.MakeDungeon. Layout: `D/LayoutProviders/LegacyDungeonLayoutProvider.cs`, `ProvideLayout(...):17-23` → `LegacyDungeonLayout(...):25-106`. Rooms: `D/Rooms/LegacyDungeonRoom.cs`, `GenerateRoom(DungeonData data):32-40` → `LegacyRoom(...):106-312`. Halls: `D/Halls/LegacyDungeonHall.cs`, `GenerateHall(...):51-64` → `LegacyHall(...):66-881`. Entrance: `D/Entrances/LegacyDungeonEntrance.cs`, `GenerateEntrance(...):23-29` → `LegacyEntrance(...):31-715`. Finishing: named global feature calls in `D/DungeonCrawler.cs:479-533`. Exact full signatures are in SOURCE_LOCATORS.json and the packet. Dependencies: mutable settings/data, local seeds, object placement and ordered feature calls. Unresolved: actual ordinary entrance subtype, complete finishing callees, late world mutations and geometry parity. See [Dungeon packet](packets/DUNGEON.md).

### 5. Jungle Temple construction and later finishing — PASS

`C/WorldGen.cs`: caller `AddPasses` at `16383-16476`; `public static void makeTemple(int x, int y, GenerationProgress progress = null):34514-35400`; direct brick/path/cleaner methods and calls indexed in the packet. `public static void templePart2():35686-35894` is invoked by `LihzahrdTemplePart2` at `18326-18335`. **The later altar pass at `22724-22755` also changes physical tiles and frames.** Dependencies: dungeon side, depths/tiles, GenVars.t* and lAltar coordinates, solidity tables, traps/chests and reset pass RNG. Unresolved: full geometry/callee proof, intervening mutations and reliable replay boundary. See [Temple packet](packets/TEMPLE.md).

### 6. Living trees and connected underground structures — PASS

`C/WorldGen.cs`: LivingTrees caller at `16023-16248`; `public static bool GrowLivingTree(int i, int j, bool patch = false):28934-29570`; `GrowLivingTree_MakePassage(...):29832-30103`; `GrowLivingTree_HorizontalTunnel(int i, int j):29594-29830`; `GrowLivingTreePassageRoom(int minl, int minr, int Y):30105-30314`. Inspected support rejection `28964-29012`, conditional passage call `29541-29569`, downstream calls `29995-30047` and room startup `30105-30134`. Dependencies: caller placement attempts, flags, wall/solidity tables, mutable passage bounds and RNG. Unresolved: full tunnel/room mutations and later-pass coverage. See [Living-tree packet](packets/LIVING_TREES.md).

### 7. Placement rejection and protection — PASS

`C/WorldBuilding/StructureMap.cs:19-75`: both `CanPlace(Rectangle area, ...)` overloads check bounds, protected intersections and valid active tile IDs. `AddStructure(Rectangle area, int padding = 0):104-112` and `AddProtectedStructure(Rectangle area, int padding = 0):114-124` populate different collections. `C/TileObject.cs:174-887`, `CanPlace(...)`, inspected at `204-227`, uses object metadata/origin and bounds. Temple and tree have additional caller-specific predicates above. Dependencies: valid-tile tables, object style/dimensions, protected shapes and world flags. Unresolved: complete anchor and object-placement rules; this does not show all generators use StructureMap. See [Tile-state packet](packets/TILE_STATE.md).

### 8. Tile presence, shape, frames, wall and liquid semantics — PASS

`C/Tile.cs:8-24` defines ushort type/wall, byte liquid, packed headers and signed-short object frames. `active():594-609`, `inActive():611-626`, `nactive():240-247`, `halfBrick():679-694`, `actuator():696-711`, `slope():713-721`, and `liquidType():216-237` supply distinct semantics. Wire overloads are indexed at `442-457,628-677`; full wire bodies were not all read. `C/ObjectData/TileObjectData.cs:5126-5154`, `CheckLiquidPlacement(...)`/`LiquidPlace(...)`, connects liquid to placement rules. Dependencies: tile/object metadata and classification tables; unknown IDs must survive import. Unresolved: full version-325 serializer, frame/style tables, canonical round trip. See [Tile-state packet](packets/TILE_STATE.md).

### 9. Matching runtime generation and save invocation — PASS for retrieval; execution BLOCKED

`S/IO/WorldFile.cs:662-705`, inside `public static void LoadWorld()`, handles missing-world autocreation, parses seed options, calls WorldGen.GenerateWorld and then SaveNewWorld. `C/IO/WorldFile.cs:854-858`, `public static void SaveNewWorld()`, and `SaveWorld(bool resetTime = false, bool useTemps = false, bool canBeSkipped = false):860-871` route to serialization. `SaveWorld_Version2(BinaryWriter writer):1186-1204` writes sections; `SaveFileFormatHeader(BinaryWriter writer):1206-1242` writes format **325** at 1210. Server config `seed`, `difficulty`, `autocreate` was inspected in `S/Main.cs:5185-5201`; `-savedirectory` is handled in `S/Program.cs:182`. Dependencies: Windows CLR/XNA, generation configuration, file metadata and save paths. Unresolved: successful generation, reliable parsing and normalized comparison. Exact invocation failure follows below.

### 10. Spawning dispatcher and biome/wall/player checks — PASS for bounded retrieval

`C/NPC.cs:94053-94062`, `public static void SpawnNPC()`, delegates to `new Spawner().SpawnNPC()`. The instance method at `290-307` iterates players and calls `CanSpawnEnemiesNear(Player player):345-364` and `TrySpawnAnNPC(Player player):309-343`. The latter checks population, random attempt rate, spawn tile, screen exclusion and post-checks before `SpawnAnNPC(...):1311-5281`. `SetSpawnFlags(Player player):366-431` reads player zones, time/weather, progression and walls, including living-tree wall 244. `FindSpawnTile(...):997-1038` rejects solid tiles/safe walls; `PostCheckChosenSpawnTile(...):1050-1076` checks Dungeon tile/wall and liquids. `GetSpawnRate(Player player, out int spawnRate, out int maxSpawns):484-941` was located, not fully inspected. Dependencies: player position/luck/status, nearby population, progression/events, wall/tile state and Main.rand. Unresolved: complete conditional selection and rate model. No spawn map or probability implementation was added.

## Retrieval tool results and repairs

Twelve corrected query routes were executed and checked for their exact intended symbol families. Query logs preserve command argument arrays, process exit codes, stdout and stderr. Existing-tool syntax was obtained from `TerrariaIndexer.exe --help` and the Query-Terraria.ps1 parameter block before execution. Small direct read-only SQLite lookups disambiguated exact names/signatures; source reading used bounded spans, never the whole WorldGen file in model context.

Initial route failures are retained rather than relabeled as successful retrieval: JungleTemple returned a bestiary field; LivingTree returned an unrelated flag/enum; CanPlace filtered to worldgen and SpawnNPC filtered to spawning returned no matches; the fully qualified dotted MakeDungeon query returned no matches. Related edges were treated as candidate syntax only. The routing-only repair is [START_HERE.md](START_HERE.md), the five substantive packets, and portable locator/pass-site records. The raw database and tools were not modified or rebuilt.

## Fresh runtime result: BLOCKED

Intended fixture: ordinary **Small 4200 by 1200**, **Classic / game mode 0**, **Corruption**, numeric seed **314159265**, no secret options, pre-hardmode. Seed transport is `1.1.1.0.314159265`, using the pinned copied-seed parser to make evil explicit. These are intended settings; the failed process never generated a world to confirm them.

Executed from the repository root:

```powershell
python audit/fidelity-bootstrap/run_reference.py
```

That helper launched exactly this argument sequence, with absolute resolutions stored in its local JSON (the relative spelling below is portable):

```text
Game Reference/01_canonical/client/TerrariaServer.exe
  -config audit/fidelity-bootstrap/runtime-attempt-1/serverconfig.txt
  -savedirectory audit/fidelity-bootstrap/runtime-attempt-1/saves
```

The child working directory was the absolute `runtime-attempt-1` directory. stdout/stderr were redirected; Windows crash dialogs were suppressed only for that process tree. A 60-second startup ceiling was configured, but the process failed in **0.672 seconds**, so timeout logic did not run. The launcher script itself returned 0 after recording the failure; **the child server exit was 3762504530 / `0xE0434352`**, not success.

```text
Unhandled Exception: System.IO.FileNotFoundException: Could not load file or assembly
'Microsoft.Xna.Framework, Version=4.0.0.0, Culture=neutral, PublicKeyToken=842cf8be1de50553'
or one of its dependencies. The system cannot find the file specified.
   at Terraria.Program.LaunchGame(String[] args, Boolean monoArgs)
   at Terraria.WindowsLaunch.Main(String[] args)
```

stdout was empty. `bootstrap-small.wld` does not exist. No TerrariaServer process remained after the attempt. Exact command, paths, settings, artifact/config hashes, stderr, exit and timing are in local-only `audit/fidelity-bootstrap/runtime-attempt-1/result.json`; separate stdout.txt, stderr.txt and serverconfig.txt are retained. This is the requested precise failure record; no substitute project-generated fixture was created.

Windows 11 build 26200 and installed .NET Framework 4.8.09221 (release 533509) were observed. Both game PE headers have machine code I386/0x14c. XNA 4.0 managed references are present in the pinned server's extracted AssemblyRef table. Probed GAC_32/GAC_MSIL locations had no Microsoft.Xna entries; the standard XNA development-reference and standard Steam Terraria paths did not exist. This is a bounded search of plausible local locations, not a whole-machine assertion. No usable per-process resolution candidate was found there.

The smallest identified dependency action is installation/repair of the **Microsoft XNA Framework Redistributable 4.0 Refresh**, preferably the signed, hash-verified `Game Reference/01_canonical/client/xnafx40_redist.msi` already paired with the snapshot. The pinned installscript explicitly uses this MSI through Windows Installer. Its [official Microsoft source](https://www.microsoft.com/en-us/download/details.aspx?id=27598) was looked up; nothing was downloaded or installed. Installation changes the system and was deliberately left for a separately authorized action under the bootstrap instructions. Additional managed/native dependencies remain untested until XNA loads.

Second fresh process: **NOT RUN**, since the first failed. World parsing/version/dimensions: **BLOCKED**, no file. Normalized semantic comparison: **BLOCKED**, no file or verified reader/exporter. Nondeterministic-field measurement: **NOT RUN**. Future comparison must explicitly handle metadata such as timestamps, UUIDs, save revisions and generator timing separately; these are candidates to inspect, not fields experimentally proven nondeterministic here.

## Commands and verification actually performed

| Action | Executed command / evidence | Result |
|---|---|---|
| Baseline | `git status --short`; `git branch --show-current`; `git rev-parse HEAD`; `git hash-object` on the eight reviewed paths | PASS: baseline recorded and blob agreement |
| Tool help and stats | `& 'Game Reference/tools/offline/TerrariaQuery/TerrariaIndexer.exe' --help`; `powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'Game Reference/Query-Terraria.ps1' -Stats` | PASS: help and counts returned |
| Queries | `python audit/fidelity-bootstrap/inspect_reference.py query <label> <term> <documented flags>` | Twelve expected-symbol route checks PASS; failed initial routes retained in query JSON |
| Exact symbol retrieval | Same helper `symbols <exact names>`; read-only SQLite parameterized lookups | PASS: exact AST signatures/spans located |
| Bounded reading | Same helper `excerpt <corpus-relative path> <first line> <last line>` | 53 distinct source excerpts recorded; hash attached to each |
| Integrity | `python audit/fidelity-bootstrap/collect_integrity.py` | PASS: selected manifest hashes, SQLite quick_check, metadata, source blob audit, 109 static registration sites |
| Environment | Get-Item/Get-ItemProperty/Get-ChildItem probes and `Get-AuthenticodeSignature` on the paired MSI | Observed CLR version, empty/missing candidate XNA paths and Valid Microsoft signature; local environment.json |
| Portable records | `python audit/fidelity-bootstrap/build_records.py` | PASS: 96 exact symbol locators, 53 excerpts, 18 source hashes, 16 matching client/server pairs and 12 checked query routes |
| Reference launch | `python audit/fidelity-bootstrap/run_reference.py` | **BLOCKED** child server; error/exit above |
| Existing project regression | `python -m pytest tests/test_pipeline.py::test_generation_is_deterministic_and_uses_independent_arrays -q -p no:cacheprovider --basetemp audit/fidelity-bootstrap/pytest-temp` | **PASS: 1 passed in 0.58s**; project self-repeatability only |
| Ignore protection | `git check-ignore 'audit/fidelity-bootstrap/runtime-attempt-1/result.json' 'Game Reference/07_index/library.sqlite'` | PASS: both ignored |
| Final handoff verification | `python audit/fidelity-bootstrap/verify_handoff.py` | PASS: 235 integrity/preservation checks, including exact source excerpts and AST locators, links, JSON, helper compilation, stable hashes and unchanged tracked/staged files |

Final artifact checks and helper syntax verification are recorded in ignored `audit/fidelity-bootstrap/handoff-verification.json`. The first artifact-check run falsely interpreted part of an HTTPS URL as a Windows drive path; the local checker was corrected and rerun successfully. Other exploratory failures included a nonexistent canonical/server directory probe and an invalid rg glob/PowerShell-flag combination; these did not produce retrieval evidence and were corrected with actual mapped paths and valid searches. Full pytest suite, lint/typecheck/build, GUI capture, corpus rebuild, historical validation script, IL validation and semantic oracle tests were **NOT RUN**. No runtime production code was changed, so the verification stayed focused on retrieval integrity, preservation and one existing regression.

## Files added and next task

Reviewable files: this report, TARGET_LOCK.json, COMPATIBILITY_GAPS.md, START_HERE.md, SOURCE_LOCATORS.json, PASS_REGISTRATION.json, and five packets under `packets/` (Dungeon, Temple, living trees, RNG and tile state). Original source excerpts and absolute-machine records are local-only in ignored audit files. All additions remain unstaged and uncommitted.

Resolve the verified XNA dependency, rerun the same pinned configuration in a fresh process, and obtain a parseable real world. Preserve this failure record: the local launcher deliberately refuses to reuse its existing attempt directory. A follow-up launch needs a fresh output directory and a full-generation timeout once startup is healthy. Then the next bounded development task is a **canonical importer/exporter plus verified Dungeon, Temple and living-tree renders in the existing art style**. Establish semantic normalization and fresh-process repetition with that trusted representation. Independent structure replay comes after those foundations. No replacement structure generator is part of this handoff.
