# Fidelity routing

**Current implementation route (2026-09-23):** before substantial generator, fidelity, importer, replay, GUI, or export work, read [../PRE_IMPLEMENTATION_QA.md](../PRE_IMPLEMENTATION_QA.md). The live pinned target is Terraria **1.4.5.7 / Steam BuildID 24825745**. The later [RUNTIME_ORACLE_REPORT.md](RUNTIME_ORACLE_REPORT.md) records successful native world generation, [CANONICAL_IMPORT_REPORT.md](CANONICAL_IMPORT_REPORT.md) records the canonical importer/reference renders, and [HIGH_FIDELITY_GENERATION_REPORT.md](HIGH_FIDELITY_GENERATION_REPORT.md) records the current source-informed Small generator. These supersede the bootstrap's old runtime-blocked state for current routing without rewriting that historical evidence.

## Historical bounded fidelity bootstrap routing

Start with [BOOTSTRAP_REPORT.md](BOOTSTRAP_REPORT.md) and [TARGET_LOCK.json](TARGET_LOCK.json). This is a retrieval and runtime bootstrap for the pinned Windows Terraria **1.4.5.7 / Steam BuildID 24825745** snapshot. It does not add a compatible generator or importer.

The corpus root is `Game Reference/`, relative to the checkout. Read its `08_agent_library/START_HERE.md`, `AGENTS.md`, `SYSTEM_MAP.md`, and the relevant existing topic guide first. Leave the raw corpus and SQLite index in place. Paths inside packets are relative to the corpus unless identified as project paths. Absolute local paths, source excerpts, query transcripts, and runtime logs are in ignored `audit/fidelity-bootstrap/`.

| Need | Packet / route |
|---|---|
| Dungeon call chain, mutable state, local RNGs | [DUNGEON.md](packets/DUNGEON.md) |
| Temple construction, finishing, later altar pass | [TEMPLE.md](packets/TEMPLE.md) |
| Living trees, passages, rooms, placement failure | [LIVING_TREES.md](packets/LIVING_TREES.md) |
| Seed conversion, stream aliasing, pass resets | [RNG.md](packets/RNG.md) |
| Canonical tile fields and placement semantics | [TILE_STATE.md](packets/TILE_STATE.md) |
| All ten acceptance questions, including runtime and spawning | [BOOTSTRAP_REPORT.md](BOOTSTRAP_REPORT.md) |
| Exact signatures, AST spans, inspected excerpt ranges and SHA-256 | [SOURCE_LOCATORS.json](SOURCE_LOCATORS.json) |
| Ordered registration sites and ordinary-world static selection | [PASS_REGISTRATION.json](PASS_REGISTRATION.json) |

Use the documented PowerShell interface, from the repository root:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'Game Reference/Query-Terraria.ps1' -CorpusRoot 'Game Reference' -Query TranslateSeed -Top 2 -ShowSource
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'Game Reference/Query-Terraria.ps1' -CorpusRoot 'Game Reference' -Query AddPasses -Top 2 -IncludeLarge
```

The following exact queries were executed successfully to retrieve the intended symbol families. `-Top` includes both client and server results; inspect the named exact match, not just the first result.

| Query | Options | Exact destination |
|---|---|---|
| `TranslateSeed` | `-Top 2 -ShowSource` | `IO/WorldFileData.cs`, `TranslateSeed` |
| `RunPass` | `-Top 2 -ShowSource` | `WorldBuilding/WorldGenerator.cs`, `RunPass` |
| `AddPasses` | `-Top 2 -IncludeLarge` | `WorldGen.cs`, `AddPasses` |
| `MakeDungeon` | `-Top 100 -IncludeLarge` | `Dungeon/DungeonCrawler.cs`, exact `MakeDungeon` |
| `makeTemple` | `-Top 20 -IncludeLarge` | `WorldGen.cs`, exact `makeTemple` |
| `templePart2` | `-Top 2` | `WorldGen.cs`, `templePart2` |
| `GrowLivingTree` | `-Top 20 -IncludeLarge` | `WorldGen.cs`, exact `GrowLivingTree` |
| `StructureMap` | `-Top 4` | `WorldBuilding/StructureMap.cs` |
| `active` | `-Topic tiles -Top 6 -ShowSource` | `Tile.cs`, `active` overloads |
| `SaveWorld` | `-Top 4` | `IO/WorldFile.cs`, `_SaveWorld`, `SaveWorld_Version2` |
| `SpawnNPC` | `-Top 12 -IncludeLarge` | `NPC.cs`, `NPC.Spawner.SpawnNPC` and dispatcher |
| `CanSpawnEnemiesNear` | `-Top 2 -ShowSource` | `NPC.cs`, player eligibility |

Known failed routes: `JungleTemple` resolves to a bestiary field; `LivingTree` resolves to a spawn flag and enum; `CanPlace -Topic worldgen` and `SpawnNPC -Topic spawning` return no matches. A dotted fully qualified `DungeonCrawler.MakeDungeon` query also returned no matches. These failures are retained in local query logs. Removing a misleading topic filter and using the actual method name fixes these routes without rebuilding the database.

The query tool uses OR-based full-text matching, excludes spans over 5,000 lines unless `-IncludeLarge` is used, and prints source only for spans at most 160 lines. `-Related` returns syntactic candidate edges, not a verified call graph. Do not treat it as exhaustive or as proof that two unqualified method names refer to the same method.

Read a bounded excerpt after locating it, for example:

```powershell
$source = 'Game Reference/02_static/decompiled/client/Terraria/WorldGen.cs'
Get-Content -LiteralPath $source | Select-Object -Skip 15283 -First 56
```

The existing index's `worldgen_passes` table has zero rows. The additive pass-registration artifact supplies a static route; runtime confirmation remains blocked. No query-tool, topic-guide, database, or raw source changes were necessary.
