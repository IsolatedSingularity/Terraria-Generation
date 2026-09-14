# Dungeon replay packet

Target: Terraria 1.4.5.7, BuildID 24825745. Prefix `C/` below means `02_static/decompiled/client/Terraria/`; `D/` means `C/GameContent/Generation/Dungeon/`. File hashes and complete indexed signatures are in `../SOURCE_LOCATORS.json` and `../TARGET_LOCK.json`. Excerpts stay in ignored local audit files. **PASS: source retrieval. NOT RUN: Dungeon replay or geometry parity.**

## Call chain and observed effects

Setup begins before the Dungeon pass: `WorldGen.AddPasses()` in `C/WorldGen.cs:11967-11972` calls `DungeonCrawler.SetupDungeonGenVarVariables(DungeonGenVars genVars, UnifiedRandom genRand)` (`D/DungeonCrawler.cs:54-211`). At the Dungeon pass (`C/WorldGen.cs:15284-15339`), `SetupDungeonData(int currentDungeon, bool clearOld = false):34-52` and `SetupDungeonDataVariables(int iteration, UnifiedRandom genRand):213-342` prepare state, then the caller chooses a start using dungeon location, surface/rock depths, solid-tile probes, and random offsets. Setup methods were located; their complete bodies were not all audited.

`public static void MakeDungeon(int x, int y, GenerationProgress progress = null)` is in **`D/DungeonCrawler.cs:344-534`**, not the old monolithic WorldGen method. Inspected spans `364-420` and `465-534` show it reading `WorldGen.genRand`, `CurrentDungeonData`, dimensions and style; mutating generating flags/position, bounds, entrance strengths and global feature scalar; and changing cracked-brick solidity. Ordinary Default Dungeon uses `LegacyDungeonLayoutProvider`; Dual Dungeon is a distinct branch, outside the fixture.

| Role | Exact symbol and source locator | Inspected evidence |
|---|---|---|
| Layout | `LegacyDungeonLayoutProvider.ProvideLayout(DungeonData data, GenerationProgress progress, UnifiedRandom genRand, ref int roomDelay)`, `D/LayoutProviders/LegacyDungeonLayoutProvider.cs:17-23`; `LegacyDungeonLayout(...):25-106` | `17-98`: room delay, global seed draws for room/hall settings, mutable position, branch-dependent hall/room creation |
| Room | `LegacyDungeonRoom.GenerateRoom(DungeonData data):32-40`; `LegacyRoom(DungeonData data, int i, int j, bool generating):106-312`, `D/Rooms/LegacyDungeonRoom.cs` | `32-40,164-188`: delegates geometry, local RNG from settings, style/paint, strength and slant scalars |
| Hall | `LegacyDungeonHall.GenerateHall(DungeonData data):51-56` and overload `GenerateHall(DungeonData data, int x, int y):58-64`; `LegacyHall(DungeonData dungeonData, int i, int j, bool generating = false):66-881`, `D/Halls/LegacyDungeonHall.cs` | `51-64,188-208`: local settings seed, tile/wall types and hall strength/step scalars |
| Entrance | `LegacyDungeonEntrance.GenerateEntrance(DungeonData data, int x, int y):23-29`; `LegacyEntrance(DungeonData data, int i, int j, bool generating):31-715`, `D/Entrances/LegacyDungeonEntrance.cs` | `23-29,85-108`: local seed and a 120 by 120 surrounding scan that clears liquids/slopes in bounds |
| Finishing | `DungeonCrawler.MakeDungeon`, `D/DungeonCrawler.cs:479-533` | entrance dispatch, platform/door calculation, then early features, spikes, doors, wall variants, platforms, biome chests, bookshelves, basic chests, lights, traps, furniture, paintings, banners and late features |

The finishing sequence enlarges the bounds by 25 before later features (`518-521`). A crop of visible rooms is not a proven replay boundary. The global finishing classes are named directly at their call sites; complete bodies and downstream framing/cleanup were not audited in this bootstrap.

## Future replay inputs and outputs

Capture the pre-pass tiles with full canonical fields, dimensions/depths, solidity and wall classification tables, all relevant seed flags, current dungeon index, DungeonGenVars, DungeonData, style/paint/scalars, entrance settings, existing room/hall collections and protection masks, global RNG and each settings seed. For `MakeDungeon` replay, capture **after** the caller's setup and coordinate-selection draws; for a pass replay, capture before them and include that work.

Compare all changed cells, shape/frames/walls/liquids, room and hall state, rejected placements, outside-bound writes, dungeon metadata, solidity changes and final RNG states. Include finishing that affects physical objects or later random draws. Chest computations may be required internally; this does not authorize a loot UI.

## Open boundaries

No claim of a complete transitive read/write set, all RNG streams, all later mutations, or default entrance subtype selection has been validated by execution. Geometry methods contain unresolved XNA decompiler annotations; establish difficult numeric/vector operations against IL or the matching runtime before implementing them. A verified real-world import is the next product step, not a replacement generator.
