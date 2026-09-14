# RNG and seed replay packet

Target: Terraria 1.4.5.7, BuildID 24825745. Source paths below start at `02_static/decompiled/client/Terraria/`. Exact file SHA-256 and indexed signatures are in `../TARGET_LOCK.json` and `../SOURCE_LOCATORS.json`; bounded original excerpts are local in `audit/fidelity-bootstrap/excerpts.jsonl`. Status: **PASS for source retrieval; NOT RUN for primitive/replay comparison**.

## Observed source behavior

`IO/WorldFileData.cs:393-410`, `public void SetSeed(string seedText)` and `public static int TranslateSeed(string seedText)`, preserve the text and convert an Int32-parsable number to its absolute value, with Int32.MinValue mapped to Int32.MaxValue. Other input uses `Crc32.Calculate`; its exact encoding/arithmetic has not been audited here. This is not the project's unsigned low-32-bit conversion.

`TryApplyingCopiedSeed(string input, bool playSound, out string processedSeed, out string seedTextIncludingSecrets, out List<string> secretSeedTexts)` at `IO/WorldFileData.cs:232-286` parses size, mode, evil, optional seed-option bits, and secret seed text. It sets the dimensions, `Main.GameMode`, and `WorldGen.WorldGenParam_Evil`, resets options, and clears secret seeds. The runtime attempt transports numeric seed 314159265 as `1.1.1.0.314159265`: Small, Classic, Corruption, zero option bits. This path was statically inspected; the failed process never reached it.

`WorldGen.cs:4597` defines **`genRand => Main.rand`**. `WorldGen.Reset()` at `11534-11919` initializes `Main.rand` from the seed (`11543`, `11573`) and resets world/structure state. `WorldBuilding/WorldGenerator.cs:488-517`, `private GenPassResult RunPass(GenPass pass)`, returns immediately for a disabled pass, otherwise constructs `Main.rand = new UnifiedRandom(_seed)` before `pass.Apply`. Its result construction calls `WorldGen.genRand.Next()` at line 515. That final call consumes a draw. It does not establish continuity into the next pass, which resets this alias again.

`Utilities/UnifiedRandom.cs:14-16,23-88` exposes the implementation's stored state: a uint `inext` and a 56-element Int32 `SeedArray`. `SetSeed(int Seed):28-61` initializes it; `InternalSample():68-89` computes the offset index from `inext`, updates the array, and advances the index. Do not import a presumed second stored index from a different Random implementation. Parameterless construction uses `Environment.TickCount` (`18-21`). Numeric overflow, casts, range overloads, and floating-point output need separate fixtures.

Dungeon does not use only this pass RNG. `Dungeon/LayoutProviders/LegacyDungeonLayoutProvider.cs:42-58,69-93` consumes it to populate `RandomSeed` settings. `Dungeon/Rooms/LegacyDungeonRoom.cs:169-183`, `Dungeon/Halls/LegacyDungeonHall.cs:192-208`, and `Dungeon/Entrances/LegacyDungeonEntrance.cs:89-108` construct **local** `UnifiedRandom` instances from those settings. Save the settings and local states as well as the aliased global state. The remaining features and callees have not been exhaustively classified.

## Future replay contract

Record processed numeric seed, original/transport seed text, all generation options, difficulty and evil, dimensions, exact pass identity/configuration, pre-call global RNG state and object identity, plus each local RNG's creation seed and state at the boundary. Snapshot relevant `Main`, `GenVars`, and Dungeon settings separately from RNG state. For a whole routine started at its entry, record creation seeds for short-lived RNGs; for an interrupted routine, capture their live states.

Compare requested primitive values, call order, branch outcomes, every stored state word, resulting global state, and generated local seeds. Preserve random work inside chests/decorations if it influences later geometry within the routine. Neither a fixed dummy draw count nor dropping the pass-result draw is justified by this bootstrap.

## Still unverified

No oracle primitive execution, held-out sequence, complete stream census, exceptional-pass behavior comparison, or Crc32 text-seed compatibility test ran. The source has decompiler warnings around unresolved XNA types. Targeted server/client source comparison is recorded in the lock, but no IL-level reconstruction or instrumented execution was performed. Shared state across passes still matters even though this build resets the principal RNG per enabled pass.
