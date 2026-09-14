# Jungle Temple replay packet

Target: Terraria 1.4.5.7, BuildID 24825745. All locators below are in `02_static/decompiled/client/Terraria/WorldGen.cs`, SHA-256 `252accee3c9c50352cceb7ac4403ab477a1299a317681aa94f8192ce55fd6c65`. Exact signatures/AST bounds and inspected ranges are in `../SOURCE_LOCATORS.json`; original excerpts are local-only. **PASS: retrieval. NOT RUN: replay, placement comparison or real-world crop.**

## Construction and finishing dependencies

`public static void AddPasses():11939-23296` registers `LihzahrdTemple` at `16383-16476`. The inspected caller samples depth between rock layer and the lower-world margin and samples horizontal position using dungeon side. Ordinary placement tests an active jungle-grass tile (type 60, `16444-16448`); after repeated unsuccessful searches it widens the search and eventually falls back to a location derived from dungeon position (`16450-16473`). Remix has additional checks, outside the first fixture. These are not a generic StructureMap-only placement rule.

`public static void makeTemple(int x, int y, GenerationProgress progress = null):34514-35400` draws a room count scaled by world width (`34559-34580`) and uses room rectangles/depths. Direct callees include `makeTemple_GenerateBricks(int numRooms, Rectangle[] roomRects, int[] roomDepths, bool forceStrictAngles, float progressCount, float progressPercentilePerLoop, GenerationProgress progress = null):35402-35478`, `templePather(Vector2D templePath, int destX, int destY):34448-34486`, and `templeCleaner(int x, int y):34411-34446`. Call sites were located at `34674`, `34854`, `34903`, `34919`, `35109`, and `35116`; full geometry bodies were not replayed. The routine stores `GenVars.tLeft/tRight/tTop/tBottom/tRooms` at `35395-35399` for subsequent work.

`public static void templePart2():35686-35894` is invoked by the later `LihzahrdTemplePart2` pass (`18326-18335`), after resetting specific solidity entries. Inspected `35686-35739` reads the saved bounds and room count, samples candidate positions, checks unsafe Temple wall 87 and tile presence, tries `mayanTrap`, and later calls `AddBuriedChest`. Placement success and failures influence loop counters and RNG calls. Trap/chest object state and internal draws matter to replay even when their gameplay presentation is out of scope.

**A still later `LihzahrdAltar` pass must be included.** At `22724-22755`, it uses `GenVars.lAltarX/lAltarY` to write a 3 by 2 altar (type 237) with 18-unit object frames, rebuilds its support in type 226, clears support slopes/half-bricks, and invokes `SquareTileFrame`. `AddLihzahrdAltar(int x, int y):33144-33166` is a related helper, not evidence that this final pass calls that helper. Keep the actual call chain distinct.

## Future replay contract

For the construction routine, record input x/y, the full pre-world or a proven dependency region, width/height, worldSurface/rockLayer, dungeon-side/position state, seed flags, tile solidity/classification data, configuration and the aliased pass RNG state. Record geometry plus `t*` bounds/count and altar coordinates after construction. For later finishing, capture the intervening world mutations and the newly reset pass RNG; do not reuse construction's final state as the next pass's initial RNG state.

Compare rejection/attempt sequences, every canonical tile field, trap wires/actuation and frames, the altar/support cells, saved bounds and all RNG outputs/states. Do not assume the current project's 190 by 120 Small envelope is a valid capture boundary or a vanilla dimension.

## Remaining uncertainty

No complete transitive geometry audit, numeric-semantics verification, reliable capture bounds, post-generation cleanup census, trap/chest internal dependency proof, or successful reference execution exists from this run. The three identified construction/finishing passes are required dependencies, not proof that no other pass can touch Temple tiles.
