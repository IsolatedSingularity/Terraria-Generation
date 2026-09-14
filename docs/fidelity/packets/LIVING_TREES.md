# Living-tree replay packet

Target: Terraria 1.4.5.7, BuildID 24825745. Locators below are in `02_static/decompiled/client/Terraria/WorldGen.cs`, SHA-256 `252accee3c9c50352cceb7ac4403ab477a1299a317681aa94f8192ce55fd6c65`. See `../SOURCE_LOCATORS.json` for complete signatures and AST ranges; original excerpts stay in local audit files. **PASS: retrieval. NOT RUN: replay or real-tree import.**

## Placement and connected geometry

`AddPasses():11939-23296` registers `LivingTrees` at `16023`. Its caller rejects certain surfaces, nearby tile classes and mountain-cave proximity, and bounds placement attempts (`16067-16158`). On success, it may attempt additional trees with `patch: true` (`16159-16248`). These caller draws and failures are inputs to a pass replay; they occur before entry to each individual tree routine.

`public static bool GrowLivingTree(int i, int j, bool patch = false):28934-29570` rejects an invalid support, occupied origin, disallowed support types and a too-high origin in the ordinary path (`28964-28988`). The normal routine then consumes `genRand` for geometry. Near its end (`29541-29565`), it inspects walls/solidity under the trunk and conditionally calls `GrowLivingTree_MakePassage`. It restores `Main.tileSolid[48]` to true and returns true at `29568-29569`.

| Connection | Exact signature / AST range | Evidence and dependency |
|---|---|---|
| Passage | `public static void GrowLivingTree_MakePassage(int j, int W, ref int minl, ref int minr, bool noSecretRoom = false):29832-30103` | Tree passes `patch` as `noSecretRoom`; `29995-30047` changes tunnel bounds, uses dungeon-wall classification, places platforms and branches to tunnel/room work |
| Horizontal tunnel | `private static bool GrowLivingTree_HorizontalTunnel(int i, int j):29594-29830` | Called at `30028`; return value chooses a different next random interval |
| Connected room | `private static void GrowLivingTreePassageRoom(int minl, int minr, int Y):30105-30314` | Called at `30034`; `30105-30134` draws direction and room width, then examines existing walls, presence and surface |
| Leaf acceptance | `private static bool GrowLivingTree_CanPlaceLeaves(int i, int j):29572-29592` | Located by the existing query tool; a separate geometry predicate, not an ellipse-only canopy rule |

The boolean called `noSecretRoom` must not be interpreted by name alone: the inspected passage still has a room call outside that particular `if` block. The caller's `patch` setting and branch state must be captured. Another `GrowLivingTree` call exists in Skyblock generation (`23611`), outside this ordinary fixture. Complete mutation coverage across later passes has not been established.

## Future replay contract

Capture origin, `patch`, pre-call RNG, dimensions/depths, seed flags, support tile/shape, wall/solidity tables and world cells spanning trunk, canopy, roots, downward passage and side rooms. For the passage helper also record width W and the mutable `minl/minr` references. A visible canopy crop cannot establish all reads/writes below ground.

Compare the boolean placement result, unsuccessful candidate attempts, full tile/wall/liquid/frame changes, updated passage bounds, physical objects and final RNG state. Preserve valid absence and failed attempts rather than forcing trees into a fixture for presentation. Derive bounds from observed reads/writes or start with a full-world snapshot.

## Remaining uncertainty

No vanilla tree output, complete room/tunnel callee audit, local fixture, cleanup-pass proof or held-out case ran. The existing TerraExplorer handler's solid trunk, ellipse and root walks were confirmed by inspection, but this bootstrap does not claim that no later project pass ever mutates those cells.
