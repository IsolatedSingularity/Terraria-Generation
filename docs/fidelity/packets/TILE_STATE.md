# Tile-state and placement replay packet

Target: Terraria 1.4.5.7, BuildID 24825745. Prefix `C/` means `02_static/decompiled/client/Terraria/`. File hashes and exact signatures/AST ranges are in `../TARGET_LOCK.json` and `../SOURCE_LOCATORS.json`. **PASS: source retrieval. NOT RUN: canonical importer/exporter or lossless round trip.**

## Stored state and distinct meanings

`C/Tile.cs:8-24` stores ushort tile **and wall IDs**, byte liquid amount, a ushort and three byte headers, and signed-short frameX/frameY. Tile type 0 is not an air sentinel: presence is separate. `active():594-609` uses header bit 0x20; `inActive():611-626` uses 0x40 for inactive/actuated state; `nactive():240-247` tests their combined meaning.

`halfBrick():679-694`, `actuator():696-711`, and `slope():713-721` expose independent packed shape/actuator fields. `wire`, `wire2`, `wire3`, and `wire4` are indexed at `628-677` and `442-457`. `liquidType():216-237` uses liquid-kind bits, with water/lava/honey/shimmer values defined at `56-62`. Preserve those bits separately from amount; zero amount does not justify silently changing all other state.

Object geometry depends on frame coordinates and placement metadata. `C/TileObject.cs`, `public static bool CanPlace(int x, int y, int type, int style, int dir, out TileObject objectData, bool onlyCheck = false, int? forcedRandom = null):174-887`, obtains TileObjectData and rejects missing metadata/out-of-world footprints at `204-215`. It transforms the placement origin using object Origin, Width and Height. `C/ObjectData/TileObjectData.cs:5126-5154`, `CheckLiquidPlacement` / `LiquidPlace`, dispatches through type/style metadata and liquid-specific destruction rules. Full anchoring/style/frame rules remain a later audit.

The later Temple altar pass illustrates why the model matters: `C/WorldGen.cs:22724-22755` writes a 3 by 2 object with distinct 18-unit frames and reshapes the supporting row. A single project ALTAR cell cannot represent this state.

## Placement rejection and protection

`C/WorldBuilding/StructureMap.cs:19-75` has `CanPlace(Rectangle area, int padding = 0)` and `CanPlace(Rectangle area, bool[] validTiles, int padding = 0)`. It checks world bounds, intersects padded bounds with protected structures, and rejects active tiles whose IDs are not in the valid-tile table. `AddStructure:104-112` adds to the general collection; `AddProtectedStructure:114-124` adds to both collections. They are not interchangeable. The bounds check and subsequent padded scan should be reproduced as actually implemented, not silently corrected based on a presumed intention.

Placement is not centralized here for every structure. Temple and tree caller-specific checks are in their packets; Dungeon rooms expose shape-specific protection (`D/Rooms/LegacyDungeonRoom.cs:58-68`, where D means the Dungeon directory). The project `_can_place_structure` only compares marker rectangles and project bounds. It is not an implementation of all these predicates.

## Future canonical representation and comparison

Keep full IDs, presence, shape, liquids, object frames, relevant wire/actuator bits, paint/coatings and required world flags in a representation separate from the existing artistic palette. Preserve unknown IDs and raw fields; visual fallback must not overwrite canonical data. Declare x/y ordering explicitly: Terraria accesses `Main.tile[x,y]`, whereas the project arrays are `[y,x]`.

Validate imported/exported field values against a matching game reader, including empty-but-typed cells, unknown IDs, slopes, half-bricks, all liquid kinds, framed objects and actuated tiles. Compare canonical fields and changed-cell masks, with a neutral diagnostic image as supporting evidence. A pretty render or raw .wld hash alone is insufficient.

`C/IO/WorldFile.cs:1186-1221` writes section pointers and format version **325**. Its tile writer, frame-importance table, RLE and load routines are indexed but have not been validated against a generated fixture. No reliable target-format reader was established in this bootstrap. Any proposed third-party reader requires its own pinned-format verification.
