# Canonical imported world schema

Target: Terraria **1.4.5.7**, Steam BuildID **24825745**, `.wld` version **325**. This milestone supports **Small 4200 × 1200** saves. Other versions/dimensions fail explicitly. The importer is separate from TerraExplorer's simplified `GeneratedWorld`, `Tile` and `Wall` simulation state.

Entry: `terraexplorer.fidelity.world.import_world(path) -> CanonicalWorld`. The single live decoder is `terraexplorer/fidelity/wld325.py`, promoted from the previously validated audit script. `scripts/fidelity_wld325.py` re-exports the same functions and retains its command-line interface.

## Cells and hashing

`CanonicalWorld.cell_bytes` is immutable `bytes`, **16 bytes per cell**, no alignment padding. Save/canonical order is **x-major, then y**. `world.cells[y, x]` is a read-only NumPy structured view over those same bytes, not a conversion to simulation IDs. Shape is `(height, width)`. `world.cells.T.copy().tobytes()` reproduces `cell_bytes` exactly.

The layout is unchanged from the runtime oracle: little-endian `struct <BHHhhBBBBBBB`, SHA-256 over all expanded cells. For the genuine seed-314159265 fixture, 5,040,000 cells occupy **80,640,000 bytes** and hash to `97d590a10019a682b04bdbf076c70b18f8f557fc9c6c0ce3964158994eac9b4c`. There is no new hashing scheme or lossy projection. Every field was also compared against a hash-preserved copy of the pre-promotion decoder.

In the source mapping below, `h1` is the first serialized cell header; bit 0 chains to `h2`, then `h3`, then `h4`. These are file-format headers, not `Tile.sTileHeader`'s in-memory layout.

| Offset | Field | Type | Exact preserved meaning / v325 source |
|---|---|---|---|
| 0 | `active` | uint8, 0/1 | Presence: `h1 & 2`. Native type **0 is Dirt**, not air. |
| 1 | `tile` | uint16 LE | Native tile ID; second byte when `h1 & 32`. Zero default when type was not serialized. |
| 3 | `wall` | uint16 LE | Low byte when `h1 & 4`; high byte after liquid payload when `h3 & 64`. Zero means no saved wall. |
| 5 | `frame_x` | int16 LE | Saved frame X if active and the file's frame-importance bit is true; otherwise sentinel -1. |
| 7 | `frame_y` | int16 LE | Same rule for frame Y. Saved negative values are retained. |
| 9 | `tile_paint` | uint8 | Saved byte when active and `h3 & 8`; otherwise zero. |
| 10 | `wall_paint` | uint8 | Saved byte when wall present and `h3 & 16`; otherwise zero. |
| 11 | `liquid_kind` | uint8 | Canonical enum: 0 absent, 1 water, 2 lava, 3 honey, 4 shimmer. `(h1 & 24) >> 3`, with `h3 & 128` selecting shimmer when a liquid payload exists. This is explicitly **not** the in-memory `Tile.liquidType()` numbering. |
| 12 | `liquid_amount` | uint8 | Payload byte when liquid is encoded. An encoded kind with amount zero remains identifiable. |
| 13 | `shape` | uint8 | `(h2 & 112) >> 4`: 0 full, 1 half-block, 2–5 native slopes 1–4. `half_block` and `slope` properties expose those meanings without redundant storage. |
| 14 | `wires_actuator_inactive` | uint8 bitset | Bits 0/1/2 = red/blue/green wires (`h2 & 2/4/8`); bit 3 = yellow wire (`h3 & 32`); bit 4 = actuator (`h3 & 2`); bit 5 = inactive/actuated (`h3 & 4`). |
| 15 | `coatings` | uint8 bitset | Preserve `h4 & 30`: bit 1 invisible block, bit 2 invisible wall, bit 3 fullbright block, bit 4 fullbright wall. |

`world.frame_saved` derives presence from the retained frame-importance table plus `active`; it does **not** infer presence from `frame_x != -1`. Unknown native tile/wall IDs stay uint16 throughout. Rendering has a separate, incomplete palette map and cannot change these IDs. Synthetic importer tests retain IDs **60000 / 50000**, frames **-32768 / 32767**, all flags, paints and an encoded zero-amount shimmer payload. Synthetic files are tests, not genuine oracle evidence.

## File-level state and validation

The immutable object also retains the exact preamble and all eleven sections: header, tiles, chests, signs, NPCs, tile entities, weighted pressure plates, town manager, bestiary, creative powers and footer. `original_bytes()` reconstructs the source bytes exactly, including original RLE and uninterpreted data. It returns bytes; there is no world editor or `.wld` writer. Source SHA-256, metadata, section hashes, pointers, frame-importance table and the full generation manifest are available through a fresh report dictionary; mutating that dictionary does not mutate the world.

Validation checks format/magic, Small dimensions, eleven ordered section pointers, nonempty frame-importance table, active tile IDs within that table, header boundaries, full 5,040,000-cell RLE expansion, no cross-column/negative runs or trailing cell bytes, and footer name/world-ID consistency. Unknown reserved header bits and invalid slope encodings fail explicitly. The terminal manifest must have a valid .NET string length, end exactly at the tile pointer, and identify the pinned version. Import verifies both the unchanged expanded-cell hash and exact reconstruction of the original file.

Opaque sections remain opaque. Their bytes survive, but this milestone does not interpret chest inventories, NPC records, spawn classifications, progression flags beyond the audited header prefix, or generation-history objects. Normalized hashes are audit comparisons only; original metadata is never normalized in storage.

## Pinned source provenance

Paths below begin `Game Reference/02_static/decompiled/client/Terraria/`:

- `IO/WorldFile.cs:1186-1272`: section order, metadata, pointers, frame-importance packing and identity/dimensions. `1274-1466`: world flags and terminal manifest. `1468-1641`: all cell fields and vertical RLE. `2559-2764`: corresponding loader. `1790-1797`: footer.
- `Tile.cs:8-24`: native ushort tile/wall storage and signed-short frames. `216-237`, `442-457`, `594-721`: native liquid, wires, presence, inactive, half-block, actuator and slope accessors.
- `IO/FileMetadata.cs:8-33`: Re-Logic metadata block.

The `WorldFile.cs` SHA-256 remains `17d45c43ad981505fb55d6067c21908bba10f5ce690c5f194cf24d951b4b83c2`; other locked hashes are unchanged in [TARGET_LOCK.json](TARGET_LOCK.json). Additional source excerpts, hashes and field-equivalence tests are recorded in `audit/canonical-import-20260914/verification.json`.

## Rendering boundary

Native cell geometry supplies all crops. The adapter reads the existing `TILE_STYLES` / `WALL_COLORS` palettes, reproduces the existing coordinate texture and edge-light treatment, and adds subcell half/slope masks. It never constructs a simulation world. Native green Dungeon brick can therefore look blue, matching TerraExplorer's original Dungeon art; its native ID remains 43.

Unknown tiles/walls receive deterministic ID-derived muted colors. Saved frames remain data, not proprietary sprites. Paint, wall visibility and fullbright are retained but do not alter colors in this view; invisible/inactive blocks are deliberately ghosted to expose structure geometry. Wire/actuator overlays, game lighting, sprite silhouettes, furniture framing and collision/pathfinding are outside this milestone. Liquids use the existing colors and amount-dependent opacity. No Re-Logic graphics were loaded or redistributed.
