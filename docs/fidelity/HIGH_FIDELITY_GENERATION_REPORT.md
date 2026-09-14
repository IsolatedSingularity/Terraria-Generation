# Small-world generation upgrade

14 September 2026. Target: **Terraria 1.4.5.7, Steam build 24825745**, ordinary
Small worlds. This is an independent generator upgrade, not compatible seed replay.

## Baseline and agreed scope

Started on clean `main`, commit `46cbbecefd06428afe7a35437dc2191820edb8b3`.
The user approved updating the nine generation figures and seed manifest,
minor presentation adjustments, native Small generation rendered into the
existing canvases, independent generation, accurate README edits, and a fresh
real GUI screenshot. Branding, legacy research directories, public APIs and
the 107-pass contract were preserved. No dependencies were added; nothing was
staged, committed, pushed or published.

Small selects the new handlers. Preview retains its compact algorithms and the
API/CLI default for compatibility. The GUI now defaults to Small. Oracle cells
are used only by comparison tools, never by these generation handlers.

## Implementation and evidence

The existing [target lock](TARGET_LOCK.json), [runtime evidence](RUNTIME_ORACLE_REPORT.md),
[canonical import](CANONICAL_IMPORT_REPORT.md), and Living Tree
[replay](LIVING_TREES_REPLAY_REPORT.md) / [object](LIVING_TREES_OBJECT_PLACEMENT_REPORT.md)
reports establish the reference boundary. Local source retrieval began at
`Game Reference/08_agent_library/START_HERE.md` and used the topic router and
`Query-Terraria.ps1`. The source ranges below are under
`Game Reference/02_static/decompiled/client/Terraria/`; that ignored corpus is
not included in public clones. No source dumps were added to the repository.

| System | Source evidence inspected | Independent implementation and limits |
|---|---|---|
| Terrain/layers | `GameContent/Biomes/TerrainPass.cs`, feature selection, offset walk and final extrema, lines 1–480 | Plateau/hill/dale/mountain/valley walk; Small 0.19H–0.26H limits, beach limit 0.23H; per-column rock walk; global surface=max+25, rock snapped in six-tile increments. Right-beach history retarget is a gradual ramp. |
| Caves | `WorldGen.cs:12459–12615`, `78906–79357` (`TileRunner`) | Source strength/step distributions, tapering clipped bounds and jittered Manhattan footprint. Vectorized PCG edge draws differ from native draw order. Small Holes and Rock Caves use 60% of the native attempt budget because supplemental legacy carving remains. This is measured hybrid calibration. |
| Snow | `WorldGen.cs:12768–12878` | Drifting boundaries, depth limit derived from lava line, Snow/Ice material conversion. Simplified boundary walk. |
| Jungle | `GameContent/Biomes/JunglePass.cs`, mud-runner stages | Multiple overlapping deep mud masses and an upward path; source scale informs dimensions. Cell geometry, placement rejection and grass transitions remain reduced. |
| Underground Desert | `GameContent/Biomes/Desert/DesertDescription.cs:119–170`; `DesertHive.cs:46–70` and influence-field construction | Native block scale and depth range; depth-two occupied-cell clusters, competitive claims, inverse-square influence field. Fourfold block-to-tile width; thresholds 3.5/1.8/0.7. A first percolating-cluster implementation produced an overly solid oval and was replaced after visual QA. |
| Evil | `WorldGen.cs:77395–77704` (`CrimStart`), `77920–78181` (`ChasmRunner`) | Surface conversion plus descending connected scars, Corruption branches, Crimson chamber/finger routes. Finite-depth geometry replaces full-height painted bands. Counts, chamber size, placement and materials remain approximations. |
| Dungeon | `GameContent/Generation/Dungeon/DungeonCrawler.cs:360–399`; `LayoutProviders/LegacyDungeonLayoutProvider.cs:25–106`; `Rooms/LegacyDungeonRoom.cs:169–312`; `Halls/LegacyDungeonHall.cs:190–490` | Advancing trunk, temporary side branches, local RNG seeds, drifting room stamps, source room/hall sizes and width/60 attempt budget. Hall occupancy rejection and entrance stairs remain simplified; measured footprint is still too small. |
| Temple | `WorldGen.cs:34514–35400`, especially `34559–34709` | 10–15 variable rooms, increasing switchback rows, overlap rejection, larger final chamber, connected halls and row envelope. Exterior/interior unions are painted once to avoid sealing earlier connections. Decoration remains representative. |
| Living Trees | `WorldGen.cs:28934–29592`; passage branch `29594–30314`; existing validated replay | Reuses tested crown/root geometry and UnifiedRandom on a temporary projection of independently generated cells. Connected shaft, side passage and room are approximations. Decorative object calls are omitted; canonical object placement is not claimed to be integrated. |
| Oceans/Underworld | `WorldGen.cs`, ocean shaping and Underworld pass starting 14164 | Local coastal shelves preserve rock below sand. Late Beaches no longer converts the entire coastal column. Ash ceiling/floor masses and discrete Hellstone replace the thin basal stratum; existing buildings remain. Liquid settling and exact native ash runners are not reproduced. |
| Hardmode | `WorldGen.cs:32648–32731`, `78241–78531` (`GERunner`) | Source origin distributions, inward 3:5 directions and 200–249-tile initial breadth replace the inverted, narrow V. Reduced tile registry retains sand/ice substrates with converted biome membership; full native material families and trajectory jitter remain absent. |
| Ground spawn opportunity | `NPC.cs:950–1038`, `1050–1076`, `5447–5490`, reference screen properties 6847/6849 | Two-by-three clearance, supported floor, downward candidate-column mass, lava and deep Honey/Shimmer exclusions. Conditional geometry only, with a 103-cell cap; no fabricated light/biome multiplier or permanent world-spawn safe bubble. |

Exact reused pieces are the **UnifiedRandom implementation and the supported
Living Tree crown/root operations under their validated inputs**. Reinitializing
RNGs and projecting TerraExplorer's reduced state means the resulting trees and
worlds are not exact native outputs. Terrain distributions and structure
dimensions are source-derived mechanics, not an exact replay claim.

Small Hive, Spider and Gem placements now respect existing structure bounds.
The checks caught a Hive overwriting Temple walls and splitting its navigable
interior. A later integrity check caught a Granite marker extending 11 tiles
past the right edge: marker creation now clips Small placement bounds to the
same grid as the already-clipped geometry.

## Multi-seed comparison

The existing saved control worlds for seeds **1, 2, 314159265** were imported
again from `audit/living-trees-replay-20260914/seed-<seed>-control/bootstrap-small.wld`.
All are 4200×1200, Classic, Corruption, pre-Hardmode, ordinary settings.
No new oracle host run was needed. SHA-256 identifiers are:

| Seed | Saved world SHA-256 |
|---|---|
| 1 | `504bbdd4c9f534c21ccbe60f6c65f2e55d3158d70038b015ac6d1c5cc8e8e2bb` |
| 2 | `119ccf8afc41638d5353b7fe463b3bca58c33bee31c9d88b4d2add4a659cbaa8` |
| 314159265 | `89b383b6d4e7d8a959c3679350271c84fec0cc311f3dc8994c43e5b48db70487` |

`python -m scripts.validate_generation --output audit/high-fidelity-generation/final --oracle`
records the comparison. Before/after metrics and screenshots are retained in
ignored `audit/high-fidelity-generation/{before,final}/`. These seeds informed
calibration, so this is **not a held-out statistical validation set**. Additional
independent integrity checks use seeds 17 and 99; Crimson seed 17 and a Small
Crimson/Hardmode CLI run cover further configurations without claiming Crimson
oracle validation.

| Seed | Old surface / rock | Updated surface / rock | Oracle surface / rock |
|---|---|---|---|
| 1 | 228 / 552 | 337 / 439 | 337 / 397 |
| 2 | 228 / 552 | 337 / 487 | 337 / 487 |
| 314159265 | 228 / 552 | 337 / 481 | 337 / 469 |

All three Underworld boundaries are 1000. Occupancy means **material presence**
(generated non-AIR versus imported active tile), not collision solidity. Mean
absolute error averages equal-width, 100-row bands over columns 350–3849, then
over the three seeds:

| Occupied fraction error | Before | Updated |
|---|---:|---:|
| All 12 depth bands | 8.68 percentage points | **4.29** |
| Cavern comparison bands, rows 500–999 | 3.21 | **2.26** |
| Underworld bands, rows 1000–1199 | 29.00 | **8.37** |

Cave connectivity uses four-neighbor components on a stride-four sampled grid,
below each world's global surface and above its Underworld. It is a coarse
topology diagnostic, not character traversability:

| Seed | Largest cave share: old / updated / oracle | Component count: old / updated / oracle |
|---|---|---|
| 1 | 21.91% / 6.25% / 2.80% | 426 / 2247 / 3199 |
| 2 | 14.78% / 4.36% / 4.99% | 476 / 2309 / 3136 |
| 314159265 | 17.10% / 3.91% / 3.48% | 476 / 2308 / 3182 |

Temple bounds changed from the same **188×120** rectangle on every seed to
**222×225, 246×190, 183×206**. Native bounds are **264×171, 236×196, 206×193**.
Living Tree support groups changed from **5/5/4** to **1/2/1**, versus native
**1/3/1**; their vertical extents increased from roughly 64–115 to 173–217 tiles.
Native tree support includes wall families absent from our registry, so its
support counts are not a strict like-for-like cell metric.

**Dungeon size did not improve.** Updated heights are 515/537/527, versus old
729/760/745 and native 806/845/738. Updated support is 57,769–61,054 cells versus
native 114,664–142,428. The room-and-branch construction is more source-informed,
but occupancy rejection, exploration extent and entrance height remain a clear
next task. Temple depth is also biased downward. Liquid coverage remains low:
updated 4.46–4.66%, native 5.46–5.96%.

The saved surface quantiles use the first selected natural material below row
120, which can include islands and altered terrain; they are a proxy, not a
reconstruction of the pre-cave native TerrainPass surface. No coordinate-wise
seed match is expected. Visual checks used an old/new/imported-world montage
with original palette colors, not Re-Logic sprites.

## Media and README

All nine figures use Small-world data. Canvas sizes and decoded frame counts
are preserved:

| File under `docs/media/` | Old = new canvas | Frames | Presentation |
|---|---|---:|---|
| `terraexplorer_generation.gif` | 960×616 | 34 | Full grid → 960×560, 56px title |
| `idle_corruption.gif` | 960×616 | 9 | 240×140 native tiles, 4px/tile |
| `idle_crimson.gif` | 960×616 | 9 | Same scale |
| `idle_opposition.gif` | 960×616 | 9 | Same scale |
| `terraexplorer_world.png` | 1560×650 | 1 | 4200×1200 → 1400×400, unchanged |
| `biome_atlas.png` | 584×2472 | 1 | Ten 48×72 crops, 6px/tile, unchanged |
| `containment_lab.gif` | 968×680 | 8 | Four 240×140 native crops, 2px/tile |
| `depth_descent.gif` | 960×294 | 54 | Overview → 960×952, 238px viewport + title |
| `spawn_heatmap.png` | 1560×610 | 1 | Same full grid and 1400×400 map as distribution |

`biome_seeds.json` was regenerated and is byte-identical: the original ten seeds
are now fixed in advance, rather than selected by a cross-seed beauty score.
Within-world crop optimization remains. The opposition seed has an explicit
`20` suffix to place both genuine conversion legs in one native detail view.
Idle crops include a simulated 32-tile halo, but 96 iterations can still be
influenced by their finite boundary; they are not a full-world temporal replay.
Containment deliberately installs front columns and barriers in generated crops;
the bastion has no hidden interior protection mask. Its retained seed name
`Three Front Siege` now has two modeled infections, Crimson and Hallow.

`gui.png` changed **1518×867 → 1502×859** by capturing visible frame bounds
instead of transparent Windows resize borders. Fit now preserves the whole
Small map and inspection coordinates, and the log scrollbar occupies its own
column. Capture waits for completed generation, checks screen bounds, and
returns failure if no completed capture is produced. The normal controls panel
remains scrollable. Original branding was not regenerated.

README edits cover version and import scope; mixed isolated RNG semantics;
explicit Small examples; crop versus overview transforms; biome construction;
sampled containment updates; extrema-derived layer equations; and conditional
spawn-column mass. The old fixed-depth formula and invented multiplicative
light/biome spawn score were removed. The spawn equation now states exactly
what is measured, with clearance, cap, pooling and player assumptions.

## Verification and observed failures

Evidence logs live under ignored `audit/high-fidelity-generation/`.

- Baseline: initial nested pytest temporary directory was missing, yielding
  **93 passed, 13 setup errors**. After creating the parent, the same baseline
  suite passed **106 tests, 93.54% coverage**.
- First new focused run: **12 passed, 1 failed**. Temple traversal exposed the
  Hive overwrite; placement guards fixed it. Recheck: **8 passed**. Later
  generation/render focused run: **13 passed**; additional Crimson/Hardmode
  tests: **2 passed, 8 deselected**.
- Full `python -m pytest --cov=terraexplorer --cov-report=term-missing
  --cov-fail-under=75 --basetemp audit/high-fidelity-generation/final-pytest`:
  **116 passed, 94.46% coverage**, 207.51s. After adding the GUI transform
  regression test, `release-pytest`: **117 passed, 94.46%**, 196.03s. This includes
  all canonical import, format-325, UnifiedRandom and Living Tree tests.
- GUI focused test: **2 passed**. Real desktop captures were inspected;
  the first exposed the old cover-style Fit and log overlap, both corrected.
  The initial sandboxed capture failed to initialize Tcl; the authorized
  visible-desktop execution succeeded.
- Ruff lint and format checks passed; mypy passed **24 source files**. Formatting
  and one lambda-style lint error were corrected during development. Final
  marker-bound verification and media repeat results are recorded below.
- Preview CLI PNG/GIF smoke passed (240×140, Corruption). Small CLI PNG/NPZ/JSON
  smoke passed (4200×1200, Crimson, Hardmode, seed `mechanical-tree`). The latter
  generated in 20.979s with other checks running.
- Standard isolated build initially failed because sandbox networking blocked
  fetching declared `hatchling`. The offline attempt confirmed it was absent.
  Authorized network-enabled isolated build then produced wheel and sdist
  successfully in `audit/high-fidelity-generation/dist/`.
- Two fresh full media directories were generated successfully. Visual QA led
  to refreshing the idle crops to show active underground fronts, then updating
  the distribution/generation outputs after clipping legacy marker bounds.
  Old/new first/final-frame contact sheets, the atlas, intermediate idle/depth
  frames, spawn map and real GUI screenshot were inspected. Decoded repeat
  results are appended after the final refresh.
- Expanded integrity checking initially failed on the out-of-world Granite
  marker described above. No tile-array out-of-bounds write was observed.

### Final verification after the marker fix

- Full suite with the same coverage arguments and
  `--basetemp audit/high-fidelity-generation/verified-pytest`:
  **117 passed, 94.45% coverage**, 94.04s (`verified-tests.log`). The gate remains
  75%. The new source handlers have 99% statement coverage; this establishes
  execution and invariants, not native behavioral parity.
- `ruff check terraexplorer terraforge tests scripts packaging`: passed.
  `ruff format --check` on those same paths: **59 files already formatted**.
  `mypy terraexplorer`: **24 source files, no issues**.
- `python -m scripts.validate_generation --output
  audit/high-fidelity-generation/integrity --seeds 1 2 314159265 17 99`: passed
  for all five seeds. Spawns are in bounds and clear; all marker bounds fit the
  grid; occupied cells are dry; empty liquids have NONE kind. The three shared
  seeds' comparison metrics are unchanged by the metadata clipping fix.
- All **nine** figures have identical decoded RGBA pixels, dimensions, frame
  counts and frame durations across the two refreshed output directories.
  Evidence: `media-signatures.json` and `qa_media.py compare` in the local audit
  folder. The final files in `docs/media/` byte-match the verified first output.
  The seed manifest reproduced identically and six unrelated media files
  byte-match the baseline copies.
- Final isolated wheel/sdist build passed (`build-verified.log`). Archive
  inspection confirmed both new runtime modules are packaged, and no `audit/`,
  `Game Reference/`, `.wld`, `.exe` or `.dll` content is included.
- `git diff --check` passed. Working changes comprise exactly the files below;
  reference corpus and audit evidence remain ignored. Review the updated README
  and GUI image before pushing, especially the changed meaning of the spawn
  figure and the documented Dungeon/performance limitations.

The comparable logged baseline generation times were 5.19–5.27s; updated runs
were 18.98–23.34s under concurrent checks. These are observations, not controlled
benchmarks. The upgrade is substantially slower. The native tree adapter and
full-grid masks allocate temporary arrays; peak process memory was not measured.
Media caching was reduced to two biome worlds to avoid retaining dozens of
native grids. GUI cancellation still happens between passes.

## Remaining limits and next task

Prioritize Dungeon look-ahead placement, exploration extent and entrance stairs
against a larger held-out native set, then profile the new Small handlers.
Floating Islands, cabins, minor biomes, liquid settling, Temple furnishing,
Living Tree passages, ore details and the reduced tile/shape registry remain
representative. Spread batches have no calibrated in-game clock. Spawn output
is conditional candidate geometry, not a 50-attempt success probability, NPC
population model or enemy distribution. Secret worlds and exact shared RNG
history remain outside this upgrade.

## Changed files

- `README.md`; this report.
- `terraexplorer/source_generation.py` (new), `spawn_opportunity.py` (new),
  `generation.py`, `pipeline.py`, `gui.py`.
- `scripts/media_views.py` (new), `validate_generation.py` (new),
  `generate_media.py`, `capture_gui.py`.
- `tests/test_source_generation.py` (new), `test_render.py`, `test_gui.py`.
- The nine media files listed above and `docs/media/gui.png`.
- No files deleted. `biome_seeds.json` was reproduced unchanged. Temporary QA
  images, logs, distributions and native reference worlds remain ignored.
