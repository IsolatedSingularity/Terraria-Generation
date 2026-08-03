# Changelog

All notable project changes are documented here. This project follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- Corrected Floating Island, Dungeon, Pyramid, Aether, Jungle Temple, and
  Underworld landmark generation, including the public `Ruined house` marker.
- Added deterministic biome-containment and meteor/granular/four-liquid
  catastrophe laboratories with generated README animations and tests.
- Replaced legacy README plots with clean active-runtime diagnostics, used
  Preview-world landmark crops, retained the full Small-world overview, and
  removed the logo's README-only bottom gap.
- Standardized every README diagram except the full biome overview on actual
  `240 x 140` Preview-world renders or exact crops, including containment,
  catastrophe, depth, landscape, cave, ore, benchmark, and fidelity media.
- Reworked containment to spread through generated terrain, replaced the
  catastrophe rectangle with irregular connected cave pools, and documented
  the public landscape rules and remaining simulation problems.
- Normalized README section titles, moved flavor text into subtitles, and
  refreshed fidelity and open-problem documentation.
- Renamed the supported product and package from TerraForge to TerraExplorer,
  with compatibility aliases for existing imports and console commands.
- Expanded modeled world generation with Underworld Ruined Houses and Hellforges,
  floating-island houses and waterfalls, spreading evil and Hallow, a buried
  Pyramid, a multi-level Jungle Temple, Shimmer, and a stronger Dungeon facade.
- Added a 26-stop GUI evolution rail with reversible playback, widened the map
  workspace, narrowed the log, and recaptured the complete desktop window.
- Rebuilt README media around a controlled same-seed comparison, landmark and
  biome atlases, a biome-spread animation, and a deeper `4200 x 1200` descent.
- Removed the repository license. No license is now granted and all rights are
  reserved.
- Replaced the shared-runner-sensitive 1.5-second test with a runaway-regression
  guard, and moved both workflows to the Node 24-based official actions.
- Changed the workshop fit mode to fill and center the map table, reshaped the
  Temple into an asymmetric stepped complex, deepened and varied the Underworld
  lava terrain, and moved the descent study to the snowy Dungeon coast.
- Replaced the project emblem with a transparent mechanical pixel-tree icon
  based on the supplied silhouette reference.
- Restyled the desktop app as a brass-and-iron world forge and refreshed its
  README screenshot.
- Added a six-biome terrain study and animated surface-to-Underworld descent,
  and restored the three-world comparison to the README.
- Reworked the README with fewer lists, no benchmark or fidelity scorecards,
  clearer project scope, and restrained Terraria-inspired easter eggs.
- Corrected stale legacy-reference documentation and removed duplicate ignore
  rules.

## [1.0.0] - 2026-07-21

### Added

- TerraExplorer package with a deterministic, named 107-pass pipeline.
- Canonical tile, wall, liquid, and biome registries with separate NumPy arrays.
- Preview and Small world sizes, both evil types, alternate ores, biome caves,
  structures, liquids, and an optional Hardmode V transformation.
- Progress, timings, cancellation, snapshot callbacks, phase controls, and
  pass-level fidelity metadata.
- Original pixel renderer, map symbols, mechanical-tree brand, PNG/GIF/NPZ/JSON
  exports, and reproducible README media.
- Lightweight Tk desktop GUI with inspection, overlays, comparisons, telemetry,
  exports, and dark/light interfaces.
- CLI, test suite, Windows/Ubuntu CI, Python package build, and Windows executable
  release workflow.
- Contributor guide, architecture, fidelity, and benchmark documentation.

### Changed

- Corrected the pass count from 103 to the 107-step public 1.4.4.9 list.
- Replaced source-fidelity claims with explicit modeled/approximated/documented
  labels.
- Reduced core dependencies to NumPy and Pillow; legacy plot dependencies moved
  to an optional extra.

### Deprecated

- `Engine/`, `Code/`, and `Advanced/` are retained only as a legacy research
  archive. New work should target `terraexplorer/`.
