# Changelog

All notable project changes are documented here. This project follows
[Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-07-21

### Added

- TerraForge package with a deterministic, named 107-pass pipeline.
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
- MIT license, contributor guide, architecture, fidelity, and benchmark docs.

### Changed

- Corrected the pass count from 103 to the 107-step public 1.4.4.9 list.
- Replaced source-fidelity claims with explicit modeled/approximated/documented
  labels.
- Reduced core dependencies to NumPy and Pillow; legacy plot dependencies moved
  to an optional extra.

### Deprecated

- `Engine/`, `Code/`, and `Advanced/` are retained only as a legacy research
  archive. New work should target `terraforge/`.
