# Open Problems: Terraria-Generation (TerraExplorer)

This document catalogs open problems, algorithmic challenges, and maintenance tasks for the **Terraria-Generation** (TerraExplorer) 107-pass procedural world generation and visualization engine (`Engine/worldgen.py`, `Engine/theme.py`, `terraexplorer/`).

---

## 1. Algorithmic & Implementation Problems

- **Multi-Scale Density Invariance across TINY, SMALL, and LARGE Worlds**
  - **Problem**: Scaling the 107 world-generation passes between `TINY` (240x140 tiles at ~6 px/tile), `SMALL` (4200x1200), and `LARGE` (8400x2400) without distorting per-tile ore vein density (`6e-05`), biome structure sizes, or cave percolation thresholds.
  - **Context**: Tracked in `audit/audit-2026-04-23-mini-world-redesign.md`. Fixed-probability spawns calibrated for LARGE worlds produce sparse, unreadable visualizations on small crop windows.
- **Tkinter GUI Event Loop Decoupling**
  - **Problem**: `terraexplorer gui` executes world generation passes on the primary Tkinter event loop. While cancellable between passes, long-running passes on LARGE worlds cause temporary interface freezes.
  - **Context**: Requires decoupling the generator into a background worker thread with asynchronous progress queues.

---

## 2. Bugs & Unresolved Issues

- **GIF Compression and Palette Banding (`saveTinyGif`)**
  - **Problem**: Maintaining compact GIF footprints (< 1.5 MB in `Plots/`) via `Engine/theme.py::saveTinyGif` using adaptive 128-color P-mode palette downsampling without introducing color banding or palette flickering across 107 dynamic passes.
- **LaTeX Math Rendering Stability in README**
  - **Problem**: Ensuring cellular automata equations (`\quad \text{if} \quad` chains) render cleanly across GitHub KaTeX without syntax parser regressions.

---

## 3. Theoretical & Scientific Problems

- **Cave Cellular Automata Percolation Thresholds**
  - **Problem**: Quantifying exact percolation thresholds and topological genus invariants for Terraria's Moore-neighborhood cellular automata cave-carving algorithms (`_carveCaves`, `cavinator`) across stratified depth layers (Surface, Underground, Cavern, Underworld).
- **Inhomogeneous Biome Spread Kinetics**
  - **Problem**: Formulating analytical differential models for the V-shaped Corruption, Crimson, and Hallow Hardmode conversion fronts propagating through mixed dirt, stone, sand, and mud soil matrices.

---

## 4. Code Maintenance & Refactoring Opportunities

- **Audit Archive Indexing**
  - **Opportunity**: `audit/` contains 17 deep audit reports from historical revision sprints. Consolidating key takeaways into `docs/architecture.md` and archiving superseded audit logs will streamline repository onboarding.
