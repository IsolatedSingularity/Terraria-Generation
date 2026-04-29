# Terraria-Generation Audit Checklist
# Used by @shinra-terraria during audit sessions.
# Read this file when running a full audit.

## Worldgen Accuracy
- [ ] Pass ordering: are implemented passes in correct sequence (5 macro-phases)?
- [ ] World dimensions: correct tile counts for Small/Medium/Large + border buffer?
- [ ] Layer depths: worldSurface, rockLayer, maxTilesY-200 strata variables correct?
- [ ] Cave carving: uses TileRunner random walks (NOT 2D Perlin noise)?
- [ ] Ore distribution: proportional scaling via 6E-05 formula? Correct Y-ranges?
- [ ] Biome placement: dungeonX polarity constraint respected?
- [ ] Structure quotas: match world-size scaling from ephemeral context?

## Algorithm Fidelity
- [ ] TileRunner: strength decay, diamond brush, directional drift parameters?
- [ ] digTunnel: sphere-cutter (not noisy diamond)?
- [ ] Cellular automata: cave smoothing separate from biome spread?
- [ ] SettleLiquids: bottom-up recursive pass?
- [ ] Evil biome distinction: Corruption chasms vs Crimson cave networks?

## Code Quality
- [ ] camelCase: all variables, functions, class methods?
- [ ] Type annotations: on all public functions?
- [ ] numpy vectorized ops: no nested Python loops for tile-level operations?
- [ ] Visualization: output to Plots/ directory?
- [ ] No hardcoded magic numbers without comments referencing source data?

## Implementation Coverage
- [ ] How many of 103 passes are implemented? List by name.
- [ ] Which macro-phases are represented?
- [ ] Gap analysis: what is missing vs existing-repo-analysis.md?
- [ ] Are implemented algorithms accurate to decompiled source references?

## Output Format
```
## Terraria-Generation Audit
**Date**: [date]
**Agent**: @shinra-terraria
**Pass Coverage**: [X/103 passes implemented]

### Worldgen Accuracy
| # | Check | Status | Evidence | Notes |
|---|-------|--------|----------|-------|

### Algorithm Fidelity
| # | Check | Status | Evidence | Notes |
|---|-------|--------|----------|-------|

### Code Quality
| # | Check | Status | Evidence | Notes |
|---|-------|--------|----------|-------|

### Implementation Gaps (severity-ranked)
| # | Severity | Missing Feature | Priority Phase | Effort |
|---|----------|----------------|----------------|--------|

### Summary
[2-3 sentences on overall state and recommended next steps]
```
