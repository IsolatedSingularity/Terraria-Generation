# Open problems

This list covers active questions in the supported `terraexplorer/` runtime.
The `Engine/`, `Code/`, and `Advanced/` directories are historical research
archives and are not implementation targets.

## Versioned fidelity

TerraExplorer deliberately retains the documented Terraria 1.4.4.9 generation
order as its 107-pass baseline. Terraria itself has moved beyond that version,
so newer structures, secret-seed branches, and placement rules need an explicit
compatibility policy before they can enter the baseline. A future migration
should distinguish updated source-inspired behavior from TerraExplorer's stable
seed reproducibility; secret-seed support is intentionally deferred.

## Multi-scale density invariance

Preview (`240 x 140`) and Small (`4200 x 1200`) worlds need materially similar
ore density, cave connectivity, biome occupancy, and structure frequency without
making Preview landmarks microscopic or Small worlds excessively dense. Fixed
per-tile probabilities and fixed tile dimensions solve opposite ends of this
problem. The next step is a documented set of dimensionless scale rules plus
statistical acceptance ranges across a seed corpus.

## Structure topology and protection

The Dungeon, Pyramid, Jungle Temple, Aether, Floating Islands, and Ruined Houses
now have more representative silhouettes and internal layouts. Their validation
is still mostly local: bounds, required materials, marker positions, and a few
topological checks. Useful future properties include guaranteed room
reachability, minimum wall thickness, protected-region non-overlap, usable
entrances, and stable landmark frequency over many seeds.

## Full-world liquid and granular simulation

The catastrophe laboratory models deterministic meteor excavation, falling Sand
and Silt, conservative liquid transport, and all four supported liquid-contact
products on a Preview world. It is intentionally bounded rather than a live
solver over every generation frame. Open work includes pressure, pumps, larger
connected basins, viscosity differences, settling convergence, performance on
Small worlds, and invariant tests for total liquid volume around reactions.

## Biome-spread calibration

The containment laboratory exposes three-tile reach, faster surface attempts,
and deterministic intervention comparisons. It does not yet claim calibrated
Terraria time. Spread rates should be estimated against documented behavior,
separated by host material and depth, and tested statistically over multiple
seeds. Sunflower and Chlorophyte effects also need finer spatial rules before
the lab can be treated as a quantitative emulator.

## Cave connectivity

Void fraction alone does not describe a playable cave network. The generator
still lacks explicit targets for percolation, connected-component size,
vertical traversal, loop count, dead ends, and access from the surface. These
metrics should be gathered by depth band and world scale before changing the
cave-carving distributions.

## Animated-media size and palette stability

Tracked GIFs must balance readable tiles, coherent global palettes, and a
reasonable repository footprint. A shared palette or modern video alternative
could reduce flicker and banding, but any change needs GitHub README support and
visual regression checks. The generation script should eventually enforce
media dimensions and file-size budgets.

## Scientific verification

Numerical changes need more than snapshot tests. The project would benefit from
a reproducible seed corpus, analytic-limit checks where possible, distribution
comparisons, topology metrics, and machine-readable experiment reports. This is
especially important for scale-dependent generation and post-generation
simulations.

## Historical audit indexing

The `audit/` directory contains useful but partly superseded investigations.
An index mapping each conclusion to current code, documentation, or a closed
decision would make it easier to separate live risks from historical context
without changing the archive itself.
