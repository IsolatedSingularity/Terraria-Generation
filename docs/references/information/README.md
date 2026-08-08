# Information reference atlas

These screenshots are the project's authoritative visual references for the
current world-generation correction cycle. They were moved intact from the
temporary root `Information` folder so their original framing and filenames
remain traceable. TerraExplorer uses them as composition and silhouette
references, not as distributable game art or renderer inputs.

Public Terraria documentation is used only as a sanity check for relationships
such as spread distance, layer placement, and structure location. When a visual
detail differs, these supplied references govern the project interpretation.

| Reference | Active TerraExplorer interpretation |
|---|---|
| `Corruption Biome.png` | Multiple narrow surface chasms, deep near-vertical scars, and underground connections. |
| `Crimson biome.png` | A slanted descent into linked chambers and a deep central bulb with branches. |
| `Granite biome.png` | Compact enclosed stone mini-biome with a distinct dark material shell. |
| `Shroom biome.png` | Open Cavern pocket bounded by Mud and exposed Mushroom Grass. |
| `Shimmer.png` | Broad stone Aether shell with a low, contained Shimmer pool. |
| `Bee Nest.png` | Irregular Hive shell with an open honey-bearing interior in the Jungle. |
| `Rails.png` | Long, mostly horizontal underground track with gradual slopes. |
| `Life tree structure.png` | Surface canopy, thick Living Wood trunk, roots, and a deep central shaft. |
| `Island Type I.png` | Forest-capped Floating Island with a compact sky house. |
| `Island Type II.png` | Separate cloud-supported Floating Lake without a house. |
| `Jungle Temple + forest in back.png` | Large asymmetric Lihzahrd shell containing layered connected passages. |
| `Hell cities (and types) + lava biome.png` | Varied central Underworld towers standing in irregular lava terrain. |
| `Dungeon (drunk world so incorreclty under groun).png` | Branching rooms and corridors; the underground entrance is treated as a secret-seed artifact, not the normal placement rule. |

The active implementations live in `terraexplorer/generation.py`. The
`Micro Biomes` pass now includes abandoned minecart tracks as the only supplied
feature that was absent from the runtime before this cycle.
