# Fidelity inventory

TerraExplorer is an educational approximation. It follows the public vanilla
1.4.4.9 pass names and ordering while selectively exploring newer world
concepts; it does not copy Terraria's source, internal tile IDs, RNG stream, or
assets.

The authoritative machine-readable inventory is `terraexplorer/passes.py`. Run
`terraexplorer passes` or `terraexplorer passes --json` to inspect it.

## Status definitions

- **Modeled**: a distinct TerraExplorer operation mutates the world or its metadata.
- **Approximated**: a simpler/shared operation represents the pass's broad role.
- **Documented**: preserved in ordering and telemetry without a grid mutation.

## Modeled (67)

Reset; Terrain; Dunes; Ocean Sand; Sand Patches; Tunnels; Mount Caves; Dirt Wall
Backgrounds; Rocks In Dirt; Dirt In Rocks; Clay; Small Holes; Dirt Layer Caves;
Rock Layer Caves; Surface Caves; Wavy Caves; Generate Ice Biome; Grass; Jungle;
Mud Caves To Grass; Full Desert; Floating Islands; Mushroom Patches; Marble;
Granite; Dirt To Mud; Silt; Shinies; Webs; Underworld; Corruption; Lakes; Dungeon;
Beaches; Gems; Create Ocean Caves; Shimmer; Pyramids; Living Trees; Altars;
Jungle Temple; Hives; Settle Liquids; Smooth World; Life Crystals; Buried Chests;
Surface Chests; Spider Caves; Gem Caves; Cave Walls; Pots; Spreading Grass; Traps;
Spawn Point; Planting Trees; Vines; Flowers; Settle Liquids Again; Cactus, Palm
Trees, & Coral; Tile Cleanup; Stalac; Remove Broken Traps; Final Cleanup;
Waterfalls; Temple; Floating Island Houses; Hellforge.

## Approximated (39)

Slush; Mountain Caves; Gravitating Sand; Clean Up Dirt; Dirt Rock Wall Runner;
Wood Tree Walls; Wet Jungle; Jungle Chests; Remove Water From Sand; Oasis; Shell
Piles; Ice; Wall Variety; Statues; Jungle Chests Placement; Water Chests; Moss;
Jungle Trees; Quick Cleanup; Surface Ore and Stone; Place Fallen Log; Piles;
Grass Wall; Sunflowers; Herbs; Dye Plants; Webs And Honey; Weeds; Glowing
Mushrooms and Jungle Plants; Jungle Plants; Mushrooms; Gems In Ice Biome; Random
Gems; Moss Grass; Muds Walls In Jungle; Larva; Lihzahrd Altars; Micro Biomes;
Water Plants.

## Documented (1)

Guide.

## Compatibility boundary

- Project text seeds are reproducible inside TerraExplorer, not Terraria-compatible.
- Small dimensions match the familiar `4200 x 1200`, but output is not `.wld`.
- Difficulty is recorded for experiment metadata; it does not yet reproduce all
  difficulty- or secret-seed-specific generation branches.
- Hardmode V is an optional post-generation event and not counted among the 107
  vanilla creation passes.
- Simulation IDs in `tiles.py` are TerraExplorer IDs, never claimed Terraria IDs.

## Recent modeled improvements

- Dungeon generation now builds a visible surface keep before descending into
  clipped rooms and corridors.
- Floating islands can carry sky-brick houses, reservoirs, and falling water.
- Jungle Temples use a sealed multi-level brick maze with corridors, traps, and
  an altar chamber.
- Pyramids sit mostly below the desert surface and include a shaft and treasure
  chamber.
- The Aether has a stone and gem shell around an actual Shimmer liquid pool.
- Underworld cities form obsidian-brick ruins, bridges, rooms, and Hellforge
  placements.
- Corruption, Crimson, and Hallow can advance into adjacent natural materials
  without wrapping across map edges.

## References

- [Vanilla World Generation Steps](https://github.com/tModLoader/tModLoader/wiki/Vanilla-World-Generation-Steps)
- [WorldGenerator reference](https://docs.tmodloader.net/docs/stable/class_world_generator.html)
- [WorldGen reference](https://docs.tmodloader.net/docs/stable/class_world_gen.html)
