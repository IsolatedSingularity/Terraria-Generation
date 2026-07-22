# Fidelity inventory

TerraForge is an educational approximation. It follows the public vanilla
1.4.4.9 pass names and ordering; it does not copy Terraria's source, internal
tile IDs, RNG stream, or assets.

The authoritative machine-readable inventory is `terraforge/passes.py`. Run
`terraforge passes` or `terraforge passes --json` to inspect it.

## Status definitions

- **Modeled**: a distinct TerraForge operation mutates the world or its metadata.
- **Approximated**: a simpler/shared operation represents the pass's broad role.
- **Documented**: preserved in ordering and telemetry without a grid mutation.

## Modeled (63)

Reset; Terrain; Dunes; Ocean Sand; Sand Patches; Tunnels; Mount Caves; Dirt Wall
Backgrounds; Rocks In Dirt; Dirt In Rocks; Clay; Small Holes; Dirt Layer Caves;
Rock Layer Caves; Surface Caves; Wavy Caves; Generate Ice Biome; Grass; Jungle;
Mud Caves To Grass; Full Desert; Floating Islands; Mushroom Patches; Marble;
Granite; Dirt To Mud; Silt; Shinies; Webs; Underworld; Corruption; Lakes; Dungeon;
Beaches; Gems; Create Ocean Caves; Shimmer; Pyramids; Living Trees; Altars;
Jungle Temple; Hives; Settle Liquids; Smooth World; Life Crystals; Buried Chests;
Surface Chests; Spider Caves; Gem Caves; Cave Walls; Pots; Spreading Grass; Traps;
Spawn Point; Planting Trees; Vines; Flowers; Settle Liquids Again; Cactus, Palm
Trees, & Coral; Tile Cleanup; Stalac; Remove Broken Traps; Final Cleanup.

## Approximated (43)

Slush; Mountain Caves; Gravitating Sand; Clean Up Dirt; Dirt Rock Wall Runner;
Wood Tree Walls; Wet Jungle; Jungle Chests; Remove Water From Sand; Oasis; Shell
Piles; Waterfalls; Ice; Wall Variety; Statues; Jungle Chests Placement; Water
Chests; Moss; Temple; Jungle Trees; Floating Island Houses; Quick Cleanup;
Hellforge; Surface Ore and Stone; Place Fallen Log; Piles; Grass Wall; Sunflowers;
Herbs; Dye Plants; Webs And Honey; Weeds; Glowing Mushrooms and Jungle Plants;
Jungle Plants; Mushrooms; Gems In Ice Biome; Random Gems; Moss Grass; Muds Walls
In Jungle; Larva; Lihzahrd Altars; Micro Biomes; Water Plants.

## Documented (1)

Guide.

## Compatibility boundary

- Project text seeds are reproducible inside TerraForge, not Terraria-compatible.
- Small dimensions match the familiar `4200 x 1200`, but output is not `.wld`.
- Difficulty is recorded for experiment metadata; it does not yet reproduce all
  difficulty- or secret-seed-specific generation branches.
- Hardmode V is an optional post-generation event and not counted among the 107
  vanilla creation passes.
- Simulation IDs in `tiles.py` are TerraForge IDs, never claimed Terraria IDs.

## References

- [Vanilla World Generation Steps](https://github.com/tModLoader/tModLoader/wiki/Vanilla-World-Generation-Steps)
- [WorldGenerator reference](https://docs.tmodloader.net/docs/stable/class_world_generator.html)
- [WorldGen reference](https://docs.tmodloader.net/docs/stable/class_world_gen.html)
