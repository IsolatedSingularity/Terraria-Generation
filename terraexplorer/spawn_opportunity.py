"""Ground-candidate opportunity, not a spawn rate or NPC selection simulator.

Source: NPC.Spawner.FindSpawnTile 997-1038, PostCheckChosenSpawnTile 1050-1076,
HasTileSpawnSpace/CanSpawnInTile 5447-5486 in the pinned 1.4.5.7 NPC.cs.
The hypothetical player is off screen with the candidate floor in range. No
town population, safe player-placed walls, buffs, events or special sky/aquatic
dispatch is assumed. Difficulty/time do not weight this geometric quantity.
"""

from __future__ import annotations

import numpy as np

from terraexplorer.tiles import Liquid, Tile

# Original simulation registry has no native tileSolid table. These modeled
# decorations are passable; platforms are excluded from ordinary solid ground.
PASSABLE = (
    Tile.AIR,
    Tile.COBWEB,
    Tile.ALTAR,
    Tile.CHEST,
    Tile.LIFE_CRYSTAL,
    Tile.POT,
    Tile.GEM,
    Tile.CACTUS,
    Tile.TREE,
    Tile.PLATFORM,
    Tile.STALACTITE,
    Tile.FLOWER,
    Tile.VINE,
    Tile.HELLFORGE,
    Tile.GEM_TREE,
    Tile.MINECART_TRACK,
    Tile.SUNFLOWER,
)


def ground_candidate_mass(world):
    """Number (0..103) of eligible starting cells above each supported floor.

    A source 1920x1200 reference screen gives a 168x104 spawn rectangle. We
    cap the vertical search to 103 cells, require the source 2x3 clear space,
    and reject lava and two-cell-deep honey/shimmer. Values live on floor cells.
    Player-safe-screen exclusion is conditional, not a global spawn-point mask.
    """
    solid = ~np.isin(world.tiles, PASSABLE)
    lava = (world.liquid_kind == Liquid.LAVA) & (world.liquid_amount > 0)
    clear = ~solid & ~lava
    h, w = world.shape
    footprint = np.zeros(world.shape, dtype=bool)
    footprint[3:, 1:] = (
        clear[2:-1, 1:]
        & clear[1:-2, 1:]
        & clear[:-3, 1:]
        & clear[2:-1, :-1]
        & clear[1:-2, :-1]
        & clear[:-3, :-1]
    )
    forbidden = np.isin(world.liquid_kind, (Liquid.HONEY, Liquid.SHIMMER)) & (
        world.liquid_amount > 0
    )
    footprint[2:] &= ~(forbidden[1:-1] & forbidden[:-2])
    mass = np.zeros(world.shape, dtype=np.uint8)
    run = np.zeros(w, dtype=np.uint16)
    sky_limit = round(world.layers.world_surface * 0.35)
    for y in range(h):
        mass[y] = np.where(solid[y] & footprint[y], np.minimum(run, 103), 0)
        run = np.where(clear[y] & (y >= sky_limit), np.minimum(run + 1, 103), 0)
    return mass
