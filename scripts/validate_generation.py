"""Compare independent Small worlds with local format-325 oracle references.

No oracle cells are inputs to generation. Occupancy means material presence,
not collision solidity. Cave connectivity uses a declared four-tile sampling
grid below the global surface and above the Underworld, excluding open sky.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from terraexplorer.config import WorldConfig, WorldScale
from terraexplorer.fidelity.structures import components, extract_structures
from terraexplorer.fidelity.world import import_world
from terraexplorer.pipeline import generate_world
from terraexplorer.render import render_world
from terraexplorer.tiles import Tile, Wall


def measure(present, surface, layers, liquid, structures):
    layers = tuple(map(int, layers))
    height, width = present.shape
    interior = surface[350:-350]
    bands = [float(present[y : y + 100, 350:-350].mean()) for y in range(0, height, 100)]
    caves = ~present[layers[0] : layers[2] : 4, 350:-350:4]
    sizes = sorted((len(c) for c in components(caves)), reverse=True)
    return {
        "dimensions": [width, height],
        "layers": list(map(int, layers)),
        "surface_quantiles": np.quantile(interior, [0.05, 0.5, 0.95]).tolist(),
        "surface_mean_step": float(np.abs(np.diff(interior.astype(float))).mean()),
        "occupied_fraction_per_100_rows": bands,
        "cave_sample_stride": 4,
        "largest_cave_fraction": sizes[0] / max(1, sum(sizes)) if sizes else 0,
        "cave_component_count": len(sizes),
        "liquid_fraction": float((liquid > 0).mean()),
        "structures": structures,
    }


def generated_metrics(world):
    spawn_x, spawn_y = int(world.metadata["spawn_x"]), int(world.metadata["spawn_y"])
    assert 0 <= spawn_x < world.shape[1] and 0 <= spawn_y < world.shape[0]
    assert world.tiles[spawn_y, spawn_x] == Tile.AIR
    assert not np.any(world.liquid_amount[world.tiles != Tile.AIR])
    assert np.all(world.liquid_kind[world.liquid_amount == 0] == 0)
    invalid = [
        s
        for s in world.structures
        if not (
            0 <= s.x < s.x + s.width <= world.shape[1]
            and 0 <= s.y < s.y + s.height <= world.shape[0]
        )
    ]
    assert not invalid, invalid
    natural = np.isin(
        world.tiles,
        [
            Tile.DIRT,
            Tile.STONE,
            Tile.GRASS,
            Tile.SAND,
            Tile.MUD,
            Tile.SNOW,
            Tile.ICE,
            Tile.JUNGLE_GRASS,
            Tile.EBONSTONE,
            Tile.CRIMSTONE,
            Tile.CORRUPT_GRASS,
            Tile.CRIMSON_GRASS,
        ],
    )
    natural[:120] = False
    surface = np.argmax(natural, axis=0)
    structures = []
    for name, tile, wall in (
        ("dungeon", Tile.DUNGEON_BRICK, Wall.DUNGEON),
        ("temple", Tile.LIHZAHRD_BRICK, Wall.LIHZAHRD),
        ("living_tree_group", Tile.LIVING_WOOD, None),
    ):
        mask = world.tiles == tile
        if wall is not None:
            mask |= world.walls == wall
        else:
            mask |= world.tiles == Tile.LEAF
        for cells in components(mask, diagonal=name == "living_tree_group"):
            if len(cells) < 100:
                continue
            yy, xx = np.divmod(cells, world.shape[1])
            structures.append(
                {
                    "kind": name,
                    "bbox": [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1],
                    "support_cells": len(cells),
                }
            )
    return measure(
        world.tiles != Tile.AIR,
        surface,
        (world.layers.world_surface, world.layers.rock_layer, world.layers.underworld),
        world.liquid_amount,
        structures,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 314159265])
    parser.add_argument("--oracle", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    for seed in args.seeds:
        world = generate_world(WorldConfig(seed=seed, scale=WorldScale.SMALL))
        result = {
            "generated": generated_metrics(world),
            "generation_seconds": world.metadata["generation_seconds"],
            "integrity": "Clear spawn; valid marker bounds; dry occupied cells; coherent liquids",
        }
        render_world(world, scale=1, markers=False).resize((1050, 300)).save(
            args.output / f"generated-{seed}.png"
        )
        if args.oracle:
            path = Path(
                f"audit/living-trees-replay-20260914/seed-{seed}-control/bootstrap-small.wld"
            )
            reference = import_world(path)
            cells = reference.cells
            natural = (cells["active"] != 0) & np.isin(
                cells["tile"], [0, 1, 2, 23, 25, 53, 59, 60, 147, 161, 199, 203, 396, 397]
            )
            natural[:120] = False
            structures, _ = extract_structures(reference)
            result["oracle"] = measure(
                cells["active"] != 0,
                np.argmax(natural, axis=0),
                (reference.metadata["world_surface"], reference.metadata["rock_layer"], 1000),
                cells["liquid_amount"],
                [
                    {"kind": s.kind, "bbox": list(s.bbox), "support_cells": len(s.indices)}
                    for s in structures
                    if s.kind in ("dungeon", "temple", "living_tree_group")
                ],
            )
            result["oracle_sha256"] = reference.report["sha256"]
        results[str(seed)] = result
        print(seed, round(result["generation_seconds"], 3), flush=True)
        (args.output / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
