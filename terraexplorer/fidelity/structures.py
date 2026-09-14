"""Source-backed material/connectivity measurements on immutable imported cells.

Bounds describe surviving saved structure material, not generation write bounds.
All boxes are half-open [left, top, right, bottom] in native tile coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from terraexplorer.fidelity.world import CanonicalWorld

DUNGEON_TILES = (41, 43, 44, 481, 482, 483)
DUNGEON_WALLS = (7, 8, 9, 94, 95, 96, 97, 98, 99)
TEMPLE_TILES = (226,)
TEMPLE_WALLS = (87,)
TREE_TILES = (191, 192)
TREE_WALLS = (78, 244)


def components(mask: np.ndarray, diagonal: bool = False) -> list[np.ndarray]:
    """Exact 4/8-neighbor components as sorted, row-major flat indices."""
    height, width = mask.shape
    indices = np.flatnonzero(mask)
    remaining = set(indices.tolist())
    result = []
    for start in indices:
        start = int(start)
        if start not in remaining:
            continue
        remaining.remove(start)
        stack = [start]
        found = []
        while stack:
            index = stack.pop()
            found.append(index)
            y, x = divmod(index, width)
            neighbors = []
            if x:
                neighbors.append(index - 1)
            if x + 1 < width:
                neighbors.append(index + 1)
            if y:
                neighbors.append(index - width)
                if diagonal:
                    if x:
                        neighbors.append(index - width - 1)
                    if x + 1 < width:
                        neighbors.append(index - width + 1)
            if y + 1 < height:
                neighbors.append(index + width)
                if diagonal:
                    if x:
                        neighbors.append(index + width - 1)
                    if x + 1 < width:
                        neighbors.append(index + width + 1)
            for neighbor in neighbors:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        result.append(np.asarray(sorted(found), dtype=np.int64))
    return result


def bounds(indices, width):
    y, x = np.divmod(indices, width)
    return (int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1)


def material_mask(cells, tiles, walls):
    return ((cells["active"] != 0) & np.isin(cells["tile"], tiles)) | np.isin(cells["wall"], walls)


@dataclass(frozen=True)
class Structure:
    name: str
    kind: str
    bbox: tuple[int, int, int, int]
    indices: np.ndarray
    evidence: dict


def histogram(values):
    ids, counts = np.unique(values, return_counts=True)
    return {str(int(key)): int(count) for key, count in zip(ids, counts, strict=True)}


def extract_structures(world: CanonicalWorld):
    cells = world.cells
    height, width = cells.shape
    flat = cells.reshape(-1)
    results = []
    rejected = []
    definitions = (
        ("dungeon", DUNGEON_TILES, DUNGEON_WALLS, False),
        ("temple", TEMPLE_TILES, TEMPLE_WALLS, False),
        ("living_tree_group", TREE_TILES, TREE_WALLS, True),
    )
    for kind, tiles, walls, diagonal in definitions:
        for indices in components(material_mask(cells, tiles, walls), diagonal):
            sample = flat[indices]
            present = set(sample["tile"][sample["active"] != 0].tolist())
            wall_ids = set(sample["wall"].tolist())
            bbox = bounds(indices, width)
            evidence = {
                "material_tiles": tiles,
                "material_walls": walls,
                "association_neighbors": 8 if diagonal else 4,
            }
            if kind == "dungeon":
                x, y = world.metadata["dungeon_x"], world.metadata["dungeon_y"]
                ys, xs = np.divmod(indices, width)
                distance = int((np.abs(xs - x) + np.abs(ys - y)).min())
                if not (
                    bbox[0] <= x < bbox[2]
                    and bbox[1] <= y < bbox[3]
                    and present.intersection(tiles)
                    and wall_ids.intersection(walls)
                ):
                    rejected.append(
                        {
                            "kind": kind,
                            "bbox": bbox,
                            "cells": len(indices),
                            "reason": "Not a brick/wall component enclosing the metadata anchor",
                        }
                    )
                    continue
                evidence.update(entrance=[x, y], entrance_distance_to_support=distance)
            elif kind == "temple":
                if 226 not in present or 87 not in wall_ids or 237 not in present:
                    rejected.append(
                        {
                            "kind": kind,
                            "bbox": bbox,
                            "cells": len(indices),
                            "reason": "No connected brick/wall/altar combination",
                        }
                    )
                    continue
                altar = indices[(sample["active"] != 0) & (sample["tile"] == 237)]
                evidence["altar_bbox"] = bounds(altar, width)
                evidence["altar_cells"] = len(altar)
            elif not ({191, 192} <= present and 244 in wall_ids):
                rejected.append(
                    {
                        "kind": kind,
                        "bbox": bbox,
                        "cells": len(indices),
                        "reason": "No connected living-wood + leaf + unsafe-wall combination",
                    }
                )
                continue
            number = sum(s.kind == kind for s in results) + 1
            results.append(Structure(f"{kind}_{number}", kind, bbox, indices, evidence))
    return results, rejected


def tree_stems(world: CanonicalWorld, structure: Structure):
    """Measure candidate trunks, not WorldGen call origins.

    Consecutive columns with a wood run >= ceil(canopy height/4), minimum 8,
    form a stem band. Associate shared material by nearest stem center X, ties
    left. The full connected group remains the authoritative extraction.
    """
    cells = world.cells
    left, top, right, bottom = structure.bbox
    crop = cells[top:bottom, left:right]
    leaf = (crop["active"] != 0) & (crop["tile"] == 192)
    ly, lx = np.nonzero(leaf)
    canopy_bottom, canopy_top = int(ly.max()) + 1, int(ly.min())
    threshold = max(8, (canopy_bottom - canopy_top + 3) // 4)
    wood = (crop["active"] != 0) & (crop["tile"] == 191)
    longest = np.zeros(crop.shape[1], dtype=int)
    current = longest.copy()
    for row in wood[canopy_top:canopy_bottom]:
        current = (current + 1) * row
        longest = np.maximum(longest, current)
    qualified = np.flatnonzero(longest >= threshold)
    bands = (
        np.split(qualified, np.flatnonzero(np.diff(qualified) > 1) + 1) if qualified.size else []
    )
    centers = [(int(band[0]) + int(band[-1])) / 2 + left for band in bands]
    _, x = np.divmod(structure.indices, cells.shape[1])
    association = (
        np.argmin(np.abs(x[:, None] - np.asarray(centers)[None, :]), axis=1) if centers else []
    )
    stems = []
    for index, (band, center) in enumerate(zip(bands, centers, strict=True)):
        indices = structure.indices[association == index]
        stems.append(
            {
                "center_x": center,
                "column_band": [int(band[0]) + left, int(band[-1]) + left + 1],
                "longest_wood_run": int(longest[band].max()),
                "associated_support_bbox": bounds(indices, cells.shape[1]),
                "associated_support_cells": len(indices),
            }
        )
    return {
        "canopy_bbox": [
            int(lx.min()) + left,
            canopy_top + top,
            int(lx.max()) + left + 1,
            canopy_bottom + top,
        ],
        "wood_run_threshold": threshold,
        "stems": stems,
        "ownership": "Nearest stem center X; measurement association, not generation history",
    }


def statistics(world: CanonicalWorld, structure: Structure):
    cells = world.cells
    height, width = cells.shape
    left, top, right, bottom = structure.bbox
    selected = cells.reshape(-1)[structure.indices]
    crop = cells[top:bottom, left:right]
    support = np.zeros(crop.shape, dtype=bool)
    ys, xs = np.divmod(structure.indices, width)
    support[ys - top, xs - left] = True
    active = selected["active"] != 0
    wall = selected["wall"] != 0
    # This is an empty, wall-backed passage measurement, not player pathfinding.
    empty_wall = support & (crop["active"] == 0) & (crop["wall"] != 0)
    cavities = components(empty_wall)
    result = {
        "name": structure.name,
        "kind": structure.kind,
        "bbox": list(structure.bbox),
        "width": right - left,
        "height": bottom - top,
        "support_cells": len(structure.indices),
        "occupied_support_cells": int(active.sum()),
        "wall_support_cells": int(wall.sum()),
        "tile_ids_on_support": histogram(selected["tile"][active]),
        "wall_ids_on_support": histogram(selected["wall"][wall]),
        "support_components_4": len(components(support)),
        "support_components_8": len(components(support, diagonal=True)),
        "empty_wall_components_4": len(cavities),
        "empty_wall_component_sizes": sorted([len(part) for part in cavities], reverse=True),
        "liquid_support_cells": int((selected["liquid_amount"] > 0).sum()),
        "liquid_kinds_on_support": histogram(
            selected["liquid_kind"][selected["liquid_amount"] > 0]
        ),
        "frame_bearing_support_cells": int(world.frame_saved.reshape(-1)[structure.indices].sum()),
        "crop_occupied_cells": int((crop["active"] != 0).sum()),
        "crop_tile_ids": histogram(crop["tile"][crop["active"] != 0]),
        "crop_wall_ids": histogram(crop["wall"][crop["wall"] != 0]),
        "evidence": structure.evidence,
    }
    if structure.kind == "living_tree_group":
        wood = support & (crop["active"] != 0) & (crop["tile"] == 191)
        result["wood_components_8"] = len(components(wood, diagonal=True))
        result["stem_analysis"] = tree_stems(world, structure)
    return result
