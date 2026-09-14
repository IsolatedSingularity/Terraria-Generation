"""Import the recorded oracle and export source-derived structure crops/statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

from terraexplorer.fidelity.render import font, render_crop
from terraexplorer.fidelity.structures import extract_structures, statistics
from terraexplorer.fidelity.world import import_world

ORACLE_HASHES = {
    "cae384ddc3466ca80f398912797b3dbbe241972653067c306b96142719551a8e",
    "caea7534eb437b8e9b6e11838e93e2ad2c7ee6ae491e1669f2480ab24f8efcdc",
}
CELL_HASH = "97d590a10019a682b04bdbf076c70b18f8f557fc9c6c0ce3964158994eac9b4c"


def export_references(world_path: Path, output: Path):
    world = import_world(world_path)
    if world.report["sha256"] not in ORACLE_HASHES or world.semantic_sha256 != CELL_HASH:
        raise ValueError(
            "This milestone exporter requires one of the recorded genuine oracle files"
        )
    output.mkdir(parents=True, exist_ok=False)
    structures, rejected = extract_structures(world)
    stats = [statistics(world, structure) for structure in structures]
    renders = {}
    images = {}

    def save(name, bbox, scale, title):
        image, provenance = render_crop(world, bbox, scale=scale, title=title)
        path = output / f"{name}.png"
        image.save(path)
        provenance["png_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        renders[path.name] = provenance
        images[name] = image

    def padded(box, margin=12):
        left, top, right, bottom = box
        height, width = world.cells.shape
        return (
            max(0, left - margin),
            max(0, top - margin),
            min(width, right + margin),
            min(height, bottom + margin),
        )

    for structure, stat in zip(structures, stats, strict=True):
        title = {
            "dungeon": "Dungeon - complete connected extent",
            "temple": "Jungle Temple - complete connected extent",
            "living_tree_group": "Living trees - shared canopy and passages",
        }[structure.kind]
        save(structure.name, padded(structure.bbox), 2 if structure.kind == "dungeon" else 4, title)
        if structure.kind == "dungeon":
            x, y = structure.evidence["entrance"]
            # A display window only; authoritative extracted bounds remain above.
            box = (
                max(0, x - 96),
                max(0, y - 112),
                min(world.cells.shape[1], x + 96),
                min(world.cells.shape[0], y + 80),
            )
            save("dungeon_entrance", box, 4, "Dungeon - metadata entrance detail")
        if structure.kind == "living_tree_group":
            for i, stem in enumerate(stat["stem_analysis"]["stems"], 1):
                save(
                    f"living_tree_stem_{i}",
                    padded(stem["associated_support_bbox"], 8),
                    4,
                    f"Living tree {i} - associated part of shared group",
                )

    # Place unscaled reference images on a sheet; no interpolation changes geometry.
    dungeon = images["dungeon_1"]
    temple = images["temple_1"]
    tree = images["living_tree_group_1"]
    sheet = Image.new(
        "RGB",
        (
            dungeon.width + max(temple.width, tree.width) + 48,
            max(dungeon.height, temple.height + tree.height + 16) + 96,
        ),
        (11, 18, 32),
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 12), "VANILLA GEOMETRY / TERRAEXPLORER ART", font=font(26, True), fill="#edf4ff")
    draw.text(
        (20, 48),
        "Imported Terraria 1.4.5.7 - seed 314159265 - genuine saved-world reference",
        font=font(18),
        fill="#9fb4cf",
    )
    sheet.paste(dungeon, (16, 80))
    sheet.paste(temple, (dungeon.width + 32, 80))
    sheet.paste(tree, (dungeon.width + 32, 96 + temple.height))
    sheet.save(output / "comparison_sheet.png")
    report = {
        "schema_version": 1,
        "source_wld": str(world_path),
        "source_sha256": world.report["sha256"],
        "canonical_cells_sha256": world.semantic_sha256,
        "cell_count": world.cells.size,
        "metadata": dict(world.metadata),
        "frame_importance_count": world.report["importance_count"],
        "definitions": {
            "bbox": "Half-open [left,top,right,bottom], native cells; tight support extent",
            "support": "Connected cells with active target-family tiles OR target-family walls",
            "counts": "Support includes objects on target walls; crop counts include context",
            "empty_wall_components_4": (
                "4-connected inactive cells with nonzero wall on support; not rooms or player paths"
            ),
            "frame_bearing": "Active cell and frame-importance bit true, including saved -1 frames",
            "living_tree_association": ("8-connected wood/leaf/wall group; heuristic stem split"),
            "render_padding": "12 cells around extent, 8 around stems; display context only",
        },
        "structures": stats,
        "rejected_material_components": rejected,
        "renders": renders,
        "comparison_sheet_sha256": hashlib.sha256(
            (output / "comparison_sheet.png").read_bytes()
        ).hexdigest(),
    }
    (output / "structure_statistics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("world", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = export_references(args.world, args.output)
    print(
        json.dumps(
            {
                "cells": result["cell_count"],
                "structures": [
                    {
                        key: s[key]
                        for key in (
                            "name",
                            "bbox",
                            "support_cells",
                            "occupied_support_cells",
                            "wall_support_cells",
                            "empty_wall_components_4",
                            "frame_bearing_support_cells",
                        )
                    }
                    for s in result["structures"]
                ],
                "renders": list(result["renders"]),
            },
            indent=2,
        )
    )
