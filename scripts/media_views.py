"""Presentation transforms of generated Small worlds; no resampled simulation."""

from __future__ import annotations

from dataclasses import replace

from PIL import Image

from terraexplorer.model import LayerDepths
from terraexplorer.render import render_world


def crop_world(world, left, top, width, height):
    left = max(0, min(world.shape[1] - width, int(left)))
    top = max(0, min(world.shape[0] - height, int(top)))
    region = (slice(top, top + height), slice(left, left + width))
    metadata = dict(world.metadata)
    for name in ("spawn_x", "snow_x", "jungle_x", "desert_x", "evil_x", "dungeon_x"):
        if name in metadata:
            metadata[name] = int(metadata[name]) - left
    metadata["source_crop"] = [left, top, left + width, top + height]
    return replace(
        world,
        **{
            name: getattr(world, name)[region].copy()
            for name in ("tiles", "walls", "liquid_amount", "liquid_kind", "biomes")
        },
        surface=world.surface[left : left + width].copy() - top,
        layers=LayerDepths(
            world.layers.world_surface - top,
            world.layers.rock_layer - top,
            world.layers.underworld - top,
        ),
        structures=[
            replace(s, x=s.x - left, y=s.y - top)
            for s in world.structures
            if left <= s.x
            and s.x + s.width <= left + width
            and top <= s.y
            and s.y + s.height <= top + height
        ],
        metadata=metadata,
        pass_results=list(world.pass_results),
    )


def overview(world, size=(960, 560)):
    # Preserve the README canvas. The x/y transform is presentation only.
    return render_world(world, scale=1, markers=False).resize(size, Image.Resampling.NEAREST)


def detail(world, left, top, width=48, height=72, scale=6):
    return render_world(
        crop_world(world, left, top, width, height),
        scale=scale,
        markers=False,
        biome_overlay=True,
        material_texture=False,
    )
