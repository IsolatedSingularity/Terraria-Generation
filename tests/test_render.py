import json

import numpy as np
from PIL import Image

from terraforge.config import WorldConfig
from terraforge.pipeline import generate_world
from terraforge.render import (
    add_title_bar,
    render_world,
    save_generation_gif,
    save_npz,
    save_png,
    summary_json,
)


def test_renderer_and_png_export(tmp_path) -> None:
    world = generate_world(WorldConfig(seed="render"))
    image = render_world(world, scale=2, biome_overlay=True, layer_lines=True, markers=True)
    output = save_png(world, tmp_path / "world.png", scale=2, markers=True)

    assert image.mode == "RGB"
    assert image.size == (world.shape[1] * 2, world.shape[0] * 2)
    with Image.open(output) as saved:
        assert saved.size == image.size


def test_numpy_and_json_exports_are_machine_readable(tmp_path) -> None:
    world = generate_world(WorldConfig(seed="numpy"))
    output = save_npz(world, tmp_path / "world.npz")

    with np.load(output) as archive:
        assert set(archive.files) == {
            "tiles",
            "walls",
            "liquid_amount",
            "liquid_kind",
            "biomes",
            "surface",
            "metadata",
        }
        assert np.array_equal(archive["tiles"], world.tiles)

    summary = json.loads(summary_json(world))
    assert summary["size"] == {"width": 240, "height": 140}
    assert len(summary["passes"]) == 107


def test_generation_gif_contains_real_milestones(tmp_path) -> None:
    output = save_generation_gif(WorldConfig(seed="animation"), tmp_path / "generation.gif")

    with Image.open(output) as animation:
        assert animation.n_frames == 17
        animation.seek(0)
        first = np.asarray(animation.convert("RGB"))
        animation.seek(animation.n_frames - 1)
        last = np.asarray(animation.convert("RGB"))
        assert not np.array_equal(first, last)


def test_hardmode_gif_includes_post_generation_transformation(tmp_path) -> None:
    output = save_generation_gif(
        WorldConfig(seed="hardmode-animation", hardmode=True),
        tmp_path / "hardmode.gif",
    )

    with Image.open(output) as animation:
        assert animation.n_frames == 18


def test_add_title_bar_empty_image() -> None:
    empty_image = Image.new("RGB", (0, 0))

    with_title = add_title_bar(empty_image, "Title")
    assert with_title.size == (0, 38)
    assert with_title.mode == "RGB"

    with_subtitle = add_title_bar(empty_image, "Title", "Subtitle")
    assert with_subtitle.size == (0, 56)
    assert with_subtitle.mode == "RGB"
