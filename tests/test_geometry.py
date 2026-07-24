import numpy as np

from terraforge.geometry import stamp_ellipse, stamp_walk, surface_candidates


def test_stamp_ellipse_clips_at_edges_and_respects_replace_filter() -> None:
    array = np.zeros((10, 10), dtype=np.uint8)
    array[0, 0] = 2

    changed = stamp_ellipse(array, 0, 0, 4, 4, 7, replace=(0,))

    assert changed > 0
    assert array[0, 0] == 2
    assert np.count_nonzero(array == 7) == changed


def test_stamp_walk_is_reproducible() -> None:
    first = np.zeros((30, 30), dtype=np.uint8)
    second = np.zeros_like(first)

    stamp_walk(first, np.random.default_rng(7), 15, 5, 20, (3, 1), 4, drift=(0, 0.8))
    stamp_walk(second, np.random.default_rng(7), 15, 5, 20, (3, 1), 4, drift=(0, 0.8))

    assert np.array_equal(first, second)


def test_surface_candidates_identifies_exposed_tiles() -> None:
    tiles = np.array(
        [
            [0, 1, 0, 2],
            [1, 0, 1, 1],
            [2, 1, 2, 0],
        ],
        dtype=np.uint8,
    )

    candidates = surface_candidates(tiles, [1, 2])

    expected = np.array(
        [
            [False, False, False, False],
            [True, False, True, False],
            [False, True, False, False],
        ],
        dtype=bool,
    )

    assert np.array_equal(candidates, expected)


def test_surface_candidates_empty_tile_ids() -> None:
    tiles = np.array(
        [
            [0, 0],
            [1, 2],
        ],
        dtype=np.uint8,
    )

    candidates = surface_candidates(tiles, [])

    expected = np.zeros_like(tiles, dtype=bool)

    assert np.array_equal(candidates, expected)
