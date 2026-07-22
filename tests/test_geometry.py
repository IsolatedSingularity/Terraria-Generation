import numpy as np

from terraforge.geometry import stamp_ellipse, stamp_walk


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
