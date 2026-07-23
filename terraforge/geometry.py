"""Small, allocation-conscious geometry primitives for generation passes."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import numpy.typing as npt


def fast_isin(
    elements: npt.NDArray[np.generic],
    test_elements: Iterable[int],
) -> npt.NDArray[np.bool_]:
    """Fast replacement for np.isin on small tile ID ranges."""
    test_tuple = tuple(test_elements)
    if not test_tuple:
        return np.zeros(elements.shape, dtype=bool)
    if len(test_tuple) <= 3:
        mask = (elements == test_tuple[0])
        for val in test_tuple[1:]:
            mask |= (elements == val)
        return mask
    lut = np.zeros(256, dtype=bool)
    lut[list(test_tuple)] = True
    return lut[elements]


def smooth_noise_1d(
    width: int,
    rng: np.random.Generator,
    amplitude: float,
    knot_spacing: int,
) -> npt.NDArray[np.float64]:
    knot_count = max(4, width // knot_spacing + 3)
    knot_x = np.linspace(0, width - 1, knot_count)
    knot_y = rng.normal(0.0, amplitude, knot_count)
    values = np.interp(np.arange(width), knot_x, knot_y)
    radius = max(2, knot_spacing // 5)
    kernel = np.ones(radius * 2 + 1, dtype=np.float64)
    kernel /= kernel.sum()
    return np.convolve(values, kernel, mode="same")


def stamp_ellipse(
    array: npt.NDArray[np.generic],
    center_x: int,
    center_y: int,
    radius_x: int,
    radius_y: int,
    value: int,
    replace: Iterable[int] | None = None,
) -> int:
    """Stamp a clipped ellipse and return the number of selected cells."""

    height, width = array.shape
    radius_x = max(1, int(radius_x))
    radius_y = max(1, int(radius_y))
    x0, x1 = max(0, center_x - radius_x), min(width, center_x + radius_x + 1)
    y0, y1 = max(0, center_y - radius_y), min(height, center_y + radius_y + 1)
    if x0 >= x1 or y0 >= y1:
        return 0
    yy = np.arange(y0, y1)[:, None]
    xx = np.arange(x0, x1)[None, :]
    mask = ((xx - center_x) / radius_x) ** 2 + ((yy - center_y) / radius_y) ** 2 <= 1.0
    view = array[y0:y1, x0:x1]
    if replace is not None:
        mask &= fast_isin(view, replace)
    view[mask] = value
    return int(mask.sum())


def stamp_walk(
    array: npt.NDArray[np.generic],
    rng: np.random.Generator,
    start_x: int,
    start_y: int,
    steps: int,
    radius: tuple[int, int],
    value: int,
    drift: tuple[float, float] = (0.0, 0.0),
    replace: Iterable[int] | None = None,
) -> None:
    """Stamp a tapered, directed random walk."""

    height, width = array.shape
    x, y = float(start_x), float(start_y)
    velocity_x, velocity_y = drift
    start_radius, end_radius = radius
    for step in range(max(1, steps)):
        t = step / max(1, steps - 1)
        current = max(1, round(start_radius * (1.0 - t) + end_radius * t))
        stamp_ellipse(
            array, round(x), round(y), current, max(1, round(current * 0.8)), value, replace
        )
        velocity_x = float(np.clip(velocity_x + rng.uniform(-0.35, 0.35), -1.8, 1.8))
        velocity_y = float(np.clip(velocity_y + rng.uniform(-0.25, 0.35), -1.3, 2.0))
        x = float(np.clip(x + velocity_x, 2, width - 3))
        y = float(np.clip(y + velocity_y, 2, height - 3))


def surface_candidates(
    tiles: npt.NDArray[np.uint8],
    tile_ids: Iterable[int],
) -> npt.NDArray[np.bool_]:
    """Solid cells with air immediately above."""

    candidates = np.zeros_like(tiles, dtype=bool)
    candidates[1:] = fast_isin(tiles[1:], tile_ids) & (tiles[:-1] == 0)
    return candidates
