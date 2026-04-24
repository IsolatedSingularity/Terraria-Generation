"""Terraria corruption evolution rendered as native TINY worlds.

Four-panel evolution figure plus a spread GIF, all using the 240x140 TINY
world primitive. Each phase visibly differs because they share the same
seed (so geometry stays consistent) but advance the simulation state.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from Engine.algorithms import (
    AIR,
    CORRUPT_DIRT,
    CORRUPT_ICE,
    CRIMSON_DIRT,
    CRIMSON_ICE,
    CRIMSTONE,
    DIRT,
    EBONSTONE,
    GRASS,
    HALLOW_DIRT,
    HALLOW_ICE,
    ICE,
    MUD,
    PEARLSAND,
    PEARLSTONE,
    SAND,
    SNOW,
    STONE,
    tileRunner,
)
from Engine.constants import (
    INFECTION_GAP_TILES,
    INFECTION_SPREAD_RADIUS,
)
from Engine.theme import COLORS, applyTokyoNight, saveTinyGif
from Engine.worldgen import generateMiniWorld, renderMiniWorld

applyTokyoNight()


CORRUPTION_CONVERSIONS = {
    DIRT: CORRUPT_DIRT, STONE: EBONSTONE, ICE: CORRUPT_ICE, GRASS: CORRUPT_DIRT,
}
CRIMSON_CONVERSIONS = {
    DIRT: CRIMSON_DIRT, STONE: CRIMSTONE, ICE: CRIMSON_ICE, GRASS: CRIMSON_DIRT,
}
HALLOW_CONVERSIONS = {
    STONE: PEARLSTONE, DIRT: HALLOW_DIRT, SAND: PEARLSAND,
    ICE: HALLOW_ICE, GRASS: HALLOW_DIRT,
}

CORRUPTION_TILES = frozenset({CORRUPT_DIRT, EBONSTONE, CORRUPT_ICE})
CRIMSON_TILES = frozenset({CRIMSON_DIRT, CRIMSTONE, CRIMSON_ICE})
HALLOW_TILES = frozenset({PEARLSTONE, HALLOW_DIRT, PEARLSAND, HALLOW_ICE})
ALL_INFECTED = CORRUPTION_TILES | CRIMSON_TILES | HALLOW_TILES
CONVERTIBLE = frozenset({DIRT, STONE, ICE, SAND, GRASS, SNOW, MUD})


def _conversionsFor(evilType: str) -> dict[int, int]:
    if evilType == "corruption":
        return CORRUPTION_CONVERSIONS
    if evilType == "crimson":
        return CRIMSON_CONVERSIONS
    return HALLOW_CONVERSIONS


def carveVPattern(grid: np.ndarray, evilType: str, seed: int) -> None:
    """Carve the Hardmode V-pattern (evil + hallow diagonals) at TINY scale.

    Width is 240, so the V spans ~80 tiles each side. Strength and steps
    are absolute (not world-fraction scaled) so the pattern is unmistakable.
    """
    rng = np.random.default_rng(seed)
    height, width = grid.shape
    centerX = width // 2
    surfaceY = 28
    hellY = 125
    vertDrop = hellY - surfaceY
    horzSpread = 80

    evilTile = CORRUPT_DIRT if evilType == "corruption" else CRIMSON_DIRT
    hallowTile = PEARLSTONE
    if rng.random() < 0.5:
        evilTile, hallowTile = hallowTile, evilTile

    leftDx = -horzSpread / vertDrop
    rightDx = horzSpread / vertDrop

    numPasses = 30
    for i in range(numPasses):
        t = i / numPasses
        y = int(surfaceY + vertDrop * t)
        lx = int(np.clip(centerX + leftDx * vertDrop * t, 2, width - 3))
        rx = int(np.clip(centerX + rightDx * vertDrop * t, 2, width - 3))

        tileRunner(grid, lx, y,
                   strength=float(rng.uniform(3.5, 6.0)),
                   steps=int(rng.integers(8, 18)),
                   tileType=int(evilTile), overRide=False,
                   speedX=float(leftDx * 1.5), speedY=0.5,
                   seed=int(rng.integers(0, 1 << 30)))
        tileRunner(grid, rx, y,
                   strength=float(rng.uniform(3.5, 6.0)),
                   steps=int(rng.integers(8, 18)),
                   tileType=int(hallowTile), overRide=False,
                   speedX=float(rightDx * 1.5), speedY=0.5,
                   seed=int(rng.integers(0, 1 << 30)))


def _hasAirGap(grid: np.ndarray, sx: int, sy: int, tx: int, ty: int) -> bool:
    """Return True if INFECTION_GAP_TILES consecutive air tiles block the path."""
    dx = tx - sx
    dy = ty - sy
    steps = max(abs(dx), abs(dy))
    if steps <= 1:
        return False
    consecutive = 0
    for i in range(1, steps):
        t = i / steps
        cx = int(sx + dx * t)
        cy = int(sy + dy * t)
        if grid[cy, cx] == AIR:
            consecutive += 1
            if consecutive >= INFECTION_GAP_TILES:
                return True
        else:
            consecutive = 0
    return False


def spreadInfection(grid: np.ndarray, cycles: int, seed: int) -> None:
    """Apply ``cycles`` rounds of stochastic CA infection spread.

    Each round samples every infected tile and converts one neighbor within
    ``INFECTION_SPREAD_RADIUS`` if the target is convertible and no air gap
    blocks the path.
    """
    rng = np.random.default_rng(seed)
    height, width = grid.shape

    for _ in range(cycles):
        infectedYs, infectedXs = np.where(np.isin(grid, list(ALL_INFECTED)))
        if infectedYs.size == 0:
            return
        # Sample at most 800 spread events per cycle to bound runtime.
        sampleCount = min(800, infectedYs.size)
        idxs = rng.integers(0, infectedYs.size, size=sampleCount)
        for k in idxs:
            sy, sx = int(infectedYs[k]), int(infectedXs[k])
            srcTile = grid[sy, sx]
            if srcTile in CORRUPTION_TILES:
                conv = CORRUPTION_CONVERSIONS
            elif srcTile in CRIMSON_TILES:
                conv = CRIMSON_CONVERSIONS
            else:
                conv = HALLOW_CONVERSIONS

            dy = int(rng.integers(-INFECTION_SPREAD_RADIUS,
                                  INFECTION_SPREAD_RADIUS + 1))
            dx = int(rng.integers(-INFECTION_SPREAD_RADIUS,
                                  INFECTION_SPREAD_RADIUS + 1))
            if dy == 0 and dx == 0:
                continue
            ny, nx = sy + dy, sx + dx
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            target = grid[ny, nx]
            if target in CONVERTIBLE and target in conv:
                if not _hasAirGap(grid, sx, sy, nx, ny):
                    grid[ny, nx] = conv[target]


def buildEvolutionSnapshots(evilType: str, seed: int) -> list[np.ndarray]:
    """Return four progressive snapshots: pre-HM, V-pattern, early, late spread."""
    base = generateMiniWorld(seed=seed, evilType=evilType, altarsSmashed=0)
    snap1 = base.grid.copy()  # Pre-Hardmode

    snap2 = snap1.copy()
    carveVPattern(snap2, evilType=evilType, seed=seed + 1)

    snap3 = snap2.copy()
    spreadInfection(snap3, cycles=5, seed=seed + 2)

    snap4 = snap3.copy()
    spreadInfection(snap4, cycles=30, seed=seed + 3)

    return [snap1, snap2, snap3, snap4]


def createEvolutionFigure(savePath: str | None = None,
                          evilType: str = "corruption",
                          suptitle: str = "Corruption Evolution") -> None:
    """Render the 4-phase TINY-world evolution figure."""
    if savePath is None:
        savePath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "Plots", "Advanced",
            f"{evilType}_evolution.png",
        )

    print(f"Building {evilType} evolution snapshots...")
    snaps = buildEvolutionSnapshots(evilType, seed=20260423)
    layers = generateMiniWorld(seed=20260423,
                               skipBiomes=True, skipCaves=True,
                               skipOres=True).layers

    titles = ["Pre-Hardmode", "V-Pattern", "Early Spread", "Late Spread"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5))
    for ax, snap, title in zip(axes.flat, snaps, titles):
        renderMiniWorld(snap, ax, title=title,
                        showLayers=True, layers=layers)

    fig.suptitle(suptitle, color=COLORS["fg"], fontsize=16,
                 fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(os.path.dirname(os.path.abspath(savePath)), exist_ok=True)
    plt.savefig(savePath, dpi=110, bbox_inches="tight",
                facecolor=COLORS["bg"])
    plt.close()
    print(f"Saved {savePath}")


def createSpreadAnimation(savePath: str | None = None,
                          evilType: str = "corruption") -> None:
    """Animated TINY world showing infection spreading over ~30 frames."""
    if savePath is None:
        savePath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "Plots", "Advanced",
            f"{evilType}_spread.gif",
        )

    print(f"Building {evilType} spread animation...")
    base = generateMiniWorld(seed=20260423, evilType=evilType, altarsSmashed=0)
    grid = base.grid.copy()
    carveVPattern(grid, evilType=evilType, seed=20260424)

    frames: list[np.ndarray] = [grid.copy()]
    rng = np.random.default_rng(20260425)
    for f in range(30):
        spreadInfection(grid, cycles=2, seed=int(rng.integers(0, 1 << 30)))
        frames.append(grid.copy())

    title = "Corruption Spread" if evilType == "corruption" else "Crimson Spread"
    saveTinyGif(frames, savePath, fps=6, scale=5, title=title)
    print(f"Saved {savePath}")


if __name__ == "__main__":
    print("Terraria Corruption Evolution")
    print("=" * 40)
    createEvolutionFigure()
    createSpreadAnimation()
    print("Done.")
