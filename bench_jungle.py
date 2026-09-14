import time

import numpy as np

from Engine.constants import SMALL, LayerDepths
from Engine.worldgen import DIRT, GRASS, MUD, STONE

# Setup grid and layers like generateSmallWorld does
layers = LayerDepths.forSmall()
width = SMALL.width
height = SMALL.height
hell = int(layers.hellLayer)

jungleX = width // 2
jungleHalf = 240
x_start = max(0, jungleX - jungleHalf)
x_end = min(width, jungleX + jungleHalf)

def original_jungle(grid):
    for x in range(x_start, x_end):
        for y in range(hell):
            t = grid[y, x]
            if t == DIRT or t == GRASS:
                grid[y, x] = MUD

def vectorized_jungle(grid):
    region = grid[:hell, x_start:x_end]
    mask = (region == DIRT) | (region == GRASS)
    region[mask] = MUD

def benchmark(n_iters=50):
    total_time_orig = 0
    total_time_vec = 0

    for _ in range(n_iters):
        grid = np.full((height, width), STONE, dtype=np.int32)
        grid[:int(layers.rockLayer), :] = DIRT
        grid[int(layers.worldSurface), :] = GRASS

        start = time.time()
        original_jungle(grid)
        total_time_orig += (time.time() - start)

    for _ in range(n_iters):
        grid = np.full((height, width), STONE, dtype=np.int32)
        grid[:int(layers.rockLayer), :] = DIRT
        grid[int(layers.worldSurface), :] = GRASS

        start = time.time()
        vectorized_jungle(grid)
        total_time_vec += (time.time() - start)

    print(f"Original Elapsed: {total_time_orig:.4f} seconds for {n_iters} iterations")
    print(f"Vectorized Elapsed: {total_time_vec:.4f} seconds for {n_iters} iterations")

    # Verify correctness
    grid1 = np.full((height, width), STONE, dtype=np.int32)
    grid1[:int(layers.rockLayer), :] = DIRT
    grid1[int(layers.worldSurface), :] = GRASS
    original_jungle(grid1)

    grid2 = np.full((height, width), STONE, dtype=np.int32)
    grid2[:int(layers.rockLayer), :] = DIRT
    grid2[int(layers.worldSurface), :] = GRASS
    vectorized_jungle(grid2)

    assert np.array_equal(grid1, grid2), "Implementations differ!"
    print("Correctness verified.")

benchmark()
