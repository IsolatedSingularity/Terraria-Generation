import numpy as np
import timeit

width = 240
desertX = 165
desertHalf = 22
rock = 80
DIRT = 1
GRASS = 2
SAND = 3
STONE = 4
HARDENED_SAND = 5
SANDSTONE_BLOCK = 6

def setup_grid():
    grid = np.random.choice([DIRT, GRASS, STONE, 0], size=(140, 240)).astype(np.int32)
    return grid

def orig(grid):
    for x in range(max(0, desertX - desertHalf), min(width, desertX + desertHalf)):
        for y in range(rock + 12):
            t = grid[y, x]
            if t == DIRT or t == GRASS:
                grid[y, x] = SAND
            elif t == STONE and y < rock + 4:
                grid[y, x] = HARDENED_SAND
            elif t == STONE:
                grid[y, x] = SANDSTONE_BLOCK

def vect(grid):
    x_start = max(0, desertX - desertHalf)
    x_end = min(width, desertX + desertHalf)
    y_end = rock + 12

    if x_start < x_end and y_end > 0:
        region = grid[:y_end, x_start:x_end]
        mask_sand = (region == DIRT) | (region == GRASS)
        region[mask_sand] = SAND

        y_split = min(y_end, max(0, rock + 4))

        if y_split > 0:
            upper_region = grid[:y_split, x_start:x_end]
            upper_stone_mask = (upper_region == STONE)
            upper_region[upper_stone_mask] = HARDENED_SAND

        if y_end > y_split:
            lower_region = grid[y_split:y_end, x_start:x_end]
            lower_stone_mask = (lower_region == STONE)
            lower_region[lower_stone_mask] = SANDSTONE_BLOCK

g1 = setup_grid()
g2 = g1.copy()

orig(g1)
vect(g2)
assert np.array_equal(g1, g2)

print("Original:", timeit.timeit("orig(g)", setup="from __main__ import orig, setup_grid; g = setup_grid()", number=10000))
print("Vectorized:", timeit.timeit("vect(g)", setup="from __main__ import vect, setup_grid; g = setup_grid()", number=10000))
