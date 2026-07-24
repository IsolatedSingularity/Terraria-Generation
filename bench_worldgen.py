import time

import numpy as np

from Engine.constants import SMALL, LayerDepths
from Engine.worldgen import _placeBiomes

# Setup grid and layers like generateSmallWorld does
layers = LayerDepths.forSmall()
width = SMALL.width
height = SMALL.height

def benchmark(n_iters=5):
    rng = np.random.default_rng(123)

    total_time = 0
    for _ in range(n_iters):
        grid = np.full((height, width), 2, dtype=np.int32) # STONE

        # mock some dirt and grass
        grid[:int(layers.rockLayer), :] = 0 # DIRT
        grid[int(layers.worldSurface), :] = 1 # GRASS

        start = time.time()
        _placeBiomes(grid, layers, rng, "corruption", compact=False)
        total_time += (time.time() - start)

    print(f"Elapsed: {total_time:.4f} seconds for {n_iters} iterations")

benchmark()
