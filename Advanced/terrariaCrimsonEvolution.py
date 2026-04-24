"""Terraria crimson evolution: thin wrapper around the corruption module.

Same 4-phase TINY-world layout, same spread dynamics, but with
``evilType="crimson"`` so the converters swap to Crimstone, Crimson Dirt,
and Crimson Ice instead of Ebonstone variants.
"""

import os

from Advanced.terrariaCorruptionEvolution import (
    createEvolutionFigure,
    createSpreadAnimation,
)


if __name__ == "__main__":
    print("Terraria Crimson Evolution")
    print("=" * 40)
    plotsDir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "Plots", "Advanced",
    )
    createEvolutionFigure(
        savePath=os.path.join(plotsDir, "crimson_evolution.png"),
        evilType="crimson",
        suptitle="Crimson Evolution",
    )
    createSpreadAnimation(
        savePath=os.path.join(plotsDir, "crimson_spread.gif"),
        evilType="crimson",
    )
    print("Done.")
