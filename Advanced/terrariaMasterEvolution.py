"""
Terraria World Evolution Master Animation
==========================================

Master orchestrator combining world generation, corruption/hallow evolution,
and hardmode transformations into a single animation timeline.

Runs at 1/10 scale (840x240) by default. The three subsystems
(TerrariaWorldGenerator, TerrariaCorruptionEvolution,
TerrariaHardmodeTransformation) all use Engine tile IDs, enabling
seamless grid handoff between phases.

Animation Sequence (10 phases, ~26 frames):
 1. Base Terrain Generation
 2. Cave Carving + Smoothing
 3. Biome Painting
 4. Pre-Hardmode Structures
 5. Corruption/Crimson Initial
 6. Hardmode V-Pattern
 7. Altar Smashing (3-Cycle Ore)
 8. Infection Spread
 9. Chlorophyte Growth
10. Final Hardmode State
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import List, Tuple, Dict, Optional

from Engine.algorithms import (
    tileRunner, AIR, STONE, DIRT, MUD, GRASS, SAND, ASH, HELLSTONE,
    SNOW, ICE, WATER, LAVA, COPPER, IRON, SILVER, GOLD,
    COBALT, PALLADIUM, MYTHRIL, ORICHALCUM,
    ADAMANTITE, TITANIUM, CHLOROPHYTE,
    EBONSTONE, CRIMSTONE, CORRUPT_DIRT, CRIMSON_DIRT,
    PEARLSTONE, HALLOW_DIRT, PEARLSAND,
    CORRUPT_ICE, CRIMSON_ICE, HALLOW_ICE, DUNGEON_BRICK,
)
from Engine.constants import (
    LARGE, LayerDepths, StructureQuotas, OreConfig,
    INFECTION_GAP_TILES, SURFACE_UPDATE_RATE, UNDERGROUND_UPDATE_RATE,
    LIFE_CRYSTAL, ALTAR,
)
from Engine.theme import applyDarkTheme, COLORS, TILE_COLORS as ENGINE_TILE_COLORS

from terrariaWorldGeneration import TerrariaWorldGenerator
from terrariaCorruptionEvolution import TerrariaCorruptionEvolution
from terrariaHardmodeStructures import TerrariaHardmodeTransformation

applyDarkTheme()


# ---------------------------------------------------------------------------
# Unified color palette (Engine tile IDs)
# ---------------------------------------------------------------------------
TILE_COLORS: dict[int, tuple[float, float, float]] = {
    AIR: (0.05, 0.05, 0.10),
    DIRT: (0.45, 0.32, 0.18),
    STONE: (0.50, 0.50, 0.50),
    GRASS: (0.20, 0.60, 0.20),
    SAND: (0.90, 0.80, 0.50),
    ASH: (0.30, 0.30, 0.30),
    HELLSTONE: (0.95, 0.25, 0.00),
    MUD: (0.35, 0.28, 0.40),
    SNOW: (0.88, 0.90, 0.95),
    ICE: (0.70, 0.85, 1.00),
    WATER: (0.10, 0.40, 0.90),
    LAVA: (1.00, 0.25, 0.00),
    COPPER: (0.72, 0.45, 0.20),
    IRON: (0.67, 0.67, 0.67),
    SILVER: (0.75, 0.75, 0.75),
    GOLD: (1.00, 0.84, 0.00),
    CORRUPT_DIRT: (0.35, 0.10, 0.45),
    EBONSTONE: (0.30, 0.05, 0.50),
    CRIMSON_DIRT: (0.55, 0.08, 0.15),
    CRIMSTONE: (0.70, 0.05, 0.20),
    CORRUPT_ICE: (0.50, 0.20, 0.70),
    CRIMSON_ICE: (0.80, 0.20, 0.30),
    PEARLSTONE: (0.95, 0.85, 1.00),
    HALLOW_DIRT: (0.90, 0.80, 0.95),
    PEARLSAND: (1.00, 0.95, 0.80),
    HALLOW_ICE: (0.85, 0.90, 1.00),
    COBALT: (0.00, 0.35, 0.85),
    PALLADIUM: (0.90, 0.45, 0.15),
    MYTHRIL: (0.20, 0.80, 0.30),
    ORICHALCUM: (0.85, 0.40, 0.65),
    ADAMANTITE: (0.85, 0.15, 0.15),
    TITANIUM: (0.55, 0.55, 0.60),
    CHLOROPHYTE: (0.10, 0.95, 0.20),
    DUNGEON_BRICK: (0.30, 0.20, 0.35),
    LIFE_CRYSTAL: (1.00, 0.20, 0.70),
    ALTAR: (0.60, 0.10, 0.10),
}

TILE_NAMES: dict[int, str] = {
    AIR: "Air", DIRT: "Dirt", STONE: "Stone", GRASS: "Grass",
    SAND: "Sand", ASH: "Ash", HELLSTONE: "Hellstone", MUD: "Mud",
    SNOW: "Snow", ICE: "Ice", WATER: "Water", LAVA: "Lava",
    COPPER: "Copper", IRON: "Iron", SILVER: "Silver", GOLD: "Gold",
    CORRUPT_DIRT: "Corrupt Dirt", EBONSTONE: "Ebonstone",
    CRIMSON_DIRT: "Crimson Dirt", CRIMSTONE: "Crimstone",
    PEARLSTONE: "Pearlstone", HALLOW_DIRT: "Hallow Dirt",
    PEARLSAND: "Pearlsand",
    COBALT: "Cobalt", PALLADIUM: "Palladium",
    MYTHRIL: "Mythril", ORICHALCUM: "Orichalcum",
    ADAMANTITE: "Adamantite", TITANIUM: "Titanium",
    CHLOROPHYTE: "Chlorophyte",
    LIFE_CRYSTAL: "Life Crystal", ALTAR: "Altar",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _gridToRgb(grid: np.ndarray) -> np.ndarray:
    """Convert tile-ID grid to (H, W, 3) float32 RGB image."""
    h, w = grid.shape
    rgb = np.full((h, w, 3), TILE_COLORS[AIR], dtype=np.float32)
    for tileId, color in TILE_COLORS.items():
        mask = grid == tileId
        if np.any(mask):
            rgb[mask] = color
    return rgb


def _tileStats(grid: np.ndarray, tileIds: List[int]) -> str:
    """Return compact tile count string for the given IDs."""
    parts = []
    for tid in tileIds:
        count = int(np.sum(grid == tid))
        if count > 0:
            name = TILE_NAMES.get(tid, f"T{tid}")
            parts.append(f"{name}: {count:,}")
    return "  |  ".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class TerrariaWorldEvolutionMaster:
    """Master orchestrator: world generation -> corruption -> hardmode.

    Runs TerrariaWorldGenerator for the initial 19-pass build, then hands
    the grid to TerrariaCorruptionEvolution for the V-pattern and spread,
    and TerrariaHardmodeTransformation for ore placement and chlorophyte.
    """

    PHASES = [
        "Base Terrain Generation",
        "Cave Carving + Smoothing",
        "Biome Painting",
        "Pre-Hardmode Structures",
        "Corruption/Crimson Initial",
        "Hardmode V-Pattern",
        "Altar Smashing (3-Cycle Ore)",
        "Infection Spread",
        "Chlorophyte Growth",
        "Final Hardmode State",
    ]

    # Which worldgen pass names map to which animation phase
    _PHASE_PASSES: Dict[str, List[str]] = {
        "Base Terrain Generation": ["Terrain", "Stone Layer", "Sand Patches"],
        "Cave Carving + Smoothing": [
            "Surface Caves", "Rock Layer Caves", "Smooth World",
        ],
        "Biome Painting": ["Snow Biome", "Jungle", "Corruption"],
        "Pre-Hardmode Structures": ["Shinies", "Life Crystals"],
    }

    def __init__(
        self,
        worldWidth: int = 840,
        worldHeight: int = 240,
        seed: int = 42,
    ):
        self.worldWidth = worldWidth
        self.worldHeight = worldHeight
        self.seed = seed

        self.worldGen = TerrariaWorldGenerator(worldWidth, worldHeight, seed)
        self.corruptionEvo = TerrariaCorruptionEvolution(
            worldWidth, worldHeight, evilType="corruption", seed=seed,
        )
        self.hardmodeTrans = TerrariaHardmodeTransformation(
            worldWidth, worldHeight, seed,
        )
        # Force corruption to match worldgen (always places EBONSTONE)
        self.hardmodeTrans.isCorruption = True

        self.frames: List[Tuple[str, np.ndarray]] = []

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def runEvolution(self) -> List[Tuple[str, np.ndarray]]:
        """Execute all 10 phases, capturing ~26 frames total.

        Returns list of (phaseName, gridSnapshot) tuples.
        """
        self.frames.clear()

        # -- Phases 1-4: world generation --------------------------------
        print("Phases 1-4: Running 19-pass world generation...")
        self.worldGen.generate()
        snapDict: Dict[str, np.ndarray] = {
            name: grid.copy()
            for name, grid in self.worldGen.snapshots
        }

        for phase, passNames in self._PHASE_PASSES.items():
            for passName in passNames:
                if passName in snapDict:
                    self.frames.append((phase, snapDict[passName]))

        grid = self.worldGen.grid.copy()

        # -- Phase 5: corruption/crimson initial state -------------------
        print("Phase 5: Corruption/Crimson initial state...")
        self.frames.append(("Corruption/Crimson Initial", grid.copy()))

        # Add extra evil pockets via corruption evolution
        self.corruptionEvo.grid = grid.copy()
        self.corruptionEvo.placePreHardmodeEvil()
        grid = self.corruptionEvo.grid.copy()
        self.frames.append(("Corruption/Crimson Initial", grid.copy()))

        # -- Phase 6: hardmode V-pattern ---------------------------------
        print("Phase 6: Hardmode V-pattern carving...")
        self.corruptionEvo.grid = grid.copy()
        self.corruptionEvo.triggerHardmode()
        grid = self.corruptionEvo.grid.copy()
        self.frames.append(("Hardmode V-Pattern", grid.copy()))
        self.frames.append(("Hardmode V-Pattern", grid.copy()))

        # -- Phase 7: altar smashing (3-cycle ore) -----------------------
        print("Phase 7: Altar smashing (3-cycle hardmode ore)...")
        self.hardmodeTrans.grid = grid.copy()
        self.hardmodeTrans.altarsSmashed = 0

        # Align evil region with worldgen's actual evil center
        evilCenter = self.worldGen._evilCenter
        evilHalf = max(20, int(100 * (self.worldWidth / LARGE.width)))
        self.hardmodeTrans.evilXMin = max(0, evilCenter - evilHalf)
        self.hardmodeTrans.evilXMax = min(
            self.worldWidth, evilCenter + evilHalf,
        )
        self.hardmodeTrans._placeAltars()

        # Smash 3 altars: one per ore tier (Cobalt/Palladium,
        # Mythril/Orichalcum, Adamantite/Titanium)
        for _ in range(3):
            self.hardmodeTrans.smashAltar()
            self.frames.append((
                "Altar Smashing (3-Cycle Ore)",
                self.hardmodeTrans.grid.copy(),
            ))
        grid = self.hardmodeTrans.grid.copy()

        # -- Phase 8: infection spread -----------------------------------
        print("Phase 8: Infection spread (tile update cycle)...")
        self.corruptionEvo.grid = grid.copy()
        for _ in range(5):
            self.corruptionEvo.simulateSpread(3000.0)
            self.frames.append((
                "Infection Spread",
                self.corruptionEvo.grid.copy(),
            ))
        grid = self.corruptionEvo.grid.copy()

        # -- Phase 9: chlorophyte growth ---------------------------------
        print("Phase 9: Chlorophyte growth in jungle cavern...")
        self.hardmodeTrans.grid = grid.copy()

        # Derive jungle x-range from worldgen's dungeon side
        if self.worldGen._dungeonLeft:
            jungleXMin = int(self.worldWidth * 0.72)
            jungleXMax = self.worldWidth
        else:
            jungleXMin = 0
            jungleXMax = max(1, int(self.worldWidth * 0.28))

        self.hardmodeTrans.placeChlorophyte(jungleXMin, jungleXMax)
        grid = self.hardmodeTrans.grid.copy()
        self.frames.append(("Chlorophyte Growth", grid.copy()))
        self.frames.append(("Chlorophyte Growth", grid.copy()))

        # -- Phase 10: final state ---------------------------------------
        self.frames.append(("Final Hardmode State", grid.copy()))

        print(f"Evolution complete: {len(self.frames)} frames across "
              f"{len(self.PHASES)} phases")
        return self.frames

    # ------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------
    def createAnimation(
        self,
        savePath: Optional[str] = None,
        interval: int = 400,
        dpi: int = 150,
    ) -> Optional[FuncAnimation]:
        """Create FuncAnimation with PillowWriter and save as .gif.

        Args:
            savePath: Output .gif path. Defaults to Plots/Code+/.
            interval: Milliseconds per frame.
            dpi: Output resolution.

        Returns:
            The FuncAnimation object, or None if no frames exist.
        """
        if not self.frames:
            print("No frames. Call runEvolution() first.")
            return None

        fig, ax = plt.subplots(figsize=(18, 6))
        fig.patch.set_facecolor("#0d0d1a")
        ax.set_facecolor("#0d0d1a")

        initRgb = _gridToRgb(self.frames[0][1])
        img = ax.imshow(initRgb, aspect="auto", interpolation="nearest")
        ax.set_xlabel("X (tiles)", color="white")
        ax.set_ylabel("Y (tiles)", color="white")

        phaseText = ax.text(
            0.02, 0.96, "", transform=ax.transAxes,
            fontsize=13, fontweight="bold", color="white",
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="#1a1a2e", alpha=0.85,
            ),
        )
        statsText = ax.text(
            0.98, 0.04, "", transform=ax.transAxes,
            fontsize=7, color="white",
            verticalalignment="bottom", horizontalalignment="right",
            fontfamily="monospace",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="#1a1a2e", alpha=0.75,
            ),
        )
        frameText = ax.text(
            0.02, 0.04, "", transform=ax.transAxes,
            fontsize=8, color="gray", verticalalignment="bottom",
        )

        scaleLabel = f"1/10 scale ({self.worldWidth}x{self.worldHeight})"
        ax.set_title(
            f"Terraria World Evolution    [{scaleLabel}]",
            fontsize=14, fontweight="bold", color="white",
        )

        # Compact legend with key tile types
        legendIds = [
            DIRT, STONE, GRASS, MUD, SNOW, ICE, SAND,
            EBONSTONE, CORRUPT_DIRT, PEARLSTONE, HALLOW_DIRT,
            COBALT, MYTHRIL, ADAMANTITE, CHLOROPHYTE, LIFE_CRYSTAL,
        ]
        handles = [
            mpatches.Patch(color=TILE_COLORS[tid], label=TILE_NAMES[tid])
            for tid in legendIds if tid in TILE_COLORS
        ]
        ax.legend(
            handles=handles, loc="lower left", fontsize=5,
            ncol=8, framealpha=0.6, bbox_to_anchor=(0.0, -0.18),
        )

        # Tile IDs tracked in stats overlay
        statIds = [
            DIRT, STONE, EBONSTONE, CORRUPT_DIRT, PEARLSTONE, HALLOW_DIRT,
            COBALT, PALLADIUM, MYTHRIL, ORICHALCUM,
            ADAMANTITE, TITANIUM, CHLOROPHYTE, LIFE_CRYSTAL,
        ]

        def _update(frame: int):
            phase, grid = self.frames[frame]
            img.set_data(_gridToRgb(grid))
            phaseText.set_text(f"Phase: {phase}")
            frameText.set_text(f"Frame {frame + 1}/{len(self.frames)}")
            statsText.set_text(_tileStats(grid, statIds))
            return [img, phaseText, statsText, frameText]

        anim = FuncAnimation(
            fig, _update, frames=len(self.frames),
            interval=interval, blit=False, repeat=True,
        )

        if savePath is None:
            savePath = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "Plots", "Advanced", "terraria_master_evolution.gif",
            )

        os.makedirs(os.path.dirname(savePath), exist_ok=True)
        print(f"Saving animation ({len(self.frames)} frames) to {savePath}")
        writer = PillowWriter(fps=max(1, 1000 // interval))
        anim.save(savePath, writer=writer, dpi=dpi)
        plt.close(fig)
        print(f"Saved: {savePath}")

        return anim


# ---------------------------------------------------------------------------
# Backward-compatible alias
# ---------------------------------------------------------------------------
master_evolution = TerrariaWorldEvolutionMaster


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    SCALE = 10
    w = LARGE.width // SCALE   # 840
    h = LARGE.height // SCALE  # 240

    master = TerrariaWorldEvolutionMaster(worldWidth=w, worldHeight=h, seed=42)
    master.runEvolution()
    master.createAnimation()
