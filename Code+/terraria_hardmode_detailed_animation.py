"""
Terraria Hardmode Transformation Master Animation
===============================================

This module creates a detailed animation showing the Hardmode transformation process
in Terraria, including the V-pattern corruption/hallow spread, altar breaking mechanics,
and new ore generation systems.

Animation Features:
1. Wall of Flesh defeat trigger
2. V-pattern diagonal stripe generation
3. Altar breaking and ore scattering
4. Biome conversion acceleration
5. Environmental transformation visualization

Mathematical Models:
- V-pattern generation using linear diagonal functions
- Poisson distribution for altar placement
- Exponential decay for ore probability by depth
- Accelerated cellular automata for evil biome spread

Author: Generated for Terraria Generation Analysis
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle, Polygon
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Set visualization preferences
plt.style.use('dark_background')
sns.set_palette("rocket")

class TerrariaHardmodeTransformation:
    """Detailed Hardmode transformation animation system."""
    
    def __init__(self, width: int = 400, height: int = 100):
        """
        Initialize the Hardmode transformation system.
        
        Args:
            width: World width in blocks
            height: World height in blocks
        """
        self.width = width
        self.height = height
        self.surface_level = height // 4
        self.cavern_level = int(height * 0.6)
        self.hell_level = int(height * 0.85)
        
        # Enhanced block type system
        self.AIR = 0
        self.DIRT = 1
        self.STONE = 2
        self.GRASS = 3
        self.CORRUPTION = 4
        self.CRIMSON = 5
        self.HALLOW = 6
        self.JUNGLE = 7
        self.SNOW = 8
        self.SAND = 9
        self.HELLSTONE = 10
        self.COBALT = 11
        self.MYTHRIL = 12
        self.ADAMANTITE = 13
        self.CHLOROPHYTE = 14
        self.PEARLSTONE = 15
        self.EBONSTONE = 16
        self.CRIMSTONE = 17
        
        # Enhanced color scheme for hardmode visualization
        self.colors = {
            self.AIR: '#000022',          # Deep night blue
            self.DIRT: '#8B4513',         # Saddle brown
            self.STONE: '#708090',        # Slate gray
            self.GRASS: '#32CD32',        # Lime green
            self.CORRUPTION: '#6A0DAD',   # Blue violet
            self.CRIMSON: '#DC143C',      # Crimson red
            self.HALLOW: '#FF1493',       # Deep pink
            self.JUNGLE: '#228B22',       # Forest green
            self.SNOW: '#F0F8FF',         # Alice blue
            self.SAND: '#F4A460',         # Sandy brown
            self.HELLSTONE: '#FF4500',    # Orange red
            self.COBALT: '#0066CC',       # Bright blue
            self.MYTHRIL: '#00FF00',      # Lime
            self.ADAMANTITE: '#FF0066',   # Bright red
            self.CHLOROPHYTE: '#7FFF00',  # Chartreuse
            self.PEARLSTONE: '#FFB6C1',   # Light pink
            self.EBONSTONE: '#483D8B',    # Dark slate blue
            self.CRIMSTONE: '#8B0000'     # Dark red
        }
        
        # Hardmode transformation state
        self.altars_broken = 0
        self.max_altars = 6
        self.v_pattern_complete = False
        self.world_states = []
        
    def generate_pre_hardmode_world(self) -> np.ndarray:
        """
        Generate a pre-hardmode world ready for transformation.
        
        Returns:
            2D numpy array representing the pre-hardmode world
        """
        world = np.full((self.height, self.width), self.AIR, dtype=int)
        
        # Generate terrain with established biomes
        surface_heights = self._generate_surface_terrain()
        
        # Fill basic terrain
        for x in range(self.width):
            surface_y = surface_heights[x]
            biome = self._determine_biome(x)
            
            for y in range(surface_y, self.height):
                if y == surface_y:
                    if biome == self.JUNGLE:
                        world[y, x] = self.JUNGLE
                    elif biome == self.SNOW:
                        world[y, x] = self.SNOW
                    elif biome == self.SAND:
                        world[y, x] = self.SAND
                    else:
                        world[y, x] = self.GRASS
                elif y < surface_y + 40:  # Dirt layer
                    world[y, x] = self.DIRT
                elif y < self.hell_level:  # Stone layer
                    world[y, x] = self.STONE
                else:  # Hell layer
                    world[y, x] = self.HELLSTONE
        
        # Add existing corruption/crimson
        self._add_existing_corruption(world)
        
        # Carve cave systems
        self._carve_caves(world)
        
        return world
    
    def _generate_surface_terrain(self) -> List[int]:
        """Generate surface terrain heights."""
        heights = []
        for x in range(self.width):
            height = self.surface_level
            height += 20 * np.sin(x * 0.02) * np.cos(x * 0.01)
            height += 10 * np.sin(x * 0.05) * np.cos(x * 0.03)
            height = int(np.clip(height, 15, self.surface_level + 25))
            heights.append(height)
        return heights
    
    def _determine_biome(self, x: int) -> int:
        """Determine biome type based on position."""
        pos = x / self.width
        if pos < 0.2:
            return self.SNOW
        elif 0.7 < pos < 0.9:
            return self.JUNGLE
        elif pos > 0.9:
            return self.SAND
        else:
            return self.GRASS
    
    def _add_existing_corruption(self, world: np.ndarray) -> None:
        """Add pre-existing corruption patches."""
        # Left side corruption
        corruption_center = self.width // 4
        for x in range(corruption_center - 30, corruption_center + 30):
            for y in range(self.surface_level, self.cavern_level):
                if 0 <= x < self.width and world[y, x] in [self.DIRT, self.STONE]:
                    if np.random.random() < 0.6:
                        world[y, x] = self.CORRUPTION
    
    def _carve_caves(self, world: np.ndarray) -> None:
        """Carve cave systems."""
        for _ in range(80):
            start_x = np.random.randint(0, self.width)
            start_y = np.random.randint(self.surface_level, self.hell_level)
            self._carve_tunnel(world, start_x, start_y, 6, 25)
    
    def _carve_tunnel(self, world: np.ndarray, x: int, y: int, strength: int, steps: int) -> None:
        """Carve a single tunnel."""
        for _ in range(steps):
            dx = np.random.randint(-1, 2)
            dy = np.random.randint(-1, 2)
            x = np.clip(x + dx, 0, self.width - 1)
            y = np.clip(y + dy, 0, self.height - 1)
            
            for i in range(-strength//2, strength//2 + 1):
                for j in range(-strength//2, strength//2 + 1):
                    if i*i + j*j <= (strength//2)**2:
                        nx, ny = x + i, y + j
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            world[ny, nx] = self.AIR
    
    def create_v_pattern_stripes(self, world: np.ndarray, progress: float) -> np.ndarray:
        """
        Create the V-pattern diagonal stripes characteristic of hardmode.
        
        Args:
            world: Current world state
            progress: Animation progress (0.0 to 1.0)
        
        Returns:
            World with V-pattern stripes
        """
        new_world = world.copy()
        center_x = self.width // 2
        max_reach = int(progress * min(center_x, self.hell_level - self.surface_level))
        
        # Left diagonal (corruption/crimson enhancement)
        for i in range(max_reach):
            x = center_x - i
            y = self.surface_level + i
            
            if 0 <= x < self.width and 0 <= y < self.height:
                # Create stripe width
                stripe_width = 8
                for offset in range(-stripe_width//2, stripe_width//2 + 1):
                    stripe_x = x + offset
                    if 0 <= stripe_x < self.width:
                        # Convert convertible blocks to corruption
                        if new_world[y, stripe_x] in [self.DIRT, self.STONE, self.SAND]:
                            new_world[y, stripe_x] = self.CORRUPTION
                        elif new_world[y, stripe_x] == self.STONE:
                            new_world[y, stripe_x] = self.EBONSTONE
        
        # Right diagonal (hallow)
        for i in range(max_reach):
            x = center_x + i
            y = self.surface_level + i
            
            if 0 <= x < self.width and 0 <= y < self.height:
                # Create stripe width
                stripe_width = 8
                for offset in range(-stripe_width//2, stripe_width//2 + 1):
                    stripe_x = x + offset
                    if 0 <= stripe_x < self.width:
                        # Convert convertible blocks to hallow
                        if new_world[y, stripe_x] in [self.DIRT, self.SAND]:
                            new_world[y, stripe_x] = self.HALLOW
                        elif new_world[y, stripe_x] == self.STONE:
                            new_world[y, stripe_x] = self.PEARLSTONE
        
        return new_world
    
    def break_altars_and_generate_ores(self, world: np.ndarray, altar_number: int) -> np.ndarray:
        """
        Simulate altar breaking and consequent ore generation.
        
        Args:
            world: Current world state
            altar_number: Which altar is being broken (1-6)
        
        Returns:
            World with new hardmode ores
        """
        new_world = world.copy()
        
        # Determine which ore tier to generate
        if altar_number <= 2:
            ore_type = self.COBALT
            ore_count = 300
        elif altar_number <= 4:
            ore_type = self.MYTHRIL
            ore_count = 200
        else:
            ore_type = self.ADAMANTITE
            ore_count = 100
        
        # Generate ore veins with depth-based probability
        for _ in range(ore_count):
            # Deeper ores are more common for hardmode
            depth_factor = np.random.exponential(0.4)
            depth_factor = min(1.0, depth_factor)
            
            x = np.random.randint(0, self.width)
            y = int(self.cavern_level + depth_factor * (self.hell_level - self.cavern_level))
            y = min(y, self.hell_level - 5)
            
            if 0 <= y < self.height and new_world[y, x] == self.STONE:
                # Create small vein
                vein_size = np.random.randint(2, 6)
                for _ in range(vein_size):
                    vein_x = x + np.random.randint(-2, 3)
                    vein_y = y + np.random.randint(-1, 2)
                    
                    if (0 <= vein_x < self.width and 0 <= vein_y < self.height and
                        new_world[vein_y, vein_x] == self.STONE):
                        new_world[vein_y, vein_x] = ore_type
        
        self.altars_broken = altar_number
        return new_world
    
    def accelerate_biome_spread(self, world: np.ndarray, intensity: float) -> np.ndarray:
        """
        Accelerate corruption and hallow spreading in hardmode.
        
        Args:
            world: Current world state
            intensity: Spreading intensity multiplier
        
        Returns:
            World with accelerated spreading
        """
        new_world = world.copy()
        
        # Enhanced spreading algorithm
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                current_block = world[y, x]
                
                if current_block in [self.CORRUPTION, self.HALLOW]:
                    # Count neighbors and spread more aggressively
                    spread_range = 2  # Larger spread range in hardmode
                    
                    for dy in range(-spread_range, spread_range + 1):
                        for dx in range(-spread_range, spread_range + 1):
                            nx, ny = x + dx, y + dy
                            
                            if (0 <= nx < self.width and 0 <= ny < self.height and
                                world[ny, nx] in [self.DIRT, self.STONE, self.SAND, self.GRASS]):
                                
                                # Distance-based probability
                                distance = np.sqrt(dx*dx + dy*dy)
                                prob = intensity * np.exp(-distance * 0.5)
                                
                                if np.random.random() < prob:
                                    if current_block == self.CORRUPTION:
                                        new_world[ny, nx] = self.CORRUPTION
                                    else:  # Hallow
                                        new_world[ny, nx] = self.HALLOW
        
        return new_world
    
    def generate_chlorophyte(self, world: np.ndarray, progress: float) -> np.ndarray:
        """
        Generate chlorophyte ore in jungle areas (post-mech bosses).
        
        Args:
            world: Current world state
            progress: Generation progress (0.0 to 1.0)
        
        Returns:
            World with chlorophyte ore
        """
        new_world = world.copy()
        
        # Chlorophyte only spawns in jungle areas, deep underground
        jungle_start = int(0.7 * self.width)
        jungle_end = int(0.9 * self.width)
        
        chlorophyte_count = int(progress * 50)  # Gradual appearance
        
        for _ in range(chlorophyte_count):
            x = np.random.randint(jungle_start, jungle_end)
            y = np.random.randint(self.cavern_level + 10, self.hell_level - 10)
            
            # Chlorophyte converts mud and stone in jungle
            if new_world[y, x] == self.STONE:
                new_world[y, x] = self.CHLOROPHYTE
        
        return new_world
    
    def create_colormap(self) -> ListedColormap:
        """Create enhanced colormap for hardmode visualization."""
        colors = [self.colors[i] for i in range(len(self.colors))]
        return ListedColormap(colors)

def create_hardmode_transformation_animation(save_path: str):
    """
    Create detailed hardmode transformation animation.
    
    Args:
        save_path: Path to save the animation
    """
    print("Creating Detailed Hardmode Transformation Animation...")
    print("=" * 55)
    
    # Initialize transformation system
    hardmode = TerrariaHardmodeTransformation(width=400, height=100)
    
    # Generate pre-hardmode world
    world = hardmode.generate_pre_hardmode_world()
    
    # Animation phases
    world_states = []
    phase_markers = []
    
    # Phase 1: Pre-hardmode state (frames 0-20)
    for _ in range(20):
        world_states.append(world.copy())
        phase_markers.append("Pre-Hardmode: Awaiting Wall of Flesh...")
    
    # Phase 2: V-pattern formation (frames 21-60)
    for frame in range(40):
        progress = frame / 39
        world = hardmode.create_v_pattern_stripes(world, progress)
        world_states.append(world.copy())
        phase_markers.append(f"V-Pattern Formation: {progress*100:.0f}% Complete")
    
    # Phase 3: Altar breaking sequence (frames 61-120)
    for altar in range(1, 7):  # 6 altars
        for sub_frame in range(10):
            if sub_frame == 0:  # Break altar on first frame
                world = hardmode.break_altars_and_generate_ores(world, altar)
            world_states.append(world.copy())
            phase_markers.append(f"Altar {altar}/6 Broken - Generating Hardmode Ores")
    
    # Phase 4: Accelerated spreading (frames 121-180)
    for frame in range(60):
        intensity = 0.08 + (frame / 60) * 0.12  # Increasing intensity
        world = hardmode.accelerate_biome_spread(world, intensity)
        
        # Add chlorophyte after frame 140 (post-mech bosses simulation)
        if frame > 20:
            chlorophyte_progress = (frame - 20) / 40
            world = hardmode.generate_chlorophyte(world, chlorophyte_progress)
        
        world_states.append(world.copy())
        phase_markers.append(f"Hardmode Spread: Intensity {intensity:.2f}")
    
    # Phase 5: Final equilibrium (frames 181-200)
    for _ in range(20):
        world_states.append(world.copy())
        phase_markers.append("Hardmode Complete: World Fully Transformed")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 10))
    fig.patch.set_facecolor('#0D1117')
    ax.set_facecolor('#0D1117')
    
    # Create colormap
    cmap = hardmode.create_colormap()
    
    def animate(frame):
        ax.clear()
        
        # Get current world state
        world = world_states[frame]
        phase_desc = phase_markers[frame]
        
        # Display world
        im = ax.imshow(world, cmap=cmap, aspect='auto', vmin=0, vmax=17,
                      extent=[0, hardmode.width, hardmode.height, 0])
        
        # Add layer indicators
        ax.axhline(y=hardmode.surface_level, color='cyan', linestyle=':', 
                  alpha=0.6, linewidth=2, label='Surface')
        ax.axhline(y=hardmode.cavern_level, color='yellow', linestyle=':', 
                  alpha=0.6, linewidth=2, label='Cavern')
        ax.axhline(y=hardmode.hell_level, color='red', linestyle=':', 
                  alpha=0.6, linewidth=2, label='Hell')
        
        # Dynamic title
        ax.set_title(f'Terraria Hardmode Transformation\n{phase_desc}\nFrame: {frame+1}/200', 
                    fontsize=16, fontweight='bold', color='white', pad=20)
        
        # Add V-pattern indicators if in that phase
        if 20 < frame <= 60:
            center_x = hardmode.width // 2
            progress = (frame - 20) / 40
            max_reach = int(progress * min(center_x, hardmode.hell_level - hardmode.surface_level))
            
            # Draw V-pattern guidelines
            if max_reach > 0:
                # Left diagonal
                x1, y1 = center_x, hardmode.surface_level
                x2, y2 = center_x - max_reach, hardmode.surface_level + max_reach
                ax.plot([x1, x2], [y1, y2], 'r--', linewidth=3, alpha=0.8)
                
                # Right diagonal
                x3, y3 = center_x + max_reach, hardmode.surface_level + max_reach
                ax.plot([x1, x3], [y1, y3], 'm--', linewidth=3, alpha=0.8)
        
        # Add altar breaking indicators
        if 60 < frame <= 120:
            altar_number = (frame - 60) // 10 + 1
            ax.text(0.98, 0.85, f'Breaking Altar {altar_number}/6',
                   transform=ax.transAxes, fontsize=14, color='yellow',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.8),
                   horizontalalignment='right', fontweight='bold')
        
        # Statistics panel
        corruption_count = np.sum(world == hardmode.CORRUPTION)
        hallow_count = np.sum(world == hardmode.HALLOW)
        cobalt_count = np.sum(world == hardmode.COBALT)
        mythril_count = np.sum(world == hardmode.MYTHRIL)
        adamantite_count = np.sum(world == hardmode.ADAMANTITE)
        chlorophyte_count = np.sum(world == hardmode.CHLOROPHYTE)
        
        total_blocks = hardmode.width * hardmode.height
        
        stats_text = (
            f"World Statistics:\n"
            f"Corruption: {corruption_count/total_blocks*100:.1f}%\n"
            f"Hallow: {hallow_count/total_blocks*100:.1f}%\n"
            f"Cobalt Ore: {cobalt_count} blocks\n"
            f"Mythril Ore: {mythril_count} blocks\n"
            f"Adamantite: {adamantite_count} blocks\n"
            f"Chlorophyte: {chlorophyte_count} blocks"
        )
        
        ax.text(0.02, 0.98, stats_text,
               transform=ax.transAxes, fontsize=10, color='white',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8),
               verticalalignment='top')
        
        # Mathematical formulas
        if frame <= 60:
            formula = r'V-Pattern: $\{(x,y) : |x-x_c| + |y-y_s| = t\}$'
        elif frame <= 120:
            formula = r'Ore Density: $\rho(d) = \rho_0 e^{-\lambda d}$'
        else:
            formula = r'Spread Rate: $dC/dt = \alpha C(1-C) + \beta \nabla^2 C$'
        
        ax.text(0.98, 0.02, formula,
               transform=ax.transAxes, fontsize=12, color='white',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='purple', alpha=0.8),
               horizontalalignment='right', verticalalignment='bottom')
        
        # Styling
        ax.set_xlabel('World X Coordinate (blocks)', fontsize=12, color='white', fontweight='bold')
        ax.set_ylabel('World Depth (blocks)', fontsize=12, color='white', fontweight='bold')
        ax.tick_params(colors='white')
        
        return [im]
    
    # Create animation
    print("Rendering hardmode transformation frames...")
    anim = FuncAnimation(fig, animate, frames=len(world_states), 
                        interval=200, repeat=True, blit=False)
    
    # Save animation
    print(f"Saving hardmode transformation animation to {save_path}")
    writer = PillowWriter(fps=6)
    anim.save(save_path, writer=writer, dpi=150)
    plt.close(fig)
    
    print("Hardmode transformation animation completed!")

def create_biome_evolution_timeline(save_path: str):
    """
    Create a timeline animation showing biome evolution through all phases.
    
    Args:
        save_path: Path to save the animation
    """
    print("Creating Biome Evolution Timeline Animation...")
    print("=" * 45)
    
    # Initialize system
    hardmode = TerrariaHardmodeTransformation(width=300, height=60)
    
    # Generate timeline of biome states
    world = hardmode.generate_pre_hardmode_world()
    timeline_states = []
    
    # Pre-hardmode (natural state)
    timeline_states.append(("Pre-Hardmode World Generation", world.copy()))
    
    # Add initial corruption
    hardmode._add_existing_corruption(world)
    timeline_states.append(("Initial Corruption Seeding", world.copy()))
    
    # V-pattern application
    world = hardmode.create_v_pattern_stripes(world, 1.0)
    timeline_states.append(("V-Pattern Corruption/Hallow", world.copy()))
    
    # Full altar breaking
    for altar in range(1, 7):
        world = hardmode.break_altars_and_generate_ores(world, altar)
    timeline_states.append(("All Altars Broken - Ores Generated", world.copy()))
    
    # Heavy spreading
    for _ in range(5):
        world = hardmode.accelerate_biome_spread(world, 0.15)
    timeline_states.append(("Accelerated Evil Spread", world.copy()))
    
    # Final chlorophyte
    world = hardmode.generate_chlorophyte(world, 1.0)
    timeline_states.append(("Chlorophyte Generation Complete", world.copy()))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(18, 8))
    fig.patch.set_facecolor('#161B22')
    ax.set_facecolor('#161B22')
    
    cmap = hardmode.create_colormap()
    
    def animate_timeline(frame):
        ax.clear()
        
        # Cycle through timeline states
        state_index = frame % len(timeline_states)
        phase_name, world_state = timeline_states[state_index]
        
        # Display world
        im = ax.imshow(world_state, cmap=cmap, aspect='auto', vmin=0, vmax=17,
                      extent=[0, hardmode.width, hardmode.height, 0])
        
        # Timeline progress indicator
        progress = (state_index + 1) / len(timeline_states)
        
        ax.set_title(f'Terraria Biome Evolution Timeline\n{phase_name}\nPhase {state_index + 1}/{len(timeline_states)}', 
                    fontsize=16, fontweight='bold', color='white', pad=20)
        
        # Add timeline progress bar
        bar_width = hardmode.width * 0.8
        bar_start = hardmode.width * 0.1
        bar_y = hardmode.height - 8
        
        # Background bar
        progress_bg = Rectangle((bar_start, bar_y), bar_width, 4,
                               facecolor='gray', alpha=0.5)
        ax.add_patch(progress_bg)
        
        # Progress fill
        progress_fill = Rectangle((bar_start, bar_y), bar_width * progress, 4,
                                 facecolor='lime', alpha=0.8)
        ax.add_patch(progress_fill)
        
        # Phase markers
        marker_spacing = bar_width / (len(timeline_states) - 1)
        for i in range(len(timeline_states)):
            marker_x = bar_start + i * marker_spacing
            marker_color = 'yellow' if i == state_index else 'white'
            marker = Circle((marker_x, bar_y + 2), 2, facecolor=marker_color, edgecolor='black')
            ax.add_patch(marker)
        
        # Biome analysis
        unique_blocks, counts = np.unique(world_state, return_counts=True)
        total_blocks = hardmode.width * hardmode.height
        
        analysis_text = "Biome Distribution:\n"
        biome_names = {
            hardmode.CORRUPTION: "Corruption",
            hardmode.HALLOW: "Hallow", 
            hardmode.JUNGLE: "Jungle",
            hardmode.SNOW: "Snow",
            hardmode.COBALT: "Cobalt",
            hardmode.MYTHRIL: "Mythril",
            hardmode.ADAMANTITE: "Adamantite",
            hardmode.CHLOROPHYTE: "Chlorophyte"
        }
        
        for block_type, count in zip(unique_blocks, counts):
            if block_type in biome_names:
                percentage = (count / total_blocks) * 100
                analysis_text += f"{biome_names[block_type]}: {percentage:.1f}%\n"
        
        ax.text(0.02, 0.98, analysis_text,
               transform=ax.transAxes, fontsize=10, color='white',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8),
               verticalalignment='top')
        
        # Mathematical model description
        model_descriptions = [
            "Terrain: Multi-octave Perlin noise",
            "Infection: Random seeding algorithm", 
            "V-Pattern: Linear diagonal transformation",
            "Ore Gen: Exponential depth distribution",
            "Spread: Cellular automata diffusion",
            "Chlorophyte: Biome-restricted generation"
        ]
        
        current_model = model_descriptions[state_index] if state_index < len(model_descriptions) else model_descriptions[-1]
        
        ax.text(0.98, 0.02, f"Mathematical Model:\n{current_model}",
               transform=ax.transAxes, fontsize=11, color='white',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='purple', alpha=0.8),
               horizontalalignment='right', verticalalignment='bottom')
        
        # Styling
        ax.set_xlabel('World X Coordinate (blocks)', fontsize=12, color='white', fontweight='bold')
        ax.set_ylabel('World Depth (blocks)', fontsize=12, color='white', fontweight='bold')
        ax.tick_params(colors='white')
        
        return [im]
    
    # Create animation with slow transitions
    print("Rendering biome evolution timeline...")
    total_frames = len(timeline_states) * 30  # Hold each state for 30 frames
    
    def timeline_frame_generator():
        for state_idx in range(len(timeline_states)):
            for _ in range(30):  # Hold each state
                yield state_idx
    
    frame_gen = timeline_frame_generator()
    frames = list(frame_gen)
    
    anim = FuncAnimation(fig, animate_timeline, frames=frames, 
                        interval=100, repeat=True, blit=False)
    
    # Save animation
    print(f"Saving biome evolution timeline to {save_path}")
    writer = PillowWriter(fps=10)
    anim.save(save_path, writer=writer, dpi=150)
    plt.close(fig)
    
    print("Biome evolution timeline completed!")

if __name__ == "__main__":
    print("Starting Terraria Hardmode Transformation Animations...")
    print("=" * 60)
    
    # Create output directory
    output_dir = r"c:\Users\hunkb\OneDrive\Desktop\Terraria Generation\Code+"
    
    # Generate hardmode transformation animation
    hardmode_path = f"{output_dir}/terraria_hardmode_transformation_detailed.gif"
    create_hardmode_transformation_animation(hardmode_path)
    
    # Generate biome evolution timeline
    timeline_path = f"{output_dir}/terraria_biome_evolution_timeline.gif"
    create_biome_evolution_timeline(timeline_path)
    
    print("\n" + "=" * 60)
    print("All Hardmode transformation animations completed successfully!")
    print("Files created:")
    print(f"- {hardmode_path}")
    print(f"- {timeline_path}")
    print("\nThese animations demonstrate the detailed mechanics of")
    print("Terraria's Hardmode transformation and biome evolution.")
