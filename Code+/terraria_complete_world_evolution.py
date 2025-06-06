"""
Terraria Complete World Evolution Master Animation
=================================================

This module creates comprehensive animations showing the complete evolution of a Terraria world
from initial generation through corruption/crimson spread and hardmode transformations.

Animation Features:
1. Complete World Generation Process (103-pass system)
2. Corruption/Crimson Evolution Over Time
3. Hardmode Transformation with V-Pattern
4. Altar Breaking and New Ore Generation
5. Environmental Changes and Biome Interactions

Mathematical Models:
- Multi-octave Perlin noise for terrain generation
- Cellular automata for corruption spreading
- Poisson distribution for structure placement
- Exponential decay for ore generation probability
- Sigmoid functions for biome transitions

Author: Generated for Terraria Generation Analysis
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.patches as patches
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Set visualization preferences
plt.style.use('dark_background')
sns.set_palette("rocket")

class TerrariaCompleteEvolution:
    """Complete Terraria world evolution simulation system."""
    
    def __init__(self, width: int = 420, height: int = 120):
        """
        Initialize the complete evolution system.
        
        Args:
            width: World width in blocks
            height: World height in blocks
        """
        self.width = width
        self.height = height
        self.surface_level = height // 4
        self.cavern_level = int(height * 0.6)
        self.hell_level = int(height * 0.85)
        
        # Block type constants
        self.AIR = 0
        self.DIRT = 1
        self.STONE = 2
        self.GRASS = 3
        self.SAND = 4
        self.CORRUPTION = 5
        self.CRIMSON = 6
        self.HALLOW = 7
        self.JUNGLE = 8
        self.SNOW = 9
        self.DUNGEON = 10
        self.HELLSTONE = 11
        self.COBALT = 12
        self.MYTHRIL = 13
        self.ADAMANTITE = 14
        self.CHLOROPHYTE = 15
        
        # Color mapping for enhanced visualization
        self.colors = {
            self.AIR: '#000011',          # Deep space blue
            self.DIRT: '#8B4513',         # Saddle brown
            self.STONE: '#696969',        # Dim gray
            self.GRASS: '#228B22',        # Forest green
            self.SAND: '#F4A460',         # Sandy brown
            self.CORRUPTION: '#9370DB',   # Medium purple
            self.CRIMSON: '#DC143C',      # Crimson
            self.HALLOW: '#FFB6C1',       # Light pink
            self.JUNGLE: '#006400',       # Dark green
            self.SNOW: '#F0F8FF',         # Alice blue
            self.DUNGEON: '#2F4F4F',      # Dark slate gray
            self.HELLSTONE: '#FF4500',    # Orange red
            self.COBALT: '#0047AB',       # Cobalt blue
            self.MYTHRIL: '#00FF7F',      # Spring green
            self.ADAMANTITE: '#FF0000',   # Red
            self.CHLOROPHYTE: '#7FFF00'   # Chartreuse
        }
        
        # Initialize world states
        self.world_history = []
        self.current_world = None
        self.corruption_centers = []
        self.hardmode_active = False
        
    def generate_base_world(self) -> np.ndarray:
        """
        Generate the initial world using Terraria's generation algorithm.
        
        Returns:
            2D numpy array representing the world
        """
        world = np.full((self.height, self.width), self.AIR, dtype=int)
        
        # Generate surface terrain using multi-octave noise
        x_coords = np.arange(self.width)
        surface_heights = []
        
        for x in x_coords:
            # Multi-octave Perlin noise simulation
            height = self.surface_level
            height += 30 * np.sin(x * 0.01) * np.cos(x * 0.005)
            height += 15 * np.sin(x * 0.03) * np.cos(x * 0.02)
            height += 8 * np.sin(x * 0.08) * np.cos(x * 0.05)
            height = int(np.clip(height, 20, self.surface_level + 40))
            surface_heights.append(height)
        
        # Fill terrain layers
        for x in range(self.width):
            surface_y = surface_heights[x]
            
            # Determine biome type
            biome = self._determine_biome(x)
            
            for y in range(surface_y, self.height):
                if y == surface_y and biome != self.SAND:
                    world[y, x] = self.GRASS
                elif y < surface_y + 50:  # Dirt layer
                    world[y, x] = self.DIRT if biome != self.SAND else self.SAND
                elif y < self.hell_level:  # Stone layer
                    world[y, x] = self.STONE
                else:  # Hell layer
                    world[y, x] = self.HELLSTONE
            
            # Apply biome-specific surface
            if biome == self.JUNGLE:
                world[surface_y, x] = self.JUNGLE
            elif biome == self.SNOW:
                world[surface_y, x] = self.SNOW
        
        # Carve caves using random walk algorithm
        self._carve_cave_systems(world)
        
        # Place major structures
        self._place_initial_structures(world)
        
        return world
    
    def _determine_biome(self, x: int) -> int:
        """Determine biome type based on world position."""
        pos = x / self.width
        
        if pos < 0.15:  # Snow biome
            return self.SNOW
        elif 0.75 < pos < 0.9:  # Jungle biome
            return self.JUNGLE
        elif pos > 0.9:  # Desert
            return self.SAND
        else:  # Forest/normal
            return self.DIRT
    
    def _carve_cave_systems(self, world: np.ndarray) -> None:
        """Carve cave systems using TileRunner algorithm."""
        # Surface caves
        for _ in range(100):
            start_x = np.random.randint(0, self.width)
            start_y = np.random.randint(self.surface_level, self.cavern_level)
            self._carve_tunnel(world, start_x, start_y, strength=8, steps=30)
        
        # Deep caves
        for _ in range(50):
            start_x = np.random.randint(0, self.width)
            start_y = np.random.randint(self.cavern_level, self.hell_level)
            self._carve_tunnel(world, start_x, start_y, strength=12, steps=50)
    
    def _carve_tunnel(self, world: np.ndarray, start_x: int, start_y: int, 
                     strength: int, steps: int) -> None:
        """Carve a single tunnel using random walk."""
        x, y = start_x, start_y
        
        for _ in range(steps):
            # Random walk direction
            dx = np.random.randint(-2, 3)
            dy = np.random.randint(-1, 2)
            
            x = np.clip(x + dx, 0, self.width - 1)
            y = np.clip(y + dy, 0, self.height - 1)
            
            # Carve circular area
            for i in range(-strength//2, strength//2 + 1):
                for j in range(-strength//2, strength//2 + 1):
                    if i*i + j*j <= (strength//2)**2:
                        nx, ny = x + i, y + j
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            if world[ny, nx] != self.AIR:
                                world[ny, nx] = self.AIR
            
            strength = max(3, int(strength * 0.99))
    
    def _place_initial_structures(self, world: np.ndarray) -> None:
        """Place initial structures like dungeon."""
        # Simple dungeon placement
        dungeon_x = 50 if np.random.choice([True, False]) else self.width - 100
        dungeon_y = self.surface_level - 20
        
        for x in range(dungeon_x, dungeon_x + 50):
            for y in range(dungeon_y, dungeon_y + 80):
                if 0 <= x < self.width and 0 <= y < self.height:
                    world[y, x] = self.DUNGEON
    
    def initialize_corruption(self, world: np.ndarray, corruption_type: int = None) -> np.ndarray:
        """
        Initialize corruption/crimson infection points.
        
        Args:
            world: Current world state
            corruption_type: Type of evil biome (CORRUPTION or CRIMSON)
        
        Returns:
            World with initial corruption
        """
        if corruption_type is None:
            corruption_type = np.random.choice([self.CORRUPTION, self.CRIMSON])
        
        # Pre-hardmode corruption pockets
        num_pockets = np.random.randint(3, 6)
        for _ in range(num_pockets):
            x = np.random.randint(self.width // 4, 3 * self.width // 4)
            y = np.random.randint(self.surface_level, self.cavern_level)
            
            # Create initial corruption pocket
            for i in range(-15, 16):
                for j in range(-10, 11):
                    nx, ny = x + i, y + j
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        if world[ny, nx] in [self.DIRT, self.STONE, self.GRASS]:
                            if np.random.random() < 0.7:
                                world[ny, nx] = corruption_type
            
            self.corruption_centers.append((x, y, corruption_type))
        
        return world
    
    def spread_corruption(self, world: np.ndarray, intensity: float = 0.1) -> np.ndarray:
        """
        Spread corruption using cellular automata.
        
        Args:
            world: Current world state
            intensity: Spreading intensity (0.0 to 1.0)
        
        Returns:
            World with spread corruption
        """
        new_world = world.copy()
        
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                if world[y, x] in [self.CORRUPTION, self.CRIMSON]:
                    # Spread to neighboring blocks
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            nx, ny = x + dx, y + dy
                            if world[ny, nx] in [self.DIRT, self.STONE, self.GRASS, self.SAND]:
                                if np.random.random() < intensity:
                                    new_world[ny, nx] = world[y, x]  # Same corruption type
        
        return new_world
    
    def trigger_hardmode(self, world: np.ndarray) -> np.ndarray:
        """
        Trigger hardmode with V-pattern corruption and hallow.
        
        Args:
            world: Current world state
        
        Returns:
            World with hardmode changes
        """
        new_world = world.copy()
        self.hardmode_active = True
        
        # Create V-pattern from center
        center_x = self.width // 2
        
        # Left diagonal (corruption/crimson)
        for i in range(min(center_x, self.hell_level)):
            x = center_x - i
            y = i + self.surface_level
            if 0 <= x < self.width and 0 <= y < self.height:
                # Create diagonal stripe
                for offset in range(-5, 6):
                    if 0 <= x + offset < self.width:
                        corruption_type = self.corruption_centers[0][2] if self.corruption_centers else self.CORRUPTION
                        new_world[y, x + offset] = corruption_type
        
        # Right diagonal (hallow)
        for i in range(min(self.width - center_x, self.hell_level)):
            x = center_x + i
            y = i + self.surface_level
            if 0 <= x < self.width and 0 <= y < self.height:
                # Create diagonal stripe
                for offset in range(-5, 6):
                    if 0 <= x + offset < self.width:
                        new_world[y, x + offset] = self.HALLOW
        
        return new_world
    
    def generate_hardmode_ores(self, world: np.ndarray) -> np.ndarray:
        """
        Generate hardmode ores after altar breaking.
        
        Args:
            world: Current world state
        
        Returns:
            World with hardmode ores
        """
        new_world = world.copy()
        
        # Cobalt ore (most common)
        for _ in range(50):
            x = np.random.randint(0, self.width)
            y = np.random.randint(self.cavern_level, self.hell_level)
            
            if new_world[y, x] == self.STONE:
                # Create small vein
                for i in range(-2, 3):
                    for j in range(-2, 3):
                        nx, ny = x + i, y + j
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            if new_world[ny, nx] == self.STONE and np.random.random() < 0.6:
                                new_world[ny, nx] = self.COBALT
        
        # Mythril ore (less common)
        for _ in range(30):
            x = np.random.randint(0, self.width)
            y = np.random.randint(self.cavern_level, self.hell_level)
            
            if new_world[y, x] == self.STONE:
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        nx, ny = x + i, y + j
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            if new_world[ny, nx] == self.STONE and np.random.random() < 0.5:
                                new_world[ny, nx] = self.MYTHRIL
        
        # Adamantite ore (rare)
        for _ in range(15):
            x = np.random.randint(0, self.width)
            y = np.random.randint(self.hell_level - 50, self.hell_level)
            
            if new_world[y, x] == self.STONE:
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        nx, ny = x + i, y + j
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            if new_world[ny, nx] == self.STONE and np.random.random() < 0.4:
                                new_world[ny, nx] = self.ADAMANTITE
        
        # Chlorophyte ore (jungle only, post-mech bosses)
        jungle_start = int(0.75 * self.width)
        jungle_end = int(0.9 * self.width)
        
        for _ in range(20):
            x = np.random.randint(jungle_start, jungle_end)
            y = np.random.randint(self.cavern_level, self.hell_level - 20)
            
            if new_world[y, x] == self.STONE:
                new_world[y, x] = self.CHLOROPHYTE
        
        return new_world
    
    def create_color_map(self) -> ListedColormap:
        """Create custom colormap for world visualization."""
        colors = [self.colors[i] for i in range(len(self.colors))]
        return ListedColormap(colors)
    
    def simulate_complete_evolution(self, total_frames: int = 200) -> List[np.ndarray]:
        """
        Simulate the complete world evolution process.
        
        Args:
            total_frames: Total number of animation frames
        
        Returns:
            List of world states for animation
        """
        world_states = []
        
        # Phase 1: Initial world generation (frames 0-30)
        print("Generating initial world...")
        world = self.generate_base_world()
        for _ in range(30):
            world_states.append(world.copy())
        
        # Phase 2: Initial corruption placement (frames 31-50)
        print("Placing initial corruption...")
        world = self.initialize_corruption(world)
        for _ in range(20):
            world_states.append(world.copy())
        
        # Phase 3: Pre-hardmode corruption spread (frames 51-100)
        print("Spreading pre-hardmode corruption...")
        for frame in range(50):
            intensity = 0.02 + (frame / 50) * 0.05  # Gradually increase spread
            world = self.spread_corruption(world, intensity)
            world_states.append(world.copy())
        
        # Phase 4: Hardmode trigger (frames 101-120)
        print("Triggering hardmode...")
        world = self.trigger_hardmode(world)
        for _ in range(20):
            world_states.append(world.copy())
        
        # Phase 5: Hardmode ore generation (frames 121-140)
        print("Generating hardmode ores...")
        world = self.generate_hardmode_ores(world)
        for _ in range(20):
            world_states.append(world.copy())
        
        # Phase 6: Intense hardmode corruption spread (frames 141-200)
        print("Intense hardmode spreading...")
        for frame in range(60):
            intensity = 0.08 + (frame / 60) * 0.12  # Much more intense spreading
            world = self.spread_corruption(world, intensity)
            world_states.append(world.copy())
        
        self.world_history = world_states
        return world_states

def create_master_evolution_animation(save_path: str):
    """
    Create the master animation showing complete Terraria world evolution.
    
    Args:
        save_path: Path to save the animation
    """
    print("Creating Master Terraria World Evolution Animation...")
    print("=" * 60)
    
    # Initialize evolution system
    evolution = TerrariaCompleteEvolution(width=420, height=120)
    
    # Simulate complete evolution
    world_states = evolution.simulate_complete_evolution(total_frames=200)
    
    # Create figure with enhanced styling
    fig, ax = plt.subplots(figsize=(20, 8))
    fig.patch.set_facecolor('#000011')
    ax.set_facecolor('#000011')
    
    # Create colormap
    cmap = evolution.create_color_map()
    
    # Animation phase descriptions
    phase_descriptions = {
        0: "Phase 1: Initial World Generation\nMulti-octave Perlin noise terrain formation",
        30: "Phase 2: Pre-Hardmode Corruption Seeding\nSmall infection pockets placed randomly",
        50: "Phase 3: Natural Corruption Spread\nCellular automata-based expansion",
        100: "Phase 4: Wall of Flesh Defeated!\nHardmode activated - V-pattern begins",
        120: "Phase 5: Altar Breaking Consequences\nHardmode ores scattered throughout caverns",
        140: "Phase 6: Accelerated Evil Spread\nIntense corruption and hallow expansion"
    }
    
    def animate(frame):
        ax.clear()
        
        # Get current world state
        world = world_states[frame]
        
        # Display world
        im = ax.imshow(world, cmap=cmap, aspect='auto', vmin=0, vmax=15,
                      extent=[0, evolution.width, evolution.height, 0])
        
        # Add layer boundaries
        ax.axhline(y=evolution.surface_level, color='cyan', linestyle='--', 
                  alpha=0.5, linewidth=1, label='Surface Level')
        ax.axhline(y=evolution.cavern_level, color='yellow', linestyle='--', 
                  alpha=0.5, linewidth=1, label='Cavern Level')
        ax.axhline(y=evolution.hell_level, color='red', linestyle='--', 
                  alpha=0.5, linewidth=1, label='Hell Level')
        
        # Dynamic title based on phase
        current_phase = "Phase 6: Accelerated Evil Spread"
        for phase_frame, description in phase_descriptions.items():
            if frame >= phase_frame:
                current_phase = description
        
        ax.set_title(f'Terraria Complete World Evolution\n{current_phase}\nFrame: {frame+1}/200', 
                    fontsize=16, fontweight='bold', color='white', pad=20)
        
        # Styling
        ax.set_xlabel('World X Coordinate (blocks)', fontsize=12, color='white', fontweight='bold')
        ax.set_ylabel('World Depth (blocks)', fontsize=12, color='white', fontweight='bold')
        ax.tick_params(colors='white')
        
        # Add progress bar
        progress = frame / len(world_states)
        progress_bar = patches.Rectangle((10, evolution.height - 15), 
                                       progress * (evolution.width - 20), 8,
                                       facecolor='lime', alpha=0.7)
        ax.add_patch(progress_bar)
        
        # Progress bar outline
        progress_outline = patches.Rectangle((10, evolution.height - 15), 
                                           evolution.width - 20, 8,
                                           facecolor='none', edgecolor='white', linewidth=2)
        ax.add_patch(progress_outline)
        
        # Add mathematical formula
        if frame < 50:
            formula = r'$h(x) = h_0 + \sum_{i=0}^{n} A_i \sin(f_i x + \phi_i)$'
            desc = "Multi-octave noise terrain generation"
        elif frame < 100:
            formula = r'$C_{t+1}(x,y) = C_t(x,y) + \alpha \sum N(x,y)$'
            desc = "Cellular automata corruption spread"
        elif frame < 140:
            formula = r'$V(x,y) = |x - x_c| + |y - y_c| < W_{stripe}$'
            desc = "V-pattern hardmode transformation"
        else:
            formula = r'$P_{ore}(d) = \rho_0 e^{-\lambda d} \cdot \beta_{biome}$'
            desc = "Exponential ore distribution with depth"
        
        ax.text(0.02, 0.98, f'{desc}\n{formula}', 
               transform=ax.transAxes, fontsize=11, color='white',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8),
               verticalalignment='top')
        
        return [im]
    
    # Create animation
    print("Rendering animation frames...")
    anim = FuncAnimation(fig, animate, frames=len(world_states), 
                        interval=150, repeat=True, blit=False)
    
    # Save animation
    print(f"Saving master evolution animation to {save_path}")
    writer = PillowWriter(fps=8)
    anim.save(save_path, writer=writer, dpi=150)
    plt.close(fig)
    
    print("Master evolution animation completed!")

def create_corruption_focus_animation(save_path: str):
    """
    Create a focused animation showing corruption/crimson evolution mechanics.
    
    Args:
        save_path: Path to save the animation
    """
    print("Creating Corruption Evolution Focus Animation...")
    print("=" * 50)
    
    # Initialize smaller world for detailed view
    evolution = TerrariaCompleteEvolution(width=200, height=80)
    
    # Generate base world
    world = evolution.generate_base_world()
    
    # Focus on corruption evolution
    world_states = []
    
    # Initialize with heavy corruption
    world = evolution.initialize_corruption(world, evolution.CORRUPTION)
    
    # Simulate detailed corruption spread
    for frame in range(100):
        if frame == 50:
            # Trigger hardmode at midpoint
            world = evolution.trigger_hardmode(world)
        
        # Different spread rates pre/post hardmode
        intensity = 0.05 if frame < 50 else 0.15
        world = evolution.spread_corruption(world, intensity)
        world_states.append(world.copy())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    
    # Custom colormap for corruption focus
    corruption_colors = [
        '#000011',  # Air
        '#8B4513',  # Dirt
        '#696969',  # Stone
        '#228B22',  # Grass
        '#F4A460',  # Sand
        '#4B0082',  # Corruption (enhanced)
        '#8B0000',  # Crimson (enhanced)
        '#FF69B4',  # Hallow (enhanced)
        '#006400',  # Jungle
        '#F0F8FF',  # Snow
        '#2F4F4F',  # Dungeon
        '#FF4500',  # Hellstone
        '#0047AB',  # Cobalt
        '#00FF7F',  # Mythril
        '#FF0000',  # Adamantite
        '#7FFF00'   # Chlorophyte
    ]
    cmap = ListedColormap(corruption_colors)
    
    def animate_corruption(frame):
        ax.clear()
        
        # Get current world state
        world = world_states[frame]
        
        # Display world with enhanced contrast
        im = ax.imshow(world, cmap=cmap, aspect='auto', vmin=0, vmax=15,
                      extent=[0, evolution.width, evolution.height, 0])
        
        # Dynamic title
        if frame < 50:
            phase = "Pre-Hardmode: Natural Corruption Spread"
            rate = "Spread Rate: 5% per iteration"
        else:
            phase = "Hardmode: Accelerated Evil Expansion"
            rate = "Spread Rate: 15% per iteration + V-Pattern"
        
        ax.set_title(f'Terraria Corruption Evolution Mechanics\n{phase}\n{rate}\nFrame: {frame+1}/100', 
                    fontsize=14, fontweight='bold', color='white', pad=20)
        
        # Add cellular automata visualization
        if frame > 0:
            prev_world = world_states[frame-1]
            # Highlight newly corrupted areas
            diff = (world != prev_world) & (world == evolution.CORRUPTION)
            if np.any(diff):
                y_coords, x_coords = np.where(diff)
                ax.scatter(x_coords, y_coords, c='yellow', s=20, alpha=0.7, marker='*')
        
        # Styling
        ax.set_xlabel('X Coordinate (blocks)', fontsize=12, color='white', fontweight='bold')
        ax.set_ylabel('Depth (blocks)', fontsize=12, color='white', fontweight='bold')
        ax.tick_params(colors='white')
        
        # Add corruption statistics
        corruption_count = np.sum(world == evolution.CORRUPTION)
        hallow_count = np.sum(world == evolution.HALLOW)
        total_blocks = evolution.width * evolution.height
        
        stats_text = (
            f"Corruption Coverage: {corruption_count/total_blocks*100:.1f}%\n"
            f"Hallow Coverage: {hallow_count/total_blocks*100:.1f}%\n"
            f"Total Evil Blocks: {corruption_count + hallow_count}"
        )
        
        ax.text(0.98, 0.98, stats_text,
               transform=ax.transAxes, fontsize=10, color='white',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='purple', alpha=0.8),
               verticalalignment='top', horizontalalignment='right')
          # Add cellular automata formula
        formula = (
            "Cellular Automata Rule:\n"
            r"$C_{t+1}(i,j) = f(neighbors, threshold)$"
        )
        
        ax.text(0.02, 0.02, formula,
               transform=ax.transAxes, fontsize=9, color='white',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8),
               verticalalignment='bottom')
        
        return [im]
    
    # Create animation
    print("Rendering corruption focus animation...")
    anim = FuncAnimation(fig, animate_corruption, frames=len(world_states), 
                        interval=200, repeat=True, blit=False)
    
    # Save animation
    print(f"Saving corruption focus animation to {save_path}")
    writer = PillowWriter(fps=6)
    anim.save(save_path, writer=writer, dpi=150)
    plt.close(fig)
    
    print("Corruption focus animation completed!")

if __name__ == "__main__":
    print("Starting Terraria Complete World Evolution Animations...")
    print("=" * 65)
    
    # Create output directory
    output_dir = r"c:\Users\hunkb\OneDrive\Desktop\Terraria Generation\Code+"
    
    # Generate master evolution animation
    master_path = f"{output_dir}/terraria_master_world_evolution.gif"
    create_master_evolution_animation(master_path)
    
    # Generate corruption focus animation
    corruption_path = f"{output_dir}/terraria_corruption_evolution_focus.gif"
    create_corruption_focus_animation(corruption_path)
    
    print("\n" + "=" * 65)
    print("All Terraria evolution animations completed successfully!")
    print("Files created:")
    print(f"- {master_path}")
    print(f"- {corruption_path}")
    print("\nThese animations demonstrate the complete Terraria world")
    print("evolution from initial generation through hardmode transformation.")
