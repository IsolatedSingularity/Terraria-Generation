"""
Terraria World Generation Visualization
=====================================

This module creates comprehensive visualizations of Terraria's 103-pass world generation system,
showing the step-by-step construction of terrain, biomes, and structures.

Mathematical Foundation:
- Surface generation: height(x) = base + Σ(amplitude_i × noise(x × frequency_i))
- Cave carving: TileRunner algorithm with random walk patterns
- Biome placement: Rule-based positioning with distance constraints
- Structure generation: Probabilistic placement with spatial validation

Author: Terraria Generation Project
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import os
import warnings
warnings.filterwarnings('ignore')

# Set seaborn style and preferred color palettes
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Custom color palettes as requested
MAKO_CMAP = sns.color_palette("mako", as_cmap=True)
CUBEHELIX_CMAP = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=True, as_cmap=True)
ROCKET_CMAP = sns.color_palette("rocket", as_cmap=True)

class TerrariaWorldGenerator:
    """
    Simulates Terraria's world generation process using mathematical models
    based on the 103-pass generation system.
    """
    
    def __init__(self, world_width=8400, world_height=2400, seed=42):
        """
        Initialize world generator for large world dimensions.
        
        Args:
            world_width: World width in blocks (8400 for large world)
            world_height: World height in blocks (2400 for large world)
            seed: Random seed for deterministic generation
        """
        np.random.seed(seed)
        self.world_width = world_width
        self.world_height = world_height
        self.surface_level = world_height // 4
        self.cavern_level = int(world_height * 0.6)
        self.hell_level = int(world_height * 0.85)
        
        # Initialize world grid
        self.world = np.zeros((world_height, world_width), dtype=int)
        self.generation_stages = []
        
        # Block type definitions
        self.block_types = {
            0: 'Air',
            1: 'Dirt', 
            2: 'Stone',
            3: 'Grass',
            4: 'Sand',
            5: 'Corruption',
            6: 'Crimson',
            7: 'Hallow',
            8: 'Jungle',
            9: 'Snow',
            10: 'Dungeon',
            11: 'Hell'
        }
        
        # Color mapping for visualization
        self.block_colors = {
            0: '#87CEEB',   # Air - Sky Blue
            1: '#8B4513',   # Dirt - Saddle Brown
            2: '#696969',   # Stone - Dim Gray
            3: '#90EE90',   # Grass - Light Green
            4: '#F4A460',   # Sand - Sandy Brown
            5: '#9370DB',   # Corruption - Medium Purple
            6: '#DC143C',   # Crimson - Crimson
            7: '#FFB6C1',   # Hallow - Light Pink
            8: '#228B22',   # Jungle - Forest Green
            9: '#F0F8FF',   # Snow - Alice Blue
            10: '#2F4F4F',  # Dungeon - Dark Slate Gray
            11: '#8B0000'   # Hell - Dark Red
        }
    
    def simple_noise(self, x, y=0, frequency=0.01, octaves=4, amplitude=1.0):
        """
        Simple noise implementation using trigonometric functions.
        
        Args:
            x, y: Coordinates
            frequency: Base frequency
            octaves: Number of noise layers
            amplitude: Maximum amplitude
            
        Returns:
            Noise value between -amplitude and amplitude
        """
        value = 0
        current_amplitude = amplitude
        current_frequency = frequency
        
        for _ in range(octaves):
            value += current_amplitude * np.sin(x * current_frequency * 2 * np.pi)
            if y != 0:
                value += current_amplitude * np.cos(y * current_frequency * 2 * np.pi)
            current_amplitude *= 0.5
            current_frequency *= 2
        
        return value
    
    def generate_surface_terrain(self):
        """
        Generate surface terrain using 1D noise for height variation.
        Pass 1-5: Terrain formation and basic height map.
        """
        print("Generating surface terrain...")
        
        # Generate height map
        x_coords = np.arange(self.world_width)
        heights = []
        
        for x in x_coords:
            # Multi-octave noise for natural terrain
            height = self.surface_level
            height += self.simple_noise(x, frequency=0.01, octaves=4, amplitude=30)
            height += self.simple_noise(x, frequency=0.05, octaves=2, amplitude=10)
            height = int(np.clip(height, 50, self.surface_level + 50))
            heights.append(height)
        
        # Fill terrain
        for x in range(self.world_width):
            surface_y = heights[x]
            
            # Fill from surface to bottom
            for y in range(surface_y, self.world_height):
                if y < surface_y + 50:
                    self.world[y, x] = 1  # Dirt layer
                else:
                    self.world[y, x] = 2  # Stone layer
            
            # Add grass on surface
            if surface_y > 0:
                self.world[surface_y, x] = 3
        
        # Store this stage
        self.generation_stages.append(('Surface Terrain', self.world.copy()))
    
    def carve_caves(self):
        """
        Carve cave systems using random walk algorithm (TileRunner).
        Pass 6-25: Cave generation with increasing size by depth.
        """
        print("Carving cave systems...")
        
        # Small caves in dirt layer
        for _ in range(200):
            start_x = np.random.randint(0, self.world_width)
            start_y = np.random.randint(self.surface_level, self.cavern_level)
            self._carve_tunnel(start_x, start_y, strength=8, steps=50)
        
        # Large caves in stone layer
        for _ in range(100):
            start_x = np.random.randint(0, self.world_width)
            start_y = np.random.randint(self.cavern_level, self.hell_level)
            self._carve_tunnel(start_x, start_y, strength=15, steps=100)
        
        self.generation_stages.append(('Cave Systems', self.world.copy()))
    
    def _carve_tunnel(self, start_x, start_y, strength, steps):
        """
        Carve a single tunnel using random walk algorithm.
        
        Args:
            start_x, start_y: Starting coordinates
            strength: Tunnel radius
            steps: Number of steps to take
        """
        x, y = start_x, start_y
        
        for _ in range(steps):
            # Random walk direction
            dx = np.random.randint(-2, 3)
            dy = np.random.randint(-1, 2)
            
            x = np.clip(x + dx, 0, self.world_width - 1)
            y = np.clip(y + dy, 0, self.world_height - 1)
            
            # Carve circular area
            for i in range(-strength//2, strength//2 + 1):
                for j in range(-strength//2, strength//2 + 1):
                    if i*i + j*j <= (strength//2)**2:
                        nx, ny = x + i, y + j
                        if 0 <= nx < self.world_width and 0 <= ny < self.world_height:
                            if self.world[ny, nx] != 0:  # Don't carve air
                                self.world[ny, nx] = 0  # Air
            
            # Gradually reduce strength
            strength = max(3, int(strength * 0.98))
    
    def place_biomes(self):
        """
        Place major biomes according to Terraria's rules.
        Pass 26-45: Biome placement and terrain conversion.
        """
        print("Placing biomes...")
        
        # Determine dungeon side (left or right)
        dungeon_left = np.random.choice([True, False])
        
        # Snow biome (same side as dungeon)
        if dungeon_left:
            snow_start = 0
            snow_end = self.world_width // 6
            jungle_start = 4 * self.world_width // 6
            jungle_end = self.world_width
        else:
            snow_start = 5 * self.world_width // 6
            snow_end = self.world_width
            jungle_start = 0
            jungle_end = self.world_width // 6
        
        # Convert snow biome
        for x in range(snow_start, snow_end):
            for y in range(self.world_height):
                if self.world[y, x] == 1:  # Dirt to snow
                    self.world[y, x] = 9
                elif self.world[y, x] == 3:  # Grass to snow
                    self.world[y, x] = 9
        
        # Convert jungle biome
        for x in range(jungle_start, jungle_end):
            for y in range(self.surface_level, self.hell_level):
                if self.world[y, x] == 1:  # Dirt to jungle
                    self.world[y, x] = 8
                elif self.world[y, x] == 3:  # Grass to jungle
                    self.world[y, x] = 8
        
        # Evil biome (corruption or crimson)
        evil_type = np.random.choice([5, 6])  # Corruption or Crimson
        evil_center = self.world_width // 3 if dungeon_left else 2 * self.world_width // 3
        evil_width = 200
        
        for x in range(max(0, evil_center - evil_width), min(self.world_width, evil_center + evil_width)):
            for y in range(0, self.cavern_level):
                if self.world[y, x] in [1, 2, 3]:
                    if np.random.random() < 0.7:  # 70% conversion rate
                        self.world[y, x] = evil_type
        
        # Carve evil chasms
        for _ in range(5):
            chasm_x = np.random.randint(max(0, evil_center - evil_width), 
                                       min(self.world_width, evil_center + evil_width))
            self._carve_chasm(chasm_x)
        
        self.generation_stages.append(('Biome Placement', self.world.copy()))
    
    def _carve_chasm(self, x):
        """
        Carve a vertical chasm for corruption/crimson.
        
        Args:
            x: X coordinate for the chasm
        """
        # Find surface
        surface_y = 0
        for y in range(self.world_height):
            if self.world[y, x] != 0:
                surface_y = y
                break
        
        # Carve downward
        current_x = x
        for y in range(surface_y, min(self.cavern_level, surface_y + 150)):
            # Add some horizontal variation
            if np.random.random() < 0.1:
                current_x += np.random.randint(-2, 3)
                current_x = np.clip(current_x, 0, self.world_width - 1)
            
            # Carve width of 5-8 blocks
            width = np.random.randint(5, 9)
            for i in range(-width//2, width//2 + 1):
                nx = current_x + i
                if 0 <= nx < self.world_width:
                    self.world[y, nx] = 0
    
    def place_structures(self):
        """
        Place major structures like dungeons, temples, etc.
        Pass 46-60: Structure generation.
        """
        print("Placing structures...")
        
        # Dungeon (simplified as a rectangular structure)
        dungeon_x = 100 if np.random.choice([True, False]) else self.world_width - 200
        dungeon_y = self.surface_level - 50
        
        for x in range(dungeon_x, dungeon_x + 100):
            for y in range(dungeon_y, dungeon_y + 200):
                if 0 <= x < self.world_width and 0 <= y < self.world_height:
                    self.world[y, x] = 10
        
        # Hell layer
        for x in range(self.world_width):
            for y in range(self.hell_level, self.world_height):
                if self.world[y, x] == 2:  # Stone to hell stone
                    self.world[y, x] = 11
        
        self.generation_stages.append(('Structure Placement', self.world.copy()))
    
    def generate_world(self):
        """
        Execute the complete world generation process.
        
        Returns:
            List of generation stages for animation
        """
        self.generate_surface_terrain()
        self.carve_caves()
        self.place_biomes()
        self.place_structures()
        
        return self.generation_stages

def create_world_generation_animation(save_path):
    """
    Create an animation showing the step-by-step world generation process.
    """
    print("Creating world generation animation...")
    
    # Generate world data
    generator = TerrariaWorldGenerator(world_width=840, world_height=240)  # Scaled down for performance
    stages = generator.generate_world()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_title('Terraria World Generation Process', fontsize=16, fontweight='bold')
    ax.set_xlabel('World X Coordinate (blocks)', fontsize=12)
    ax.set_ylabel('World Y Coordinate (blocks)', fontsize=12)
    
    # Color mapping
    colors = list(generator.block_colors.values())
    cmap = LinearSegmentedColormap.from_list('terraria', colors, N=len(colors))
    
    def animate(frame):
        ax.clear()
        stage_name, world_data = stages[frame % len(stages)]
        
        # Display world
        im = ax.imshow(world_data, cmap=cmap, aspect='auto', 
                      vmin=0, vmax=len(generator.block_types)-1)
        
        ax.set_title(f'Terraria World Generation: {stage_name}', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('World X Coordinate (blocks)', fontsize=12)
        ax.set_ylabel('World Y Coordinate (blocks)', fontsize=12)
        
        # Add generation pass information
        pass_info = {
            'Surface Terrain': 'Passes 1-5: Height map generation using multi-octave noise',
            'Cave Systems': 'Passes 6-25: TileRunner algorithm creates cave networks',
            'Biome Placement': 'Passes 26-45: Rule-based biome conversion and placement',
            'Structure Placement': 'Passes 46-60: Dungeon, temple, and hell layer generation'
        }
        
        ax.text(0.02, 0.98, pass_info.get(stage_name, ''), 
               transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               verticalalignment='top')
        
        return [im]
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=len(stages)*3, 
                                  interval=2000, repeat=True)
    
    # Save animation
    print(f"Saving world generation animation to {save_path}")
    anim.save(save_path, writer='pillow', fps=1, dpi=100)
    plt.close(fig)

def create_generation_comparison_static(save_path):
    """
    Create a static comparison showing all generation stages side by side.
    """
    print("Creating generation comparison visualization...")
    
    # Generate world data
    generator = TerrariaWorldGenerator(world_width=420, world_height=120)  # Smaller for display
    stages = generator.generate_world()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Terraria World Generation Stages', fontsize=18, fontweight='bold')
    
    # Color mapping
    colors = list(generator.block_colors.values())
    cmap = LinearSegmentedColormap.from_list('terraria', colors, N=len(colors))
    
    # Plot each stage
    for idx, (stage_name, world_data) in enumerate(stages):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]
        
        im = ax.imshow(world_data, cmap=cmap, aspect='auto',
                      vmin=0, vmax=len(generator.block_types)-1)
        ax.set_title(stage_name, fontsize=14, fontweight='bold')
        ax.set_xlabel('X Coordinate', fontsize=10)
        ax.set_ylabel('Y Coordinate', fontsize=10)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.6, aspect=30)
    cbar.set_label('Block Type', fontsize=12)
    
    # Add legend for block types
    legend_text = '\n'.join([f'{k}: {v}' for k, v in generator.block_types.items()])
    fig.text(0.02, 0.02, f'Block Types:\n{legend_text}', 
             fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    print("Starting Terraria world generation visualizations...")
    
    # Create output directory
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Generate visualizations
    create_world_generation_animation(os.path.join(output_dir, 'world_generation_process.gif'))
    create_generation_comparison_static(os.path.join(output_dir, 'world_generation_stages.png'))
    
    print("World generation visualizations completed!")
