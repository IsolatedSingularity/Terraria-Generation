"""
Terraria Hardmode Structure Transformation System
================================================

This module provides comprehensive visualization and analysis of Terraria's hardmode
structural changes, including altar breaking mechanics, ore generation patterns,
mechanical boss arenas, and environmental transformations that occur during
the hardmode transition.

Mathematical Foundation:
- Poisson distribution for ore scatter: P(k;λ) = (λᵏe^(-λ))/k!
- Spatial clustering algorithms for structure placement
- Probability density functions for altar breaking effects
- Environmental transformation matrices
- Boss arena geometric calculations

Key Hardmode Changes Modeled:
1. Demon/Crimson Altar breaking and ore generation
2. New enemy spawn mechanics and rates
3. Mechanical boss arena requirements
4. Underground corruption/hallow spread acceleration
5. New structure generation (crystal hearts, etc.)

Author: Generated for Terraria Generation Analysis
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from scipy.spatial.distance import cdist
from scipy.stats import poisson, norm
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from typing import Tuple, List, Dict, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Set visualization preferences
plt.style.use('dark_background')
sns.set_palette("cubehelix")

class TerrariaHardmodeTransformation:
    """
    Comprehensive system for modeling and visualizing the structural and mechanical
    changes that occur during Terraria's hardmode transition, including ore generation,
    altar mechanics, boss arena requirements, and environmental transformations.
    
    The system accurately models:
    - Altar breaking mechanics and ore distribution
    - Mechanical boss spawning requirements
    - Environmental hazard placement
    - Structure transformation algorithms
    - Player interaction consequences
    """
    
    def __init__(self, world_width: int = 8400, world_height: int = 2400):
        """
        Initialize the hardmode transformation system.
        
        Parameters:
        -----------
        world_width : int
            World width in blocks (8400 for large world)
        world_height : int  
            World height in blocks (2400 for large world)
        """
        self.width = world_width
        self.height = world_height
        
        # Block type definitions
        self.EMPTY = 0
        self.DIRT = 1
        self.STONE = 2
        self.CORRUPTION = 3
        self.CRIMSON = 4
        self.HALLOW = 5
        self.COBALT = 6
        self.MYTHRIL = 7
        self.ADAMANTITE = 8
        self.ALTAR = 9
        self.CRYSTAL_HEART = 10
        self.BOSS_ARENA = 11
        self.CHLOROPHYTE = 12
          # Hardmode ore generation parameters (reduced for faster processing)
        self.ore_generation_params = {
            self.COBALT: {'density': 0.04, 'cluster_size': 6, 'depth_range': (0.3, 0.8)},
            self.MYTHRIL: {'density': 0.03, 'cluster_size': 7, 'depth_range': (0.4, 0.9)},
            self.ADAMANTITE: {'density': 0.02, 'cluster_size': 8, 'depth_range': (0.5, 1.0)},
            self.CHLOROPHYTE: {'density': 0.01, 'cluster_size': 4, 'depth_range': (0.7, 1.0)}
        }
        
        # Altar breaking mechanics
        self.altars_per_breaking = 3  # Number of altars broken per wave
        self.ore_per_altar = {
            1: self.COBALT,    # First wave
            2: self.MYTHRIL,   # Second wave  
            3: self.ADAMANTITE # Third wave
        }
        
        # Boss arena specifications
        self.boss_arenas = {
            'destroyer': {'width': 800, 'height': 400, 'platform_spacing': 50},
            'twins': {'width': 1000, 'height': 600, 'flight_space': True},
            'skeletron_prime': {'width': 600, 'height': 500, 'cover_needed': True}
        }
        
        # Color mapping for visualization
        self.colors = {
            self.EMPTY: (0.05, 0.05, 0.05),
            self.DIRT: (0.4, 0.3, 0.2),
            self.STONE: (0.5, 0.5, 0.5),
            self.CORRUPTION: (0.3, 0.0, 0.5),
            self.CRIMSON: (0.7, 0.0, 0.2),
            self.HALLOW: (1.0, 0.8, 1.0),
            self.COBALT: (0.0, 0.3, 0.8),
            self.MYTHRIL: (0.2, 0.8, 0.2),
            self.ADAMANTITE: (0.8, 0.2, 0.2),
            self.ALTAR: (0.6, 0.1, 0.1),
            self.CRYSTAL_HEART: (1.0, 0.2, 0.8),
            self.BOSS_ARENA: (0.9, 0.9, 0.0),
            self.CHLOROPHYTE: (0.0, 1.0, 0.0)
        }
        
        self.world = None
        self.altars_broken = 0
        self.transformation_history = []
        
    def generate_prehardmode_world(self) -> np.ndarray:
        """
        Generate a pre-hardmode world with altars and basic structures.
        
        Returns:
        --------
        np.ndarray
            2D array representing the pre-hardmode world
        """
        world = np.full((self.height, self.width), self.EMPTY, dtype=int)
        
        # Generate terrain
        surface_noise = self._generate_terrain_noise()
        surface_height = (surface_noise * 300 + self.height // 3).astype(int)
        
        # Fill world layers
        for x in range(self.width):
            surface_y = surface_height[x]
            
            for y in range(surface_y, self.height):
                if y < surface_y + 100:  # Dirt layer
                    world[y, x] = self.DIRT
                else:  # Stone layer
                    world[y, x] = self.STONE
        
        # Place demon/crimson altars
        self._place_altars(world)
        
        return world
    
    def _generate_terrain_noise(self) -> np.ndarray:
        """Generate terrain using multi-octave noise."""
        x = np.linspace(0, 10 * np.pi, self.width)
        
        terrain = (
            0.4 * np.sin(x * 0.3) +
            0.3 * np.sin(x * 0.8 + 2.1) +
            0.2 * np.sin(x * 1.5 + 4.2) +
            0.1 * np.sin(x * 3.2 + 1.8)
        )
        
        return terrain
    
    def _place_altars(self, world: np.ndarray) -> None:
        """
        Place demon/crimson altars throughout the underground.
        
        Altars spawn in the cavern layer with specific spacing requirements.
        """
        # Altar placement parameters
        min_spacing = 200  # Minimum distance between altars
        num_altars = 12   # Total altars in world
        
        placed_altars = []
        attempts = 0
        max_attempts = 1000
        
        while len(placed_altars) < num_altars and attempts < max_attempts:
            # Random placement in cavern layer
            x = np.random.randint(100, self.width - 100)
            y = np.random.randint(self.height // 2, int(self.height * 0.8))
            
            # Check if location is valid (stone block)
            if world[y, x] == self.STONE:
                # Check minimum spacing
                valid_location = True
                for altar_x, altar_y in placed_altars:
                    distance = np.sqrt((x - altar_x)**2 + (y - altar_y)**2)
                    if distance < min_spacing:
                        valid_location = False
                        break
                
                if valid_location:
                    world[y, x] = self.ALTAR
                    placed_altars.append((x, y))
            
            attempts += 1
        
        print(f"Placed {len(placed_altars)} altars in the world")
    
    def break_altars(self, num_waves: int = 3) -> None:
        """
        Simulate breaking demon/crimson altars to generate hardmode ores.
        
        Breaking altars follows specific rules:
        - Every 3 altars broken generates new ore type
        - Each wave spawns different ore (Cobalt → Mythril → Adamantite)
        - Ore spawns use Poisson distribution with environmental clustering
        
        Parameters:
        -----------
        num_waves : int
            Number of altar breaking waves (typically 3)
        """
        print("Breaking altars to generate hardmode ores...")
        
        for wave in range(1, num_waves + 1):
            ore_type = self.ore_per_altar.get(wave, self.COBALT)
            
            # Break altars for this wave
            altars_to_break = min(self.altars_per_breaking, 
                                self._count_remaining_altars())
            
            if altars_to_break > 0:
                self._break_altar_wave(altars_to_break)
                self._generate_ore_wave(ore_type, altars_to_break)
                
                print(f"Wave {wave}: Broke {altars_to_break} altars, "
                      f"generated {ore_type} ore")
            else:
                print(f"Wave {wave}: No altars remaining to break")
    
    def _count_remaining_altars(self) -> int:
        """Count the number of altars still present in the world."""
        return np.sum(self.world == self.ALTAR)
    
    def _break_altar_wave(self, num_altars: int) -> None:
        """Break a specified number of altars randomly."""
        altar_positions = np.where(self.world == self.ALTAR)
        altar_indices = list(zip(altar_positions[0], altar_positions[1]))
        
        # Randomly select altars to break
        if len(altar_indices) >= num_altars:
            broken_altars = np.random.choice(len(altar_indices), 
                                           size=num_altars, replace=False)
            
            for idx in broken_altars:
                y, x = altar_indices[idx]
                self.world[y, x] = self.STONE  # Replace with stone
                self.altars_broken += 1
    
    def _generate_ore_wave(self, ore_type: int, num_altars: int) -> None:
        """
        Generate ore clusters based on altar breaking.
        
        Uses Poisson distribution for scatter placement with environmental
        clustering around suitable stone formations.
        
        Parameters:
        -----------
        ore_type : int
            Type of ore to generate
        num_altars : int
            Number of altars broken (affects ore quantity)
        """
        params = self.ore_generation_params[ore_type]
        
        # Calculate ore clusters to generate
        base_clusters = num_altars * 5  # 5 clusters per altar broken
        λ = base_clusters * params['density']
        num_clusters = poisson.rvs(λ)
        
        # Get depth range for this ore
        min_depth = int(self.height * params['depth_range'][0])
        max_depth = int(self.height * params['depth_range'][1])
        
        clusters_placed = 0
        for _ in range(num_clusters * 2):  # Allow extra attempts
            # Random placement within depth range
            x = np.random.randint(50, self.width - 50)
            y = np.random.randint(min_depth, max_depth)
            
            # Only place in stone
            if self.world[y, x] == self.STONE:
                self._create_ore_cluster(x, y, ore_type, params['cluster_size'])
                clusters_placed += 1
                
                if clusters_placed >= num_clusters:
                    break
    
    def _create_ore_cluster(self, center_x: int, center_y: int, 
                          ore_type: int, cluster_size: int) -> None:
        """
        Create a cluster of ore blocks around a center point.
        
        Uses normal distribution to create realistic clustering patterns.
        """
        # Generate cluster positions using normal distribution
        σ = cluster_size / 3  # Standard deviation
        
        for _ in range(cluster_size):
            # Normal distribution around center
            dx = int(norm.rvs(0, σ))
            dy = int(norm.rvs(0, σ))
            
            x, y = center_x + dx, center_y + dy
            
            # Place ore if position is valid and contains stone
            if (0 <= x < self.width and 0 <= y < self.height and 
                self.world[y, x] == self.STONE):
                self.world[y, x] = ore_type
    
    def generate_chlorophyte(self) -> None:
        """
        Generate chlorophyte ore in jungle areas.
        
        Chlorophyte has special generation rules:
        - Only spawns in jungle biomes
        - Requires specific depth (cavern layer)
        - Grows slowly over time
        - Cannot be artificially farmed easily
        """
        print("Generating chlorophyte in jungle biomes...")
        
        # Define jungle regions (simplified)
        jungle_regions = [
            (int(0.1 * self.width), int(0.3 * self.width)),  # Left jungle
            (int(0.7 * self.width), int(0.9 * self.width))   # Right jungle
        ]
        
        for start_x, end_x in jungle_regions:
            # Place chlorophyte clusters in cavern layer
            depth_start = int(0.6 * self.height)
            depth_end = int(0.9 * self.height)
            
            num_clusters = np.random.randint(20, 40)
            
            for _ in range(num_clusters):
                x = np.random.randint(start_x, end_x)
                y = np.random.randint(depth_start, depth_end)
                
                if self.world[y, x] == self.STONE:
                    self._create_ore_cluster(x, y, self.CHLOROPHYTE, 4)
    
    def create_boss_arenas(self) -> None:
        """
        Create optimal arena setups for mechanical bosses.
        
        Each mechanical boss requires different arena configurations:
        - The Destroyer: Long horizontal platforms
        - The Twins: Large vertical flight space
        - Skeletron Prime: Mixed platform and cover setup
        """
        print("Creating mechanical boss arenas...")
        
        # Surface level for arena placement
        surface_y = int(self.height * 0.25)
        
        # Arena positions (spread across world)
        arena_positions = [
            (int(0.2 * self.width), surface_y),  # Destroyer arena
            (int(0.5 * self.width), surface_y),  # Twins arena
            (int(0.8 * self.width), surface_y)   # Skeletron Prime arena
        ]
        
        boss_names = ['destroyer', 'twins', 'skeletron_prime']
        
        for i, (boss_name, (center_x, center_y)) in enumerate(zip(boss_names, arena_positions)):
            arena_spec = self.boss_arenas[boss_name]
            self._build_boss_arena(center_x, center_y, boss_name, arena_spec)
    
    def _build_boss_arena(self, center_x: int, center_y: int, 
                         boss_name: str, arena_spec: Dict) -> None:
        """
        Build a specific boss arena with appropriate layout.
        
        Parameters:
        -----------
        center_x : int
            Arena center X coordinate
        center_y : int
            Arena center Y coordinate
        boss_name : str
            Name of the boss this arena is for
        arena_spec : Dict
            Arena specifications (width, height, special requirements)
        """
        width = arena_spec['width']
        height = arena_spec['height']
        
        # Clear arena space
        start_x = center_x - width // 2
        end_x = center_x + width // 2
        start_y = center_y - height // 2
        end_y = center_y + height // 2
        
        # Clear the area
        for y in range(max(0, start_y), min(self.height, end_y)):
            for x in range(max(0, start_x), min(self.width, end_x)):
                if self.world[y, x] != self.EMPTY:
                    self.world[y, x] = self.EMPTY
        
        # Mark arena boundaries
        for x in range(max(0, start_x), min(self.width, end_x)):
            if 0 <= start_y < self.height:
                self.world[start_y, x] = self.BOSS_ARENA
            if 0 <= end_y - 1 < self.height:
                self.world[end_y - 1, x] = self.BOSS_ARENA
        
        for y in range(max(0, start_y), min(self.height, end_y)):
            if 0 <= start_x < self.width:
                self.world[y, start_x] = self.BOSS_ARENA
            if 0 <= end_x - 1 < self.width:
                self.world[y, end_x - 1] = self.BOSS_ARENA
        
        # Add boss-specific features
        if boss_name == 'destroyer':
            # Horizontal platforms for destroyer
            platform_spacing = arena_spec['platform_spacing']
            for platform_y in range(start_y + platform_spacing, 
                                  end_y, platform_spacing):
                if 0 <= platform_y < self.height:
                    for x in range(start_x + 20, end_x - 20):
                        if 0 <= x < self.width:
                            self.world[platform_y, x] = self.BOSS_ARENA
        
        elif boss_name == 'twins':
            # Minimal platforms for flight space
            platform_y = center_y + height // 4
            if 0 <= platform_y < self.height:
                for x in range(start_x + 50, end_x - 50, 100):
                    for platform_x in range(x, min(x + 50, end_x)):
                        if 0 <= platform_x < self.width:
                            self.world[platform_y, platform_x] = self.BOSS_ARENA
        
        elif boss_name == 'skeletron_prime':
            # Mixed platform and cover setup
            # Lower platforms
            platform_y = center_y + height // 3
            if 0 <= platform_y < self.height:
                for x in range(start_x + 30, end_x - 30, 80):
                    for platform_x in range(x, min(x + 40, end_x)):
                        if 0 <= platform_x < self.width:
                            self.world[platform_y, platform_x] = self.BOSS_ARENA
            
            # Cover structures
            cover_y = center_y - height // 4
            if 0 <= cover_y < self.height:
                for x in range(start_x + 100, end_x - 100, 150):
                    for cover_x in range(x, min(x + 30, end_x)):
                        if 0 <= cover_x < self.width:
                            for dy in range(3):  # Small cover blocks
                                if 0 <= cover_y + dy < self.height:
                                    self.world[cover_y + dy, cover_x] = self.BOSS_ARENA
    
    def add_crystal_hearts(self) -> None:
        """
        Add crystal hearts throughout the underground for increased mana.
        
        Crystal hearts spawn in specific patterns:
        - Underground layer placement
        - Surrounded by crystal blocks
        - Limited quantity per world
        """
        print("Adding crystal hearts...")
        
        num_hearts = 15  # Standard number per world
        placed_hearts = 0
        attempts = 0
        max_attempts = 500
        
        while placed_hearts < num_hearts and attempts < max_attempts:
            x = np.random.randint(100, self.width - 100)
            y = np.random.randint(int(self.height * 0.3), int(self.height * 0.7))
            
            # Check if location is suitable (stone)
            if self.world[y, x] == self.STONE:
                # Create small crystal formation
                self.world[y, x] = self.CRYSTAL_HEART
                
                # Add surrounding crystal blocks (simplified)
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if (dy != 0 or dx != 0):  # Don't replace the heart itself
                            ny, nx = y + dy, x + dx
                            if (0 <= nx < self.width and 0 <= ny < self.height and
                                self.world[ny, nx] == self.STONE and np.random.random() < 0.3):
                                self.world[ny, nx] = self.CRYSTAL_HEART
                
                placed_hearts += 1
            
            attempts += 1
    
    def run_hardmode_transformation(self) -> None:
        """
        Execute the complete hardmode transformation sequence.
        
        This includes:
        1. Initial world state (pre-hardmode)
        2. Altar breaking and ore generation
        3. Chlorophyte generation
        4. Boss arena creation
        5. Crystal heart placement
        6. Environmental transformations
        """
        print("Starting hardmode transformation sequence...")
        print("=" * 50)
        
        # Step 1: Generate pre-hardmode world
        print("1. Generating pre-hardmode world...")
        self.world = self.generate_prehardmode_world()
        self.transformation_history.append(('Pre-hardmode', self.world.copy()))
        
        # Step 2: Break altars and generate ores
        print("2. Breaking altars and generating hardmode ores...")
        self.break_altars(num_waves=3)
        self.transformation_history.append(('Post-altar breaking', self.world.copy()))
        
        # Step 3: Generate chlorophyte
        print("3. Generating chlorophyte ore...")
        self.generate_chlorophyte()
        self.transformation_history.append(('Post-chlorophyte', self.world.copy()))
        
        # Step 4: Create boss arenas
        print("4. Creating mechanical boss arenas...")
        self.create_boss_arenas()
        self.transformation_history.append(('Post-arenas', self.world.copy()))
        
        # Step 5: Add crystal hearts
        print("5. Adding crystal hearts...")
        self.add_crystal_hearts()
        self.transformation_history.append(('Final hardmode', self.world.copy()))
        
        print("\nHardmode transformation complete!")
        print(f"Total altars broken: {self.altars_broken}")
        print(f"Transformation stages recorded: {len(self.transformation_history)}")
    
    def create_transformation_visualization(self, save_path: str = None) -> plt.Figure:
        """
        Create comprehensive visualization of the hardmode transformation.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the visualization
            
        Returns:
        --------
        plt.Figure
            The created figure
        """
        if not self.transformation_history:
            print("No transformation history available. Run transformation first.")
            return None
        
        fig = plt.figure(figsize=(24, 20))
        
        # Create grid layout
        num_stages = len(self.transformation_history)
        rows = (num_stages + 1) // 2
        gs = fig.add_gridspec(rows + 2, 2, height_ratios=[1] * rows + [0.8, 0.6],
                             hspace=0.3, wspace=0.2)
        
        # Plot each transformation stage
        colors = [self.colors[i] for i in range(len(self.colors))]
        cmap = ListedColormap(colors)
        
        for i, (stage_name, world_state) in enumerate(self.transformation_history):
            row = i // 2
            col = i % 2
            ax = fig.add_subplot(gs[row, col])
            
            # Downsample for visualization
            sample_rate = max(1, self.width // 600)
            world_sample = world_state[::sample_rate, ::sample_rate]
            
            im = ax.imshow(world_sample, cmap=cmap, aspect='auto')
            ax.set_title(f"{stage_name}", fontsize=14, fontweight='bold')
            ax.set_xlabel("World X Position")
            ax.set_ylabel("World Y Position")
        
        # Add ore distribution analysis
        ax_ore = fig.add_subplot(gs[rows, :])
        self._plot_ore_distribution(ax_ore)
        
        # Add transformation statistics
        ax_stats = fig.add_subplot(gs[rows + 1, :])
        self._plot_transformation_stats(ax_stats)
        
        plt.suptitle("Terraria Hardmode Transformation Analysis", 
                    fontsize=20, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Transformation visualization saved to {save_path}")
        
        return fig
    
    def _plot_ore_distribution(self, ax: plt.Axes) -> None:
        """Plot the distribution of hardmode ores."""
        if self.world is None:
            return
        
        ore_types = [self.COBALT, self.MYTHRIL, self.ADAMANTITE, self.CHLOROPHYTE]
        ore_names = ['Cobalt', 'Mythril', 'Adamantite', 'Chlorophyte']
        ore_counts = []
        
        for ore_type in ore_types:
            count = np.sum(self.world == ore_type)
            ore_counts.append(count)
        
        colors = sns.color_palette("cubehelix", len(ore_types))
        bars = ax.bar(ore_names, ore_counts, color=colors)
        
        ax.set_title("Hardmode Ore Distribution", fontsize=14, fontweight='bold')
        ax.set_ylabel("Number of Blocks")
        ax.set_xlabel("Ore Type")
        
        # Add value labels on bars
        for bar, count in zip(bars, ore_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}', ha='center', va='bottom')
    
    def _plot_transformation_stats(self, ax: plt.Axes) -> None:
        """Plot key transformation statistics."""
        if not self.transformation_history:
            return
        
        # Calculate block type changes over transformation stages
        stage_names = [stage[0] for stage in self.transformation_history]
        
        # Track specific block types
        tracked_types = [self.ALTAR, self.COBALT, self.MYTHRIL, self.ADAMANTITE, 
                        self.CHLOROPHYTE, self.CRYSTAL_HEART, self.BOSS_ARENA]
        type_names = ['Altars', 'Cobalt', 'Mythril', 'Adamantite', 
                     'Chlorophyte', 'Crystal Hearts', 'Boss Arenas']
        
        # Calculate counts for each stage
        stage_data = []
        for _, world_state in self.transformation_history:
            stage_counts = []
            for block_type in tracked_types:
                count = np.sum(world_state == block_type)
                stage_counts.append(count)
            stage_data.append(stage_counts)
        
        # Create stacked area plot
        stage_data = np.array(stage_data).T
        colors = sns.color_palette("rocket", len(tracked_types))
        
        x = range(len(stage_names))
        ax.stackplot(x, *stage_data, labels=type_names, colors=colors, alpha=0.8)
        
        ax.set_title("Block Type Evolution During Hardmode Transformation", 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel("Transformation Stage")
        ax.set_ylabel("Number of Blocks")
        ax.set_xticks(x)
        ax.set_xticklabels(stage_names, rotation=45, ha='right')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

def create_hardmode_animation(hardmode_system: TerrariaHardmodeTransformation,
                            save_path: str = None) -> FuncAnimation:
    """
    Create an animated visualization of the hardmode transformation process.
    
    Parameters:
    -----------
    hardmode_system : TerrariaHardmodeTransformation
        The hardmode system with transformation data
    save_path : str, optional
        Path to save the animation
        
    Returns:
    --------
    FuncAnimation
        The created animation
    """
    if not hardmode_system.transformation_history:
        print("No transformation history available. Run transformation first.")
        return None
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Setup color mapping
    colors = [hardmode_system.colors[i] for i in range(len(hardmode_system.colors))]
    cmap = ListedColormap(colors)
    
    # Initial plot setup
    sample_rate = max(1, hardmode_system.width // 400)
    initial_world = hardmode_system.transformation_history[0][1]
    im1 = ax1.imshow(initial_world[::sample_rate, ::sample_rate], 
                     cmap=cmap, aspect='auto')
    ax1.set_title("Hardmode Transformation", fontsize=16, fontweight='bold')
    
    # Statistics setup
    tracked_types = [hardmode_system.ALTAR, hardmode_system.COBALT, 
                    hardmode_system.MYTHRIL, hardmode_system.ADAMANTITE]
    lines = []
    colors_stats = sns.color_palette("cubehelix", len(tracked_types))
    
    for i, color in enumerate(colors_stats):
        line, = ax2.plot([], [], color=color, linewidth=2, 
                        label=['Altars', 'Cobalt', 'Mythril', 'Adamantite'][i])
        lines.append(line)
    
    ax2.set_xlim(0, len(hardmode_system.transformation_history) - 1)
    ax2.set_ylim(0, 1000)  # Adjust based on expected max values
    ax2.set_title("Block Type Changes", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Transformation Stage")
    ax2.set_ylabel("Number of Blocks")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    def animate(frame):
        if frame >= len(hardmode_system.transformation_history):
            return [im1] + lines
        
        # Update world visualization
        stage_name, world_state = hardmode_system.transformation_history[frame]
        world_sample = world_state[::sample_rate, ::sample_rate]
        im1.set_array(world_sample)
        ax1.set_title(f"Hardmode Transformation - {stage_name}", 
                     fontsize=16, fontweight='bold')
        
        # Update statistics
        x_data = list(range(frame + 1))
        for i, block_type in enumerate(tracked_types):
            y_data = []
            for j in range(frame + 1):
                count = np.sum(hardmode_system.transformation_history[j][1] == block_type)
                y_data.append(count)
            lines[i].set_data(x_data, y_data)
        
        return [im1] + lines
    
    anim = FuncAnimation(fig, animate, frames=len(hardmode_system.transformation_history) + 5,
                        interval=1500, blit=False, repeat=True)
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=1)
        print(f"Hardmode transformation animation saved to {save_path}")
    
    return anim

# Example usage and testing
if __name__ == "__main__":
    print("Terraria Hardmode Transformation Visualization System")
    print("=" * 55)
      # Create hardmode transformation system (smaller size for faster processing)
    hardmode_system = TerrariaHardmodeTransformation(world_width=2100, world_height=600)
    
    # Run complete hardmode transformation
    hardmode_system.run_hardmode_transformation()
    
    # Create comprehensive visualization with save path
    save_path = "hardmode_transformation_analysis.png"
    fig = hardmode_system.create_transformation_visualization(save_path=save_path)
    
    # Don't show the plot to avoid blocking
    # plt.show()
    
    print("\nHardmode transformation visualization complete!")
    print("Key statistics:")
    print(f"- Altars broken: {hardmode_system.altars_broken}")
    print(f"- Cobalt ore generated: {np.sum(hardmode_system.world == hardmode_system.COBALT)}")
    print(f"- Mythril ore generated: {np.sum(hardmode_system.world == hardmode_system.MYTHRIL)}")
    print(f"- Adamantite ore generated: {np.sum(hardmode_system.world == hardmode_system.ADAMANTITE)}")
    print(f"- Chlorophyte ore generated: {np.sum(hardmode_system.world == hardmode_system.CHLOROPHYTE)}")
    print(f"- Crystal hearts placed: {np.sum(hardmode_system.world == hardmode_system.CRYSTAL_HEART)}")
