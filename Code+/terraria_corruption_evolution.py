"""
Terraria Corruption/Crimson Evolution Visualization System
=========================================================

This module implements comprehensive visualizations of corruption and crimson spread mechanics
in Terraria, modeling the algorithmic spread patterns, growth rates, and environmental
interactions that define these biome evolutions over time.

Mathematical Foundation:
- Cellular automata for spread simulation
- Exponential growth models: N(t) = N₀ × e^(rt)
- Distance-based influence: I(d) = I₀ × e^(-αd)
- Environmental resistance factors
- Probabilistic spread mechanics

Author: Generated for Terraria Generation Analysis
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from scipy.ndimage import binary_dilation, distance_transform_edt
from scipy.spatial.distance import cdist
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style and color preferences
plt.style.use('dark_background')
sns.set_palette("mako")

class TerrariaCorruptionEvolution:
    """
    Comprehensive system for modeling and visualizing corruption/crimson spread
    in Terraria worlds, incorporating realistic growth mechanics and environmental
    interactions.
    
    The system models:
    - Initial infection points based on world generation
    - Exponential spread with environmental resistance
    - Biome-specific spread rates and patterns
    - Player intervention effects
    - Hardmode acceleration mechanics
    """
    
    def __init__(self, world_width: int = 4200, world_height: int = 1200):
        """
        Initialize the corruption evolution system.
        
        Parameters:
        -----------
        world_width : int
            Width of the world in blocks (default: 4200 for large world)
        world_height : int
            Height of the world in blocks (default: 1200 for large world)
        """
        self.width = world_width
        self.height = world_height
        
        # Block type definitions
        self.EMPTY = 0
        self.DIRT = 1
        self.STONE = 2
        self.GRASS = 3
        self.CORRUPTION = 4
        self.CRIMSON = 5
        self.HALLOW = 6
        self.SAND = 7
        self.SNOW = 8
        self.JUNGLE = 9
        self.MUD = 10
        
        # Spread parameters based on Terraria mechanics
        self.base_spread_rate = 0.02  # Base probability per tick
        self.distance_decay = 0.1     # Distance influence decay
        self.hardmode_multiplier = 3.0  # Hardmode spread acceleration
        self.biome_resistance = {
            self.DIRT: 1.0,
            self.STONE: 0.8,
            self.GRASS: 1.2,
            self.SAND: 1.5,
            self.SNOW: 0.6,
            self.JUNGLE: 0.3,
            self.MUD: 0.4
        }
        
        # Color mapping for visualization
        self.colors = {
            self.EMPTY: (0.1, 0.1, 0.1),      # Dark background
            self.DIRT: (0.4, 0.3, 0.2),       # Brown dirt
            self.STONE: (0.5, 0.5, 0.5),      # Gray stone
            self.GRASS: (0.2, 0.7, 0.2),      # Green grass
            self.CORRUPTION: (0.3, 0.0, 0.5), # Purple corruption
            self.CRIMSON: (0.7, 0.0, 0.2),    # Red crimson
            self.HALLOW: (1.0, 0.8, 1.0),     # Pink hallow
            self.SAND: (0.9, 0.8, 0.5),       # Yellow sand
            self.SNOW: (0.9, 0.9, 1.0),       # White snow
            self.JUNGLE: (0.1, 0.5, 0.1),     # Dark green jungle
            self.MUD: (0.3, 0.4, 0.2)         # Muddy brown
        }
        
        self.world = None
        self.corruption_history = []
        self.time_steps = []
        
    def generate_base_world(self) -> np.ndarray:
        """
        Generate a base world with various biomes for corruption to spread through.
        
        Returns:
        --------
        np.ndarray
            2D array representing the world with different block types
        """
        world = np.full((self.height, self.width), self.EMPTY, dtype=int)
        
        # Create surface terrain using noise
        surface_noise = self._generate_terrain_noise()
        surface_height = (surface_noise * 200 + self.height // 2).astype(int)
        
        # Fill world with appropriate blocks
        for x in range(self.width):
            surface_y = surface_height[x]
            
            # Determine biome based on x position
            biome_type = self._determine_biome(x)
            
            # Fill from surface down
            for y in range(surface_y, self.height):
                if y == surface_y and biome_type != self.SAND:
                    world[y, x] = self.GRASS
                elif y < surface_y + 50:  # Topsoil layer
                    world[y, x] = self.DIRT if biome_type != self.SAND else self.SAND
                elif y < surface_y + 100:  # Subsoil
                    world[y, x] = self.DIRT if biome_type not in [self.JUNGLE, self.SNOW] else biome_type
                else:  # Deep stone
                    world[y, x] = self.STONE
            
            # Add biome-specific surface blocks
            if biome_type == self.JUNGLE:
                world[surface_y, x] = self.JUNGLE
            elif biome_type == self.SNOW:
                world[surface_y, x] = self.SNOW
        
        return world
    
    def _generate_terrain_noise(self) -> np.ndarray:
        """Generate terrain height using multiple octaves of noise."""
        x = np.linspace(0, 8 * np.pi, self.width)
        
        # Multi-octave noise for realistic terrain
        terrain = (
            0.5 * np.sin(x * 0.5) +
            0.3 * np.sin(x * 1.2 + 1.5) +
            0.2 * np.sin(x * 2.8 + 3.0) +
            0.1 * np.sin(x * 5.5 + 4.5)
        )
        
        return terrain
    
    def _determine_biome(self, x: int) -> int:
        """Determine biome type based on world position."""
        # Normalize position to 0-1
        pos = x / self.width
        
        if pos < 0.15:  # Left side - Snow/Tundra
            return self.SNOW
        elif pos < 0.25:  # Transition zone
            return self.DIRT
        elif 0.75 < pos < 0.85:  # Right side jungle area
            return self.JUNGLE
        elif pos > 0.9:  # Far right desert
            return self.SAND
        else:  # Central areas
            return self.DIRT
    
    def initialize_corruption_points(self, corruption_type: int = None) -> None:
        """
        Initialize corruption/crimson infection points based on world generation rules.
        
        In Terraria, corruption/crimson spawns:
        - In a V-pattern from the center after Wall of Flesh defeat
        - Small scattered pockets during world generation
        
        Parameters:
        -----------
        corruption_type : int
            Type of corruption (CORRUPTION or CRIMSON)
        """
        if corruption_type is None:
            corruption_type = np.random.choice([self.CORRUPTION, self.CRIMSON])
        
        # Pre-hardmode: Small scattered pockets (world generation)
        num_initial_pockets = np.random.randint(3, 7)
        for _ in range(num_initial_pockets):
            x = np.random.randint(self.width // 4, 3 * self.width // 4)
            y = np.random.randint(self.height // 3, 2 * self.height // 3)
            
            # Create small infection pocket
            self._create_infection_pocket(x, y, corruption_type, radius=20)
    
    def trigger_hardmode_spread(self, corruption_type: int = None) -> None:
        """
        Trigger the hardmode V-pattern corruption spread.
        
        When the Wall of Flesh is defeated, corruption/crimson spreads in a V-pattern
        from the center of the world, along with hallow on one side.
        
        Parameters:
        -----------
        corruption_type : int
            Type of corruption for the V-pattern
        """
        if corruption_type is None:
            corruption_type = np.random.choice([self.CORRUPTION, self.CRIMSON])
        
        center_x = self.width // 2
        center_y = self.height // 2
        
        # Create V-pattern spread
        v_width = self.width // 3
        v_angle = np.pi / 6  # 30 degrees
        
        # Left arm of V (corruption/crimson)
        for i in range(v_width):
            x_offset = -i * np.cos(v_angle)
            y_offset = i * np.sin(v_angle)
            
            x = int(center_x + x_offset)
            y = int(center_y + y_offset)
            
            if 0 <= x < self.width and 0 <= y < self.height:
                self._create_infection_strip(x, y, corruption_type, width=50)
        
        # Right arm of V (hallow)
        for i in range(v_width):
            x_offset = i * np.cos(v_angle)
            y_offset = i * np.sin(v_angle)
            
            x = int(center_x + x_offset)
            y = int(center_y + y_offset)
            
            if 0 <= x < self.width and 0 <= y < self.height:
                self._create_infection_strip(x, y, self.HALLOW, width=50)
    
    def _create_infection_pocket(self, x: int, y: int, infection_type: int, radius: int) -> None:
        """Create a circular infection pocket."""
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx**2 + dy**2 <= radius**2:
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < self.width and 0 <= ny < self.height and 
                        self.world[ny, nx] not in [self.EMPTY]):
                        self.world[ny, nx] = infection_type
    
    def _create_infection_strip(self, x: int, y: int, infection_type: int, width: int) -> None:
        """Create a vertical strip of infection."""
        half_width = width // 2
        for dy in range(-self.height // 4, self.height // 4):
            for dx in range(-half_width, half_width + 1):
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.width and 0 <= ny < self.height and 
                    self.world[ny, nx] not in [self.EMPTY]):
                    self.world[ny, nx] = infection_type
    
    def simulate_spread_step(self, hardmode: bool = False) -> None:
        """
        Simulate one step of corruption/crimson spread.
        
        The spread algorithm uses:
        - Distance-based probability
        - Biome resistance factors
        - Environmental constraints
        
        Parameters:
        -----------
        hardmode : bool
            Whether hardmode is active (increases spread rate)
        """
        new_world = self.world.copy()
        spread_rate = self.base_spread_rate
        
        if hardmode:
            spread_rate *= self.hardmode_multiplier
        
        # Find all corruption/crimson/hallow blocks
        infection_types = [self.CORRUPTION, self.CRIMSON, self.HALLOW]
        infected_blocks = np.where(np.isin(self.world, infection_types))
        
        # For each infected block, try to spread to neighbors
        for i in range(len(infected_blocks[0])):
            y, x = infected_blocks[0][i], infected_blocks[1][i]
            infection_type = self.world[y, x]
            
            # Check 8-connected neighbors
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    
                    ny, nx = y + dy, x + dx
                    
                    if (0 <= nx < self.width and 0 <= ny < self.height):
                        target_block = self.world[ny, nx]
                        
                        # Can only spread to non-empty, non-infected blocks
                        if (target_block not in [self.EMPTY] + infection_types):
                            
                            # Calculate spread probability
                            resistance = self.biome_resistance.get(target_block, 1.0)
                            distance_factor = 1.0  # Adjacent blocks
                            
                            spread_prob = spread_rate / resistance * distance_factor
                            
                            if np.random.random() < spread_prob:
                                new_world[ny, nx] = infection_type
        
        self.world = new_world
        
        # Record statistics
        corruption_count = np.sum(np.isin(self.world, [self.CORRUPTION, self.CRIMSON, self.HALLOW]))
        self.corruption_history.append(corruption_count)
    
    def simulate_evolution(self, time_steps: int = 100, hardmode_start: int = 50) -> None:
        """
        Run a complete evolution simulation.
        
        Parameters:
        -----------
        time_steps : int
            Total number of simulation steps
        hardmode_start : int
            Step at which hardmode begins
        """
        # Initialize world and corruption
        self.world = self.generate_base_world()
        self.initialize_corruption_points()
        
        self.corruption_history = []
        self.time_steps = list(range(time_steps))
        
        print("Running corruption evolution simulation...")
        
        for step in range(time_steps):
            # Trigger hardmode spread at specified step
            if step == hardmode_start:
                self.trigger_hardmode_spread()
                print(f"Hardmode activated at step {step}")
            
            # Simulate spread
            is_hardmode = step >= hardmode_start
            self.simulate_spread_step(hardmode=is_hardmode)
            
            if step % 20 == 0:
                print(f"Step {step}/{time_steps} - Infected blocks: {self.corruption_history[-1]}")
    
    def create_evolution_visualization(self, save_path: str = None) -> plt.Figure:
        """
        Create comprehensive visualization of corruption evolution.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the visualization
            
        Returns:
        --------
        plt.Figure
            The created figure
        """
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3, height_ratios=[2, 1, 1], width_ratios=[1, 1, 1])
        
        # Main world visualization
        ax_world = fig.add_subplot(gs[0, :])
        self._plot_world_state(ax_world, "Final Corruption State")
        
        # Evolution timeline
        ax_timeline = fig.add_subplot(gs[1, :])
        self._plot_evolution_timeline(ax_timeline)
        
        # Statistics panels
        ax_growth = fig.add_subplot(gs[2, 0])
        self._plot_growth_rate(ax_growth)
        
        ax_biome = fig.add_subplot(gs[2, 1])
        self._plot_biome_resistance(ax_biome)
        
        ax_spread = fig.add_subplot(gs[2, 2])
        self._plot_spread_pattern(ax_spread)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        return fig
    
    def _plot_world_state(self, ax: plt.Axes, title: str) -> None:
        """Plot the current world state with color-coded blocks."""
        # Create color map
        colors = [self.colors[i] for i in range(len(self.colors))]
        cmap = ListedColormap(colors)
        
        # Downsample for visualization
        sample_rate = max(1, self.width // 800)
        world_sample = self.world[::sample_rate, ::sample_rate]
        
        im = ax.imshow(world_sample, cmap=cmap, aspect='auto', vmin=0, vmax=len(colors)-1)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel("World X Position")
        ax.set_ylabel("World Y Position")
        
        # Add colorbar legend
        cbar = plt.colorbar(im, ax=ax, shrink=0.6)
        cbar.set_label("Block Type", rotation=270, labelpad=20)
    
    def _plot_evolution_timeline(self, ax: plt.Axes) -> None:
        """Plot the evolution of corruption over time."""
        ax.plot(self.time_steps, self.corruption_history, 
               color=sns.color_palette("rocket")[3], linewidth=3, 
               label='Infected Blocks')
        
        # Mark hardmode activation
        if len(self.time_steps) > 50:
            ax.axvline(x=50, color='red', linestyle='--', alpha=0.7, 
                      label='Hardmode Activation')
        
        ax.set_title("Corruption Evolution Over Time", fontsize=14, fontweight='bold')
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Number of Infected Blocks")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_growth_rate(self, ax: plt.Axes) -> None:
        """Plot the growth rate analysis."""
        if len(self.corruption_history) > 1:
            growth_rates = np.diff(self.corruption_history)
            ax.plot(self.time_steps[1:], growth_rates, 
                   color=sns.color_palette("mako")[4], linewidth=2)
            
            ax.set_title("Growth Rate", fontsize=12, fontweight='bold')
            ax.set_xlabel("Time")
            ax.set_ylabel("Blocks/Step")
            ax.grid(True, alpha=0.3)
    
    def _plot_biome_resistance(self, ax: plt.Axes) -> None:
        """Plot biome resistance factors."""
        biomes = list(self.biome_resistance.keys())
        resistances = list(self.biome_resistance.values())
        
        colors = sns.color_palette("cubehelix", len(biomes))
        ax.bar(range(len(biomes)), resistances, color=colors)
        ax.set_title("Biome Resistance", fontsize=12, fontweight='bold')
        ax.set_xlabel("Biome Type")
        ax.set_ylabel("Resistance Factor")
        ax.set_xticks(range(len(biomes)))
        ax.set_xticklabels([str(b) for b in biomes], rotation=45)
    
    def _plot_spread_pattern(self, ax: plt.Axes) -> None:
        """Plot spread pattern analysis."""
        # Calculate infection density by region
        regions = 10
        region_width = self.width // regions
        densities = []
        
        for i in range(regions):
            start_x = i * region_width
            end_x = (i + 1) * region_width
            region = self.world[:, start_x:end_x]
            
            infected = np.sum(np.isin(region, [self.CORRUPTION, self.CRIMSON, self.HALLOW]))
            total = np.sum(region != self.EMPTY)
            density = infected / max(total, 1)
            densities.append(density)
        
        colors = sns.color_palette("rocket", regions)
        ax.bar(range(regions), densities, color=colors)
        ax.set_title("Infection Density by Region", fontsize=12, fontweight='bold')
        ax.set_xlabel("World Region")
        ax.set_ylabel("Infection Density")

def create_corruption_animation(world_gen: TerrariaCorruptionEvolution, 
                              save_path: str = None) -> FuncAnimation:
    """
    Create an animated visualization of corruption spread over time.
    
    Parameters:
    -----------
    world_gen : TerrariaCorruptionEvolution
        The world generator with simulation data
    save_path : str, optional
        Path to save the animation
        
    Returns:
    --------
    FuncAnimation
        The created animation
    """
    # Run simulation and store states
    world_gen.world = world_gen.generate_base_world()
    world_gen.initialize_corruption_points()
    
    world_states = [world_gen.world.copy()]
    
    for step in range(100):
        if step == 50:  # Hardmode activation
            world_gen.trigger_hardmode_spread()
        
        world_gen.simulate_spread_step(hardmode=step >= 50)
        world_states.append(world_gen.world.copy())
    
    # Create animation
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Setup color mapping
    colors = [world_gen.colors[i] for i in range(len(world_gen.colors))]
    cmap = ListedColormap(colors)
    
    # Initial plot
    sample_rate = max(1, world_gen.width // 400)
    im1 = ax1.imshow(world_states[0][::sample_rate, ::sample_rate], 
                     cmap=cmap, aspect='auto')
    ax1.set_title("Corruption Evolution", fontsize=16, fontweight='bold')
    
    # Statistics plot
    line, = ax2.plot([], [], color=sns.color_palette("rocket")[3], linewidth=3)
    ax2.set_xlim(0, len(world_states))
    ax2.set_ylim(0, max(world_gen.corruption_history) * 1.1 if world_gen.corruption_history else 1000)
    ax2.set_title("Infection Growth", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Time Steps")
    ax2.set_ylabel("Infected Blocks")
    ax2.grid(True, alpha=0.3)
    
    def animate(frame):
        # Update world visualization
        world_sample = world_states[frame][::sample_rate, ::sample_rate]
        im1.set_array(world_sample)
        
        # Update statistics
        if frame < len(world_gen.corruption_history):
            x_data = list(range(frame + 1))
            y_data = world_gen.corruption_history[:frame + 1]
            line.set_data(x_data, y_data)
        
        # Update title with current step
        ax1.set_title(f"Corruption Evolution - Step {frame}" + 
                     (" (HARDMODE)" if frame >= 50 else ""), 
                     fontsize=16, fontweight='bold')
        
        return [im1, line]
    
    anim = FuncAnimation(fig, animate, frames=len(world_states), 
                        interval=100, blit=False, repeat=True)
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=10)
        print(f"Animation saved to {save_path}")
    
    return anim

# Example usage and testing
if __name__ == "__main__":
    print("Terraria Corruption Evolution Visualization System")
    print("=" * 50)
    
    # Create system
    corruption_system = TerrariaCorruptionEvolution(world_width=2100, world_height=600)
    
    # Run evolution simulation
    corruption_system.simulate_evolution(time_steps=100, hardmode_start=50)
      # Create visualizations
    save_path = "corruption_evolution_analysis.png"
    fig = corruption_system.create_evolution_visualization(save_path=save_path)
    plt.show()
    
    print("\nEvolution simulation complete!")
    print(f"Final infected blocks: {corruption_system.corruption_history[-1]}")
    print(f"Peak growth rate: {max(np.diff(corruption_system.corruption_history)):.1f} blocks/step")
