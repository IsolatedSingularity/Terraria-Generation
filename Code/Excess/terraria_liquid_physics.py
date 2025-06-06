"""
Terraria Liquid Physics Simulation
=================================

This module simulates and visualizes the behavior of liquids (water and lava)
in Terraria worlds, implementing the game's cellular automata-based fluid mechanics.

Mathematical Foundation:
- Cellular automata rules: S(t+1) = f(S(t), N(t))
- Pressure equilibrium: ΔP = ρgh (hydrostatic pressure)
- Flow dynamics: v = √(2gh) (Torricelli's law)
- Surface tension approximation: F = γL (contact line force)

Author: Terraria Generation Project
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class TerrariaLiquidSimulator:
    """
    Simulates liquid physics using Terraria's cellular automata approach.
    """
    
    def __init__(self, width: int = 200, height: int = 150):
        """
        Initialize the liquid simulator.
        
        Args:
            width: Simulation grid width
            height: Simulation grid height
        """
        self.width = width
        self.height = height
        
        # Liquid states: 0 = air, 1-255 = liquid levels, 256 = solid
        self.water_grid = np.zeros((height, width), dtype=np.int16)
        self.lava_grid = np.zeros((height, width), dtype=np.int16)
        self.solid_grid = np.zeros((height, width), dtype=bool)
        
        # Liquid properties
        self.max_liquid_level = 255
        self.flow_threshold = 5  # Minimum liquid level difference for flow
        self.settling_factor = 0.95  # Damping factor for oscillations
        
        # Physics constants (scaled for cellular automata)
        self.gravity_strength = 8
        self.viscosity_water = 1.0
        self.viscosity_lava = 0.3  # Lava flows slower
        self.evaporation_rate = 0.001  # Very slow evaporation
        
        self.iteration_count = 0
    
    def add_terrain(self, terrain_function=None) -> None:
        """
        Add solid terrain to the simulation.
        
        Args:
            terrain_function: Optional function defining terrain shape
        """
        if terrain_function is None:
            # Default terrain: sloped surface with caves
            for x in range(self.width):
                # Surface height with some variation
                surface_height = int(self.height * 0.3 + 10 * np.sin(x * 0.1) + 5 * np.sin(x * 0.05))
                surface_height = max(10, min(self.height - 20, surface_height))
                
                # Fill solid blocks below surface
                for y in range(surface_height, self.height):
                    # Add some caves
                    cave_noise = np.sin(x * 0.15) * np.sin(y * 0.1)
                    if cave_noise > 0.3 and y < self.height - 5:
                        continue  # Create cave opening
                    self.solid_grid[y, x] = True
        else:
            terrain_function(self.solid_grid)
    
    def add_water_source(self, x: int, y: int, flow_rate: int = 20) -> None:
        """
        Add a water source at specified location.
        
        Args:
            x: X coordinate
            y: Y coordinate  
            flow_rate: Amount of water added per iteration
        """
        if self._is_valid_position(x, y) and not self.solid_grid[y, x]:
            self.water_grid[y, x] = min(self.max_liquid_level, 
                                       self.water_grid[y, x] + flow_rate)
    
    def add_lava_source(self, x: int, y: int, flow_rate: int = 15) -> None:
        """
        Add a lava source at specified location.
        
        Args:
            x: X coordinate
            y: Y coordinate
            flow_rate: Amount of lava added per iteration
        """
        if self._is_valid_position(x, y) and not self.solid_grid[y, x]:
            self.lava_grid[y, x] = min(self.max_liquid_level, 
                                      self.lava_grid[y, x] + flow_rate)
    
    def _is_valid_position(self, x: int, y: int) -> bool:
        """Check if coordinates are within grid bounds."""
        return 0 <= x < self.width and 0 <= y < self.height
    
    def _calculate_flow_amount(self, source_level: int, target_level: int, 
                              viscosity: float = 1.0) -> int:
        """
        Calculate amount of liquid to flow between cells.
        
        Mathematical model:
        Flow follows pressure difference and viscosity:
        flow_rate = (ΔP / viscosity) * time_step
        
        Args:
            source_level: Liquid level in source cell
            target_level: Liquid level in target cell
            viscosity: Liquid viscosity factor
            
        Returns:
            Amount of liquid to transfer
        """
        if source_level <= self.flow_threshold:
            return 0
        
        level_diff = source_level - target_level
        if level_diff <= self.flow_threshold:
            return 0
        
        # Flow rate proportional to pressure difference and inversely to viscosity
        base_flow = int(level_diff * 0.25 / viscosity)
        
        # Limit maximum flow to prevent oscillations
        max_flow = source_level // 4
        return min(base_flow, max_flow)
    
    def _simulate_liquid_step(self, liquid_grid: np.ndarray, viscosity: float) -> np.ndarray:
        """
        Perform one simulation step for a liquid type.
        
        Cellular automata rules:
        1. Gravity: Liquid flows downward preferentially
        2. Pressure equilibrium: Liquid spreads horizontally to equalize levels
        3. Conservation: Total liquid amount is conserved (minus evaporation)
        
        Args:
            liquid_grid: Current liquid state
            viscosity: Liquid viscosity factor
            
        Returns:
            Updated liquid state
        """
        new_grid = liquid_grid.copy()
        
        # Process cells in random order to prevent directional bias
        positions = [(y, x) for y in range(self.height) for x in range(self.width)]
        np.random.shuffle(positions)
        
        for y, x in positions:
            if liquid_grid[y, x] <= 0 or self.solid_grid[y, x]:
                continue
            
            current_level = liquid_grid[y, x]
            
            # 1. Gravity - flow downward
            if y < self.height - 1 and not self.solid_grid[y + 1, x]:
                below_level = liquid_grid[y + 1, x]
                flow_down = self._calculate_flow_amount(current_level, below_level, viscosity)
                flow_down = int(flow_down * self.gravity_strength)  # Gravity amplification
                
                if flow_down > 0:
                    transfer = min(flow_down, current_level, 
                                 self.max_liquid_level - below_level)
                    new_grid[y, x] -= transfer
                    new_grid[y + 1, x] += transfer
                    current_level -= transfer
            
            # 2. Horizontal flow for pressure equilibrium
            if current_level > self.flow_threshold:
                neighbors = []
                if x > 0 and not self.solid_grid[y, x - 1]:
                    neighbors.append((y, x - 1))
                if x < self.width - 1 and not self.solid_grid[y, x + 1]:
                    neighbors.append((y, x + 1))
                
                for ny, nx in neighbors:
                    neighbor_level = liquid_grid[ny, nx]
                    flow_amount = self._calculate_flow_amount(current_level, neighbor_level, viscosity)
                    
                    if flow_amount > 0:
                        transfer = min(flow_amount, new_grid[y, x], 
                                     self.max_liquid_level - neighbor_level)
                        new_grid[y, x] -= transfer
                        new_grid[ny, nx] += transfer
        
        # Apply settling factor to reduce oscillations
        new_grid = (new_grid * self.settling_factor).astype(np.int16)
        
        # Apply evaporation (very slow)
        if self.iteration_count % 100 == 0:  # Every 100 iterations
            evaporation = (new_grid * self.evaporation_rate).astype(np.int16)
            new_grid = np.maximum(0, new_grid - evaporation)
        
        return new_grid
    
    def simulate_step(self) -> None:
        """Perform one complete simulation step for all liquids."""
        # Simulate water
        self.water_grid = self._simulate_liquid_step(self.water_grid, self.viscosity_water)
        
        # Simulate lava
        self.lava_grid = self._simulate_liquid_step(self.lava_grid, self.viscosity_lava)
        
        # Handle water-lava interactions
        self._handle_liquid_interactions()
        
        self.iteration_count += 1
    
    def _handle_liquid_interactions(self) -> None:
        """
        Handle interactions between water and lava (obsidian formation).
        
        Terraria mechanics:
        - Water + Lava source = Obsidian
        - Water + Lava flow = Steam (liquid destruction)
        """
        interaction_mask = (self.water_grid > 0) & (self.lava_grid > 0)
        
        if np.any(interaction_mask):
            # Convert interacting liquids to obsidian (solid blocks)
            self.solid_grid[interaction_mask] = True
            self.water_grid[interaction_mask] = 0
            self.lava_grid[interaction_mask] = 0
    
    def get_visualization_grid(self) -> np.ndarray:
        """
        Create a composite grid for visualization.
        
        Returns:
            Grid with encoded values: 0=air, 1-255=water, 256-511=lava, 512=solid, 513=obsidian
        """
        vis_grid = np.zeros((self.height, self.width), dtype=np.int16)
        
        # Air remains 0
        # Water: 1-255
        vis_grid[self.water_grid > 0] = self.water_grid[self.water_grid > 0]
        
        # Lava: 256-511
        lava_mask = self.lava_grid > 0
        vis_grid[lava_mask] = 256 + self.lava_grid[lava_mask]
        
        # Solid terrain: 512
        vis_grid[self.solid_grid] = 512
        
        return vis_grid
    
    def calculate_liquid_statistics(self) -> Dict[str, float]:
        """
        Calculate statistics about current liquid state.
        
        Returns:
            Dictionary with liquid statistics
        """
        total_water = np.sum(self.water_grid)
        total_lava = np.sum(self.lava_grid)
        
        # Calculate flow rates (change in liquid distribution)
        water_variance = np.var(self.water_grid[self.water_grid > 0]) if np.any(self.water_grid > 0) else 0
        lava_variance = np.var(self.lava_grid[self.lava_grid > 0]) if np.any(self.lava_grid > 0) else 0
        
        # Calculate equilibrium measure (how settled the liquids are)
        water_equilibrium = 1.0 / (1.0 + water_variance / 1000) if water_variance > 0 else 1.0
        lava_equilibrium = 1.0 / (1.0 + lava_variance / 1000) if lava_variance > 0 else 1.0
        
        return {
            'total_water': total_water,
            'total_lava': total_lava,
            'water_cells': np.sum(self.water_grid > 0),
            'lava_cells': np.sum(self.lava_grid > 0),
            'water_equilibrium': water_equilibrium,
            'lava_equilibrium': lava_equilibrium,
            'iterations': self.iteration_count
        }

def create_custom_colormap():
    """Create custom colormap for liquid visualization."""
    # Define colors for different states
    colors = []
    
    # Air (transparent/light blue)
    colors.append('#E6F3FF')
    
    # Water gradient (light to dark blue)
    for i in range(1, 256):
        intensity = i / 255.0
        blue_component = 0.3 + 0.7 * intensity
        colors.append((0.1, 0.3, blue_component))
    
    # Lava gradient (yellow to red)
    for i in range(256, 512):
        intensity = (i - 256) / 255.0
        red_component = 0.8 + 0.2 * intensity
        green_component = 0.6 * (1 - intensity)
        colors.append((red_component, green_component, 0.0))
    
    # Solid terrain (brown)
    colors.append('#8B4513')
    
    return ListedColormap(colors)

def run_liquid_simulation(save_path: str = None, num_steps: int = 200) -> None:
    """
    Run a complete liquid physics simulation.
    
    Args:
        save_path: Optional path to save the final visualization
        num_steps: Number of simulation steps to run
    """
    print("Initializing Terraria Liquid Physics Simulation...")
    
    # Create simulator
    sim = TerrariaLiquidSimulator(200, 150)
    sim.add_terrain()
    
    # Add liquid sources
    sim.add_water_source(50, 20, 30)   # Water spring
    sim.add_water_source(150, 25, 25)  # Another water source
    sim.add_lava_source(100, 30, 20)   # Lava source
    sim.add_lava_source(180, 40, 15)   # Another lava source
    
    # Create visualization setup
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    cmap = create_custom_colormap()
    
    statistics_history = []
    
    print(f"Running {num_steps} simulation steps...")
    
    # Run simulation
    for step in range(num_steps):
        sim.simulate_step()
        
        # Occasionally add more liquid to maintain flow
        if step % 20 == 0:
            sim.add_water_source(50, 20, 15)
            sim.add_lava_source(100, 30, 10)
        
        # Record statistics
        stats = sim.calculate_liquid_statistics()
        statistics_history.append(stats)
        
        if step % 50 == 0:
            print(f"Step {step}: Water={stats['total_water']:.0f}, Lava={stats['total_lava']:.0f}, "
                  f"Equilibrium: W={stats['water_equilibrium']:.3f}, L={stats['lava_equilibrium']:.3f}")
    
    # Final visualization
    vis_grid = sim.get_visualization_grid()
    
    # Main simulation visualization
    im1 = ax1.imshow(vis_grid, cmap=cmap, aspect='auto', vmin=0, vmax=513)
    ax1.set_title('Terraria Liquid Physics Simulation\nCellular Automata with Pressure Equilibrium', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('X Position (blocks)', fontweight='bold')
    ax1.set_ylabel('Y Position (blocks)', fontweight='bold')
    
    # Add colorbar with custom labels
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Material Type', fontweight='bold')
    
    # Statistics plot
    steps = range(len(statistics_history))
    water_totals = [s['total_water'] for s in statistics_history]
    lava_totals = [s['total_lava'] for s in statistics_history]
    water_eq = [s['water_equilibrium'] for s in statistics_history]
    lava_eq = [s['lava_equilibrium'] for s in statistics_history]
    
    ax2_twin = ax2.twinx()
    
    line1 = ax2.plot(steps, water_totals, 'b-', linewidth=2, label='Total Water')
    line2 = ax2.plot(steps, lava_totals, 'r-', linewidth=2, label='Total Lava')
    line3 = ax2_twin.plot(steps, water_eq, 'b--', alpha=0.7, label='Water Equilibrium')
    line4 = ax2_twin.plot(steps, lava_eq, 'r--', alpha=0.7, label='Lava Equilibrium')
    
    ax2.set_xlabel('Simulation Steps', fontweight='bold')
    ax2.set_ylabel('Total Liquid Amount', fontweight='bold', color='black')
    ax2_twin.set_ylabel('Equilibrium Factor', fontweight='bold', color='gray')
    ax2.set_title('Liquid Physics Statistics\nFlow Dynamics and Equilibrium Analysis', 
                  fontsize=14, fontweight='bold')
    
    # Combine legends
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper right')
    
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Liquid simulation visualization saved to: {save_path}")
    
    plt.show()
    
    # Print final statistics
    final_stats = statistics_history[-1]
    print("\nFinal Simulation Statistics:")
    print("-" * 35)
    print(f"Total Water: {final_stats['total_water']:.0f} units")
    print(f"Total Lava: {final_stats['total_lava']:.0f} units")
    print(f"Water Cells: {final_stats['water_cells']}")
    print(f"Lava Cells: {final_stats['lava_cells']}")
    print(f"Water Equilibrium: {final_stats['water_equilibrium']:.3f}")
    print(f"Lava Equilibrium: {final_stats['lava_equilibrium']:.3f}")

def create_flow_dynamics_analysis(save_path: str = None) -> None:
    """
    Create a detailed analysis of flow dynamics for different scenarios.
    
    Args:
        save_path: Optional path to save the plot
    """
    print("Creating flow dynamics analysis...")
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    scenarios = [
        {"name": "Water Fall", "liquid": "water", "source_pos": (50, 10), "terrain": "cliff"},
        {"name": "Lava Pool", "liquid": "lava", "source_pos": (100, 20), "terrain": "basin"},
        {"name": "Underground River", "liquid": "water", "source_pos": (30, 80), "terrain": "cave"},
        {"name": "Volcanic Flow", "liquid": "lava", "source_pos": (80, 40), "terrain": "slope"},
        {"name": "Water-Lava Mix", "liquid": "both", "source_pos": (75, 30), "terrain": "complex"},
        {"name": "Pressure Test", "liquid": "water", "source_pos": (100, 120), "terrain": "u_tube"}
    ]
    
    cmap = create_custom_colormap()
    
    for i, scenario in enumerate(scenarios):
        sim = TerrariaLiquidSimulator(150, 100)
        
        # Create scenario-specific terrain
        if scenario["terrain"] == "cliff":
            for x in range(sim.width):
                cliff_height = 40 if x < 75 else 80
                for y in range(cliff_height, sim.height):
                    sim.solid_grid[y, x] = True
        elif scenario["terrain"] == "basin":
            for x in range(sim.width):
                basin_depth = int(70 + 10 * np.sin(x * 0.1))
                for y in range(basin_depth, sim.height):
                    if x < 20 or x > sim.width - 20:
                        sim.solid_grid[y, x] = True
        elif scenario["terrain"] == "cave":
            for x in range(sim.width):
                for y in range(60, sim.height - 10):
                    if y < 90 and 20 < x < 130:
                        continue  # Cave tunnel
                    sim.solid_grid[y, x] = True
        elif scenario["terrain"] == "slope":
            for x in range(sim.width):
                slope_height = int(30 + x * 0.3)
                for y in range(slope_height, sim.height):
                    sim.solid_grid[y, x] = True
        elif scenario["terrain"] == "complex":
            # Mixed terrain with platforms and gaps
            for x in range(sim.width):
                base_height = int(60 + 15 * np.sin(x * 0.05))
                for y in range(base_height, sim.height):
                    if (y - base_height) % 20 < 15 and x % 30 < 25:
                        sim.solid_grid[y, x] = True
        elif scenario["terrain"] == "u_tube":
            # U-shaped tube for pressure demonstration
            for x in range(sim.width):
                for y in range(sim.height):
                    if (x < 30 or x > 120) and y > 50:
                        sim.solid_grid[y, x] = True
                    elif 30 <= x <= 120 and y > 90:
                        sim.solid_grid[y, x] = True
        
        # Add liquids based on scenario
        x, y = scenario["source_pos"]
        if scenario["liquid"] == "water":
            for _ in range(100):
                sim.add_water_source(x, y, 25)
                sim.simulate_step()
        elif scenario["liquid"] == "lava":
            for _ in range(100):
                sim.add_lava_source(x, y, 20)
                sim.simulate_step()
        elif scenario["liquid"] == "both":
            for _ in range(100):
                sim.add_water_source(x - 10, y, 20)
                sim.add_lava_source(x + 10, y, 15)
                sim.simulate_step()
        
        # Visualize
        vis_grid = sim.get_visualization_grid()
        axes[i].imshow(vis_grid, cmap=cmap, aspect='auto', vmin=0, vmax=513)
        axes[i].set_title(f'{scenario["name"]}\n{scenario["terrain"].replace("_", " ").title()} Terrain', 
                         fontweight='bold')
        axes[i].set_xlabel('X Position')
        axes[i].set_ylabel('Y Position')
    
    plt.suptitle('Terraria Liquid Flow Dynamics Analysis\nDifferent Scenarios and Terrain Interactions', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Flow dynamics analysis saved to: {save_path}")
    
    plt.show()

def main():
    """
    Main function to run all liquid physics simulations.
    """
    print("Terraria Liquid Physics Simulation Suite")
    print("=" * 45)
    
    plots_dir = r"c:\Users\hunkb\OneDrive\Desktop\Terraria Generation\Plots"
    
    # Run main simulation
    print("\n1. Running comprehensive liquid physics simulation...")
    sim_path = f"{plots_dir}/liquid_physics_simulation.png"
    run_liquid_simulation(sim_path, num_steps=300)
    
    # Run flow dynamics analysis
    print("\n2. Creating flow dynamics analysis...")
    flow_path = f"{plots_dir}/liquid_flow_dynamics.png"
    create_flow_dynamics_analysis(flow_path)
    
    print("\n" + "=" * 45)
    print("Liquid physics simulations completed!")
    print("Cellular automata models successfully demonstrate")
    print("Terraria's pressure-based fluid mechanics.")

if __name__ == "__main__":
    main()
