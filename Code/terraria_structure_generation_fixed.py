"""
Terraria Structure Generation Visualization
==========================================

This module visualizes the placement and generation patterns of various structures
in Terraria worlds, including dungeons, temples, sky islands, and underground cabins.
Focuses on large world generation (8400x2400 blocks).

Mathematical Foundation:
- Structure placement uses weighted probability distributions
- Distance constraints follow Euclidean metrics: d = √((x₂-x₁)² + (y₂-y₁)²)
- Biome-specific placement rules use conditional probability matrices
- Sky island generation follows Poisson distribution for spacing

Author: Terraria Generation Project
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import random
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set seaborn style for beautiful plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class TerrariaStructureGenerator:
    """
    Generates and visualizes Terraria world structures following game mechanics.
    Focuses on large world generation (8400x2400 blocks).
    """
    
    def __init__(self, world_width: int = 8400, world_height: int = 2400):
        """
        Initialize the structure generator for large world.
        
        Args:
            world_width: Width of the world in blocks (default: large world)
            world_height: Height of the world in blocks (default: large world)
        """
        self.world_width = world_width
        self.world_height = world_height
        self.surface_level = world_height // 4  # Approximate surface level
        self.cavern_level = int(world_height * 0.6)  # Cavern layer
        self.hell_level = int(world_height * 0.85)  # Hell layer
        self.structures = {}
        
        # Enhanced structure parameters for large world
        self.structure_params = {
            'dungeon': {'size': (200, 300), 'depth_range': (0.2, 0.5), 'count': 1},
            'jungle_temple': {'size': (150, 120), 'depth_range': (0.5, 0.7), 'count': 1},
            'sky_islands': {'size': (80, 40), 'height_range': (0.1, 0.25), 'count': (8, 12)},
            'underground_cabins': {'size': (25, 20), 'depth_range': (0.2, 0.6), 'count': (25, 35)},
            'ore_veins': {'size': (12, 10), 'depth_range': (0.1, 0.9), 'count': (400, 600)},
            'corruption_chasms': {'size': (40, 150), 'depth_range': (0.0, 0.4), 'count': (5, 8)},
            'floating_islands': {'size': (60, 35), 'height_range': (0.15, 0.3), 'count': (6, 10)},
            'hell_structures': {'size': (30, 25), 'depth_range': (0.85, 1.0), 'count': (15, 25)},
            'ruined_houses': {'size': (20, 15), 'depth_range': (0.85, 0.95), 'count': (8, 12)},
            'hell_towers': {'size': (15, 40), 'depth_range': (0.88, 1.0), 'count': (5, 8)}
        }
    
    def generate_structure_positions(self, structure_type: str) -> List[Tuple[int, int]]:
        """
        Generate positions for a specific structure type using Terraria's placement rules.
        
        Mathematical approach:
        - Weighted random sampling with distance constraints
        - Biome-specific probability modifiers
        - Minimum separation distances to prevent overlap
        
        Args:
            structure_type: Type of structure to generate
            
        Returns:
            List of (x, y) positions for the structures
        """
        params = self.structure_params[structure_type]
        positions = []
        
        if structure_type == 'dungeon':
            # Dungeon spawns on one side of the world
            side = random.choice(['left', 'right'])
            x = self.world_width // 6 if side == 'left' else 5 * self.world_width // 6
            y = int(self.surface_level * (1 + params['depth_range'][0]))
            positions.append((x, y))
            
        elif structure_type == 'jungle_temple':
            # Temple spawns in jungle biome (typically right side, deep underground)
            x = int(0.7 * self.world_width + random.randint(-200, 200))
            y = int(self.surface_level * (1 + params['depth_range'][0]))
            positions.append((x, y))
            
        elif 'islands' in structure_type or structure_type == 'sky_islands':
            # Sky structures use Poisson distribution for spacing
            count = random.randint(*params['count']) if isinstance(params['count'], tuple) else params['count']
            min_distance = self.world_width // (count + 2)
            
            for i in range(count):
                attempts = 0
                while attempts < 50:  # Prevent infinite loops
                    x = random.randint(params['size'][0], self.world_width - params['size'][0])
                    y = int(self.surface_level * (1 - random.uniform(*params['height_range'])))
                    
                    # Check minimum distance from other structures
                    valid = True
                    for pos_x, pos_y in positions:
                        distance = np.sqrt((x - pos_x)**2 + (y - pos_y)**2)
                        if distance < min_distance:
                            valid = False
                            break
                    
                    if valid:
                        positions.append((x, y))
                        break
                    attempts += 1
                    
        elif structure_type == 'underground_cabins':
            # Cabins distributed throughout underground with clustering
            count = random.randint(*params['count'])
            
            # Create clusters of cabins
            num_clusters = count // 4
            for cluster in range(num_clusters):
                cluster_x = random.randint(200, self.world_width - 200)
                cluster_y = int(self.surface_level * (1 + random.uniform(*params['depth_range'])))
                
                # Generate cabins around cluster center
                cabins_per_cluster = random.randint(2, 6)
                for _ in range(cabins_per_cluster):
                    x = cluster_x + random.randint(-150, 150)
                    y = cluster_y + random.randint(-50, 50)
                    x = max(params['size'][0], min(self.world_width - params['size'][0], x))
                    y = max(params['size'][1], min(self.world_height - params['size'][1], y))
                    positions.append((x, y))
                    
        elif structure_type == 'ore_veins':
            # Ore veins follow depth-based probability distribution
            count = random.randint(*params['count'])
            for _ in range(count):
                # Deeper ores are rarer but more valuable
                depth_factor = np.random.exponential(0.5)  # Exponential distribution
                depth_factor = min(1.0, depth_factor)
                x = random.randint(0, self.world_width)
                y = int(self.surface_level * (1 + depth_factor * 0.8))
                positions.append((x, y))
                    
        elif structure_type == 'corruption_chasms':
            # Chasms spawn in corruption biome with specific spacing
            count = random.randint(*params['count'])
            corruption_center = self.world_width // 4  # Assume corruption on left side
            
            for i in range(count):
                x = corruption_center + random.randint(-300, 300)
                y = int(self.surface_level * (1 + random.uniform(*params['depth_range'])))
                positions.append((x, y))
        
        elif structure_type in ['hell_structures', 'ruined_houses', 'hell_towers']:
            # Hell layer structures (new additions)
            count = random.randint(*params['count']) if isinstance(params['count'], tuple) else params['count']
            
            for _ in range(count):
                x = random.randint(params['size'][0], self.world_width - params['size'][0])
                # Hell layer positioning with some variation
                base_hell_y = int(self.world_height * params['depth_range'][0])
                max_hell_y = int(self.world_height * params['depth_range'][1])
                y = random.randint(base_hell_y, max_hell_y)
                
                # Ensure minimum spacing for hell structures
                valid = True
                for pos_x, pos_y in positions:
                    distance = np.sqrt((x - pos_x)**2 + (y - pos_y)**2)
                    if distance < 100:  # Minimum 100 block spacing
                        valid = False
                        break
                
                if valid:
                    positions.append((x, y))
        
        return positions
    
    def generate_all_structures(self) -> Dict[str, List[Tuple[int, int]]]:
        """
        Generate positions for all structure types.
        
        Returns:
            Dictionary mapping structure types to their positions
        """
        self.structures = {}
        for structure_type in self.structure_params.keys():
            self.structures[structure_type] = self.generate_structure_positions(structure_type)
        
        return self.structures
    
    def visualize_structure_overview(self, save_path: str = None) -> None:
        """
        Create a comprehensive overview of all structures in the large world.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.structures:
            self.generate_all_structures()
        
        fig, ax = plt.subplots(figsize=(24, 16))
        
        # Enhanced color scheme using seaborn palette
        structure_colors = {
            'dungeon': '#4A4A4A',
            'jungle_temple': '#8B4513',
            'sky_islands': '#87CEEB',
            'underground_cabins': '#DEB887',
            'ore_veins': '#FFD700',
            'corruption_chasms': '#800080',
            'floating_islands': '#98FB98',
            'hell_structures': '#FF4500',
            'ruined_houses': '#8B0000',
            'hell_towers': '#DC143C'
        }
        
        # Plot world boundaries with enhanced styling
        world_rect = patches.Rectangle((0, 0), self.world_width, self.world_height, 
                                     linewidth=3, edgecolor='black', facecolor='lightsteelblue', alpha=0.2)
        ax.add_patch(world_rect)
        
        # Draw layer boundaries
        ax.axhline(y=self.surface_level, color='saddlebrown', linestyle='-', linewidth=3, alpha=0.8, label='Surface Level')
        ax.axhline(y=self.cavern_level, color='darkgray', linestyle='--', linewidth=2, alpha=0.7, label='Cavern Layer')
        ax.axhline(y=self.hell_level, color='darkred', linestyle='-.', linewidth=3, alpha=0.8, label='Hell Layer')
        
        # Plot each structure type with enhanced markers
        structure_counts = {}
        for structure_type, positions in self.structures.items():
            if not positions:
                continue
                
            structure_counts[structure_type] = len(positions)
            color = structure_colors.get(structure_type, '#000000')
            
            # Enhanced markers for different structure types
            if 'hell' in structure_type or structure_type == 'ruined_houses':
                marker = 's'  # Square for hell structures
                size = 80
            elif 'island' in structure_type or structure_type == 'sky_islands':
                marker = 'o'  # Circle for sky structures
                size = 60
            elif structure_type == 'dungeon':
                marker = 'D'  # Diamond for dungeon
                size = 200
            elif structure_type == 'jungle_temple':
                marker = '^'  # Triangle for temple
                size = 150
            else:
                marker = '.'  # Dot for smaller structures
                size = 40
            
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            
            ax.scatter(x_coords, y_coords, c=color, s=size, marker=marker, 
                      alpha=0.8, edgecolors='black', linewidth=1, 
                      label=f'{structure_type.replace("_", " ").title()} ({len(positions)})')
        
        # Enhanced styling
        ax.set_xlim(0, self.world_width)
        ax.set_ylim(0, self.world_height)
        ax.invert_yaxis()  # Terraria coordinate system
        ax.set_xlabel('X Coordinate (blocks)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Y Coordinate (blocks)', fontsize=14, fontweight='bold')
        ax.set_title('Terraria Large World Structure Distribution Analysis\n' +
                    'Complete Structure Placement & Density Visualization', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#F0F8FF')
        
        # Enhanced legend
        legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=11, 
                          frameon=True, fancybox=True, shadow=True, ncol=1)
        legend.set_title('Structure Types & Counts', prop={'size': 12, 'weight': 'bold'})
        
        # Add mathematical information
        total_structures = sum(structure_counts.values())
        density = total_structures / (self.world_width * self.world_height) * 1e6  # per million blocks
        
        info_text = (
            f"Large World Analysis:\n"
            f"• World size: {self.world_width}×{self.world_height} blocks\n"
            f"• Total structures: {total_structures}\n"
            f"• Structure density: {density:.2f} per million blocks\n"
            f"• Surface layer: 0-{self.surface_level} blocks\n"
            f"• Cavern layer: {self.surface_level}-{self.hell_level} blocks\n"
            f"• Hell layer: {self.hell_level}-{self.world_height} blocks"
        )
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=12,
               verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
               facecolor='white', alpha=0.9, edgecolor='gray', linewidth=2))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"Enhanced structure overview saved to {save_path}")
        else:
            plt.show()

    def visualize_structure_density_heatmap(self, save_path: str = None) -> None:
        """
        Create a density heatmap showing structure concentration across the world.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.structures:
            self.generate_all_structures()
        
        # Create density grid
        grid_size = 50
        x_bins = np.linspace(0, self.world_width, grid_size)
        y_bins = np.linspace(0, self.world_height, grid_size)
        density_grid = np.zeros((grid_size-1, grid_size-1))
        
        # Count structures in each grid cell
        for structure_type, positions in self.structures.items():
            for x, y in positions:
                x_idx = np.digitize(x, x_bins) - 1
                y_idx = np.digitize(y, y_bins) - 1
                
                # Ensure indices are within bounds
                x_idx = max(0, min(grid_size-2, x_idx))
                y_idx = max(0, min(grid_size-2, y_idx))
                
                density_grid[y_idx, x_idx] += 1
        
        # Create the heatmap
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Custom colormap for structure density
        colors = ['#000033', '#000066', '#003399', '#0066CC', '#3399FF', '#66CCFF', '#FFFF66', '#FFCC00', '#FF6600', '#FF0000']
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('structure_density', colors, N=n_bins)
        
        im = ax.imshow(density_grid, extent=[0, self.world_width, self.world_height, 0], 
                      cmap=cmap, aspect='auto', interpolation='bilinear')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Structure Density', fontsize=12, fontweight='bold')
        
        # Overlay structure positions
        structure_colors = {
            'dungeon': 'white', 'jungle_temple': 'yellow', 'sky_islands': 'cyan',
            'underground_cabins': 'orange', 'ore_veins': 'gold', 
            'corruption_chasms': 'magenta', 'floating_islands': 'lime',
            'hell_structures': 'red', 'ruined_houses': 'darkred', 'hell_towers': 'crimson'
        }
        
        for structure_type, positions in self.structures.items():
            if positions:
                xs, ys = zip(*positions)
                ax.scatter(xs, ys, c=structure_colors.get(structure_type, 'white'), 
                          s=20, alpha=0.8, edgecolors='black', linewidth=0.5)
        
        # Customize plot
        ax.set_xlabel('World Width (blocks)', fontsize=12, fontweight='bold')
        ax.set_ylabel('World Depth (blocks)', fontsize=12, fontweight='bold')
        ax.set_title('Terraria Large World Structure Density Heatmap\n' +
                    'Showing Concentration Patterns Across World Layers', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Add layer boundaries
        ax.axhline(y=self.surface_level, color='white', linestyle='--', linewidth=2, alpha=0.8)
        ax.axhline(y=self.cavern_level, color='white', linestyle='--', linewidth=2, alpha=0.8)
        ax.axhline(y=self.hell_level, color='white', linestyle='-.', linewidth=3, alpha=0.8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"Structure density heatmap saved to: {save_path}")
        else:
            plt.show()
    
    def analyze_structure_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze statistical properties of structure placement.
        
        Returns:
            Dictionary containing statistics for each structure type
        """
        if not self.structures:
            self.generate_all_structures()
        
        stats = {}
        
        for structure_type, positions in self.structures.items():
            if not positions:
                continue
                
            xs, ys = zip(*positions) if positions else ([], [])
            
            # Calculate statistics
            structure_stats = {
                'count': len(positions),
                'mean_x': np.mean(xs) if xs else 0,
                'mean_y': np.mean(ys) if ys else 0,
                'std_x': np.std(xs) if xs else 0,
                'std_y': np.std(ys) if ys else 0,
                'depth_ratio': np.mean([y / self.world_height for y in ys]) if ys else 0
            }
            
            # Calculate nearest neighbor distances
            if len(positions) > 1:
                distances = []
                for i, (x1, y1) in enumerate(positions):
                    min_dist = float('inf')
                    for j, (x2, y2) in enumerate(positions):
                        if i != j:
                            dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                            min_dist = min(min_dist, dist)
                    distances.append(min_dist)
                
                structure_stats['mean_nearest_distance'] = np.mean(distances)
                structure_stats['clustering_coefficient'] = np.std(distances) / np.mean(distances)
            else:
                structure_stats['mean_nearest_distance'] = 0
                structure_stats['clustering_coefficient'] = 0
            
            stats[structure_type] = structure_stats
        
        return stats

def main():
    """
    Main function to run structure generation visualizations for large world only.
    """
    print("Generating Terraria Large World Structure Visualizations...")
    print("=" * 55)
    
    plots_dir = r"c:\Users\hunkb\OneDrive\Desktop\Terraria Generation\Plots"
    
    print(f"\nGenerating Large World Structure Analysis...")
    
    generator = TerrariaStructureGenerator(8400, 2400)  # Large world only
    
    # Generate structure overview
    overview_path = f"{plots_dir}/terraria_structure_overview_large.png"
    generator.visualize_structure_overview(overview_path)
    
    # Generate density heatmap
    heatmap_path = f"{plots_dir}/terraria_structure_density_large.png"
    generator.visualize_structure_density_heatmap(heatmap_path)
    
    # Print statistics
    stats = generator.analyze_structure_statistics()
    print(f"\nLarge World Structure Statistics:")
    print("-" * 40)
    for structure_type, structure_stats in stats.items():
        print(f"{structure_type.replace('_', ' ').title()}:")
        print(f"  Count: {structure_stats['count']}")
        print(f"  Average Depth Ratio: {structure_stats['depth_ratio']:.3f}")
        print(f"  Clustering Coefficient: {structure_stats['clustering_coefficient']:.3f}")
    
    print("\n" + "=" * 55)
    print("Large world structure generation visualizations completed!")
    print("Mathematical models successfully demonstrate Terraria's")
    print("structure placement algorithms and probability distributions.")

if __name__ == "__main__":
    main()
