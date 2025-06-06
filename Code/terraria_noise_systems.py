"""
Terraria Noise Systems Visualization

This script visualizes the core noise algorithms used in Terraria's world generation:
1. Surface terrain generation using 1D Perlin noise
2. Cave generation using 2D noise with threshold-based carving
3. Multi-octave fractal noise for complex terrain features
4. Biome transition zones using gradient interpolation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import os

# Set seaborn style for beautiful plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Simple noise implementation to replace opensimplex
class SimpleNoise:
    def __init__(self, seed=12345):
        np.random.seed(seed)
        self.perm = np.random.permutation(256)
        
    def noise2d(self, x, y):
        # Simple noise implementation using sine functions
        return np.sin(x * 6.283) * np.cos(y * 6.283) * 0.5 + \
               np.sin(x * 12.566) * np.cos(y * 12.566) * 0.25 + \
               np.sin(x * 25.132) * np.cos(y * 25.132) * 0.125

def perlin_1d(x, frequency=0.01, octaves=4, amplitude=50, persistence=0.5):
    """
    Generate 1D Perlin noise for surface terrain
    
    Args:
        x: Input coordinates
        frequency: Base frequency of the noise
        octaves: Number of noise layers
        amplitude: Maximum height variation
        persistence: Amplitude reduction per octave
    
    Returns:
        height values for terrain surface
    """
    noise_gen = SimpleNoise(seed=12345)
    height = 0
    current_amplitude = amplitude
    current_frequency = frequency
    
    for _ in range(octaves):
        height += current_amplitude * noise_gen.noise2d(x * current_frequency, 0)
        current_amplitude *= persistence
        current_frequency *= 2
    
    return height

def cave_noise_2d(x, y, frequency=0.05, threshold=0.25):
    """
    Generate 2D noise for cave systems
    
    Args:
        x, y: 2D coordinate arrays
        frequency: Noise frequency
        threshold: Cave formation threshold
    
    Returns:
        Boolean array where True = cave (air), False = solid
    """
    noise_gen = SimpleNoise(seed=67890)
    noise_values = np.zeros_like(x)
    
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            noise_values[i, j] = noise_gen.noise2d(x[i, j] * frequency, y[i, j] * frequency)
    
    return np.abs(noise_values) < threshold

def biome_transition(x, biome1_value, biome2_value, transition_center, transition_width):
    """
    Create smooth biome transitions using gradient interpolation
    
    Args:
        x: X coordinates
        biome1_value, biome2_value: Values for each biome
        transition_center: Center of transition zone
        transition_width: Width of transition zone
    
    Returns:
        Interpolated values across transition
    """
    noise_gen = SimpleNoise(seed=11111)
    
    # Base transition using distance from center
    distance = np.abs(x - transition_center)
    base_factor = np.clip(distance / transition_width, 0, 1)
    
    # Add noise for natural variation
    noise_factor = 0.1 * np.array([noise_gen.noise2d(xi * 0.005, 0) for xi in x])
    
    # Combine factors
    blend_factor = np.clip(base_factor + noise_factor, 0, 1)
    
    return biome1_value * (1 - blend_factor) + biome2_value * blend_factor

def create_surface_terrain_visualization(save_path):
    """
    Visualize surface terrain generation using 1D noise for all biome types (Large World)
    """
    print("Creating comprehensive surface terrain visualization for large world...")
    
    # Generate terrain data for large world (8400 blocks wide)
    world_width = 8400
    x = np.linspace(0, world_width, world_width // 4)  # Subsample for performance
    
    # Different biome terrain types with distinct characteristics
    forest_terrain = 100 + np.array([perlin_1d(xi, frequency=0.01, octaves=4, amplitude=30) for xi in x])
    desert_terrain = 90 + np.array([perlin_1d(xi, frequency=0.008, octaves=3, amplitude=15) for xi in x])
    jungle_terrain = 95 + np.array([perlin_1d(xi, frequency=0.015, octaves=5, amplitude=45) for xi in x])
    snow_terrain = 110 + np.array([perlin_1d(xi, frequency=0.012, octaves=4, amplitude=35) for xi in x])
    corruption_terrain = 105 + np.array([perlin_1d(xi, frequency=0.02, octaves=6, amplitude=40) for xi in x])
    crimson_terrain = 105 + np.array([perlin_1d(xi, frequency=0.018, octaves=6, amplitude=38) for xi in x])
    mushroom_terrain = 98 + np.array([perlin_1d(xi, frequency=0.01, octaves=3, amplitude=20) for xi in x])
    hallow_terrain = 103 + np.array([perlin_1d(xi, frequency=0.025, octaves=5, amplitude=35) for xi in x])
    
    # Create beautiful color palette using seaborn
    palette = sns.color_palette("viridis", 8)
    biome_colors = {
        'forest': '#2E8B57',      # Sea Green
        'desert': '#DEB887',      # Burlywood  
        'jungle': '#228B22',      # Forest Green
        'snow': '#E0F6FF',        # Alice Blue
        'corruption': '#9370DB',   # Medium Purple
        'crimson': '#DC143C',      # Crimson
        'mushroom': '#8A2BE2',     # Blue Violet
        'hallow': '#FFB6C1'        # Light Pink
    }
      # Create visualization with better spacing and styling
    fig, axes = plt.subplots(8, 1, figsize=(18, 16))
    fig.suptitle('Terraria Surface Terrain Generation - Large World Analysis\nMathematical Noise Models Across All Biomes', 
                fontsize=20, fontweight='bold', y=0.98)
    
    terrains = [
        (forest_terrain, 'Forest Biome', biome_colors['forest'], '#90EE90'),
        (desert_terrain, 'Desert Biome', biome_colors['desert'], '#FFFF99'), 
        (jungle_terrain, 'Jungle Biome', biome_colors['jungle'], '#ADFF2F'),
        (snow_terrain, 'Snow Biome', biome_colors['snow'], '#F0F8FF'),
        (corruption_terrain, 'Corruption Biome', biome_colors['corruption'], '#E6E6FA'),
        (crimson_terrain, 'Crimson Biome', biome_colors['crimson'], '#FFB6C1'),
        (mushroom_terrain, 'Mushroom Biome', biome_colors['mushroom'], '#DDA0DD'),
        (hallow_terrain, 'Hallow Biome', biome_colors['hallow'], '#FFF0F5')
    ]
    
    for i, (terrain, title, base_color, surface_color) in enumerate(terrains):
        axes[i].fill_between(x, 0, terrain, color=base_color, alpha=0.8, label='Base Terrain')
        axes[i].fill_between(x, terrain, terrain + 8, color=surface_color, alpha=0.7, label='Surface Layer')
        axes[i].set_title(title, fontsize=14, fontweight='bold', pad=10)
        axes[i].set_ylabel('Height (blocks)', fontsize=11, fontweight='bold')
        axes[i].legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        axes[i].grid(True, alpha=0.3, linestyle='--')
        axes[i].set_xlim(0, world_width)
        
        # Add subtle background gradient
        axes[i].set_facecolor('#FAFAFA')
    
    axes[-1].set_xlabel('World Position (blocks)', fontsize=12, fontweight='bold')
    
    # Add mathematical formula with better positioning
    fig.text(0.02, 0.02, 
             r'$height(x) = base + \sum_{i=0}^{octaves} amplitude \cdot persistence^i \cdot noise(x \cdot frequency \cdot 2^i)$',
             fontsize=13, ha='left', va='bottom', 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Enhanced surface terrain visualization saved to {save_path}")

def create_cave_system_visualization(save_path):
    """
    Visualize cave generation using 2D noise for large world
    """
    print("Creating cave system visualization for large world...")
    
    # Generate cave data for large world cross-section
    width, height = 500, 300
    x = np.linspace(0, width, width)
    y = np.linspace(0, height, height)
    X, Y = np.meshgrid(x, y)
    
    # Different cave densities with seaborn color palette
    palette = sns.color_palette("mako_r", 4)
    sparse_caves = cave_noise_2d(X, Y, frequency=0.03, threshold=0.15)
    normal_caves = cave_noise_2d(X, Y, frequency=0.05, threshold=0.25)
    dense_caves = cave_noise_2d(X, Y, frequency=0.08, threshold=0.35)
    hell_caves = cave_noise_2d(X, Y, frequency=0.12, threshold=0.45)  # New: Hell layer caves
    
    # Create visualization with seaborn styling
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('Terraria Cave System Generation - Large World Cross-Section\nNoise-Based Cave Carving Algorithms', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Color maps: True (cave/air) = black, False (solid) = earth tones
    cave_colors = sns.blend_palette(['#8B4513', '#000000'], as_cmap=True)
    
    # Sparse caves (Surface/Underground)
    axes[0,0].imshow(sparse_caves, cmap=cave_colors, origin='upper', extent=[0, width, height, 0])
    axes[0,0].set_title('Surface & Underground Caves\n(freq=0.03, threshold=0.15)', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('X Coordinate (blocks)', fontweight='bold')
    axes[0,0].set_ylabel('Depth (blocks)', fontweight='bold')
    
    # Normal caves (Cavern layer)
    axes[0,1].imshow(normal_caves, cmap=cave_colors, origin='upper', extent=[0, width, height, 0])
    axes[0,1].set_title('Cavern Layer Caves\n(freq=0.05, threshold=0.25)', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('X Coordinate (blocks)', fontweight='bold')
    
    # Dense caves (Deep underground)
    axes[1,0].imshow(dense_caves, cmap=cave_colors, origin='upper', extent=[0, width, height, 0])
    axes[1,0].set_title('Deep Underground Caves\n(freq=0.08, threshold=0.35)', fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel('X Coordinate (blocks)', fontweight='bold')
    axes[1,0].set_ylabel('Depth (blocks)', fontweight='bold')
    
    # Hell layer caves
    hell_cmap = sns.blend_palette(['#8B0000', '#FF4500', '#000000'], as_cmap=True)
    axes[1,1].imshow(hell_caves, cmap=hell_cmap, origin='upper', extent=[0, width, height, 0])
    axes[1,1].set_title('Hell Layer Caves\n(freq=0.12, threshold=0.45)', fontsize=14, fontweight='bold')
    axes[1,1].set_xlabel('X Coordinate (blocks)', fontweight='bold')
    
    # Add styling to all subplots
    for ax in axes.flat:
        ax.grid(True, alpha=0.3, color='white', linestyle='--')
        ax.set_facecolor('#2F2F2F')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#8B4513', label='Solid Block'),
        Patch(facecolor='#000000', label='Cave (Air)'),
        Patch(facecolor='#FF4500', label='Lava/Hell')
    ]
    fig.legend(handles=legend_elements, loc='center', ncol=3, 
               bbox_to_anchor=(0.5, 0.02), fontsize=12, frameon=True, 
               fancybox=True, shadow=True)
    
    # Add mathematical formula
    fig.text(0.02, 0.95, 
             r'$cave(x,y) = |noise_{2D}(x \cdot freq, y \cdot freq)| < threshold$' + '\n' +
             r'$density \propto frequency \times threshold$',
             fontsize=14, ha='left', va='top', color='white',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8, edgecolor='gray'))
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.94])
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Enhanced cave system visualization saved to {save_path}")

def create_biome_transition_visualization(save_path):
    """
    Visualize biome transitions using gradient interpolation for large world
    """
    print("Creating biome transition visualization for large world...")
    
    # Generate biome transition data for large world
    world_width = 8400  # Large world width
    x = np.linspace(0, world_width, world_width // 4)  # Subsample for performance
    
    # Define multiple biome transition zones across large world
    forest_to_desert = biome_transition(x, 1.0, 0.0, 1000, 200)    # Forest (1) to Desert (0)
    desert_to_jungle = biome_transition(x, 0.0, 2.0, 2500, 300)    # Desert (0) to Jungle (2)
    jungle_to_corruption = biome_transition(x, 2.0, 3.0, 4200, 250)  # Jungle (2) to Corruption (3)
    corruption_to_snow = biome_transition(x, 3.0, 4.0, 6000, 400)   # Corruption (3) to Snow (4)
    snow_to_hallow = biome_transition(x, 4.0, 5.0, 7500, 350)      # Snow (4) to Hallow (5)
    
    # Create enhanced color palette using seaborn
    biome_palette = sns.color_palette("Set2", 8)
    
    # Create visualization with enhanced styling
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))
    fig.suptitle('Terraria Biome Transitions - Large World Analysis\nMathematical Gradient Interpolation Models', 
                fontsize=20, fontweight='bold', y=0.98)
    
    # Top plot: Biome values with transitions
    transition_colors = sns.color_palette("husl", 5)
    ax1.plot(x, forest_to_desert, color=transition_colors[0], linewidth=3, label='Forest → Desert', alpha=0.8)
    ax1.plot(x, desert_to_jungle, color=transition_colors[1], linewidth=3, label='Desert → Jungle', alpha=0.8) 
    ax1.plot(x, jungle_to_corruption, color=transition_colors[2], linewidth=3, label='Jungle → Corruption', alpha=0.8)
    ax1.plot(x, corruption_to_snow, color=transition_colors[3], linewidth=3, label='Corruption → Snow', alpha=0.8)
    ax1.plot(x, snow_to_hallow, color=transition_colors[4], linewidth=3, label='Snow → Hallow', alpha=0.8)
    
    ax1.set_title('Biome Transition Functions - Large World Scale', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Biome Blend Factor', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=12, frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.4, linestyle='--')
    ax1.set_facecolor('#FAFAFA')
    
    # Bottom plot: Enhanced visual biome map
    biome_map = np.zeros((100, len(x), 3))  # RGB image with more height
    
    for i, xi in enumerate(x):
        # Determine dominant biome at each position with smooth transitions
        if xi < 800:
            # Forest region
            color = np.array([34/255, 139/255, 34/255])  # Forest Green
        elif xi < 1200:
            # Forest to Desert transition
            blend = (xi - 800) / 400
            forest_color = np.array([34/255, 139/255, 34/255])
            desert_color = np.array([244/255, 164/255, 96/255])
            color = forest_color * (1 - blend) + desert_color * blend
        elif xi < 2200:
            # Desert region  
            color = np.array([244/255, 164/255, 96/255])  # Sandy Brown
        elif xi < 2800:
            # Desert to Jungle transition
            blend = (xi - 2200) / 600
            desert_color = np.array([244/255, 164/255, 96/255])
            jungle_color = np.array([139/255, 69/255, 19/255])
            color = desert_color * (1 - blend) + jungle_color * blend
        elif xi < 3950:
            # Jungle region
            color = np.array([139/255, 69/255, 19/255])  # Saddle Brown
        elif xi < 4450:
            # Jungle to Corruption transition
            blend = (xi - 3950) / 500
            jungle_color = np.array([139/255, 69/255, 19/255])
            corruption_color = np.array([138/255, 43/255, 226/255])
            color = jungle_color * (1 - blend) + corruption_color * blend
        elif xi < 5600:
            # Corruption region
            color = np.array([138/255, 43/255, 226/255])  # Blue Violet
        elif xi < 6400:
            # Corruption to Snow transition
            blend = (xi - 5600) / 800
            corruption_color = np.array([138/255, 43/255, 226/255])
            snow_color = np.array([224/255, 224/255, 224/255])
            color = corruption_color * (1 - blend) + snow_color * blend
        elif xi < 7150:
            # Snow region
            color = np.array([224/255, 224/255, 224/255])  # Light Gray
        elif xi < 7850:
            # Snow to Hallow transition
            blend = (xi - 7150) / 700
            snow_color = np.array([224/255, 224/255, 224/255])
            hallow_color = np.array([255/255, 182/255, 193/255])
            color = snow_color * (1 - blend) + hallow_color * blend
        else:
            # Hallow region
            color = np.array([255/255, 182/255, 193/255])  # Light Pink
            
        biome_map[:, i, :] = color
    
    ax2.imshow(biome_map, extent=[0, world_width, 0, 100], aspect='auto')
    ax2.set_title('Large World Biome Distribution with Smooth Transitions', fontsize=16, fontweight='bold')
    ax2.set_xlabel('X Coordinate (blocks)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Surface Layer', fontsize=14, fontweight='bold')
    
    # Add biome labels with better positioning
    label_positions = [400, 1600, 3100, 4800, 6300, 7500]
    labels = ['Forest', 'Desert', 'Jungle', 'Corruption', 'Snow', 'Hallow']
    text_colors = ['white', 'black', 'white', 'white', 'black', 'black']
    
    for pos, label, color in zip(label_positions, labels, text_colors):
        ax2.text(pos, 50, label, ha='center', va='center', fontweight='bold', 
                color=color, fontsize=12, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Add mathematical formula with enhanced styling
    formula_text = (
        r'$blend(x) = biome_1 \cdot (1-t) + biome_2 \cdot t$' + '\n' +
        r'where $t = \text{sigmoid}(\frac{distance - center}{width}) + noise(x)$'
    )
    fig.text(0.02, 0.96, formula_text, fontsize=14, ha='left', va='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, 
                      edgecolor='gray', linewidth=2))
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Enhanced biome transition visualization saved to {save_path}")

if __name__ == "__main__":
    print("Starting Terraria noise systems visualization generation")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Plots')
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Generate surface terrain visualization
        create_surface_terrain_visualization(os.path.join(output_dir, 'terraria_surface_terrain.png'))
        
        # Generate cave system visualization  
        create_cave_system_visualization(os.path.join(output_dir, 'terraria_cave_systems.png'))
        
        # Generate biome transition visualization
        create_biome_transition_visualization(os.path.join(output_dir, 'terraria_biome_transitions.png'))
        
        print("All noise system visualizations completed successfully")
        
    except Exception as e:
        print(f"Error in noise systems visualization generation: {e}")
