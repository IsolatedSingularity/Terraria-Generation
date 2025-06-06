"""
Terraria Biome Distribution Analysis

This script visualizes the spatial distribution and mathematical relationships
of different biomes in Terraria worlds:
1. Biome placement rules and constraints
2. Size distributions and probability analysis
3. Spatial relationships between major biomes
4. Evil biome vs Hallow positioning patterns
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle, Ellipse
import seaborn as sns
import os

# Set seaborn style for beautiful plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

def generate_world_layout(world_size='large', seed=12345):
    """
    Generate a typical Terraria world layout following placement rules
    
    Args:
        world_size: 'small', 'medium', or 'large'
        seed: Random seed for consistent generation
    
    Returns:
        Dictionary containing biome positions and properties
    """
    np.random.seed(seed)
      # World dimensions (in tiles) - Focus on large world only
    world_dims = {
        'large': {'width': 8400, 'height': 2400}
    }
    
    width = world_dims[world_size]['width']
    height = world_dims[world_size]['height']
    
    # Spawn point (always near center)
    spawn_x = width // 2 + np.random.randint(-100, 100)
    
    # Dungeon side (left or right, 50/50 chance)
    dungeon_side = np.random.choice(['left', 'right'])
    
    if dungeon_side == 'left':
        dungeon_x = np.random.randint(100, width//4)
        jungle_x = np.random.randint(3*width//4, width-100)
        snow_x = np.random.randint(100, width//3)
    else:
        dungeon_x = np.random.randint(3*width//4, width-100)
        jungle_x = np.random.randint(100, width//4)  
        snow_x = np.random.randint(2*width//3, width-100)
    
    # Evil biome (Corruption or Crimson, same side as dungeon typically)
    evil_type = np.random.choice(['corruption', 'crimson'])
    if dungeon_side == 'left':
        evil_x = np.random.randint(100, width//2)
    else:
        evil_x = np.random.randint(width//2, width-100)
      # Desert biomes (3 for large world)
    num_deserts = 3
    desert_positions = []
    for _ in range(num_deserts):
        desert_x = np.random.randint(200, width-200)
        desert_positions.append(desert_x)
    
    # Ocean positions (always at edges)
    ocean_left = 0
    ocean_right = width
    
    # Floating islands (8 for large world)
    num_islands = 8
    island_positions = []
    for _ in range(num_islands):
        island_x = np.random.randint(200, width-200)
        island_y = np.random.randint(150, 300)  # Sky layer
        island_positions.append((island_x, island_y))
    
    return {
        'world_size': world_size,
        'dimensions': (width, height),
        'spawn': (spawn_x, height//2),
        'dungeon': (dungeon_x, 100),
        'dungeon_side': dungeon_side,
        'jungle': (jungle_x, height//2),
        'snow': (snow_x, height//2),
        'evil': (evil_x, height//2),
        'evil_type': evil_type,
        'deserts': desert_positions,
        'oceans': [(ocean_left, height//2), (ocean_right, height//2)],
        'floating_islands': island_positions
    }

def create_biome_layout_visualization(save_path):
    """
    Create a visualization showing typical Terraria large world biome layouts
    """
    print("Creating biome layout visualization for large worlds...")
    
    # Generate three different large world examples with different seeds
    layouts = [
        generate_world_layout('large', seed=111),
        generate_world_layout('large', seed=222), 
        generate_world_layout('large', seed=333)
    ]
    
    fig, axes = plt.subplots(3, 1, figsize=(20, 15))
    fig.suptitle('Terraria Large World Biome Layouts - Mathematical Placement Analysis\n' +
                'World Generation Rules & Spatial Distribution Patterns', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Enhanced color scheme using seaborn palette
    biome_palette = sns.color_palette("Set2", 10)
    colors = {
        'forest': '#2E8B57',      # Sea Green
        'jungle': '#228B22',      # Forest Green  
        'desert': '#DEB887',      # Burlywood
        'snow': '#E0F6FF',        # Alice Blue
        'corruption': '#9370DB',  # Medium Purple
        'crimson': '#DC143C',     # Crimson
        'ocean': '#4682B4',       # Steel Blue
        'dungeon': '#696969',     # Dim Gray
        'spawn': '#FFD700',       # Gold
        'floating_island': '#87CEEB',  # Sky Blue
        'hallow': '#FFB6C1'       # Light Pink
    }
    
    for i, layout in enumerate(layouts):
        ax = axes[i]
        width, height = layout['dimensions']
        
        # Draw background (forest by default)
        forest_rect = Rectangle((0, 0), width, height, 
                               facecolor=colors['forest'], alpha=0.3, label='Forest')
        ax.add_patch(forest_rect)
        
        # Draw oceans
        ocean_width = 300
        ocean_left = Rectangle((0, 0), ocean_width, height,
                              facecolor=colors['ocean'], alpha=0.7)
        ocean_right = Rectangle((width-ocean_width, 0), ocean_width, height,
                               facecolor=colors['ocean'], alpha=0.7)
        ax.add_patch(ocean_left)
        ax.add_patch(ocean_right)
        
        # Draw major biomes
        biome_width = 400
        
        # Jungle
        jungle_x, jungle_y = layout['jungle']
        jungle_rect = Rectangle((jungle_x-biome_width//2, 0), biome_width, height*0.8,
                               facecolor=colors['jungle'], alpha=0.8)
        ax.add_patch(jungle_rect)
        ax.text(jungle_x, height*0.4, 'Jungle', ha='center', va='center', 
                fontweight='bold', color='white', fontsize=10)
        
        # Snow biome
        snow_x, snow_y = layout['snow']
        snow_rect = Rectangle((snow_x-biome_width//2, 0), biome_width, height*0.6,
                             facecolor=colors['snow'], alpha=0.8)
        ax.add_patch(snow_rect)
        ax.text(snow_x, height*0.3, 'Snow', ha='center', va='center',
                fontweight='bold', color='black', fontsize=10)
        
        # Evil biome
        evil_x, evil_y = layout['evil']
        evil_color = colors[layout['evil_type']]
        evil_rect = Rectangle((evil_x-biome_width//3, 0), biome_width//1.5, height*0.7,
                             facecolor=evil_color, alpha=0.8)
        ax.add_patch(evil_rect)
        evil_name = layout['evil_type'].capitalize()
        ax.text(evil_x, height*0.35, evil_name, ha='center', va='center',
                fontweight='bold', color='white', fontsize=10)
        
        # Desert biomes
        for desert_x in layout['deserts']:
            desert_rect = Rectangle((desert_x-150, 0), 300, height*0.4,
                                   facecolor=colors['desert'], alpha=0.8)
            ax.add_patch(desert_rect)
            ax.text(desert_x, height*0.2, 'Desert', ha='center', va='center',
                    fontweight='bold', color='black', fontsize=8)
        
        # Dungeon
        dungeon_x, dungeon_y = layout['dungeon']
        dungeon_rect = Rectangle((dungeon_x-50, dungeon_y-50), 100, 100,
                                facecolor=colors['dungeon'], alpha=0.9)
        ax.add_patch(dungeon_rect)
        ax.text(dungeon_x, dungeon_y, 'D', ha='center', va='center',
                fontweight='bold', color='white', fontsize=12)
        
        # Spawn point
        spawn_x, spawn_y = layout['spawn']
        spawn_circle = Circle((spawn_x, spawn_y), 30, 
                             facecolor=colors['spawn'], edgecolor='black', linewidth=2)
        ax.add_patch(spawn_circle)
        ax.text(spawn_x, spawn_y, 'S', ha='center', va='center',
                fontweight='bold', color='black', fontsize=10)
        
        # Floating islands
        for island_x, island_y in layout['floating_islands']:
            island_ellipse = Ellipse((island_x, island_y), 60, 20,
                                    facecolor=colors['floating_island'], alpha=0.8)
            ax.add_patch(island_ellipse)
          # Configure subplot with enhanced styling
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.invert_yaxis()  # Terraria coordinates (0,0 at top-left)
        ax.set_title(f'Large World Layout #{i+1} ({width}×{height} blocks)\n' +
                    f'Dungeon: {layout["dungeon_side"].title()}, Evil: {layout["evil_type"].title()}',
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('X Coordinate (blocks)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y Coordinate (blocks)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#F8F8FF')
        
        # Add world generation rules text (only on first subplot)
        if i == 0:
            rules_text = (
                "Large World Generation Rules:\n"
                "• Jungle always opposite Dungeon side\n"
                "• Snow biome on same side as Dungeon\n"
                "• Evil biome often near Snow/Dungeon\n"
                "• Spawn point near world center\n"
                "• Oceans at both edges\n"
                "• 3 Desert biomes distributed\n"
                "• 8 Floating islands in sky layer\n"
                "• Mathematical biome constraints"
            )
            ax.text(0.02, 0.98, rules_text, transform=ax.transAxes,
                   fontsize=10, va='top', ha='left', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, 
                            edgecolor='navy', linewidth=2))
    
    # Create legend
    legend_elements = [
        patches.Patch(color=colors['forest'], alpha=0.7, label='Forest'),
        patches.Patch(color=colors['jungle'], alpha=0.7, label='Jungle'),
        patches.Patch(color=colors['desert'], alpha=0.7, label='Desert'),
        patches.Patch(color=colors['snow'], alpha=0.7, label='Snow'),
        patches.Patch(color=colors['corruption'], alpha=0.7, label='Corruption'),
        patches.Patch(color=colors['crimson'], alpha=0.7, label='Crimson'),
        patches.Patch(color=colors['ocean'], alpha=0.7, label='Ocean'),
        patches.Patch(color=colors['dungeon'], alpha=0.7, label='Dungeon'),
        patches.Patch(color=colors['spawn'], alpha=0.7, label='Spawn'),
        patches.Patch(color=colors['floating_island'], alpha=0.7, label='Sky Island')
    ]
    
    fig.legend(handles=legend_elements, loc='center right', 
               bbox_to_anchor=(0.98, 0.5), ncol=1)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Biome layout visualization saved to {save_path}")

def create_biome_statistics_visualization(save_path):
    """
    Create statistical analysis of biome distributions for large worlds
    """
    print("Creating biome statistics visualization for large worlds...")
    
    # Generate statistical data from multiple large world samples
    num_samples = 150
    
    # Collect statistics for large worlds only
    stats = {
        'jungle_distances': [],
        'dungeon_distances': [], 
        'evil_distances': [],
        'snow_distances': [],
        'desert_counts': [],
        'island_counts': [],
        'dungeon_sides': {'left': 0, 'right': 0},
        'evil_types': {'corruption': 0, 'crimson': 0},
        'biome_spacings': [],
        'world_balance_scores': []
    }
    
    # Generate samples
    for i in range(num_samples):
        layout = generate_world_layout('large', seed=i*42)
        width = layout['dimensions'][0]
        spawn_x = layout['spawn'][0]
        
        # Calculate distances from spawn (normalized)
        jungle_dist = abs(layout['jungle'][0] - spawn_x) / width
        dungeon_dist = abs(layout['dungeon'][0] - spawn_x) / width  
        evil_dist = abs(layout['evil'][0] - spawn_x) / width
        snow_dist = abs(layout['snow'][0] - spawn_x) / width
        
        stats['jungle_distances'].append(jungle_dist)
        stats['dungeon_distances'].append(dungeon_dist)
        stats['evil_distances'].append(evil_dist)
        stats['snow_distances'].append(snow_dist)
        stats['desert_counts'].append(len(layout['deserts']))
        stats['island_counts'].append(len(layout['floating_islands']))
        
        # Count categorical data
        stats['dungeon_sides'][layout['dungeon_side']] += 1
        stats['evil_types'][layout['evil_type']] += 1
        
        # Calculate biome spacing (average distance between major biomes)
        major_biomes = [layout['jungle'][0], layout['dungeon'][0], layout['evil'][0], layout['snow'][0]]
        spacings = []
        for j in range(len(major_biomes)):
            for k in range(j+1, len(major_biomes)):
                spacings.append(abs(major_biomes[j] - major_biomes[k]) / width)
        stats['biome_spacings'].append(np.mean(spacings))
        
        # Calculate world balance score (how evenly distributed biomes are)
        positions = sorted([layout['jungle'][0], layout['dungeon'][0], layout['evil'][0], layout['snow'][0]])
        expected_spacing = width / 5  # Ideal spacing
        actual_spacings = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
        balance_score = 1 - np.std(actual_spacings) / expected_spacing
        stats['world_balance_scores'].append(max(0, balance_score))
    
    # Create enhanced visualization with seaborn styling
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Terraria Large World Biome Statistics & Mathematical Analysis\n' +
                'Statistical Distribution Patterns from 150 Generated Worlds', 
                fontsize=18, fontweight='bold', y=0.98)
    
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)
    
    # Use seaborn color palette
    palette = sns.color_palette("husl", 8)
    
    # 1. Distance distributions from spawn
    ax1 = fig.add_subplot(gs[0, :])
    distances = [stats['jungle_distances'], stats['dungeon_distances'], 
                stats['evil_distances'], stats['snow_distances']]
    labels = ['Jungle', 'Dungeon', 'Evil Biome', 'Snow']
    colors = [palette[0], palette[1], palette[2], palette[3]]
    
    # Create violin plots for better distribution visualization
    parts = ax1.violinplot(distances, positions=range(1, 5), showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    ax1.set_xlabel('Biome Type', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Distance from Spawn (normalized)', fontsize=12, fontweight='bold')
    ax1.set_title('Biome Distance Distributions from Spawn Point', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(1, 5))
    ax1.set_xticklabels(labels)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#FAFAFA')
    
    # 2. Structure count distributions
    ax2 = fig.add_subplot(gs[1, 0])
    desert_counts = np.bincount(stats['desert_counts'])
    ax2.bar(range(len(desert_counts)), desert_counts, color=palette[4], alpha=0.8)
    ax2.set_title('Desert Count Distribution', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Number of Deserts')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[1, 1])
    island_counts = np.bincount(stats['island_counts'])
    ax3.bar(range(len(island_counts)), island_counts, color=palette[5], alpha=0.8)
    ax3.set_title('Floating Island Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Number of Islands')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3)
    
    # 3. Categorical distributions
    ax4 = fig.add_subplot(gs[1, 2])
    dungeon_data = list(stats['dungeon_sides'].values())
    ax4.pie(dungeon_data, labels=['Left', 'Right'], autopct='%1.1f%%', 
           colors=[palette[6], palette[7]], startangle=90)
    ax4.set_title('Dungeon Side Distribution', fontsize=12, fontweight='bold')
    
    # 4. Evil biome distribution
    ax5 = fig.add_subplot(gs[2, 0])
    evil_data = list(stats['evil_types'].values())
    ax5.pie(evil_data, labels=['Corruption', 'Crimson'], autopct='%1.1f%%',
           colors=['#9370DB', '#DC143C'], startangle=90)
    ax5.set_title('Evil Biome Type Distribution', fontsize=12, fontweight='bold')
    
    # 5. Biome spacing analysis
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.hist(stats['biome_spacings'], bins=20, color=palette[0], alpha=0.7, edgecolor='black')
    ax6.axvline(np.mean(stats['biome_spacings']), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(stats["biome_spacings"]):.3f}')
    ax6.set_title('Average Biome Spacing', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Normalized Distance')
    ax6.set_ylabel('Frequency')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 6. World balance scores
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.hist(stats['world_balance_scores'], bins=20, color=palette[1], alpha=0.7, edgecolor='black')
    ax7.axvline(np.mean(stats['world_balance_scores']), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {np.mean(stats["world_balance_scores"]):.3f}')
    ax7.set_title('World Balance Distribution', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Balance Score (0-1)')
    ax7.set_ylabel('Frequency')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 7. Correlation matrix
    ax8 = fig.add_subplot(gs[3, :])
    correlation_data = np.array([
        stats['jungle_distances'],
        stats['dungeon_distances'], 
        stats['evil_distances'],
        stats['snow_distances'],
        stats['biome_spacings'],
        stats['world_balance_scores']
    ])
    
    corr_matrix = np.corrcoef(correlation_data)
    corr_labels = ['Jungle Dist', 'Dungeon Dist', 'Evil Dist', 'Snow Dist', 'Avg Spacing', 'Balance Score']
    
    im = ax8.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax8.set_xticks(range(len(corr_labels)))
    ax8.set_yticks(range(len(corr_labels)))
    ax8.set_xticklabels(corr_labels, rotation=45, ha='right')
    ax8.set_yticklabels(corr_labels)
    ax8.set_title('Biome Parameter Correlation Matrix', fontsize=14, fontweight='bold')
    
    # Add correlation values to the plot
    for i in range(len(corr_labels)):
        for j in range(len(corr_labels)):
            text = ax8.text(j, i, f'{corr_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black" if abs(corr_matrix[i, j]) < 0.5 else "white")
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax8, orientation='horizontal', pad=0.1, shrink=0.8)
    cbar.set_label('Correlation Coefficient', fontsize=12)
    
    # Add statistical summary text
    summary_text = (
        f"Statistical Summary (n={num_samples}):\n"
        f"• Average jungle distance: {np.mean(stats['jungle_distances']):.3f} ± {np.std(stats['jungle_distances']):.3f}\n"
        f"• Average dungeon distance: {np.mean(stats['dungeon_distances']):.3f} ± {np.std(stats['dungeon_distances']):.3f}\n"        f"• Desert count mode: {stats['desert_counts'].count(max(set(stats['desert_counts']), key=stats['desert_counts'].count))}\n"
        f"• Island count mode: {stats['island_counts'].count(max(set(stats['island_counts']), key=stats['island_counts'].count))}\n"
        f"• Dungeon side preference: {max(stats['dungeon_sides'], key=stats['dungeon_sides'].get).title()}\n"
        f"• Evil biome preference: {max(stats['evil_types'], key=stats['evil_types'].get).title()}"
    )
    
    fig.text(0.02, 0.15, summary_text, fontsize=11, ha='left', va='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8, 
                      edgecolor='navy', linewidth=2))
    
    plt.tight_layout(rect=[0, 0.2, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Enhanced biome statistics visualization saved to {save_path}")

if __name__ == "__main__":
    print("Starting Terraria biome distribution analysis")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Plots')
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Generate biome layout visualization
        create_biome_layout_visualization(os.path.join(output_dir, 'terraria_biome_layouts.png'))
        
        # Generate biome statistics visualization
        create_biome_statistics_visualization(os.path.join(output_dir, 'terraria_biome_statistics.png'))
        
        print("All biome distribution visualizations completed successfully")
        
    except Exception as e:
        print(f"Error in biome distribution visualization generation: {e}")
