"""
Terraria Complete World Evolution Master Animation
=================================================

This module creates the ultimate comprehensive animation showing the complete
evolution of a Terraria world from initial generation through corruption/crimson
spread and hardmode transformations. It integrates all visualization systems
into a cohesive narrative of world development.

Animation Sequence:
1. Initial World Generation (103-pass system)
2. Pre-hardmode Biome Establishment
3. Corruption/Crimson Initial Infection
4. Wall of Flesh Defeat & Hardmode Trigger
5. V-Pattern Corruption/Hallow Spread
6. Altar Breaking & Ore Generation
7. Mechanical Boss Arena Construction
8. Environmental Transformation Completion

Mathematical Integration:
- Synchronized timing algorithms
- Multi-system state management
- Interpolated transitions between phases
- Statistical tracking across all systems
- Performance optimization for large-scale visualization

Author: Generated for Terraria Generation Analysis
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
from typing import Tuple, List, Dict, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
try:
    from terraria_world_generation import TerrariaWorldGenerator
    from terraria_corruption_evolution import TerrariaCorruptionEvolution
    from terraria_hardmode_structures import TerrariaHardmodeTransformation
except ImportError:
    print("Warning: Could not import all custom modules. Make sure all files are in the same directory.")

# Set visualization preferences
plt.style.use('dark_background')
sns.set_palette("mako")

class TerrariaWorldEvolutionMaster:
    """
    Master class that orchestrates the complete evolution of a Terraria world,
    integrating world generation, corruption spread, and hardmode transformations
    into a single comprehensive visualization system.
    
    This system provides:
    - Synchronized multi-system simulation
    - Comprehensive animation capabilities
    - Statistical analysis across all phases
    - Performance-optimized rendering
    - Modular component integration
    """
    
    def __init__(self, world_width: int = 4200, world_height: int = 1200):
        """
        Initialize the master world evolution system.
        
        Parameters:
        -----------
        world_width : int
            World width in blocks (4200 for large world)
        world_height : int
            World height in blocks (1200 for large world)
        """
        self.width = world_width
        self.height = world_height
        
        # Initialize component systems
        self.world_generator = TerrariaWorldGenerator(world_width, world_height)
        self.corruption_system = TerrariaCorruptionEvolution(world_width, world_height)
        self.hardmode_system = TerrariaHardmodeTransformation(world_width, world_height)
        
        # Evolution phases
        self.phases = [
            'Initial Generation',
            'Surface Formation', 
            'Cave Carving',
            'Biome Placement',
            'Structure Generation',
            'Pre-hardmode Stabilization',
            'Initial Corruption',
            'Corruption Spread',
            'Wall of Flesh Defeat',
            'Hardmode V-Pattern',
            'Altar Breaking Wave 1',
            'Altar Breaking Wave 2', 
            'Altar Breaking Wave 3',
            'Chlorophyte Generation',
            'Boss Arena Construction',
            'Final Hardmode State'
        ]
        
        # Evolution timeline
        self.phase_durations = {
            'Initial Generation': 5,
            'Surface Formation': 8,
            'Cave Carving': 10,
            'Biome Placement': 6,
            'Structure Generation': 8,
            'Pre-hardmode Stabilization': 3,
            'Initial Corruption': 5,
            'Corruption Spread': 20,
            'Wall of Flesh Defeat': 3,
            'Hardmode V-Pattern': 8,
            'Altar Breaking Wave 1': 6,
            'Altar Breaking Wave 2': 6,
            'Altar Breaking Wave 3': 6,
            'Chlorophyte Generation': 8,
            'Boss Arena Construction': 10,
            'Final Hardmode State': 5
        }
        
        # State tracking
        self.world_states = []
        self.phase_markers = []
        self.statistics_history = []
        self.current_phase = 0
        
        # Enhanced color system combining all modules
        self.master_colors = {
            0: (0.02, 0.02, 0.02),    # Empty/Air
            1: (0.4, 0.3, 0.2),       # Dirt
            2: (0.5, 0.5, 0.5),       # Stone
            3: (0.2, 0.7, 0.2),       # Grass
            4: (0.1, 0.5, 0.1),       # Jungle
            5: (0.9, 0.8, 0.5),       # Sand
            6: (0.9, 0.9, 1.0),       # Snow
            7: (0.7, 0.9, 1.0),       # Ice
            8: (0.2, 0.4, 0.8),       # Water
            9: (0.8, 0.3, 0.1),       # Lava
            10: (0.3, 0.0, 0.5),      # Corruption
            11: (0.7, 0.0, 0.2),      # Crimson
            12: (1.0, 0.8, 1.0),      # Hallow
            13: (0.0, 0.3, 0.8),      # Cobalt
            14: (0.2, 0.8, 0.2),      # Mythril
            15: (0.8, 0.2, 0.2),      # Adamantite
            16: (0.0, 1.0, 0.0),      # Chlorophyte
            17: (0.6, 0.1, 0.1),      # Altars
            18: (1.0, 0.2, 0.8),      # Crystal Hearts
            19: (0.9, 0.9, 0.0),      # Boss Arenas
            20: (0.8, 0.6, 0.4),      # Structures
        }
        
    def run_complete_evolution(self, fast_mode: bool = False) -> None:
        """
        Execute the complete world evolution from generation to final hardmode state.
        
        Parameters:
        -----------
        fast_mode : bool
            If True, reduces animation frames for faster processing
        """
        print("Starting Complete Terraria World Evolution")
        print("=" * 50)
        
        frame_multiplier = 0.5 if fast_mode else 1.0
        
        # Phase 1-5: World Generation
        print("Phase 1-5: Initial World Generation")
        self._run_world_generation_phases(frame_multiplier)
        
        # Phase 6: Pre-hardmode stabilization
        print("Phase 6: Pre-hardmode Stabilization")
        self._run_prehardmode_phase(frame_multiplier)
        
        # Phase 7-8: Corruption spread
        print("Phase 7-8: Initial Corruption and Spread")
        self._run_corruption_phases(frame_multiplier)
        
        # Phase 9: Wall of Flesh defeat trigger
        print("Phase 9: Wall of Flesh Defeat")
        self._run_wof_defeat_phase(frame_multiplier)
        
        # Phase 10-15: Hardmode transformation
        print("Phase 10-15: Hardmode Transformation")
        self._run_hardmode_phases(frame_multiplier)
        
        print(f"\nComplete evolution finished!")
        print(f"Total frames generated: {len(self.world_states)}")
        print(f"Evolution phases: {len(self.phases)}")
    
    def _run_world_generation_phases(self, frame_multiplier: float) -> None:
        """Run the initial world generation phases (1-5)."""
        # Initialize world generator
        world = np.zeros((self.height, self.width), dtype=int)
        
        # Phase 1: Initial Generation
        phase_frames = int(self.phase_durations['Initial Generation'] * frame_multiplier)
        for frame in range(phase_frames):
            self.world_states.append(world.copy())
            self._record_statistics(world, 'Initial Generation')
          # Phase 2: Surface Formation
        print("  Generating surface terrain...")
        self.world_generator.generate_surface_terrain()
        surface_world = self.world_generator.world.copy()
        phase_frames = int(self.phase_durations['Surface Formation'] * frame_multiplier)
        
        for frame in range(phase_frames):
            # Interpolate between empty and surface
            progress = frame / max(1, phase_frames - 1)
            interpolated_world = self._interpolate_worlds(world, surface_world, progress)
            self.world_states.append(interpolated_world)
            self._record_statistics(interpolated_world, 'Surface Formation')
        
        world = surface_world.copy()
        
        # Phase 3: Cave Carving
        print("  Carving cave systems...")
        self.world_generator.carve_caves()
        cave_world = self.world_generator.world.copy()
        phase_frames = int(self.phase_durations['Cave Carving'] * frame_multiplier)
        
        for frame in range(phase_frames):
            progress = frame / max(1, phase_frames - 1)
            interpolated_world = self._interpolate_worlds(world, cave_world, progress)
            self.world_states.append(interpolated_world)
            self._record_statistics(interpolated_world, 'Cave Carving')
        
        world = cave_world.copy()
        
        # Phase 4: Biome Placement
        print("  Placing biomes...")
        self.world_generator.place_biomes()
        biome_world = self.world_generator.world.copy()
        phase_frames = int(self.phase_durations['Biome Placement'] * frame_multiplier)
        
        for frame in range(phase_frames):
            progress = frame / max(1, phase_frames - 1)
            interpolated_world = self._interpolate_worlds(world, biome_world, progress)
            self.world_states.append(interpolated_world)
            self._record_statistics(interpolated_world, 'Biome Placement')
        
        world = biome_world.copy()
        
        # Phase 5: Structure Generation
        print("  Generating structures...")
        self.world_generator.place_structures()
        final_world = self.world_generator.world.copy()
        phase_frames = int(self.phase_durations['Structure Generation'] * frame_multiplier)
        
        for frame in range(phase_frames):
            progress = frame / max(1, phase_frames - 1)
            interpolated_world = self._interpolate_worlds(world, final_world, progress)
            self.world_states.append(interpolated_world)
            self._record_statistics(interpolated_world, 'Structure Generation')
        
        # Store final world for next phases
        self.base_world = final_world.copy()
    
    def _run_prehardmode_phase(self, frame_multiplier: float) -> None:
        """Run the pre-hardmode stabilization phase."""
        phase_frames = int(self.phase_durations['Pre-hardmode Stabilization'] * frame_multiplier)
        
        for frame in range(phase_frames):
            self.world_states.append(self.base_world.copy())
            self._record_statistics(self.base_world, 'Pre-hardmode Stabilization')
    
    def _run_corruption_phases(self, frame_multiplier: float) -> None:
        """Run the corruption initialization and spread phases."""
        # Initialize corruption system with base world
        self.corruption_system.world = self._adapt_world_for_corruption(self.base_world.copy())
        
        # Phase 7: Initial Corruption
        print("  Initializing corruption points...")
        self.corruption_system.initialize_corruption_points()
        
        phase_frames = int(self.phase_durations['Initial Corruption'] * frame_multiplier)
        for frame in range(phase_frames):
            world_adapted = self._adapt_corruption_for_display(self.corruption_system.world)
            self.world_states.append(world_adapted)
            self._record_statistics(world_adapted, 'Initial Corruption')
        
        # Phase 8: Corruption Spread
        print("  Simulating corruption spread...")
        spread_frames = int(self.phase_durations['Corruption Spread'] * frame_multiplier)
        
        for frame in range(spread_frames):
            self.corruption_system.simulate_spread_step(hardmode=False)
            world_adapted = self._adapt_corruption_for_display(self.corruption_system.world)
            self.world_states.append(world_adapted)
            self._record_statistics(world_adapted, 'Corruption Spread')
    
    def _run_wof_defeat_phase(self, frame_multiplier: float) -> None:
        """Run the Wall of Flesh defeat phase."""
        phase_frames = int(self.phase_durations['Wall of Flesh Defeat'] * frame_multiplier)
        
        print("  Wall of Flesh defeated - Hardmode activated!")
        
        for frame in range(phase_frames):
            # Add dramatic effect for WoF defeat
            world_adapted = self._adapt_corruption_for_display(self.corruption_system.world)
            # Could add visual effects here (flashing, color changes, etc.)
            self.world_states.append(world_adapted)
            self._record_statistics(world_adapted, 'Wall of Flesh Defeat')
    
    def _run_hardmode_phases(self, frame_multiplier: float) -> None:
        """Run all hardmode transformation phases."""
        # Phase 10: Hardmode V-Pattern
        print("  Triggering hardmode V-pattern spread...")
        self.corruption_system.trigger_hardmode_spread()
        
        phase_frames = int(self.phase_durations['Hardmode V-Pattern'] * frame_multiplier)
        for frame in range(phase_frames):
            world_adapted = self._adapt_corruption_for_display(self.corruption_system.world)
            self.world_states.append(world_adapted)
            self._record_statistics(world_adapted, 'Hardmode V-Pattern')
        
        # Initialize hardmode system
        self.hardmode_system.world = self._adapt_world_for_hardmode(self.corruption_system.world.copy())
        
        # Phases 11-13: Altar Breaking
        for wave in range(1, 4):
            phase_name = f'Altar Breaking Wave {wave}'
            print(f"  {phase_name}...")
            
            # Break altars and generate ore
            self._simulate_altar_breaking_wave(wave)
            
            phase_frames = int(self.phase_durations[phase_name] * frame_multiplier)
            for frame in range(phase_frames):
                world_adapted = self._adapt_hardmode_for_display(self.hardmode_system.world)
                self.world_states.append(world_adapted)
                self._record_statistics(world_adapted, phase_name)
        
        # Phase 14: Chlorophyte Generation
        print("  Generating chlorophyte...")
        self.hardmode_system.generate_chlorophyte()
        
        phase_frames = int(self.phase_durations['Chlorophyte Generation'] * frame_multiplier)
        for frame in range(phase_frames):
            world_adapted = self._adapt_hardmode_for_display(self.hardmode_system.world)
            self.world_states.append(world_adapted)
            self._record_statistics(world_adapted, 'Chlorophyte Generation')
        
        # Phase 15: Boss Arena Construction
        print("  Constructing boss arenas...")
        self.hardmode_system.create_boss_arenas()
        
        phase_frames = int(self.phase_durations['Boss Arena Construction'] * frame_multiplier)
        for frame in range(phase_frames):
            world_adapted = self._adapt_hardmode_for_display(self.hardmode_system.world)
            self.world_states.append(world_adapted)
            self._record_statistics(world_adapted, 'Boss Arena Construction')
        
        # Phase 16: Final State
        print("  Finalizing hardmode state...")
        self.hardmode_system.add_crystal_hearts()
        
        phase_frames = int(self.phase_durations['Final Hardmode State'] * frame_multiplier)
        for frame in range(phase_frames):
            world_adapted = self._adapt_hardmode_for_display(self.hardmode_system.world)
            self.world_states.append(world_adapted)
            self._record_statistics(world_adapted, 'Final Hardmode State')
    
    def _interpolate_worlds(self, world1: np.ndarray, world2: np.ndarray, progress: float) -> np.ndarray:
        """Interpolate between two world states for smooth transitions."""
        if progress <= 0:
            return world1.copy()
        elif progress >= 1:
            return world2.copy()
        
        # Create probabilistic interpolation
        result = world1.copy()
        differences = world1 != world2
        
        # Apply changes gradually based on progress
        change_probability = progress
        random_mask = np.random.random(world1.shape) < change_probability
        
        # Apply changes where both difference exists and random allows
        change_mask = differences & random_mask
        result[change_mask] = world2[change_mask]
        
        return result
    
    def _adapt_world_for_corruption(self, world: np.ndarray) -> np.ndarray:
        """Adapt base world format for corruption system."""
        # Map world generator block types to corruption system types
        adapted = np.zeros_like(world)
        
        # Simple mapping - extend as needed
        adapted[world == 1] = self.corruption_system.DIRT  # Dirt
        adapted[world == 2] = self.corruption_system.STONE  # Stone
        adapted[world == 3] = self.corruption_system.GRASS  # Grass
        adapted[world == 4] = self.corruption_system.JUNGLE  # Jungle
        adapted[world == 5] = self.corruption_system.SAND  # Sand
        adapted[world == 6] = self.corruption_system.SNOW  # Snow
        
        return adapted
    
    def _adapt_corruption_for_display(self, world: np.ndarray) -> np.ndarray:
        """Adapt corruption system world for display."""
        # Map corruption system types to display types
        display_world = np.zeros_like(world)
        
        display_world[world == self.corruption_system.EMPTY] = 0
        display_world[world == self.corruption_system.DIRT] = 1
        display_world[world == self.corruption_system.STONE] = 2
        display_world[world == self.corruption_system.GRASS] = 3
        display_world[world == self.corruption_system.JUNGLE] = 4
        display_world[world == self.corruption_system.SAND] = 5
        display_world[world == self.corruption_system.SNOW] = 6
        display_world[world == self.corruption_system.CORRUPTION] = 10
        display_world[world == self.corruption_system.CRIMSON] = 11
        display_world[world == self.corruption_system.HALLOW] = 12
        
        return display_world
    
    def _adapt_world_for_hardmode(self, world: np.ndarray) -> np.ndarray:
        """Adapt corruption world for hardmode system."""
        # Map corruption types to hardmode types
        adapted = np.zeros_like(world)
        
        adapted[world == self.corruption_system.EMPTY] = self.hardmode_system.EMPTY
        adapted[world == self.corruption_system.DIRT] = self.hardmode_system.DIRT
        adapted[world == self.corruption_system.STONE] = self.hardmode_system.STONE
        adapted[world == self.corruption_system.CORRUPTION] = self.hardmode_system.CORRUPTION
        adapted[world == self.corruption_system.CRIMSON] = self.hardmode_system.CRIMSON
        adapted[world == self.corruption_system.HALLOW] = self.hardmode_system.HALLOW
        
        # Place altars randomly in stone areas
        stone_positions = np.where(adapted == self.hardmode_system.STONE)
        if len(stone_positions[0]) > 0:
            num_altars = 12
            indices = np.random.choice(len(stone_positions[0]), 
                                     size=min(num_altars, len(stone_positions[0])), 
                                     replace=False)
            for idx in indices:
                y, x = stone_positions[0][idx], stone_positions[1][idx]
                adapted[y, x] = self.hardmode_system.ALTAR
        
        return adapted
    
    def _adapt_hardmode_for_display(self, world: np.ndarray) -> np.ndarray:
        """Adapt hardmode system world for display."""
        display_world = np.zeros_like(world)
        
        display_world[world == self.hardmode_system.EMPTY] = 0
        display_world[world == self.hardmode_system.DIRT] = 1
        display_world[world == self.hardmode_system.STONE] = 2
        display_world[world == self.hardmode_system.CORRUPTION] = 10
        display_world[world == self.hardmode_system.CRIMSON] = 11
        display_world[world == self.hardmode_system.HALLOW] = 12
        display_world[world == self.hardmode_system.COBALT] = 13
        display_world[world == self.hardmode_system.MYTHRIL] = 14
        display_world[world == self.hardmode_system.ADAMANTITE] = 15
        display_world[world == self.hardmode_system.CHLOROPHYTE] = 16
        display_world[world == self.hardmode_system.ALTAR] = 17
        display_world[world == self.hardmode_system.CRYSTAL_HEART] = 18
        display_world[world == self.hardmode_system.BOSS_ARENA] = 19
        
        return display_world
    
    def _simulate_altar_breaking_wave(self, wave: int) -> None:
        """Simulate one wave of altar breaking."""
        ore_type = {1: self.hardmode_system.COBALT,
                   2: self.hardmode_system.MYTHRIL,
                   3: self.hardmode_system.ADAMANTITE}[wave]
        
        # Break 3 altars
        self.hardmode_system._break_altar_wave(3)
        
        # Generate corresponding ore
        self.hardmode_system._generate_ore_wave(ore_type, 3)
    
    def _record_statistics(self, world: np.ndarray, phase: str) -> None:
        """Record statistics for the current world state."""
        stats = {
            'phase': phase,
            'frame': len(self.statistics_history),
            'empty_blocks': np.sum(world == 0),
            'dirt_blocks': np.sum(world == 1),
            'stone_blocks': np.sum(world == 2),
            'corruption_blocks': np.sum(world == 10),
            'crimson_blocks': np.sum(world == 11),
            'hallow_blocks': np.sum(world == 12),
            'ore_blocks': np.sum(np.isin(world, [13, 14, 15, 16])),
            'special_blocks': np.sum(np.isin(world, [17, 18, 19]))
        }
        
        self.statistics_history.append(stats)
    
    def create_master_animation(self, save_path: str = None, fps: int = 10) -> FuncAnimation:
        """
        Create the master animation showing complete world evolution.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the animation
        fps : int
            Frames per second for the animation
            
        Returns:
        --------
        FuncAnimation
            The created master animation
        """
        if not self.world_states:
            print("No world states available. Run complete evolution first.")
            return None
        
        print(f"Creating master animation with {len(self.world_states)} frames...")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], width_ratios=[3, 1])
        
        # Main world view
        ax_world = fig.add_subplot(gs[0, :])
        
        # Statistics views
        ax_blocks = fig.add_subplot(gs[1, 0])
        ax_special = fig.add_subplot(gs[1, 1])
        
        # Setup color mapping
        colors = [self.master_colors[i] for i in range(len(self.master_colors))]
        cmap = ListedColormap(colors)
        
        # Initial setup
        sample_rate = max(1, self.width // 600)
        initial_world = self.world_states[0][::sample_rate, ::sample_rate]
        
        im_world = ax_world.imshow(initial_world, cmap=cmap, aspect='auto')
        ax_world.set_title("Terraria Complete World Evolution", 
                          fontsize=18, fontweight='bold')
        ax_world.set_xlabel("World X Position")
        ax_world.set_ylabel("World Y Position")
        
        # Setup statistics plots
        ax_blocks.set_title("Block Distribution", fontsize=12, fontweight='bold')
        ax_blocks.set_xlabel("Frame")
        ax_blocks.set_ylabel("Block Count")
        
        ax_special.set_title("Special Features", fontsize=12, fontweight='bold')
        ax_special.set_xlabel("Frame") 
        ax_special.set_ylabel("Feature Count")
        
        # Initialize plot lines
        stat_lines = {}
        colors_stats = sns.color_palette("cubehelix", 6)
        
        for i, (stat_name, color) in enumerate(zip(['dirt_blocks', 'stone_blocks', 'corruption_blocks', 
                                                   'ore_blocks', 'hallow_blocks', 'special_blocks'], 
                                                  colors_stats)):
            line, = ax_blocks.plot([], [], color=color, linewidth=2, label=stat_name.replace('_', ' ').title())
            stat_lines[stat_name] = line
        
        ax_blocks.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_blocks.grid(True, alpha=0.3)
        
        # Special features line
        special_line, = ax_special.plot([], [], color=sns.color_palette("rocket")[3], linewidth=3)
        ax_special.grid(True, alpha=0.3)
        
        def animate(frame):
            if frame >= len(self.world_states):
                return [im_world] + list(stat_lines.values()) + [special_line]
            
            # Update world visualization
            world_sample = self.world_states[frame][::sample_rate, ::sample_rate]
            im_world.set_array(world_sample)
            
            # Update title with current phase
            current_stats = self.statistics_history[frame] if frame < len(self.statistics_history) else self.statistics_history[-1]
            phase_name = current_stats['phase']
            ax_world.set_title(f"Terraria World Evolution - {phase_name} (Frame {frame})", 
                             fontsize=18, fontweight='bold')
            
            # Update statistics
            if frame < len(self.statistics_history):
                x_data = list(range(frame + 1))
                
                # Update each statistic line
                for stat_name, line in stat_lines.items():
                    y_data = [self.statistics_history[i][stat_name] for i in range(frame + 1)]
                    line.set_data(x_data, y_data)
                
                # Update special features
                special_data = [self.statistics_history[i]['special_blocks'] for i in range(frame + 1)]
                special_line.set_data(x_data, special_data)
                
                # Auto-scale axes
                if frame > 0:
                    ax_blocks.relim()
                    ax_blocks.autoscale_view()
                    ax_special.relim()
                    ax_special.autoscale_view()
            
            return [im_world] + list(stat_lines.values()) + [special_line]
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=len(self.world_states), 
                           interval=1000//fps, blit=False, repeat=True)
        
        if save_path:
            print(f"Saving master animation to {save_path}...")
            writer = PillowWriter(fps=fps)
            anim.save(save_path, writer=writer)
            print(f"Master animation saved successfully!")
        
        return anim
    
    def create_summary_visualization(self, save_path: str = None) -> plt.Figure:
        """
        Create a comprehensive summary visualization of the complete evolution.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the visualization
            
        Returns:
        --------
        plt.Figure
            The created summary figure
        """
        if not self.world_states:
            print("No world states available. Run complete evolution first.")
            return None
        
        # Create figure with complex layout
        fig = plt.figure(figsize=(24, 18))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Key snapshots
        snapshot_indices = [0, len(self.world_states)//4, len(self.world_states)//2, -1]
        snapshot_titles = ['Initial Generation', 'Pre-hardmode', 'Early Hardmode', 'Final State']
        
        colors = [self.master_colors[i] for i in range(len(self.master_colors))]
        cmap = ListedColormap(colors)
        sample_rate = max(1, self.width // 400)
        
        # Plot key snapshots
        for i, (idx, title) in enumerate(zip(snapshot_indices, snapshot_titles)):
            ax = fig.add_subplot(gs[0, i])
            world_sample = self.world_states[idx][::sample_rate, ::sample_rate]
            im = ax.imshow(world_sample, cmap=cmap, aspect='auto')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Evolution timeline
        ax_timeline = fig.add_subplot(gs[1, :])
        self._plot_evolution_timeline(ax_timeline)
        
        # Detailed statistics
        ax_stats1 = fig.add_subplot(gs[2, :2])
        self._plot_block_evolution(ax_stats1)
        
        ax_stats2 = fig.add_subplot(gs[2, 2:])
        self._plot_phase_analysis(ax_stats2)
        
        # Final analysis
        ax_final1 = fig.add_subplot(gs[3, :2])
        self._plot_final_composition(ax_final1)
        
        ax_final2 = fig.add_subplot(gs[3, 2:])
        self._plot_evolution_metrics(ax_final2)
        
        plt.suptitle("Terraria Complete World Evolution Summary", 
                    fontsize=20, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Summary visualization saved to {save_path}")
        
        return fig
    
    def _plot_evolution_timeline(self, ax: plt.Axes) -> None:
        """Plot the complete evolution timeline."""
        frames = [stat['frame'] for stat in self.statistics_history]
        phases = [stat['phase'] for stat in self.statistics_history]
        
        # Create phase regions
        phase_changes = []
        current_phase = ""
        
        for i, phase in enumerate(phases):
            if phase != current_phase:
                phase_changes.append((i, phase))
                current_phase = phase
        
        # Plot phase regions
        colors = sns.color_palette("cubehelix", len(phase_changes))
        for i, ((start_frame, phase_name), color) in enumerate(zip(phase_changes, colors)):
            end_frame = phase_changes[i+1][0] if i+1 < len(phase_changes) else len(frames)
            ax.axvspan(start_frame, end_frame, alpha=0.3, color=color, label=phase_name)
        
        ax.set_title("Evolution Timeline", fontsize=14, fontweight='bold')
        ax.set_xlabel("Frame Number")
        ax.set_ylabel("Evolution Phase")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    def _plot_block_evolution(self, ax: plt.Axes) -> None:
        """Plot the evolution of different block types."""
        frames = [stat['frame'] for stat in self.statistics_history]
        
        block_types = ['dirt_blocks', 'stone_blocks', 'corruption_blocks', 'hallow_blocks', 'ore_blocks']
        colors = sns.color_palette("mako", len(block_types))
        
        for block_type, color in zip(block_types, colors):
            counts = [stat[block_type] for stat in self.statistics_history]
            ax.plot(frames, counts, color=color, linewidth=2, 
                   label=block_type.replace('_', ' ').title())
        
        ax.set_title("Block Type Evolution", fontsize=14, fontweight='bold')
        ax.set_xlabel("Frame Number")
        ax.set_ylabel("Block Count")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_phase_analysis(self, ax: plt.Axes) -> None:
        """Plot phase-by-phase analysis."""
        # Calculate average statistics per phase
        phase_stats = {}
        for stat in self.statistics_history:
            phase = stat['phase']
            if phase not in phase_stats:
                phase_stats[phase] = []
            phase_stats[phase].append(stat['ore_blocks'] + stat['special_blocks'])
        
        phases = list(phase_stats.keys())
        avg_features = [np.mean(stats) for stats in phase_stats.values()]
        
        colors = sns.color_palette("rocket", len(phases))
        bars = ax.bar(range(len(phases)), avg_features, color=colors)
        
        ax.set_title("Features by Phase", fontsize=14, fontweight='bold')
        ax.set_xlabel("Phase")
        ax.set_ylabel("Average Special Features")
        ax.set_xticks(range(len(phases)))
        ax.set_xticklabels(phases, rotation=45, ha='right')
    
    def _plot_final_composition(self, ax: plt.Axes) -> None:
        """Plot final world composition."""
        if not self.statistics_history:
            return
        
        final_stats = self.statistics_history[-1]
        
        # Get composition data
        composition = {
            'Empty': final_stats['empty_blocks'],
            'Dirt': final_stats['dirt_blocks'],
            'Stone': final_stats['stone_blocks'],
            'Corruption': final_stats['corruption_blocks'],
            'Hallow': final_stats['hallow_blocks'],
            'Ores': final_stats['ore_blocks'],
            'Special': final_stats['special_blocks']
        }
        
        # Remove zero values
        composition = {k: v for k, v in composition.items() if v > 0}
        
        colors = sns.color_palette("cubehelix", len(composition))
        wedges, texts, autotexts = ax.pie(composition.values(), labels=composition.keys(), 
                                         autopct='%1.1f%%', colors=colors)
        
        ax.set_title("Final World Composition", fontsize=14, fontweight='bold')
    
    def _plot_evolution_metrics(self, ax: plt.Axes) -> None:
        """Plot key evolution metrics."""
        if len(self.statistics_history) < 2:
            return
        
        frames = [stat['frame'] for stat in self.statistics_history]
        
        # Calculate diversity index (simplified Shannon diversity)
        diversity_scores = []
        for stat in self.statistics_history:
            total_blocks = sum([stat['dirt_blocks'], stat['stone_blocks'], 
                              stat['corruption_blocks'], stat['hallow_blocks'], 
                              stat['ore_blocks'], stat['special_blocks']])
            
            if total_blocks > 0:
                proportions = np.array([stat['dirt_blocks'], stat['stone_blocks'], 
                                      stat['corruption_blocks'], stat['hallow_blocks'], 
                                      stat['ore_blocks'], stat['special_blocks']]) / total_blocks
                proportions = proportions[proportions > 0]  # Remove zeros
                diversity = -np.sum(proportions * np.log(proportions))
                diversity_scores.append(diversity)
            else:
                diversity_scores.append(0)
        
        ax.plot(frames, diversity_scores, color=sns.color_palette("rocket")[3], 
               linewidth=3, label='Block Diversity')
        
        ax.set_title("World Complexity Evolution", fontsize=14, fontweight='bold')
        ax.set_xlabel("Frame Number")
        ax.set_ylabel("Diversity Index")
        ax.legend()
        ax.grid(True, alpha=0.3)

# Example usage and testing
if __name__ == "__main__":
    print("Terraria Complete World Evolution Master System")
    print("=" * 55)
    
    # Create master evolution system
    master_system = TerrariaWorldEvolutionMaster(world_width=2100, world_height=600)  # Smaller for demo
    
    # Run complete evolution (fast mode for demo)
    master_system.run_complete_evolution(fast_mode=True)
      # Create summary visualization
    summary_fig = master_system.create_summary_visualization(save_path="master_evolution_summary.png")
    
    if summary_fig:
        plt.savefig("master_evolution_summary.png", dpi=300, bbox_inches='tight')
        print("Master evolution summary saved to master_evolution_summary.png")
        plt.close()  # Close instead of show to prevent blocking
        
        # Optionally create master animation (uncomment if desired)
        # master_anim = master_system.create_master_animation(save_path="master_evolution_animation.gif", fps=5)
    
    print("\nMaster evolution system demonstration complete!")
    print("Summary:")
    print(f"- Total frames generated: {len(master_system.world_states)}")
    print(f"- Evolution phases completed: {len(set(stat['phase'] for stat in master_system.statistics_history))}")
    print(f"- Final world complexity: {len(master_system.statistics_history)} data points")
