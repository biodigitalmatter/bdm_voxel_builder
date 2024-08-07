# ruff: noqa: F401

from bdm_voxel_builder.agent_algorithms.algo_10_b_slicer_agents import (
    Algo10b_VoxelSlicer,
)
from bdm_voxel_builder.config_setup import Config
from bdm_voxel_builder.visualizer.compas_viewer import CompasViewerVisualizer



# test slice
grid_size = 30
iterations = 3000
agent_count = 1
interval = iterations
name = f"test_config_slice_b_i{iterations}a{agent_count}"

config = Config(
    iterations=iterations,
    grid_size=grid_size,
    algo=Algo10b_VoxelSlicer(
        agent_count=agent_count,
        grid_size=grid_size,
        name=name,
    ),
    visualizer=CompasViewerVisualizer(save_file=True, skip_grids=("pheromon_move")),
    save_interval=interval,
    visualize_interval=interval,
)
