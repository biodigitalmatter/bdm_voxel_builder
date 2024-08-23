# ruff: noqa: F401

from bdm_voxel_builder.agent_algorithms.algo_10_b_slicer_agents import (
    Algo10b_VoxelSlicer,
)
from bdm_voxel_builder.agent_algorithms.algo_10_c_slicer_agents_import import (
    Algo10c_VoxelSlicer,
)
from bdm_voxel_builder.agent_algorithms.algo_10_d_slicer_agents_3x3 import (
    Algo10d_VoxelSlicer,
)
from bdm_voxel_builder.agent_algorithms.algo_10_e_slicer_bullets import (
    Algo10e_VoxelSlicer,
)
from bdm_voxel_builder.agent_algorithms.algo_10_f_slicer_bullets_2 import (
    Algo10f_VoxelSlicer,
)
from bdm_voxel_builder.config_setup import Config
from bdm_voxel_builder.visualizer.compas_viewer import CompasViewerVisualizer

# test slice
grid_size = [30, 30, 30]
iterations = 2000
agent_count = 1
interval = 10
name = f"slicer_f_walk_r_7-1_{iterations}a{agent_count}"

config = Config(
    iterations=iterations,
    grid_size=grid_size,
    algo=Algo10f_VoxelSlicer(
        agent_count=agent_count,
        grid_size=grid_size,
        name=name,
    ),
    visualizer=CompasViewerVisualizer(save_file=True, skip_grids=("pheromon_move")),
    save_interval=interval,
    visualize_interval=interval,
)
