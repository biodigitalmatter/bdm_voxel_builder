# ruff: noqa: F401
from bdm_voxel_builder.agent_algorithms.algo_10_a_slicer_agents import Algo10a_VoxelSlicer
from bdm_voxel_builder.config_setup import Config
from bdm_voxel_builder.visualizer.compas_viewer import CompasViewerVisualizer

# SHORT RUN test compasview
grid_size = 30
iterations = 4000
agent_count = 1
interval = iterations / 10
name = f"slicer_test_simple_i{iterations}a{agent_count}"

config = Config(
    iterations=iterations,
    grid_size=grid_size,
    algo=Algo10a_VoxelSlicer(
        agent_count=agent_count,
        grid_size=grid_size,
        name=name,
    ),
    visualizer=CompasViewerVisualizer(save_file=True, skip_grids=("pheromon_move")),
    save_interval=interval,
    visualize_interval=interval,
)
