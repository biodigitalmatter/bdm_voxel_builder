# ruff: noqa: F401
from bdm_voxel_builder.agent_algorithms.algo_15_build_density_2 import Algo15_Build
from bdm_voxel_builder.config_setup import Config
from bdm_voxel_builder.visualizer.compas_viewer import CompasViewerVisualizer

# setup 2 test
grid_size = [200, 200, 80]
iterations = 1000
agent_count = 10
interval = 50
name = f"algo_15_i{iterations}a{agent_count}"

config = Config(
    iterations=iterations,
    grid_size=grid_size,
    algo=Algo15_Build(
        agent_count=agent_count,
        grid_size=grid_size,
        name=name,
    ),
    visualizer=CompasViewerVisualizer(save_file=True, skip_grids=("pheromon_move")),
    save_interval=interval,
    visualize_interval=interval,
)
