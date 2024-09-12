# ruff: noqa: F401
from bdm_voxel_builder.agent_algorithms.algo_20_build_by_sense import Algo20_Build
from bdm_voxel_builder.config_setup import Config
from bdm_voxel_builder.visualizer.compas_viewer import CompasViewerVisualizer

# setup 2 test
grid_size = [300, 250, 150]

grid_size = [120, 120, 100]

iterations = 4000
agent_count = 10
interval = 100
name = f"algo_20_central_follow_built_i{iterations}a{agent_count}"

config = Config(
    iterations=iterations,
    grid_size=grid_size,
    algo=Algo20_Build(
        agent_count=agent_count,
        grid_size=grid_size,
        name=name,
    ),
    visualizer=CompasViewerVisualizer(save_file=True, skip_grids=("agent_space")),
    save_interval=interval,
    visualize_interval=interval,
)
