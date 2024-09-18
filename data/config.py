# ruff: noqa: F401

from bdm_voxel_builder.agent_algorithms.algo_20_build_by_sense_c_4view import (
    Algo20_Build_c,
)
from bdm_voxel_builder.agent_algorithms.algo_20_build_by_sense_d2_goal_density_v2 import (
    Algo20_Build_d2,
)
from bdm_voxel_builder.agent_algorithms.algo_20_build_by_sense_d_goal_density import (
    Algo20_Build_d,
)
from bdm_voxel_builder.config_setup import Config
from bdm_voxel_builder.visualizer.compas_viewer import CompasViewerVisualizer

# setup 2 test
# grid_size = [300, 250, 150]

# grid_size = [200, 200, 100]
grid_size = [100, 100, 50]
grid_size = [50, 50, 50]

iterations = 5000
agent_count = 30
save_interval = 100
visualize_interval = 1000
name = f"algo_20_d2_small_test3_i{iterations}a{agent_count}"
skip = ("agent_space", "ground", "follow_grid")
config = Config(
    iterations=iterations,
    grid_size=grid_size,
    algo=Algo20_Build_d2(
        agent_count=agent_count,
        grid_size=grid_size,
        name=name,
    ),
    visualizer=CompasViewerVisualizer(save_file=True, skip_grids=skip),
    save_interval=save_interval,
    visualize_interval=visualize_interval,
)
