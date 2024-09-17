# ruff: noqa: F401

from bdm_voxel_builder.agent_algorithms.algo_20_build_by_sense_c_4view import (
    Algo20_Build_c,
)
from bdm_voxel_builder.config_setup import Config
from bdm_voxel_builder.visualizer.compas_viewer import CompasViewerVisualizer

# setup 2 test
# grid_size = [300, 250, 150]

# grid_size = [200, 200, 100]
grid_size = [100, 100, 50]

iterations = 500
agent_count = 30
save_interval = 100
visualize_interval = 100
name = f"algo_20_test_c_follownew10_moveup0_random1_buildnextto1__i{iterations}a{agent_count}"
skip = ("agent_space", "ground", "follow_grid")
config = Config(
    iterations=iterations,
    grid_size=grid_size,
    algo=Algo20_Build_c(
        agent_count=agent_count,
        grid_size=grid_size,
        name=name,
    ),
    visualizer=CompasViewerVisualizer(save_file=True, skip_grids=skip),
    save_interval=save_interval,
    visualize_interval=visualize_interval,
)
