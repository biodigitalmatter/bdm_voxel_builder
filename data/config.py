# ruff: noqa: F401
from bdm_voxel_builder.agent_algorithms.algo_20_build_by_sense_b_walls import Algo20_Build_b
from bdm_voxel_builder.agent_algorithms.algo_20_build_by_sense_a_vortex import Algo20_Build_a

from bdm_voxel_builder.config_setup import Config
from bdm_voxel_builder.visualizer.compas_viewer import CompasViewerVisualizer

# setup 2 test
# grid_size = [300, 250, 150]

grid_size = [100,100,80]

iterations = 40
agent_count = 30
interval = 10
name = f"algo_20_b_test_nozzle_and_overhang_i{iterations}a{agent_count}"

config = Config(
    iterations=iterations,
    grid_size=grid_size,
    algo=Algo20_Build_b(
        agent_count=agent_count,
        grid_size=grid_size,
        name=name,
    ),
    visualizer=CompasViewerVisualizer(save_file=True, skip_grids=("agent_space", 'ground', 'follow_grid', 'move_map_grid')),
    save_interval=interval,
    visualize_interval=interval,
)
