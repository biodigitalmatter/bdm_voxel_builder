# ruff: noqa: F401
import compas.geometry as cg

from bdm_voxel_builder.agent_algorithms.algo_20_build_by_sense_c_4view import (
    Algo20_Build_c,
)
from bdm_voxel_builder.config_setup import Config
from bdm_voxel_builder.visualizer.compas_viewer import CompasViewerVisualizer

# setup 2 test
# grid_size = [300, 250, 150]

# grid_size = [200, 200, 100]
grid_size = [100, 100, 50]

iterations = 200
agent_count = 30
save_interval = 100
visualize_interval = 5000
name = f"algo_20_test_c_wall_radar_deploy_anywhere_False_i{iterations}a{agent_count}"
skip = ("agent_space", "ground", "follow_grid")
agent_count = 40
save_interval = 50
visualize_interval = 500
name = f"algo_20_b_walls_B_i{iterations}a{agent_count}"

clipping_box = cg.Box.from_diagonal(([0, 0, 0], grid_size))
# skip = ("agent_space", 'ground', 'follow_grid', 'move_map_grid')
skip = ("agent_space", "follow_grid")
config = Config(
    iterations=iterations,
    grid_size=grid_size,
    algo=Algo20_Build_c(
        agent_count=agent_count,
        name=name,
        clipping_box=clipping_box,
    ),
    save_interval=save_interval,
    visualizer=CompasViewerVisualizer(save_file=True, skip_grids=skip),
    visualize_interval=visualize_interval,
)
