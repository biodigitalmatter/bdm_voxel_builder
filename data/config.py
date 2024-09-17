# ruff: noqa: F401
import compas.geometry as cg

from bdm_voxel_builder.agent_algorithms.algo_20_build_by_sense import Algo20_Build
from bdm_voxel_builder.config_setup import Config
from bdm_voxel_builder.visualizer.compas_viewer import CompasViewerVisualizer

# setup 2 test
# grid_size = [300, 250, 150]

# grid_size = [200, 200, 80]
grid_size = [100, 100, 80]

iterations = 400
agent_count = 40
save_interval = 50
visualize_interval = 500
name = f"algo_20_b_walls_B_i{iterations}a{agent_count}"

clipping_box = cg.Box.from_diagonal(([0, 0, 0], grid_size))
# skip = ("agent_space", 'ground', 'follow_grid', 'move_map_grid')
skip = ("agent_space", "follow_grid")
config = Config(
    iterations=iterations,
    clipping_box=clipping_box,
    algo=Algo20_Build(
        agent_count=agent_count,
        name=name,
        clipping_box=clipping_box,
    ),
    visualizer=CompasViewerVisualizer(save_file=True, skip_grids=skip),
    save_interval=save_interval,
    visualize_interval=visualize_interval,
)
