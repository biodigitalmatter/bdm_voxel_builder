# ruff: noqa: F401
import compas.geometry as cg

from bdm_voxel_builder.agent_algorithms.algo_21_print_path import Algo21PrintPath
from bdm_voxel_builder.config_setup import Config
from bdm_voxel_builder.visualizer.compas_viewer import CompasViewerVisualizer

grid_size = [50, 50, 50]

iterations = 1
agent_count = 1
name = f"algo_21_i{iterations}a{agent_count}"
skip = ("agent_space", "ground", "follow")
save_interval = 50
visualize_interval = 100

clipping_box = cg.Box.from_diagonal(([0, 0, 0], grid_size))
# skip = ("agent_space", 'ground', 'follow_grid', 'move_map_grid')
config = Config(
    iterations=iterations,
    clipping_box=clipping_box,
    algo=Algo21PrintPath(
        agent_count=agent_count,
        name=name,
        clipping_box=clipping_box,
        grids_to_dump=[("track")],
    ),
    save_interval=save_interval,
    visualizer=CompasViewerVisualizer(save_file=True, skip_grids=skip),
    visualize_interval=visualize_interval,
    mockup_scan=True,
    add_initial_box=True,
)
