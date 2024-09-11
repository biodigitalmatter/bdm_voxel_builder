# ruff: noqa: F401
import compas.geometry as cg

from bdm_voxel_builder.agent_algorithms.algo_14_build_density import (
    Algo14_Build_DensRange,
)
from bdm_voxel_builder.config_setup import Config
from bdm_voxel_builder.visualizer.compas_viewer import CompasViewerVisualizer

# setup 2 test
grid_size = [100, 100, 30]
iterations = 100
agent_count = 15
interval = 200
name = f"algo_13_build_prob_i{iterations}a{agent_count}"

clipping_box = cg.Box.from_diagonal(([0, 0, 0], grid_size))

config = Config(
    iterations=iterations,
    clipping_box=clipping_box,
    algo=Algo14_Build_DensRange(
        agent_count=agent_count,
        name=name,
        clipping_box=clipping_box,
    ),
    visualizer=CompasViewerVisualizer(save_file=True, skip_grids=("pheromone_move")),
    save_interval=interval,
    visualize_interval=interval,
)
