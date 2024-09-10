# ruff: noqa: F401
import compas.geometry as cg

from bdm_voxel_builder.agent_algorithms.algo_12_just_go_and_build import (
    Algo12_Random_builder,
)
from bdm_voxel_builder.agent_algorithms.algo_13_build_probability import (
    Algo13_Build_Prob,
)
from bdm_voxel_builder.agent_algorithms.algo_14_build_density import (
    Algo14_Build_DensRange,
)
from bdm_voxel_builder.agent_algorithms.make_gyroid_1 import Make_Gyroid
from bdm_voxel_builder.config_setup import Config
from bdm_voxel_builder.visualizer.compas_viewer import CompasViewerVisualizer

# setup 2 test
grid_size = [100, 100, 30]
iterations = 100
agent_count = 15
interval = 200
name = f"algo_13_build_prob_i{iterations}a{agent_count}"

config = Config(
    iterations=iterations,
    clipping_box=cg.Box.from_diagonal(([0, 0, 0], grid_size)),
    algo=Algo13_Build_Prob(
        agent_count=agent_count,
        name=name,
    ),
    visualizer=CompasViewerVisualizer(save_file=True, skip_grids=("pheromone_move")),
    save_interval=interval,
    visualize_interval=interval,
)
