# ruff: noqa: F401
from bdm_voxel_builder.agent_algorithms.algo_8_d_build_fresh import Algo8d
from bdm_voxel_builder.agent_algorithms.algo_8_e_build_ridge import Algo8eRidge
from bdm_voxel_builder.agent_algorithms.algo_10_a_slicer_agents import (
    Algo10a_VoxelSlicer,
)
from bdm_voxel_builder.agent_algorithms.algo_12_just_go_and_build import (
    Algo12_Random_builder,
)
from bdm_voxel_builder.agent_algorithms.algo_13_build_probability import (
    Algo13_Build_Prob,
)
from bdm_voxel_builder.agent_algorithms.algo_14_build_density import (
    Algo14_Build_DensRange,
)
from bdm_voxel_builder.config_setup import Config
from bdm_voxel_builder.visualizer.compas_viewer import CompasViewerVisualizer

# setup 2 test
grid_size = [200, 300, 200]
iterations = 5000
agent_count = 15
interval = 200
name = f"algo_13_build_prob_i{iterations}a{agent_count}"

config = Config(
    iterations=iterations,
    grid_size=grid_size,
    algo=Algo13_Build_Prob(
        agent_count=agent_count,
        grid_size=grid_size,
        name=name,
    ),
    visualizer=CompasViewerVisualizer(save_file=True, skip_grids=("pheromon_move")),
    save_interval=interval,
    visualize_interval=interval,
)
