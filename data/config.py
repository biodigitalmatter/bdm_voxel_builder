from bdm_voxel_builder.config_setup import Config

from bdm_voxel_builder.agent_algorithms.algo_7_queen_box import Algo7QueenBox

from bdm_voxel_builder.visualizers.matplotlib import MPLVisualizer

voxel_size = 40

config = Config(
    iterations=5,
    scale=voxel_size,
    algo=Algo7QueenBox(agent_count=25, voxel_size=voxel_size),
    visualizer=MPLVisualizer(save_file=True, scale=voxel_size, save_animation=True),
)
