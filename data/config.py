from bdm_voxel_builder.config_setup import Config

from bdm_voxel_builder.agent_algorithms import algo_7_queen_box

from bdm_voxel_builder.visualizers.matplotlib import MPLVisualizer

config = Config(
    iterations=100,
    algo=algo_7_queen_box,
    visualizer=MPLVisualizer(save_file=True, filename="Test", scale=10),
)
