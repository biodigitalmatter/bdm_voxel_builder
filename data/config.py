from bdm_voxel_builder.agent_algorithms.algo_8_b_build_on_2 import Algo8b
from bdm_voxel_builder.config_setup import Config

# from bdm_voxel_builder.agent_algorithms.algo_7_queen_box import Algo7QueenBox
# from bdm_voxel_builder.agent_algorithms.algo_8_build_on import Algo8
from bdm_voxel_builder.agent_algorithms.algo_8_c_build_on_3 import Algo8c

from bdm_voxel_builder.visualizers.compas_viewer import CompasViewerVisualizer

# SHORT RUN test compasview
voxel_size = 7
iterations = 20
interval = 4

config = Config(
    iterations=iterations,
    scale=voxel_size,
    algo=Algo8c(agent_count=2, voxel_size=voxel_size),
    visualizer=CompasViewerVisualizer(save_file=True),
    save_interval=interval,
    visualize_interval=interval
)
