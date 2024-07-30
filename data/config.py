# ruff: noqa: F401
from bdm_voxel_builder.agent_algorithms.algo_8_b_build_on_2 import Algo8b
from bdm_voxel_builder.agent_algorithms.algo_8_d_build_fresh import Algo8d
from bdm_voxel_builder.agent_algorithms.algo_8_c_build_on_3 import Algo8c
from bdm_voxel_builder.config_setup import Config
from bdm_voxel_builder.visualizer.compas_viewer import CompasViewerVisualizer

# SHORT RUN test compasview
grid_size = 100
iterations = 500
agent_count = 35
interval = iterations / 4
info = 25
name = f'algo_8d_test_i{iterations}a{agent_count}'

config = Config(
    iterations=iterations,
    grid_size=grid_size,
    algo=Algo8d(
        agent_count=agent_count,
        grid_size=grid_size,
        name=name,
    ),
    visualizer=CompasViewerVisualizer(
        save_file=True, 
        skip_layers=('ground', 'pheromon_layer_move')
    ),
    save_interval=interval,
    visualize_interval=interval,
)
