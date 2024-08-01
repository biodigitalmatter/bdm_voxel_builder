# ruff: noqa: F401
from bdm_voxel_builder.agent_algorithms.algo_9_a_carve import Algo9a
from bdm_voxel_builder.config_setup import Config
from bdm_voxel_builder.visualizer.compas_viewer import CompasViewerVisualizer

# SHORT RUN test compasview
grid_size = 30
iterations = 100
agent_count = 5
interval = iterations / 10
info = 25
name = f'algo_9a_test_carving_i{iterations}a{agent_count}'

config = Config(
    iterations=iterations,
    grid_size=grid_size,
    algo=Algo9a(
        agent_count=agent_count,
        grid_size=grid_size,
        name=name,
    ),
    visualizer=CompasViewerVisualizer(
        save_file=True, 
        skip_layers=( 'pheromon_layer_move')
    ),
    save_interval=interval,
    visualize_interval=interval,
)
