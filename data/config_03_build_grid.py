from bdm_voxel_builder.agent_algorithms.algo_8_e_build_ridge import Algo8eRidge
from bdm_voxel_builder.config_setup import Config
from bdm_voxel_builder.visualizer.compas_viewer import CompasViewerVisualizer

# setup 2 test
grid_size = 50  # not apply if imported
iterations = 50
agent_count = 20
interval = 500
name = f"build_grid_wall_{iterations}a{agent_count}"

config = Config(
    iterations=iterations,
    grid_size=grid_size,
    algo=Algo8eRidge(
        agent_count=agent_count,
        grid_size=grid_size,
        name=name,
    ),
    visualizer=CompasViewerVisualizer(
        save_file=True, skip_grids=("pheromon_move", "deploy_zone")
    ),
    save_interval=interval,
    visualize_interval=interval,
)
