# ruff: noqa: F401
from bdm_voxel_builder.agent_algorithms.algo_8_e_build_ridge import (
    Algo8eRidge,
)
from bdm_voxel_builder.agent_algorithms.algo_10_a_slicer_agents import (
    Algo10a_VoxelSlicer,
)
from bdm_voxel_builder.agent_algorithms.algo_11_test_scan_import import (
    Algo11a_TestScanImport,
)
from bdm_voxel_builder.config_setup import Config
from bdm_voxel_builder.visualizer.compas_viewer import CompasViewerVisualizer

# # setup 1
# grid_size = 100
# iterations = 15
# agent_count = 10
# interval = iterations / 4
# name = f"test_import_ply{iterations}a{agent_count}"

# config = Config(
#     iterations=iterations,
#     grid_size=grid_size,
#     algo=Algo11a_TestScanImport(
#         agent_count=agent_count,
#         grid_size=grid_size,
#         name=name,
#     ),
#     visualizer=CompasViewerVisualizer(save_file=True, skip_grids=("pheromon_move")),
#     save_interval=interval,
#     visualize_interval=interval,
# )


# setup 2 test
grid_size = 50
iterations = 25
agent_count = 1
interval = iterations / 100
name = f"test_8_e_build_ridge{iterations}a{agent_count}"

config = Config(
    iterations=iterations,
    grid_size=grid_size,
    algo=Algo8eRidge(
        agent_count=agent_count,
        grid_size=grid_size,
        name=name,
    ),
    visualizer=CompasViewerVisualizer(save_file=True, skip_grids=("pheromon_move")),
    save_interval=interval,
    visualize_interval=interval,
)
