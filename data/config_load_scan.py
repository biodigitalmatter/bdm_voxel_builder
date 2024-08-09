# ruff: noqa: F401

from bdm_voxel_builder.agent_algorithms.algo_11_test_scan_import import (
    Algo11a_TestScanImport,
)
from bdm_voxel_builder.config_setup import Config
from bdm_voxel_builder.visualizer.compas_viewer import CompasViewerVisualizer

grid_size = 100
iterations = 1
agent_count = 1
interval = 1
name = f"test_import{iterations}a{agent_count}"

config = Config(
    iterations=iterations,
    grid_size=grid_size,
    algo=Algo11a_TestScanImport(
        agent_count=agent_count,
        grid_size=grid_size,
        name=name,
    ),
    visualizer=CompasViewerVisualizer(save_file=False, skip_grids=()),
    save_interval=interval,
    visualize_interval=interval,
)
