# ruff: noqa: F401
import compas.geometry as cg

from bdm_voxel_builder.agent_algorithms.algo_20_build_by_sense import Algo20_Build
from bdm_voxel_builder.config_setup import Config
from bdm_voxel_builder.visualizer.compas_viewer import CompasViewerVisualizer

# setup 2 test
grid_size = [200, 200, 100]
iterations = 15
agent_count = 10
interval = 200
name = f"algo_20_on_scan_i{iterations}a{agent_count}"

clipping_box = cg.Box.from_diagonal(([0, 0, 0], grid_size))

config = Config(
    iterations=iterations,
    clipping_box=clipping_box,
    algo=Algo20_Build(
        agent_count=agent_count,
        name=name,
        clipping_box=clipping_box,
    ),
    visualizer=CompasViewerVisualizer(
        save_file=True, skip_grids=("scan", "agent_space")
    ),
    save_interval=interval,
    visualize_interval=interval,
)
