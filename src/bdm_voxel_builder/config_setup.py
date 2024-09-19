from dataclasses import dataclass

import compas.geometry as cg

from bdm_voxel_builder.agent_algorithms.base import AgentAlgorithm
from bdm_voxel_builder.visualizer.base import Visualizer


@dataclass
class Config:
    algo: AgentAlgorithm
    visualizer: Visualizer
    clipping_box: cg.Box
    xform: cg.Transformation = None
    iterations: int = 200
    save_interval: int = 100
    visualize_interval: int = 1
    grids_to_visualize: list[str] = None
    mockup_scan: bool = False
    add_initial_box: bool = False
