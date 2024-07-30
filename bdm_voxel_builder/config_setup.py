from dataclasses import dataclass

from bdm_voxel_builder.agent_algorithms.base import AgentAlgorithm
from bdm_voxel_builder.visualizer.base import Visualizer


@dataclass
class Config:
    algo: AgentAlgorithm
    visualizer: Visualizer
    grid_size: int = 40
    iterations: int = 200
    save_interval: int = 100
    visualize_interval: int = 1
    datalayers_to_visualize: list[str] = None
