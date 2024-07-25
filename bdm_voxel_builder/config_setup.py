
from dataclasses import dataclass
from typing import List

from bdm_voxel_builder.agent_algorithms.base import AgentAlgorithm
from bdm_voxel_builder.visualizers.base import Visualizer


@dataclass
class Config:
    algo: AgentAlgorithm
    visualizer: Visualizer
    scale: int = 40
    iterations: int=200
    save_interval: int=100
    datalayers_to_visualize: List[str] = None

