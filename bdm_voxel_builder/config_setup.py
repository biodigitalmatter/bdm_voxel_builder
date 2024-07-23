
from dataclasses import dataclass

from bdm_voxel_builder.agent_algorithms.base import AgentAlgorithm
from bdm_voxel_builder.visualizers.base import Visualizer


@dataclass
class Config:
    iterations: int
    algo: AgentAlgorithm
    visualizer: Visualizer

