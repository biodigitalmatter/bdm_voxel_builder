from typing import List
from bdm_voxel_builder.agent import Agent
from bdm_voxel_builder.agent_algorithms.base import AgentAlgorithm
from bdm_voxel_builder.data_layer import DataLayer


class SimulationState:
    def __init__(self, algo: AgentAlgorithm, iterations):
        self.counter: int = 0

        self.data_layers: List[DataLayer] = algo.layer_setup(iterations)

        # prediffuse
        for i in range(algo.seed_iterations):
            algo.diffuse_environment(self.data_layers)

        # MAKE AGENTS
        self.agents: List[Agent] = algo.setup_agents(self.data_layers)
