from typing import List
from bdm_voxel_builder.agent import Agent
from bdm_voxel_builder.data_layer.diffusive_layer import DiffusiveLayer


class SimulationState:
    def __init__(self, config):
        self.counter: int = 0

        self.data_layers: List[DiffusiveLayer] = config.algo.initialization()

        # prediffuse
        for _ in range(config.algo.seed_iterations):
            config.algo.update_environment(self)

        # MAKE AGENTS
        self.agents: List[Agent] = config.algo.setup_agents(self.data_layers)
