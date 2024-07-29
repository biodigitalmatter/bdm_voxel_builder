from typing import List
from bdm_voxel_builder.agent import Agent
from bdm_voxel_builder.data_layer.diffusive_layer import DiffusiveLayer


class Environment:
    def __init__(self, config):
        self.iteration_count: int = 0

        self.data_layers: List[DiffusiveLayer] = config.algo.initialization(
            iterations=config.iterations
        )

        # prediffuse
        for _ in range(config.algo.seed_iterations):
            config.algo.update_environment(self)

        # MAKE AGENTS
        self.agents: List[Agent] = config.algo.setup_agents(self.data_layers)
