import compas.geometry as cg

from bdm_voxel_builder.agent import Agent
from bdm_voxel_builder.grid import Grid


class Environment:
    def __init__(self, config):
        self.clipping_box = config.clipping_box
        self.xform = config.xform or cg.Transformation()
        self.iteration_count: int = 0
        self.end_state: bool = False
        self.grids: dict[str, Grid] = config.algo.initialization(
            iterations=config.iterations,
            clipping_box=self.clipping_box,
            xform=self.xform,
        )

        # prediffuse
        for _ in range(config.algo.seed_iterations):
            config.algo.update_environment(self)

        # MAKE AGENTS
        self.agents: list[Agent] = config.algo.setup_agents(self)
