from compas.geometry import oriented_bounding_box_numpy

from bdm_voxel_builder.agent import Agent
from bdm_voxel_builder.grid import DiffusiveGrid


class Environment:
    def __init__(self, config):
        self.iteration_count: int = 0
        self.end_state: bool = False
        self.grids: list[DiffusiveGrid] = config.algo.initialization(
            iterations=config.iterations
        )

        # prediffuse
        for _ in range(config.algo.seed_iterations):
            config.algo.update_environment(self)

        # MAKE AGENTS
        self.agents: list[Agent] = config.algo.setup_agents(self.grids)

    def get_compound_bbox(self):
        bboxes = [grid.get_world_bbox() for grid in self.grids]

        pts = [box.points for box in bboxes]

        return oriented_bounding_box_numpy(pts)
