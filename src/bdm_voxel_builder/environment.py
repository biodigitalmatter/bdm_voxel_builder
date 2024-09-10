import compas.geometry as cg
from compas.geometry import oriented_bounding_box_numpy

from bdm_voxel_builder.agent import Agent
from bdm_voxel_builder.grid import Grid


class Environment:
    def __init__(self, config):
        self.clipping_box = config.clipping_box
        self.xform = config.xform or cg.Transformation()
        self.iteration_count: int = 0
        self.end_state: bool = False
        self.grids: dict[Grid] = config.algo.initialization(
            iterations=config.iterations,
            clipping_box=self.clipping_box,
            xform=self.xform,
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
