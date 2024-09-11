import abc

import compas.geometry as cg

from bdm_voxel_builder.environment import Environment
from bdm_voxel_builder.grid.diffusive_grid import DiffusiveGrid


class AgentAlgorithm(abc.ABC):
    def __init__(
        self,
        agent_count: int,
        grid_to_dump: str,
        clipping_box: cg.Box,
        name: str | None = None,
        grids_to_decay: list[str] | None = None,
    ) -> None:
        self.agent_count = agent_count
        self.grid_to_dump = grid_to_dump
        self.clipping_box = clipping_box
        self.name = name
        self.grids_to_decay = grids_to_decay

    @abc.abstractmethod
    def setup_agents(self, state: Environment):
        raise NotImplementedError

    def update_environment(self, state: Environment):
        for grid_name in self.grids_to_decay or []:
            grid = state.grids[grid_name]
            assert isinstance(grid, DiffusiveGrid)
            grid.decay()
