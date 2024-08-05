from dataclasses import dataclass

import numpy as np
from compas.colors import Color

from bdm_voxel_builder.agent import Agent
from bdm_voxel_builder.agent_algorithms.base import AgentAlgorithm
from bdm_voxel_builder.environment import Environment
from bdm_voxel_builder.grid import DiffusiveGrid, Grid
from bdm_voxel_builder.helpers.common import get_nth_newest_file_in_folder

@dataclass
class Algo11a_TestScanImport(AgentAlgorithm):
    """
    # Voxel Builder Algorithm: Algo_10 SLICER BASIC:

    ## Summary

    default voxel builder algorithm
    agents build on and around an initial 'clay' volume on a 'ground' surface
    inputs: solid_ground_volume, clay_volume
    output:


    ## Agent behaviour

    1. find the built clay
    2. climb <up> on it
    3. build after a while of climbing
    4. reset or not

    ## Features

    - move on solid array
    - move direction is controlled with the mix of the pheromon environment and
      a global direction preference
    - move randomness controlled by setting the number of best directions for
      the random choice
    - build on existing volume
    - build and erase is controlled by gaining rewards
    - move and build both is regulated differently at different levels of
      environment grid density

    ## NEW in 8_d
    agents aim more towards the freshly built volumes.
    the clay array values slowly decay

    ## Observations:

    """

    agent_count: int
    grid_size: int | tuple[int, int, int]
    name: str = "algo_8_d"
    relevant_data_grids: str = "design"
    grid_to_dump = 'ground'
    seed_iterations = 0
    ground_level_Z = 0

    scan_ply_folder_path = "temp/ply"
    unit_in_mm = 10

    def __post_init__(self):
        """Initialize values held in parent class.

        Run in __post_init__ since @dataclass creates __init__ method"""
        super().__init__(
            agent_count=self.agent_count,
            grid_size=self.grid_size,
            grid_to_dump=self.grid_to_dump,
            name=self.name
        )

    def initialization(self, **kwargs):
        """
        creates the simulation environment setup
        with preset values in the definition

        returns: grids
        """
        print('algorithm 11 started')
        ground = DiffusiveGrid(
            name="ground",
            grid_size=self.grid_size,
        )

        imported_grid = Grid(self.grid_size)
        file = get_nth_newest_file_in_folder(self.scan_ply_folder_path,1)
        imported_grid.array_from_ply(file, self.unit_in_mm)
        print('imported from ply')
        ground.array = imported_grid.array

        grids = {
            "ground": ground
        }
        print(f'ground.array:\n{ground.array}')
        return grids

    def update_environment(self, state: Environment):
        pass

    def setup_agents(self, grids: dict[str, DiffusiveGrid]):
        agents = []
        return agents

    def reset_agent(self, agent: Agent):
        return False
    
    def move_agent(self, agent: Agent, state: Environment):
        moved = False
        return moved
    
    def agent_action(self, agent, state: Environment):
        """MOVE BUILD .RESET"""
        pass
