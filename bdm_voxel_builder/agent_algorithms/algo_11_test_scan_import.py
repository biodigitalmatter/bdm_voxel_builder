from dataclasses import dataclass

from compas.colors import Color

from bdm_voxel_builder.agent import Agent
from bdm_voxel_builder.agent_algorithms.base import AgentAlgorithm
from bdm_voxel_builder.agent_algorithms.common import diffuse_diffusive_grid
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
    grid_to_dump = "ground"
    seed_iterations = 0
    ground_level_Z = 0

    scan_ply_folder_path = "temp/ply"
    file_index_to_load = 2
    unit_in_mm = 10

    def __post_init__(self):
        """Initialize values held in parent class.

        Run in __post_init__ since @dataclass creates __init__ method"""
        super().__init__(
            agent_count=self.agent_count,
            grid_size=self.grid_size,
            grid_to_dump=self.grid_to_dump,
            name=self.name,
        )

    def initialization(self, **kwargs):
        """
        creates the simulation environment setup
        with preset values in the definition

        returns: grids
        """
        print("algorithm 11 started")
        ground = DiffusiveGrid(
            name="ground",
            grid_size=self.grid_size,
        )
        offset = DiffusiveGrid(
            name="offset",
            grid_size=self.grid_size,
            color=Color.from_rgb255(25, 100, 55),
            gradient_resolution=100,
            decay_ratio=1 / 10,
        )

        file = get_nth_newest_file_in_folder(
            self.scan_ply_folder_path, self.file_index_to_load
        )
        imported_grid = Grid.from_ply(
            file, self.grid_size, voxel_edge_length=self.unit_in_mm, name=file.name
        )

        ground.array = imported_grid.array
        # trim scan to gridsize
        # trim_array = imported_array[0:self.grid_size][0:self.grid_size][0:self.grid_size]
        # print(f'trim_array shape{trim_array.shape}')
        # ground.array = trim_array
        print("imported from ply")

        grids = {"ground": ground, "offset": offset}
        return grids

    def update_environment(self, state: Environment):
        ground = state.grids["ground"]
        offset = state.grids["offset"]
        diffuse_diffusive_grid(
            offset,
            emmission_array=ground.array,
            decay_linear=True,
            decay=False,
            grade=True,
            gravity_shift_bool=True,
        )
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
