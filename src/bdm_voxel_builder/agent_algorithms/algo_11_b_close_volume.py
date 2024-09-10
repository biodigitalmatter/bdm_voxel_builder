from dataclasses import dataclass

from compas.colors import Color

from bdm_voxel_builder import REPO_DIR
from bdm_voxel_builder.agent import Agent
from bdm_voxel_builder.agent_algorithms.base import AgentAlgorithm
from bdm_voxel_builder.environment import Environment
from bdm_voxel_builder.grid import DiffusiveGrid, Grid
from bdm_voxel_builder.helpers import get_nth_newest_file_in_folder
from bdm_voxel_builder.helpers.file import save_ndarray


@dataclass
class Algo11b_CloseVolume(AgentAlgorithm):
    """
    imports ply scan from:
    //data/live/scan_ply

    dumps pointcloud to:
    data/live/build_grid/01_scanned

    """

    agent_count: int
    grid_size: int | tuple[int, int, int]
    name: str = "algo_8_d"
    relevant_data_grids: str = "scan"
    grid_to_dump = "scan"
    seed_iterations = 0

    # directory import
    dir_scan_import = REPO_DIR / "data/live/build_grid/01_scanned"
    dir_scan_import_npy = REPO_DIR / "data/live/build_grid/01_scanned/npy"
    dir_save_solid = REPO_DIR / "data/live/build_grid/02_solid"
    dir_save_solid_npy = REPO_DIR / "data/live/build_grid/02_solid/npy"

    file_index_to_load = 0
    # unit_in_mm = 10

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
        # radius = kwargs.get('radius')
        # mode = kwargs.get('mode')
        print("algorithm 11 started")

        # filename = "2024-08-09_16_15_12_grid_20240731_stone_scan_5mm__01.vdb"
        # file_path = os.path.join(self.dir_scan_import, filename)

        # load vdb
        # file_path = get_nth_newest_file_in_folder(self.dir_scan_import)
        # loaded_grid = Grid.from_vdb(grid=file_path)
        # shape = loaded_grid.pad_array(
        #     pad_width=5, values=0
        # )  # TODO not sure is a good idea...
        # self.grid_size = shape
        # load npy
        file_path = get_nth_newest_file_in_folder(self.dir_scan_import_npy)
        loaded_grid = Grid.from_npy(file_path)

        # shape = loaded_grid.pad_array(
        #     pad_width=5, values=0
        # )  # TODO not sure is a good idea...
        # self.grid_size = shape

        scan = DiffusiveGrid(
            name="scan",
            grid_size=self.grid_size,
        )
        solid = DiffusiveGrid(
            name="solid",
            grid_size=self.grid_size,
            color=Color.from_rgb255(125, 170, 185),
        )

        scan.array = loaded_grid.array

        solid.array = scan.array.copy()

        # SOLID METHOD OPTIONS
        # solid.extrude_along_vector([-15, 3, -20], 5)
        solid.extrude_unit([0, 0, -1], 50)

        # solid.offset_radial(3)

        # solid.extrude_from_point([100, 20, 120], 50)
        save_ndarray(solid.array, note="", folder_path=self.dir_save_solid_npy)

        grids = {"scan": scan, "solid": solid}
        return grids

    def update_environment(self, state: Environment):
        # scan = state.grids["scan"]
        # solid = state.grids["solid"]
        # diffuse_diffusive_grid(
        #     solid,
        #     emmission_array=scan.array,
        #     decay_linear=True,
        #     decay=False,
        #     grade=True,
        #     gravity_shift_bool=True,
        # )

        pass

    def setup_agents(self, grids: dict[str, DiffusiveGrid]):
        agents = []
        return agents

    def move_agent(self, agent: Agent, state: Environment):
        moved = False
        return moved

    def agent_action(self, agent, state: Environment):
        """MOVE BUILD .RESET"""
        pass
