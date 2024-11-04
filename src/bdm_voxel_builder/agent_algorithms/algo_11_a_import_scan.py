from dataclasses import dataclass

import compas.geometry as cg
from compas.colors import Color

from bdm_voxel_builder import REPO_DIR
from bdm_voxel_builder.agent_algorithms.base import AgentAlgorithm
from bdm_voxel_builder.agents import Agent
from bdm_voxel_builder.environment import Environment
from bdm_voxel_builder.grid import DiffusiveGrid, Grid
from bdm_voxel_builder.helpers import get_nth_newest_file_in_folder


@dataclass
class Algo11a_ImportScan(AgentAlgorithm):
    """
    imports ply scan from:
    //data/live/scan_ply

    dumps pointcloud to:
    data/live/build_grid/01_scanned

    """

    agent_count: int
    clipping_box: cg.Box
    name: str = "algo_11_a_import_scan"
    relevant_data_grids: str = "scan"
    grid_to_dump = "scan"
    seed_iterations = 0

    scan_ply_folder_path = REPO_DIR / "data/live/scan_ply"
    dir_save_scan = REPO_DIR / "data/live/build_grid/01_scanned"
    dir_save_scan_npy = REPO_DIR / "data/live/build_grid/01_scanned/npy"

    file_index_to_load = 0
    # unit_in_mm = 10

    def __post_init__(self):
        """Initialize values held in parent class.

        Run in __post_init__ since @dataclass creates __init__ method"""
        super().__init__(
            clipping_box=self.clipping_box,
            agent_count=self.agent_count,
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
        offset = DiffusiveGrid(
            name="offset",
            clipping_box=self.clipping_box,
            color=Color.from_rgb255(25, 100, 55),
            gradient_resolution=100,
            decay_ratio=1 / 10,
        )
        file = get_nth_newest_file_in_folder(
            self.scan_ply_folder_path, self.file_index_to_load
        )
        imported_grid = Grid.from_ply(
            file, clipping_box=self.clipping_box, name=file.name
        )

        grids = {
            "scan": imported_grid,
            "offset": offset,
            "imported_grid": imported_grid,
        }
        return grids

    def update_environment(self, state: Environment):
        scan = state.grids["scan"]
        offset = state.grids["offset"]

        assert isinstance(offset, DiffusiveGrid)

        offset.diffuse_diffusive_grid(
            scan.to_numpy(),
            decay_linear=True,
            decay=False,
            grade=True,
            gravity_shift_bool=True,
        )

    def setup_agents(self, state: Environment):
        agents = []
        return agents

    def move_agent(self, agent: Agent, state: Environment):
        moved = False
        return moved

    def agent_action(self, agent, state: Environment):
        """MOVE BUILD .RESET"""
        pass
