from dataclasses import dataclass

import numpy as np
from compas.colors import Color

from bdm_voxel_builder import REPO_DIR
from bdm_voxel_builder.agent import Agent
from bdm_voxel_builder.agent_algorithms.base import AgentAlgorithm
from bdm_voxel_builder.agent_algorithms.common import diffuse_diffusive_grid
from bdm_voxel_builder.environment import Environment
from bdm_voxel_builder.grid import DiffusiveGrid, Grid
from bdm_voxel_builder.helpers import get_nth_newest_file_in_folder
from bdm_voxel_builder.helpers.file import save_ndarray


@dataclass
class Algo11a_ImportScan(AgentAlgorithm):
    """
    imports ply scan from:
    //data/live/scan_ply

    dumps pointcloud to:
    data/live/build_grid/01_scanned

    """

    agent_count: int
    grid_size: int | tuple[int, int, int]
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
        scan = DiffusiveGrid(
            name="scan",
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
        imported_grid = Grid.from_ply(file, self.grid_size, name=file.name)

        print(f"imported file: {file}")
        save_ndarray(imported_grid.array, note="", folder_path=self.dir_save_scan_npy)
        imported_grid.save_vdb(self.dir_save_scan)
        print(f"saved to {self.dir_save_scan}")
        scan.array = imported_grid.array

        print(f"imported from ply, sum = {np.sum(scan.array)}")

        grids = {"scan": scan, "offset": offset, "imported_grid": imported_grid}
        return grids

    def update_environment(self, state: Environment):
        scan = state.grids["scan"]
        offset = state.grids["offset"]
        diffuse_diffusive_grid(
            offset,
            emmission_array=scan.array,
            decay_linear=True,
            decay=False,
            grade=True,
            gravity_shift_bool=True,
        )
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
