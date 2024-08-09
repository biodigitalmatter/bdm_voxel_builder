from dataclasses import dataclass
import os

import numpy as np

from bdm_voxel_builder import REPO_DIR
from bdm_voxel_builder.agent import Agent
from bdm_voxel_builder.agent_algorithms.base import AgentAlgorithm
from bdm_voxel_builder.agent_algorithms.common import diffuse_diffusive_grid
from bdm_voxel_builder.environment import Environment
from bdm_voxel_builder.grid import DiffusiveGrid, Grid
from bdm_voxel_builder.helpers import get_nth_newest_file_in_folder
from compas.colors import Color


@dataclass
class Algo11b_CloseVolume(AgentAlgorithm):
    """
    imports ply scan from:
    //data/live/scan_ply

    dumps pointcloud to:
    data/live/work/01_scanned

    """

    agent_count: int
    grid_size: int | tuple[int, int, int]
    name: str = "algo_8_d"
    relevant_data_grids: str = "scan"
    grid_to_dump = "scan"
    seed_iterations = 0

    scan_ply_folder_path = REPO_DIR / "data/live/scan_ply"
    dir_save_scan = REPO_DIR / "data/live/work/01_scanned"
    dir_save_scan_npy = REPO_DIR / "data/live/work/01_scanned/npy"

    file_index_to_load = 0
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

        # filename = "2024-08-09_16_15_12_grid_20240731_stone_scan_5mm__01.vdb"
        # file_path = os.path.join(self.dir_save_scan, filename)

        # load vdb
        file_path = get_nth_newest_file_in_folder(self.dir_save_scan)
        loaded_grid = Grid.from_vdb(grid=file_path)
        shape = loaded_grid.pad_array(
            pad_width=5, values=0
        )  # TODO not sure is a good idea...
        self.grid_size = shape
        # load npy
        # file_path = get_nth_newest_file_in_folder(self.dir_save_scan_npy)
        # loaded_grid = loaded_grid.from_npy(file_path)

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

        scan.array = loaded_grid.array
        offset.array = np.zeros_like(scan.array)
        grids = {"scan": scan, "offset": offset}
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

    def reset_agent(self, agent: Agent):
        return False

    def move_agent(self, agent: Agent, state: Environment):
        moved = False
        return moved

    def agent_action(self, agent, state: Environment):
        """MOVE BUILD .RESET"""
        pass
