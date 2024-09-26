from math import floor

import compas.geometry as cg
from compas.colors import Color

from bdm_voxel_builder.agent import Agent
from bdm_voxel_builder.grid import DiffusiveGrid, Grid
from bdm_voxel_builder.helpers import (
    get_nth_newest_file_in_folder,
    index_map_box_xxyyzz,
)


class Environment:
    def __init__(self, config):
        self.clipping_box = config.clipping_box
        self.xform = config.xform or cg.Transformation()
        self.iteration_count: int = 0
        self.end_state: bool = False
        self.grids: dict[str, Grid] = self.setup_grids(
            config.iterations,
            mockup_scan=config.mockup_scan,
            add_initial_box=config.add_initial_box,
        )

        config.algo.initialization(self)

        # prediffuse
        for _ in range(config.algo.seed_iterations):
            config.algo.update_environment(self)

        # MAKE AGENTS
        self.agents: list[Agent] = config.algo.setup_agents(self)

    def setup_grids(
        self, iterations, mockup_scan: bool = False, add_initial_box: bool = False
    ):
        shared_grid_kwargs = {
            "clipping_box": self.clipping_box,
            "xform": self.xform,
        }

        default_decay = 1e-4
        slow_decay = 1e-13

        scan = Grid(
            name="scan",
            color=Color.from_rgb255(210, 220, 230),
            **shared_grid_kwargs,
        )

        if mockup_scan:
            scan.set_values(self.make_ground_mockup(), values=1.0)
        else:
            scan_arr = get_nth_newest_file_in_folder(self.dir_save_solid_npy)
            scan.set_values_using_array(scan_arr)

        ground = scan.copy()
        ground.name = "ground"
        ground.color = Color.from_rgb255(97, 92, 97)

        agent_space = Grid(
            name="agent_space",
            color=Color.from_rgb255(34, 116, 240),
            **shared_grid_kwargs,
        )
        track = DiffusiveGrid(
            name="track",
            color=Color.from_rgb255(34, 116, 240),
            diffusion_ratio=default_decay,
            **shared_grid_kwargs,
        )
        centroids = DiffusiveGrid(
            name="built_centroids",
            color=Color.from_rgb255(252, 25, 0),
            flip_colors=True,
            decay_linear_value=1 / (iterations * 10),
            diffusion_ratio=default_decay,
            **shared_grid_kwargs,
        )
        built_volume = DiffusiveGrid(
            name="density",
            color=Color.from_rgb255(219, 26, 206),
            flip_colors=True,
            decay_ratio=slow_decay,
            **shared_grid_kwargs,
        )

        if add_initial_box:
            built_volume.set_values(self.make_init_box_mockup(), values=1.0)

        follow_grid = DiffusiveGrid(
            name="follow_grid",
            color=Color.from_rgb255(232, 226, 211),
            flip_colors=True,
            decay_ratio=slow_decay,
            gradient_resolution=10e22,
            **shared_grid_kwargs,
        )

        sense_maps_grid = Grid(
            name="sense_maps_grid",
            color=Color.from_rgb255(200, 195, 0),
            flip_colors=True,
            **shared_grid_kwargs,
        )
        move_map_grid = Grid(
            name="move_map_grid",
            color=Color.from_rgb255(180, 180, 195),
            flip_colors=True,
            **shared_grid_kwargs,
        )

        return {
            "agent": agent_space,
            "ground": ground,
            "track": track,
            "centroids": centroids,
            "built_volume": built_volume,
            "follow_grid": follow_grid,
            "scan": scan,
            "sense_maps_grid": sense_maps_grid,
            "move_map_grid": move_map_grid,
        }

    def make_ground_mockup(self):
        a, b, c = (floor(v) - 1 for v in self.clipping_box.dimensions)

        base_layer = [0, a, 0, b, 0, 10]

        ground_zones = [base_layer]

        mask_set = set()

        for zone in ground_zones:
            indices = index_map_box_xxyyzz(zone)
            mask_set.update([tuple(ijk) for ijk in indices])

        return list(mask_set)

    def make_init_box_mockup(self):
        a, b, c = (floor(v) - 1 for v in self.clipping_box.dimensions)

        box_1 = [floor(v) for v in (a / 2, a / 2 + 3, b / 2, b / 2 + 3, 10, 13)]

        ground_zones = [box_1]

        mask_set = set()

        for zone in ground_zones:
            indices = index_map_box_xxyyzz(zone)
            mask_set.update([tuple(ijk) for ijk in indices])

        return list(mask_set)
