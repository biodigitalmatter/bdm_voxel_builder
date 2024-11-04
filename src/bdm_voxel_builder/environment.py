import compas.geometry as cg
import numpy as np
from compas.colors import Color

from bdm_voxel_builder.agents import Agent
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
        self,
        iterations,
        mockup_scan: bool = False,
        add_initial_box: bool = False,
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
            scan.set_values(self.make_ground_mockup_index_grid(), values=1.0)
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
        built = DiffusiveGrid(
            name="density",
            color=Color.from_rgb255(219, 26, 206),
            flip_colors=True,
            decay_ratio=slow_decay,
            **shared_grid_kwargs,
        )

        if add_initial_box:
            built.set_values(self.make_init_box_mockup_index_grid(), values=1.0)

        follow = DiffusiveGrid(
            name="follow",
            color=Color.from_rgb255(232, 226, 211),
            flip_colors=True,
            decay_ratio=slow_decay,
            gradient_resolution=10e22,
            **shared_grid_kwargs,
        )

        sense_maps = Grid(
            name="sense_maps",
            color=Color.from_rgb255(200, 195, 0),
            flip_colors=True,
            **shared_grid_kwargs,
        )
        move_maps = Grid(
            name="move_maps",
            color=Color.from_rgb255(180, 180, 195),
            flip_colors=True,
            **shared_grid_kwargs,
        )
        goal_density = DiffusiveGrid(
            name="goal_density",
            color=Color.from_rgb255(232, 226, 211),
            flip_colors=True,
            decay_ratio=slow_decay,
            gradient_resolution=1e23,
            **shared_grid_kwargs,
        )

        return {
            "agent": agent_space,
            "ground": ground,
            "track": track,
            "centroids": centroids,
            "built": built,
            "follow": follow,
            "scan": scan,
            "sense_maps": sense_maps,
            "move_map": move_maps,
            "goal_density": goal_density,
        }

    def make_mockup_index_map_from_xxyyzz_factors(
        self, xxyyzz_factors: list[float, float, float, float, float, float]
    ):
        b_min, b_max = self.clipping_box.diagonal

        xxyyzz = []

        for i in range(3):
            min_ = xxyyzz_factors[i * 2]
            max_ = xxyyzz_factors[i * 2 + 1]

            xxyyzz.extend(np.interp([min_, max_], [0, 1], [b_min[i], b_max[i]]))

        return index_map_box_xxyyzz(xxyyzz)

    def make_ground_mockup_index_grid(self):
        zsize = self.clipping_box.dimensions[2]
        return self.make_mockup_index_map_from_xxyyzz_factors(
            [0, 1, 0, 1, 0, 10 / zsize]
        )

    def make_init_box_mockup_index_grid(self):
        xsize, ysize, zsize = self.clipping_box.dimensions
        return self.make_mockup_index_map_from_xxyyzz_factors(
            [0.5, 0.5 + 3 / xsize, 0.5, 0.5 + 3 / ysize, 10 / zsize, 13 / zsize]
        )

    def make_goal_density_box_mockup_A(self):
        return self.make_mockup_index_map_from_xxyyzz_factors(
            [0.2, 0.5, 0.2, 0.8, 0, 1]
        )

    def make_goal_density_box_mockup_B(self):
        xsize, ysize, _ = self.clipping_box.dimensions
        return self.make_mockup_index_map_from_xxyyzz_factors(
            [0.5, 0.8, 0.2, 0.8, 0, 0.7]
        )

    def make_goal_density_box_mockup_C(self):
        return self.make_mockup_index_map_from_xxyyzz_factors([0, 0.2, 0, 0.8, 0, 1])

    def make_goal_density_box_mockup_D(self):
        return self.make_mockup_index_map_from_xxyyzz_factors(
            [0.4, 0.6, 0.4, 0.6, 0, 1]
        )
