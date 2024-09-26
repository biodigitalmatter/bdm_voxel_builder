import random as r
from dataclasses import dataclass

import numpy as np
from compas import json_dumps
from compas.colors import Color

from bdm_voxel_builder import REPO_DIR
from bdm_voxel_builder.agent import Agent
from bdm_voxel_builder.agent_algorithms.base import AgentAlgorithm
from bdm_voxel_builder.agent_algorithms.common import diffuse_diffusive_grid
from bdm_voxel_builder.environment import Environment
from bdm_voxel_builder.grid import DiffusiveGrid
from bdm_voxel_builder.grid.base import Grid
from bdm_voxel_builder.helpers import (
    get_nth_newest_file_in_folder,
    get_savepath,
    index_map_cylinder,
    index_map_sphere,
    set_value_by_index_map,
)

# ultimate_parameters - test_1 - absolut random build
overhang_density = 0.35
move_up = 0
move_random = 1
follow_goal = 0.05
follow_built = 0.5

build_next_to_bool = True
sense_wall_radar_bool = True
sense_goal_density = True

build_probability_absolut_random = 0.001
build_probability_next_to = [0, 0.3]  # (no, yes)
build_probability_too_dense = [0, -10]  # (no, yes)

goal_density__modifier_A = -0.1
goal_density__modifier_B = 0.35
goal_density__modifier_C = -5

max_build_density = 0.5
wall_radar_radius = 15
deploy_anywhere = False
add_initial_box = False
reset = True


@dataclass
class Algo20_Build_d(AgentAlgorithm):
    """
    # Voxel Builder Algorithm: Algo 20

    the agents build based on sensor feedback
    overhang
    print_nozzle access

    20_d:
    build next to
    build more in goal density
    """

    agent_count: int
    grid_size: int | tuple[int, int, int]
    name: str = "algo_20"
    relevant_data_grids: str = "ground"
    grid_to_dump: str = "ground"

    # TODO
    vdb_to_dump: str = "built_volume"  # not implemented

    # global settings

    n = 50 if follow_built else 1
    seed_iterations: int = n

    walk_region_thickness = 1
    deploy_anywhere = deploy_anywhere  # only on the initial scan volume

    # import scan
    import_scan = False
    dir_solid_npy = REPO_DIR / "data/live/build_grid/02_solid/npy"

    # save fab_planes
    fab_planes = []
    fab_planes_file_path = get_savepath(
        dir=REPO_DIR / "temp/fabplanes/", note=name, suffix="_fabplanes.txt"
    )
    # fab_planes_file = open(fab_planes_file_path, "a")

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
        iterations = kwargs.get("iterations")
        ground = DiffusiveGrid(
            name="ground",
            grid_size=self.grid_size,
            color=Color.from_rgb255(97, 92, 97),
        )
        scan = DiffusiveGrid(
            name="scan",
            grid_size=self.grid_size,
            color=Color.from_rgb255(210, 220, 230),
        )
        if self.import_scan:
            file_path = get_nth_newest_file_in_folder(self.dir_solid_npy)
            loaded_grid = Grid.from_npy(file_path).array
            loaded_grid = np.array(
                loaded_grid, dtype=np.float64
            )  # TODO set dtype in algo_11_a_self.import_scan.py  # noqa: E501

            # the imported array is already cropped to the BBOX of interest

            self.grid_size = np.shape(loaded_grid)
            scan.array = loaded_grid
        else:
            arr = make_ground_mockup(self.grid_size)
            if add_initial_box:
                arr += make_init_box_mockup(self.grid_size)
            arr = np.clip(arr, 0, 1)
            scan.array = arr

        # IMPORT SCAN
        ground.array = scan.array.copy()
        print(f"ground grid_size = {ground.grid_size}")

        agent_space = DiffusiveGrid(
            name="agent_space",
            grid_size=self.grid_size,
            color=Color.from_rgb255(34, 116, 240),
        )
        track = DiffusiveGrid(
            name="track",
            grid_size=self.grid_size,
            color=Color.from_rgb255(34, 116, 240),
            decay_ratio=1 / 10000,
        )
        centroids = DiffusiveGrid(
            name="centroids",
            grid_size=self.grid_size,
            color=Color.from_rgb255(252, 25, 0),
            flip_colors=True,
            decay_ratio=1 / 10000,
            decay_linear_value=1 / (iterations * 10),
        )
        self.print_dot_counter = 0
        built_volume = DiffusiveGrid(
            name="built_volume",
            grid_size=self.grid_size,
            color=Color.from_rgb255(219, 26, 206),
            flip_colors=True,
            decay_ratio=1 / 10e12,
        )
        if add_initial_box:
            built_volume.array = make_init_box_mockup(self.grid_size)

        follow_grid = DiffusiveGrid(
            name="follow_grid",
            grid_size=self.grid_size,
            color=Color.from_rgb255(232, 226, 211),
            flip_colors=True,
            decay_ratio=1 / 10e12,
            gradient_resolution=10e22,
        )
        goal_density = DiffusiveGrid(
            name="goal_density",
            grid_size=self.grid_size,
            color=Color.from_rgb255(232, 226, 211),
            flip_colors=True,
            decay_ratio=1 / 10e12,
            gradient_resolution=10e22,
        )
        sense_maps_grid = DiffusiveGrid(
            name="sense_maps_grid",
            grid_size=self.grid_size,
            color=Color.from_rgb255(200, 195, 0),
            flip_colors=True,
        )
        move_map_grid = DiffusiveGrid(
            name="move_map_grid",
            grid_size=self.grid_size,
            color=Color.from_rgb255(180, 180, 195),
            flip_colors=True,
        )

        # define goal density
        goal_density.array += make_goal_density_box_mockup_A(
            self.grid_size, goal_density__modifier_A
        )
        goal_density.array += make_goal_density_box_mockup_B(
            self.grid_size, goal_density__modifier_B
        )
        goal_density.array += make_goal_density_box_mockup_B(
            self.grid_size, goal_density__modifier_C
        )

        # update walk region
        self.update_offset_regions(ground.array.copy(), scan.array.copy())

        print("initialized")

        # WRAP ENVIRONMENT
        grids = {
            "agent": agent_space,
            "ground": ground,
            "track": track,
            "centroids": centroids,
            "built_volume": built_volume,
            "follow_grid": follow_grid,
            "scan": scan,
            "sense_maps_grid": sense_maps_grid,
            "move_map_grid": move_map_grid,
            "goal_density": goal_density,
        }
        return grids

    def update_environment(self, state: Environment, **kwargs):
        grids = state.grids
        grids["built_volume"].decay()
        if follow_built > 0:
            emmision_array = (
                grids["goal_density"].array * follow_goal
                + grids["built_volume"].array * follow_built
            )  # noqa: E501
            diffuse_diffusive_grid(
                grids["follow_grid"],
                emmission_array=emmision_array,
                blocking_grids=[grids["ground"]],
                grade=False,
            )

    def setup_agents(self, grids: dict[str, DiffusiveGrid]):
        agent_space = grids["agent"]
        track = grids["track"]
        ground_grid = grids["ground"]
        agents = []

        # generate agents based on agent_type_dicts
        categories = ["a", "b"]
        category_sizes = [1, 0.1]
        d = np.array(category_sizes)
        d_normalized = d * 1 / np.sum(d)
        id = 0
        for category, n in enumerate(d_normalized):
            print(category, n)
            type_size = int(n * self.agent_count)
            # print(type_size)
            for _j in range(type_size):
                basic_agent = Agent()
                # TYPE A
                basic_agent.agent_type_summary = categories[category]
                # movement settings
                basic_agent.walk_radius = 4
                basic_agent.move_mod_z = move_up
                basic_agent.move_mod_random = move_random
                basic_agent.move_mod_follow = 1
                # build settings
                basic_agent.build_radius = 3
                basic_agent.build_h = 3
                basic_agent.reset_after_build = reset
                basic_agent.inactive_step_count_limit = None
                # sensor settings
                basic_agent.sense_radius = 3
                basic_agent.build_random_chance = build_probability_absolut_random
                basic_agent.build_random_gain = 1
                basic_agent.max_build_angle = 30
                basic_agent.overhang_density = overhang_density

                # NEW
                basic_agent.build_next_to_bool = build_next_to_bool
                basic_agent.build_probability_next_to = build_probability_next_to

                basic_agent.sense_wall_radar_bool = sense_wall_radar_bool
                basic_agent.sense_goal_density = sense_goal_density
                basic_agent.wall_radar_radius = 5
                basic_agent.build_probability_too_dense = build_probability_too_dense

                basic_agent.max_build_density = max_build_density

                # ALTER VERSIONS
                if category == 1:
                    basic_agent.sense_radius = 6

                # create shape maps
                basic_agent.move_map = index_map_sphere(
                    basic_agent.walk_radius, basic_agent.min_walk_radius
                )
                basic_agent.build_map = index_map_cylinder(
                    basic_agent.build_radius, basic_agent.build_h, 0, -1
                )
                basic_agent.sense_map = index_map_sphere(basic_agent.sense_radius)

                basic_agent.sense_overhang_map = index_map_cylinder(
                    radius=1, height=1, z_lift=-1
                )

                basic_agent.sense_nozzle_map = index_map_cylinder(
                    radius=0, height=40, z_lift=0
                )
                basic_agent.sense_wall_radar_map = index_map_cylinder(
                    radius=basic_agent.wall_radar_radius, height=1, z_lift=0
                )

                # set grids
                basic_agent.space_grid = agent_space
                basic_agent.track_grid = track
                basic_agent.ground_grid = ground_grid
                basic_agent.id = id
                # print(f"created agent_{basic_agent.id}")

                # deploy agent
                basic_agent.deploy_in_region(self.region_deploy_agent)

                # append
                agents.append(basic_agent)
                id += 1

        return agents

    def build(self, agent: Agent, state: Environment):
        """fill built volume in built_shape if agent.build_probability >= build_limit"""

        self.print_dot_counter += 1

        built_volume = state.grids["built_volume"]
        centroids = state.grids["centroids"]
        ground = state.grids["ground"]

        x, y, z = agent.pose

        # update print dot array
        centroids.array[x, y, z] = self.print_dot_counter

        # orient shape map
        build_map = agent.orient_build_map()
        # update built_volume_volume_array
        built_volume.array = set_value_by_index_map(
            built_volume.array,
            build_map,
            value=1,
        )
        ground.array = set_value_by_index_map(
            ground.array,
            build_map,
            value=1,
        )

        print(f"built at: {agent.pose}")

        # update fab_planes
        plane = agent.fab_plane
        self.fab_planes.append(plane)

        # write fabplane to file
        data = json_dumps(agent.fab_plane)
        with open(self.fab_planes_file_path, "a") as f:
            f.write(data + "\n")

    def get_agent_build_probability(self, agent, state):
        # BUILD CONSTRAINTS:
        ground = state.grids["ground"]

        nozzle_map = agent.orient_sense_nozzle_map(world_z=False)
        nozzle_access_density = agent.get_array_density_by_oriented_index_map(
            ground.array, nozzle_map, nonzero=True
        )
        # print(
        #     f"nozzle_acces_density: {nozzle_access_density}, degree: {agent.normal_angle}"
        # )
        nozzle_access_collision = nozzle_access_density >= 0.01
        overhang_map = agent.orient_sense_overhang_map()
        density = agent.get_array_density_by_oriented_index_map(
            ground.array, overhang_map, nonzero=True
        )
        too_low_overhang = density < agent.overhang_density
        if nozzle_access_collision or too_low_overhang:
            build_probability = 0
        else:
            # RANDOM FACTOR
            if r.random() < agent.build_random_chance:  # noqa: SIM108
                bp_random = agent.build_random_gain
            else:
                bp_random = 0

            # BUILD NEXT TO built
            if agent.build_next_to_bool:
                built_volume = state.grids["built_volume"]
                build_map = agent.orient_move_map()
                built_density = agent.get_array_density_by_oriented_index_map(
                    built_volume.array, build_map, nonzero=True
                )
                if built_density < 0.001:  # noqa: SIM108
                    bp_build_next_to = agent.build_probability_next_to[0]
                else:
                    bp_build_next_to = agent.build_probability_next_to[1]
            else:
                bp_build_next_to = 0

            # BUILD BY WALL RADAR
            if agent.sense_wall_radar_bool:
                ground = state.grids["ground"]
                wall_radar_map = agent.orient_sense_wall_radar_map()
                radar_density = agent.get_array_density_by_oriented_index_map(
                    ground.array, wall_radar_map, nonzero=True
                )
                if radar_density < agent.max_build_density:  # walking on wall
                    bp_wall_radar = agent.build_probability_too_dense[0]
                else:
                    bp_wall_radar = agent.build_probability_too_dense[1]

            # BUILD BY GOAL DENSITY
            if agent.sense_goal_density:
                goal_density = state.grids["goal_density"]
                sense_map = agent.orient_sense_map()
                bp_goal_density_modifier = (
                    agent.get_array_density_by_oriented_index_map(
                        goal_density.array, sense_map, nonzero=False
                    )
                )
                # print(bp_goal_density_modifier)

            build_probability = (
                bp_random + bp_build_next_to + bp_goal_density_modifier + bp_wall_radar
            )
        # print(f"build_probability:{build_probability}")
        return build_probability
