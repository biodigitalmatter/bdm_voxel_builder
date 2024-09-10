import random as r
from dataclasses import dataclass

import numpy as np
from bdm_voxel_builder import REPO_DIR
from bdm_voxel_builder.agent import Agent
from bdm_voxel_builder.agent_algorithms.base import AgentAlgorithm
from bdm_voxel_builder.environment import Environment
from bdm_voxel_builder.grid import DiffusiveGrid
from bdm_voxel_builder.grid.base import Grid
from bdm_voxel_builder.helpers.array import (
    get_mask_zone_xxyyzz,
    get_surrounding_offset_region,
    get_values_by_index_map,
    index_map_cylinder,
    index_map_sphere,
    set_value_by_index_map,
)
from bdm_voxel_builder.helpers.file import get_nth_newest_file_in_folder
from compas.colors import Color

import_scan = False


def make_ground_mockup(grid_size):
    a, b, c = grid_size
    box_1 = [10, 13, 10, 60, 1, 30]
    box_2 = [10, 60, 10, 30, 1, 30]

    base_layer = [a * 0.35, a * 0.75, b * 0.35, b * 0.65, 0, 4]
    base_layer = [0, a, 0, b / 2, 0, 3]
    base_layer_2 = [0, a, b / 2, b, 0, 6]
    base_layer = np.array(base_layer, dtype=np.int32)
    base_layer_2 = np.array(base_layer_2, dtype=np.int32)

    mockup_ground = np.zeros(grid_size)
    ground_zones = [box_1, box_2, base_layer, base_layer_2]
    # ground_zones = [base_layer]
    for zone in ground_zones:
        mask = get_mask_zone_xxyyzz(grid_size, zone, return_bool=True)
        mockup_ground[mask] = 1
    return mockup_ground


@dataclass
class Algo20_Build(AgentAlgorithm):
    """
    # Voxel Builder Algorithm: Algo 20

    the agents randomly build until a density is reached in a radius

    """

    agent_count: int
    grid_size: int | tuple[int, int, int]
    name: str = "algo_20"
    relevant_data_grids: str = "density"
    grid_to_dump: str = "built_centroids"

    seed_iterations: int = 1

    # directory import
    dir_scan_import = REPO_DIR / "data/live/build_grid/01_scanned"
    dir_scan_import_npy = REPO_DIR / "data/live/build_grid/01_scanned/npy"
    dir_save_solid = REPO_DIR / "data/live/build_grid/02_solid"
    dir_save_solid_npy = REPO_DIR / "data/live/build_grid/02_solid/npy"

    # Agent deployment
    legal_move_region_thickness = 1

    print_dot_counter = 0
    legal_move_region = None

    walk_region_thickness = 1

    density_check_radius = 10

    # agent settings

    # settings
    inplane_radius, inplane_top, inplane_bottom = [3, 2, 1]
    depth_radius, depth_top, depth_bottom = [1, 5, 1]

    agent_type_A = {
        "build_probability": 0.7,
        "walk_radius": 4,
        "min_walk_radius": 0,
        "build_radius": 3,
        "build_h": 2,
        "inactive_step_count_limit": None,
        "reset_after_build": 0.2,
        "move_mod_z": 0.2,
        "move_mod_random": 0.5,
        "min_build_density": 0,
        "max_build_density": 0.5,
        "build_by_density_random_factor": 0.01,
        "build_by_density": True,
        "sense_range_radius": 3,
        "inplane_p": [inplane_radius, inplane_top, inplane_bottom],
        "depth_p": [depth_radius, depth_top, depth_bottom],
    }
    agent_types = [agent_type_A, agent_type_A]
    agent_type_dividors = [0, 0.5]

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
        if import_scan:
            file_path = get_nth_newest_file_in_folder(self.dir_save_solid_npy)
            loaded_grid = Grid.from_npy(file_path).array
            loaded_grid = np.array(
                loaded_grid, dtype=np.float64
            )  # TODO set dtype in algo_11_a_import_scan.py  # noqa: E501

            # the imported array is already cropped to the BBOX of interest

            self.grid_size = np.shape(loaded_grid)
            scan.array = loaded_grid
        else:
            scan.array = make_ground_mockup(self.grid_size)

        # IMPORT SCAN
        ground.array = scan.array
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
        built_centroids = DiffusiveGrid(
            name="built_centroids",
            grid_size=self.grid_size,
            color=Color.from_rgb255(252, 25, 0),
            flip_colors=True,
            decay_ratio=1 / 10000,
            decay_linear_value=1 / (iterations * 10),
        )
        density = DiffusiveGrid(
            name="density",
            grid_size=self.grid_size,
            color=Color.from_rgb255(219, 26, 206),
            flip_colors=True,
            decay_ratio=1 / 10000,
        )
        follow_grid = DiffusiveGrid(
            name="follow_grid",
            grid_size=self.grid_size,
            color=Color.from_rgb255(232, 226, 211),
            flip_colors=True,
            decay_ratio=1 / 10000,
        )

        # init legal_move_mask
        self.region_legal_move = get_surrounding_offset_region(
            [ground.array], self.walk_region_thickness
        )

        # WRAP ENVIRONMENT
        grids = {
            "agent": agent_space,
            "ground": ground,
            "track": track,
            "built_centroids": built_centroids,
            "density": density,
            "follow_grid": follow_grid,
            "scan": scan,
        }
        return grids

    def update_environment(self, state: Environment):
        # grids = state.grids
        pass
        # grids["built_centroids"].decay()
        # grids["density"].decay()
        # diffuse_diffusive_grid(grids.follow_grid, )

    def setup_agents(self, grids: dict[str, DiffusiveGrid]):
        agent_space = grids["agent"]
        track = grids["track"]
        ground_grid = grids["ground"]

        agents: list[Agent] = []

        for i in range(self.agent_count):
            # create object
            agent = Agent(
                space_grid=agent_space,
                track_grid=track,
                ground_grid=ground_grid,
                leave_trace=True,
                save_move_history=True,
            )

            # deploy agent
            agent.deploy_in_region(self.region_legal_move)

            # agent settings
            div = self.agent_type_dividors + [1]
            for j in range(len(self.agent_types)):
                u, v = div[j], div[j + 1]
                if u <= i / self.agent_count < v:
                    d = self.agent_types[j]
            agent.id = i
            agent.build_probability = d["build_probability"]
            agent.walk_radius = d["walk_radius"]
            agent.min_walk_radius = d["min_walk_radius"]
            agent.build_radius = d["build_radius"]
            agent.build_h = d["build_h"]
            agent.inactive_step_count_limit = d["inactive_step_count_limit"]
            agent.reset_after_build = d["reset_after_build"]
            agent.reset_after_erase = False
            agent.move_mod_z = d["move_mod_z"]
            agent.move_mod_random = d["move_mod_random"]

            agent.min_build_density = d["min_build_density"]
            agent.max_build_density = d["max_build_density"]
            agent.build_by_density = d["build_by_density"]
            agent.build_by_density_random_factor = d["build_by_density_random_factor"]
            agent.sense_radius = d["sense_range_radius"]

            # create shape maps
            agent.move_shape_map = index_map_sphere(
                agent.walk_radius, agent.min_walk_radius
            )
            agent.build_shape_map = index_map_cylinder(
                agent.build_radius, agent.build_h, z_bottom=-1 * agent.build_h
            )
            agent.sense_range_map = index_map_sphere(agent.sense_radius, 0)
            inplane_radius, inplane_top, inplane_bottom = d["inplane_p"]
            agent.inplane_map = index_map_cylinder(
                inplane_radius, z_top=inplane_top, z_bottom=inplane_bottom
            )
            depth_radius, depth_top, depth_bottom = d["depth_p"]
            agent.depth_map = index_map_cylinder(
                depth_radius, z_top=depth_top, z_bottom=depth_bottom
            )
            agents.append(agent)
        return agents

    def calculate_move_values_random_and_Z(self, agent: Agent, state: Environment):
        """moves agents in a calculated direction
        calculate weigthed sum of slices of grids makes the direction_cube
        check and excludes illegal moves by replace values to -1
        move agent
        return True if moved, False if not or in ground
        """
        move_map_in_place = agent.move_map_in_place

        # random map
        map_size = len(move_map_in_place)
        random_map_values = np.random.random(map_size) + 0.5

        # global direction preference
        move_z_coordinate = (
            np.array(move_map_in_place, dtype=np.float64)[:, 2] - agent.pose[2]
        )

        # MOVE PREFERENCE SETTINGS
        move_z_coordinate *= agent.move_mod_z
        random_map_values *= agent.move_mod_random
        move_values = move_z_coordinate + random_map_values  # + follow_map
        return move_values

    def calculate_move_values_random_and_Z_f(self, agent: Agent, state: Environment):
        """moves agents in a calculated direction
        calculate weigthed sum of slices of grids makes the direction_cube
        check and excludes illegal moves by replace values to -1
        move agent
        return True if moved, False if not or in ground
        """
        move_map_in_place = agent.move_map_in_place

        # random map
        random_map_values = np.random.random(len(move_map_in_place)) + 0.5

        # global direction preference
        move_z_coordinate = (
            np.array(move_map_in_place, dtype=np.float64)[:, 2] - agent.pose[2]
        )

        # follow pheromones
        follow_map = get_values_by_index_map(
            state.grids["follow_grid"].array, agent.move_shape_map, agent.pose
        )

        # MOVE PREFERENCE SETTINGS
        move_z_coordinate *= agent.move_mod_z
        random_map_values *= agent.move_mod_random
        follow_map *= agent.move_mod_follow

        move_values = move_z_coordinate + random_map_values + follow_map

        return move_values

    def get_legal_move_mask(self, agent: Agent, state: Environment):
        """moves agents in a calculated direction
        calculate weigthed sum of slices of grids makes the direction_cube
        check and excludes illegal moves by replace values to -1
        move agent
        return True if moved, False if not or in ground
        """

        # legal move mask
        filter = get_values_by_index_map(
            self.region_legal_move, agent.move_shape_map, agent.pose, dtype=np.float64
        )
        legal_move_mask = filter == 1
        return legal_move_mask

    def build(self, agent: Agent, state: Environment):
        """fill built volume in built_shape if agent.build_probability >= build_limit"""

        self.print_dot_counter += 1

        density = state.grids["density"]
        built_centroids = state.grids["built_centroids"]
        ground = state.grids["ground"]

        x, y, z = agent.pose

        # update print dot array
        built_centroids.array[x, y, z] = self.print_dot_counter

        # orient shape map
        build_map = agent.orient_build_shape_map()
        # update density_volume_array
        density.array = set_value_by_index_map(
            density.array,
            build_map,
            value=self.print_dot_counter,
        )
        ground.array = set_value_by_index_map(
            ground.array,
            build_map,
            value=self.print_dot_counter,
        )

        print(f"built at: {agent.pose}")

    def get_agent_build_probability(self, agent, state):
        # BUILD CONSTRAINTS:
        # cone_angle = 45
        # cone_height = 100
        # cone_division = 4

        # nozzle_access_condition = agent.check_accessibility_in_cone_divisions(
        #     cone_angle, cone_height, cone_division
        # ) TODO
        nozzle_access_condition = True
        max_build_angle = 50
        build_angle_condition = agent.normal_angle > max_build_angle

        if not nozzle_access_condition and build_angle_condition:
            build_probability = 0
        else:
            # RANDOM FACTOR
            random_factor = 0.1
            bg_random = (r.random() - 0.5) * random_factor

            # NORMAL ANGLE PREFERENCE
            angle1, angle2 = [0, 25]
            angle_factor = 0.05
            if angle1 <= agent.normal_angle <= angle2:
                bg_angle_factor = angle_factor
            else:
                bg_angle_factor = angle_factor * -2
            # print(f"angle_gain {bg_angle_factor}")

            # TOPOLOGY SENSATION:
            # agent.orient_topology_maps()
            # topology gains
            topology_gain_inplane = 0.9
            topology_gain_edge = 0.9
            # thick_wall_pain = -0.75
            # thin_wall_gain = 0.75
            # edge_gain = 0.75
            # wall thickness and shell edge
            density = state.grids["density"]
            ground = state.grids["ground"]
            depth_map = agent.orient_depth_map()
            depth_density = agent.get_array_density_by_oriented_index_map(
                density.array + ground.array, depth_map, nonzero=True
            )
            inplane_map = agent.orient_inplane_map()
            inplane_density = agent.get_array_density_by_oriented_index_map(
                density.array, ground.array, inplane_map, nonzero=True
            )
            # access topology type
            # topology sensor values
            d_wall_inplane = 0.4
            d_wall_max_depth = 0.6
            d_min_ridge_depth = 0.1
            if inplane_density >= d_wall_inplane:  # walking on wall
                if depth_density > d_wall_max_depth:
                    # too thick wall
                    bg_shell_topology = topology_gain_inplane * -1
                    print("THICK SHELL")
                elif depth_density <= d_wall_max_depth:
                    # thin wall >> build
                    bg_shell_topology = topology_gain_inplane
                    print("THIN SHELL")
            elif (
                inplane_density < d_wall_inplane and depth_density >= d_min_ridge_depth
            ):
                # being on ridge edge of the shell
                bg_shell_topology = topology_gain_edge
                print("EDGE")
            else:
                bg_shell_topology = 0

            build_probability = bg_random + bg_angle_factor + bg_shell_topology

        return build_probability

    # ACTION FUNCTION
    def agent_action(self, agent, state: Environment):
        """
        GET_PROPABILITY
        BUILD
        MOVE
        *RESET
        """
        # BUILD
        build_probability = self.get_agent_build_probability(agent, state)

        if build_probability > r.random():
            print(f"""\n## agent_{agent.id} - {agent.pose} ##""")
            print(f"build_probability = {build_probability}")

            print(f"angle {agent.normal_angle}")

            # build
            self.build(agent, state)

            # update walk region
            self.region_legal_move = get_surrounding_offset_region(
                [state.grids["ground"].array, state.grids["density"].array],
                self.walk_region_thickness,
            )
            # reset if
            agent.step_counter = 0
            if agent.reset_after_build:
                if isinstance(agent.reset_after_build, float):
                    if r.random() < agent.reset_after_build:
                        agent.deploy_in_region(self.region_legal_move)
                elif agent.reset_after_build is True:
                    agent.deploy_in_region(self.region_legal_move)
                else:
                    pass

        # MOVE
        # check collision
        collision = agent.check_solid_collision(
            [state.grids["density"].array, state.grids["ground"].array]
        )
        # move
        if not collision:
            move_values = self.calculate_move_values_random_and_Z(agent, state)
            move_map_in_place = agent.move_map_in_place

            legal_move_mask = self.get_legal_move_mask(agent, state)

            agent.move_by_index_map(
                index_map_in_place=move_map_in_place[legal_move_mask],
                move_values=move_values[legal_move_mask],
                random_batch_size=1,
            )
            agent.step_counter += 1

        # RESET
        else:
            # reset if stuck
            agent.deploy_in_region(self.region_legal_move)

        # reset if inactive
        if agent.inactive_step_count_limit:  # noqa: SIM102
            if agent.step_counter >= agent.inactive_step_count_limit:
                agent.deploy_in_region(self.region_legal_move)
