import random as r
from dataclasses import dataclass

import numpy as np
from compas.colors import Color

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


def make_ground_mockup(grid_size):
    a, b, c = grid_size
    box_1 = [a / 2, a / 2 + 3, b / 3, b / 3 + 20, 0, 30]
    box_1 = np.array(box_1, dtype=np.int32)

    base_layer = [0, a, 0, b, 0, 10]
    base_layer = np.array(base_layer, dtype=np.int32)

    mockup_ground = np.zeros(grid_size)
    ground_zones = [box_1, base_layer]
    # ground_zones = [base_layer]
    for zone in ground_zones:
        mask = get_mask_zone_xxyyzz(grid_size, zone, return_bool=True)
        mockup_ground[mask] = 1
    return mockup_ground


# CREATE AGENT TYPES

basic_agent = Agent()
# TYPE A
basic_agent.agent_type_summary = "A work_on_shell_and_edge"
# movement settings
basic_agent.walk_radius = 4
basic_agent.move_mod_z = 0.08
basic_agent.move_mod_random = 0.5
# build settings
basic_agent.build_radius = 3
basic_agent.build_h = 3
basic_agent.reset_after_build = True
basic_agent.inactive_step_count_limit = None
# sensor settings
basic_agent.sense_radius = 3
basic_agent.build_random_chance = 0.04
basic_agent.build_random_gain = 10
basic_agent.pref_build_angle = 25
basic_agent.pref_build_angle_gain = 0
basic_agent.max_shell_thickness = 5
basic_agent.max_build_angle = 91

# create shape maps
basic_agent.move_map = index_map_sphere(
    basic_agent.walk_radius, basic_agent.min_walk_radius
)
basic_agent.build_map = index_map_cylinder(
    basic_agent.build_radius, basic_agent.build_h, 0, -1
)
basic_agent.sense_map = index_map_sphere(basic_agent.sense_radius)
basic_agent.sense_inplane_map = index_map_cylinder(
    radius=3, height=2, min_radius=0, z_lift=1
)
basic_agent.sense_depth_map = index_map_cylinder(
    1, basic_agent.max_shell_thickness * 2, 0, 1
)
# __dict__
agent_dict_A = basic_agent.__dict__.copy()

# # TYPE B
# basic_agent.agent_type_summary = "B random_build_if_vertical"
# # movement settings
# basic_agent.max_build_angle = 10
# basic_agent.walk_radius = 2
# basic_agent.build_random_chance = 0.1
# basic_agent.sense_topology_bool = False
# basic_agent.reset_after_build = True
# agent_dict_B = basic_agent.__dict__.copy()

# # TYPE C
# basic_agent.agent_type_summary = "C tower_up"
# # movement settings
# basic_agent.move_mod_z = 0.3
# basic_agent.build_random_chance = 0.1
# basic_agent.sense_topology_bool = False
# basic_agent.reset_after_build = False
# # basic_agent.die_chance = 0.2
# agent_dict_C = basic_agent.__dict__.copy()

# dict list
# agent_type_dicts = [agent_dict_A, agent_dict_B, agent_dict_C]
agent_type_dicts = [agent_dict_A]

agent_type_distribution = [1]


@dataclass
class Algo20_Build(AgentAlgorithm):
    """
    # Voxel Builder Algorithm: Algo 20

    the agents randomly build until a density is reached in a radius

    """

    agent_count: int
    grid_size: int | tuple[int, int, int]
    name: str = "algo_20"
    relevant_data_grids: str = "ground"
    grid_to_dump: str = "ground"

    # TODO
    vdb_to_dump: str = "built_volume"  # not implemented
    point_cloud_to_dump: str = "centroids"  # not implemented

    seed_iterations: int = 1

    # global settings

    walk_region_thickness = 1

    # import scan
    import_scan = False
    dir_solid_npy = REPO_DIR / "data/live/build_grid/02_solid/npy"

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
            scan.array = make_ground_mockup(self.grid_size)

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
            decay_ratio=1 / 10000,
        )
        follow_grid = DiffusiveGrid(
            name="follow_grid",
            grid_size=self.grid_size,
            color=Color.from_rgb255(232, 226, 211),
            flip_colors=True,
            decay_ratio=1 / 10000,
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

        # init legal_move_mask
        self.region_legal_move = get_surrounding_offset_region(
            [ground.array], self.walk_region_thickness
        )

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
        }
        return grids

    def update_environment(self, state: Environment):
        # grids = state.grids
        pass
        # grids["centroids"].decay()
        # grids["built_volume"].decay()
        # diffuse_diffusive_grid(grids.follow_grid, )

    def setup_agents(self, grids: dict[str, DiffusiveGrid]):
        agent_space = grids["agent"]
        track = grids["track"]
        ground_grid = grids["ground"]
        agents = []

        # generate agents based on agent_type_dicts
        d = np.array(agent_type_distribution)
        sum_ = np.sum(d)
        d_normalized = d * 1 / sum_
        # print(d_normalized)
        id = 0
        for i, n in enumerate(d_normalized):
            print(i, n)
            type_size = int(n * self.agent_count)
            # print(type_size)
            for _j in range(type_size):
                data_dict = agent_type_dicts[i]
                # print("type:", data_dict["agent_type_summary"])
                # create object
                # print("id in the loop", id)
                agent = Agent()
                agent.__dict__ = data_dict
                # set grids
                agent.space_grid = agent_space
                agent.track_grid = track
                agent.ground_grid = ground_grid
                agent.id = id
                # deploy agent
                agent.deploy_in_region(self.region_legal_move)
                agents.append(agent)
                del agent
                id += 1
        # for agent in agents:
        #     print(agent.id, agent.agent_type_summary)

        print("build_random_chance", agents[0].build_random_chance)
        print("pref_build_angle_gain", agents[0].pref_build_angle_gain)

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
            state.grids["follow_grid"].array, agent.move_map, agent.pose
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
            self.region_legal_move, agent.move_map, agent.pose, dtype=np.float64
        )
        legal_move_mask = filter == 1
        return legal_move_mask

    def build(self, agent: Agent, state: Environment):
        """fill built volume in built_shape if agent.build_probability >= build_limit"""

        self.print_dot_counter += 1

        built_volume = state.grids["built_volume"]
        centroids = state.grids["centroids"]
        ground = state.grids["ground"]
        sense_maps_grid = state.grids["sense_maps_grid"]
        move_map_grid = state.grids["move_map_grid"]

        x, y, z = agent.pose

        # update print dot array
        centroids.array[x, y, z] = self.print_dot_counter

        # orient shape map
        build_map = agent.orient_build_map()
        # update built_volume_volume_array
        built_volume.array = set_value_by_index_map(
            built_volume.array,
            build_map,
            value=self.print_dot_counter,
        )
        ground.array = set_value_by_index_map(
            ground.array,
            build_map,
            value=self.print_dot_counter,
        )

        # update grids for index_map visualisation
        move_map_grid.array = set_value_by_index_map(
            move_map_grid.array, agent.orient_sense_map()
        )
        sense_maps_grid.array = set_value_by_index_map(
            sense_maps_grid.array, agent.orient_sense_inplane_map()
        )
        sense_maps_grid.array = set_value_by_index_map(
            sense_maps_grid.array, agent.orient_sense_depth_map()
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
        nozzle_access_collision = False
        print(f"angle: {agent.normal_angle}")
        overhanging = agent.normal_angle > agent.max_build_angle
        if nozzle_access_collision or overhanging:
            build_probability = 0
        else:
            # RANDOM FACTOR
            if r.random() < agent.build_random_chance:  # noqa: SIM108
                bp_random = agent.build_random_gain
            else:
                bp_random = 0

            # NORMAL ANGLE PREFERENCE
            if agent.normal_angle <= agent.pref_build_angle:
                bp_angle_factor = agent.pref_build_angle_gain
            else:
                bp_angle_factor = agent.pref_build_angle_gain * -2

            if agent.sense_topology_bool:
                # TOPOLOGY SENSATION:
                topology_gain_inplane = 0.8
                topology_gain_edge = 0.8
                # topology sensor values
                shell_planarity_max_fill = 0.75
                shell_thickness_max_fill = 0.6
                edge_depth_min_fill = 0.2
                # wall thickness and shell edge
                ground = state.grids["ground"]
                sense_depth_map = agent.orient_sense_depth_map()
                depth_density = agent.get_array_density_by_oriented_index_map(
                    ground.array, sense_depth_map, nonzero=True
                )
                sense_inplane_map = agent.orient_sense_inplane_map()
                inplane_density = agent.get_array_density_by_oriented_index_map(
                    ground.array, sense_inplane_map, nonzero=True
                )
                # access topology type
                if inplane_density >= shell_planarity_max_fill:  # walking on wall
                    if depth_density > shell_thickness_max_fill:
                        # too thick wall
                        bp_shell_topology = topology_gain_inplane * -1
                        # print("THICK SHELL")
                    elif depth_density <= shell_thickness_max_fill:
                        # thin wall >> build
                        bp_shell_topology = topology_gain_inplane
                        print("\nTHIN SHELL")
                elif (
                    inplane_density < shell_planarity_max_fill
                    and depth_density >= edge_depth_min_fill
                ):
                    # being on ridge edge of the shell
                    bp_shell_topology = topology_gain_edge
                    print("\nEDGE ")
                else:
                    bp_shell_topology = 0
            else:
                bp_shell_topology = 0

            build_probability = bp_random + bp_angle_factor + bp_shell_topology
        print(f"build_probability:{build_probability}")
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
        # print(
        #     f"""\n## agent_{agent.id} - {agent.pose} ##\n{agent.agent_type_summary}\n {agent.normal_vector} ,{agent.normal_angle} degree"""  # noqa: E501
        # )
        if build_probability > r.random():
            print(
                f"""\n## agent_{agent.id} - {agent.pose} ##\n{agent.agent_type_summary}\n {agent.normal_vector}"""  # noqa: E501
            )
            print(f"build_probability = {build_probability}")

            print(f"angle {agent.normal_angle}")

            # build
            self.build(agent, state)

            # update walk region
            self.region_legal_move = get_surrounding_offset_region(
                arrays=[state.grids["ground"].array],
                offset_thickness=self.walk_region_thickness,
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
        collision = agent.check_solid_collision([state.grids["built_volume"].array])
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
