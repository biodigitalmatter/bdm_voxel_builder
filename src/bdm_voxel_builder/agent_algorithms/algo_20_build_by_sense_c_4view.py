import random as r
from dataclasses import dataclass

import compas.geometry as cg
import numpy as np
from compas import json_dumps

from bdm_voxel_builder import REPO_DIR
from bdm_voxel_builder.agent import Agent
from bdm_voxel_builder.agent_algorithms.base import AgentAlgorithm
from bdm_voxel_builder.environment import Environment
from bdm_voxel_builder.helpers import (
    get_mask_zone_xxyyzz,
    get_savepath,
    get_surrounding_offset_region,
    index_map_cylinder,
    index_map_sphere,
    set_value_using_index_map,
)
from bdm_voxel_builder.helpers.array import get_array_density_using_index_map

# ultimate_parameters - test_1 - absolut random build
overhang_density = 0.35
move_up = 0
move_random = 1
follow_newly_built = 10

build_next_to_bool = True
sense_wall_radar_bool = True

build_probability_absolut_random = 0.001
build_probability_next_to = 1
build_probability_wall_radar_low = 0.5
build_probability_wall_radar_high = -2
max_radar_density = 0.33

wall_radar_radius = 15
deploy_anywhere = False
add_initial_box = False
reset = True


@dataclass
class Algo20_Build_c(AgentAlgorithm):
    """
    # Voxel Builder Algorithm: Algo 20

    the agents build based on sensor feedback
    overhang
    print_nozzle access

    20_c:
    build next to
    no topology sensation

    """

    agent_count: int
    clipping_box: cg.Box
    name: str = "algo_20"
    relevant_data_grids: str = "ground"
    grid_to_dump: str = "ground"

    # TODO
    vdb_to_dump: str = "built_volume"  # not implemented

    # global settings

    n = 50 if follow_newly_built else 1
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
            clipping_box=self.clipping_box,
            grid_to_dump=self.grid_to_dump,
            name=self.name,
        )

    def initialization(self, state: Environment):
        """
        creates the simulation environment setup
        with preset values in the definition

        returns: grids

        """
        ground = state.grids["ground"]
        scan = state.grids["scan"]

        # update walk region
        self.update_offset_regions(ground.to_numpy(), scan.to_numpy())

        print("initialized")

    def update_environment(self, state: Environment, **kwargs):
        grids = state.grids
        pass
        # grids["centroids"].decay()
        grids["built_volume"].decay()
        if follow_newly_built > 0:
            grids["follow_grid"].diffuse_diffusive_grid(
                emission_array=grids["built_volume"].to_numpy(),
                blocking_grids=[grids["ground"]],
                grade=False,
            )

    def setup_agents(self, state: Environment):
        agent_space = state.grids["agent"]
        track = state.grids["track"]
        ground_grid = state.grids["ground"]
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
                basic_agent.move_mod_follow = follow_newly_built
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
                basic_agent.wall_radar_radius = 5
                basic_agent.build_probability_wall_radar = [
                    build_probability_wall_radar_low,
                    build_probability_wall_radar_high,
                ]
                basic_agent.max_radar_density = max_radar_density

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
        follow_map = get_values_using_index_map(
            state.grids["follow_grid"].array, agent.move_map, agent.pose
        )

        built_density = agent.get_array_density_by_oriented_index_map(
            state.grids["built_volume"].array, move_map_in_place, nonzero=True
        )
        # MOVE PREFERENCE SETTINGS
        move_z_coordinate *= agent.move_mod_z
        random_map_values *= agent.move_mod_random
        follow_map *= agent.move_mod_follow
        # print(f"follow_map min:{np.min(follow_map)} max: {np.max(follow_map)}")
        # print(
        #     f"radnom_map min:{np.min(random_map_values)} max: {np.max(random_map_values)}"
        # )
        # print(
        #     f"move_z_coordinate min:{np.min(move_z_coordinate)} max: {np.max(move_z_coordinate)}"
        # )
        if built_density < 0.1:
            follow_map *= 10000000

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
        built_volume.array = set_value_using_index_map(
            built_volume.array,
            build_map,
            value=1,
        )
        ground.array = set_value_by_index_map(
            ground.array,
            build_map,
            value=1,
        )

        # update grids for index_map visualisation
        # move_map_grid.array = set_value_by_index_map(
        #     move_map_grid.array, agent.orient_sense_map()
        # )
        # sense_maps_grid.array = set_value_by_index_map(
        #     sense_maps_grid.array, agent.orient_sense_inplane_map()
        # )
        # sense_maps_grid.array = set_value_by_index_map(
        #     sense_maps_grid.array, agent.orient_sense_depth_map()
        # )
        # move_map_grid.array = set_value_by_index_map(
        #     move_map_grid.array, agent.orient_sense_overhang_map()
        # )
        # move_map_grid.array = set_value_by_index_map(
        #     move_map_grid.array, agent.orient_sense_nozzle_map()
        # )

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
        ground_array = state.grids["ground"].to_numpy()

        nozzle_map = agent.orient_sense_nozzle_map(world_z=False)
        nozzle_access_density = get_array_density_using_index_map(
            ground_array, nozzle_map, nonzero=True
        )

        # print(
        #     f"nozzle_acces_density: {nozzle_access_density}, degree: {agent.normal_angle}"
        # )
        nozzle_access_collision = nozzle_access_density >= 0.01
        overhang_map = agent.orient_sense_overhang_map()
        density = get_array_density_using_index_map(
            ground_array, overhang_map, nonzero=True
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
                built_density = get_array_density_using_index_map(
                    built_volume.to_numpy(), build_map, nonzero=True
                )

                if built_density < 0.001:  # noqa: SIM108
                    bp_build_next_to = 0
                else:
                    print(f"built_density = {built_density}")
                    bp_build_next_to = agent.build_probability_next_to

            # BUILD BY WALL RADAR
            if agent.sense_wall_radar_bool:
                wall_radar_map = agent.orient_sense_wall_radar_map()
                radar_density = get_array_density_using_index_map(
                    ground_array, wall_radar_map, nonzero=True
                )
                if radar_density < agent.max_radar_density:  # walking on wall
                    bp_wall_radar = agent.build_probability_wall_radar[0]
                else:
                    bp_wall_radar = agent.build_probability_wall_radar[1]

            build_probability = bp_random + bp_build_next_to + bp_wall_radar
        # print(f"build_probability:{build_probability}")
        return build_probability

    def update_offset_regions(self, ground_array, scan_array):
        self.region_legal_move = get_surrounding_offset_region(
            arrays=[ground_array],
            offset_thickness=self.walk_region_thickness,
        )
        if self.deploy_anywhere:
            self.region_deploy_agent = self.region_legal_move
        else:
            self.region_deploy_agent = get_surrounding_offset_region(
                arrays=[scan_array],
                offset_thickness=self.walk_region_thickness,
                exclude_arrays=[ground_array],
            )

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
            print(f"""## agent_{agent.id} built at {agent.pose}""")
            print(f"## build_probability = {build_probability}")

            # build
            self.build(agent, state)

            # update offset regions
            self.update_offset_regions(
                ground_array=state.grids["ground"].array.copy(),
                scan_array=state.grids["scan"].array.copy(),
            )

            # reset if
            agent.step_counter = 0
            if agent.reset_after_build:
                if isinstance(agent.reset_after_build, float):
                    if r.random() < agent.reset_after_build:
                        agent.deploy_in_region(self.region_deploy_agent)
                elif agent.reset_after_build is True:
                    agent.deploy_in_region(self.region_deploy_agent)
                else:
                    pass
        # MOVE
        # check collision
        collision = agent.check_solid_collision([state.grids["built_volume"].array])
        # move
        if not collision:
            move_values = self.calculate_move_values_random_and_Z_f(agent, state)
            move_map_in_place = agent.move_map_in_place

            legal_move_mask = self.get_legal_move_mask(agent, state)

            agent.move_by_index_map(
                index_map_in_place=move_map_in_place[legal_move_mask],
                move_values=move_values[legal_move_mask],
                random_batch_size=4,
            )
            agent.step_counter += 1

        # RESET
        else:
            # reset if stuck
            agent.deploy_in_region(self.region_deploy_agent)

        # reset if inactive
        if agent.inactive_step_count_limit:  # noqa: SIM102
            if agent.step_counter >= agent.inactive_step_count_limit:
                agent.deploy_in_region(self.region_deploy_agent)


def make_ground_mockup(grid_size):
    a, b, c = grid_size

    base_layer = [0, a, 0, b, 0, 5]
    base_layer = np.array(base_layer, dtype=np.int32)

    mockup_ground = np.zeros(grid_size)
    ground_zones = [base_layer]
    # ground_zones = [base_layer]
    for zone in ground_zones:
        mask = get_mask_zone_xxyyzz(grid_size, zone, return_bool=True)
        mockup_ground[mask] = 1
    return mockup_ground


def make_init_box_mockup(grid_size):
    a, b, c = grid_size
    box_1 = [a / 2, a / 2 + 3, b / 2, b / 2 + 3, 5, 13]
    box_1 = np.array(box_1, dtype=np.int32)

    mockup_ground = np.zeros(grid_size)
    ground_zones = [box_1]
    # ground_zones = [base_layer]
    for zone in ground_zones:
        mask = get_mask_zone_xxyyzz(grid_size, zone, return_bool=True)
        mockup_ground[mask] = 1
    return mockup_ground
