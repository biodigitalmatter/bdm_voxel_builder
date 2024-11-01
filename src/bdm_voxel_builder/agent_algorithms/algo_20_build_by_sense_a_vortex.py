import random as r
from dataclasses import dataclass

import numpy as np

from bdm_voxel_builder import REPO_DIR
from bdm_voxel_builder.agent import Agent
from bdm_voxel_builder.agent_algorithms.base import AgentAlgorithm
from bdm_voxel_builder.environment import Environment
from bdm_voxel_builder.grid import DiffusiveGrid
from bdm_voxel_builder.helpers import (
    get_mask_zone_xxyyzz,
    get_savepath,
    index_map_cylinder,
    index_map_sphere,
)


def make_ground_mockup(grid_size):
    a, b, c = grid_size

    base_layer = [0, a, 0, b, 0, 10]
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
    box_1 = [a / 2, a / 2 + 3, b / 2, b / 2 + 3, 10, 13]
    box_1 = np.array(box_1, dtype=np.int32)

    mockup_ground = np.zeros(grid_size)
    ground_zones = [box_1]
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
basic_agent.move_mod_z = 0.1
basic_agent.move_mod_random = 0.6
basic_agent.move_mod_follow = 1
# build settings
basic_agent.build_radius = 3
basic_agent.build_h = 3
basic_agent.reset_after_build = True
basic_agent.inactive_step_count_limit = None
# sensor settings
basic_agent.sense_radius = 3
basic_agent.build_random_chance = 0.04
basic_agent.build_random_gain = 0
basic_agent.pref_build_angle = 25
basic_agent.pref_build_angle_gain = 0
basic_agent.max_shell_thickness = 15
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

print("started")


@dataclass
class Algo20_Build_a(AgentAlgorithm):
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
    vdb_to_dump: str = "built"  # not implemented
    point_cloud_to_dump: str = "centroids"  # not implemented

    seed_iterations: int = 1

    # global settings

    walk_region_thickness = 1
    deploy_anywhere = True

    # import scan
    import_scan = False
    dir_solid_npy = REPO_DIR / "data/live/build_grid/02_solid/npy"

    to_decay = ["built"]

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

    def update_environment(self, state: Environment):
        self.decay_environment(state)
        self.diffuse_follow_grid(state, state.grids["built"].to_numpy())

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
                agent.deploy_in_region(self.region_deploy_agent)
                agents.append(agent)
                del agent
                id += 1
        # for agent in agents:
        #     print(agent.id, agent.agent_type_summary)

        print("build_random_chance", agents[0].build_random_chance)
        print("pref_build_angle_gain", agents[0].pref_build_angle_gain)

        return agents

    def get_agent_build_probability(self, agent, state):
        # BUILD CONSTRAINTS:
        # cone_angle = 45
        # cone_height = 100
        # cone_division = 4

        # nozzle_access_condition = agent.check_accessibility_in_cone_divisions(
        #     cone_angle, cone_height, cone_division
        # ) TODO
        nozzle_access_collision = False
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
        # print(f"build_probability:{build_probability}")
        return build_probability
