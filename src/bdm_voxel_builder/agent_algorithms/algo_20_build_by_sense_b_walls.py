import random as r
from dataclasses import dataclass

import numpy as np

from bdm_voxel_builder import REPO_DIR
from bdm_voxel_builder.agent import Agent
from bdm_voxel_builder.agent_algorithms.base import AgentAlgorithm
from bdm_voxel_builder.environment import Environment
from bdm_voxel_builder.grid import DiffusiveGrid
from bdm_voxel_builder.helpers.array import (
    get_savepath,
    index_map_cylinder,
    index_map_sphere,
)

# ultimate_parameters - walls_B
overhang = 0.35
move_up = 0.4
follow_newly_built = 100
start_to_build_new_volume_chance = 0.01
max_shell_thickness = 15
deploy_anywhere = True
add_initial_box = False
reset = True


# # ultimate_parameters - walls_A
# overhang = 0.45
# move_up = 1
# follow_newly_built = 100
# start_to_build_new_volume_chance = 0.01
# max_shell_thickness = 15
# deploy_anywhere = False
# add_initial_box = False
# reset = False


@dataclass
class Algo20_Build_b(AgentAlgorithm):
    """
    # Voxel Builder Algorithm: Algo 20

    the agents build based on sensor feedback
    overhang
    wall thickness
    print_nozzle access

    """

    agent_count: int
    grid_size: int | tuple[int, int, int]
    name: str = "algo_20"
    relevant_data_grids: str = "ground"
    grid_to_dump: str = "ground"

    # TODO
    vdb_to_dump: str = "built"  # not implemented

    # global settings

    n = 50 if follow_newly_built else 1
    seed_iterations: int = n

    walk_region_thickness = 1
    deploy_anywhere = deploy_anywhere  # only on the initial scan volume

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

    def initialization(self, state: Environment):
        """
        creates the simulation environment setup
        with preset values in the definition

        returns: grids

        """
        # update walk region
        self.update_offset_regions(
            state.grids["ground"].to_numpy(), state.grids["scan"].to_numpy()
        )

    def update_environment(self, state: Environment):
        self.decay_environment(state)
        if follow_newly_built > 0:
            self.diffuse_follow_grid(state, state.grids["built"].to_numpy())

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
                basic_agent.move_mod_random = 1
                basic_agent.move_mod_follow = follow_newly_built
                # build settings
                basic_agent.build_radius = 3
                basic_agent.build_h = 3
                basic_agent.reset_after_build = reset
                basic_agent.inactive_step_count_limit = None
                # sensor settings
                basic_agent.sense_radius = 3
                basic_agent.build_random_chance = 0.01
                basic_agent.build_random_gain = 0
                basic_agent.max_shell_thickness = max_shell_thickness
                basic_agent.max_build_angle = 30
                basic_agent.overhang_density = overhang

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
                basic_agent.sense_inplane_map = index_map_cylinder(
                    radius=3, height=2, min_radius=0, z_lift=1
                )
                basic_agent.sense_depth_map = index_map_cylinder(
                    1, basic_agent.max_shell_thickness * 2, 0, 1
                )
                basic_agent.sense_overhang_map = index_map_cylinder(
                    radius=1, height=1, z_lift=-1
                )

                basic_agent.sense_nozzle_map = index_map_cylinder(
                    radius=0, height=40, z_lift=0
                )

                # set grids
                basic_agent.space_grid = agent_space
                basic_agent.track_grid = track
                basic_agent.ground_grid = ground_grid
                basic_agent.id = id
                print(f"created agent_{basic_agent.id}")

                # deploy agent
                basic_agent.deploy_in_region(self.region_deploy_agent)

                # append
                agents.append(basic_agent)
                id += 1

        return agents

    def build(self, agent: Agent, state: Environment):
        """fill built volume in built_shape if agent.build_probability >=
        build_limit"""
        super().build(agent, state)

        move_map_grid = state.grids["move_map"]

        move_map_grid.set_value_using_index_map(
            agent.orient_sense_overhang_map(), values=1
        )

        move_map_grid.set_value_using_index_map(
            agent.orient_sense_nozzle_map(), values=1
        )

    def get_agent_build_probability(self, agent, state):
        # BUILD CONSTRAINTS:
        ground = state.grids["ground"]

        nozzle_map = agent.orient_sense_nozzle_map(world_z=False)
        nozzle_access_density = agent.get_array_density_by_oriented_index_map(
            ground.array, nozzle_map, nonzero=True
        )

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

            if agent.sense_topology_bool:
                built = state.grids["built"]
                move_map = agent.orient_move_map()
                built_density = agent.get_array_density_by_oriented_index_map(
                    built.array, move_map, nonzero=True
                )
                # TOPOLOGY SENSATION:
                if built_density < 0.05:  # noqa: SIM108
                    topology_gain_inplane = start_to_build_new_volume_chance
                else:
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
                    ground.array,
                    sense_depth_map,
                    nonzero=True,
                    density_of_original_index_map=True,
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

            build_probability = bp_random + bp_shell_topology
        # print(f"build_probability:{build_probability}")
        return build_probability
