import random as r
from dataclasses import dataclass

import numpy as np

from bdm_voxel_builder import REPO_DIR
from bdm_voxel_builder.agent import Agent
from bdm_voxel_builder.agent_algorithms.base import AgentAlgorithm
from bdm_voxel_builder.environment import Environment
from bdm_voxel_builder.grid import DiffusiveGrid
from bdm_voxel_builder.helpers import (
    get_savepath,
    index_map_cylinder,
    index_map_sphere,
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
    vdb_to_dump: str = "built"  # not implemented

    # global settings

    n = 50 if follow_built else 1
    seed_iterations: int = n

    walk_region_thickness = 1
    deploy_anywhere = deploy_anywhere  # only on the initial scan volume

    to_decay = ["built"]

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

    def initialization(self, state: Environment):
        """
        creates the simulation environment setup
        with preset values in the definition

        returns: grids

        """
        goal_density = state.grids["goal_density"]

        goal_density.set_values_using_index_map(
            self.make_goal_density_box_mockup_A(), goal_density__modifier_A
        )

        goal_density.set_values_using_index_map(
            self.make_goal_density_box_mockup_B(), goal_density__modifier_B
        )

        goal_density.set_values_using_index_map(
            self.make_goal_density_box_mockup_C(), goal_density__modifier_C
        )

        # update walk region
        self.update_offset_regions(
            state.grids["ground"].to_numpy(), state.grids["scan"].to_numpy()
        )

    def update_environment(self, state: Environment):
        self.decay_environment(state)

        emission_array = (
            state.grids["goal_density"].to_numpy() * follow_goal
            + state.grids["built"].to_numpy() * follow_built
        )

        self.diffuse_follow_grid(state, emission_array)

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

            # BUILD NEXT TO built
            if agent.build_next_to_bool:
                built_volume = state.grids["built"]
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
