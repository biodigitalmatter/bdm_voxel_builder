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
follow_newly_built = 10

build_next_to_bool = False
sense_wall_radar_bool = True
sense_smoothing_map_bool = True

build_probability_absolut_random = 0
build_probability_next_to = 0
build_probability_wall_radar = [0, -2]
max_radar_density = 0.5
min_density = 0.2
build_probability_smoothing = [0.5, 0.31]
smoothing_density_limits = [0.4, 0.8]


deploy_anywhere = False
add_initial_box = False
reset = False


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
                basic_agent.wall_radar_radius = 3 * basic_agent.build_radius
                basic_agent.build_probability_wall_radar = build_probability_wall_radar
                basic_agent.max_radar_density = max_radar_density

                # NEW
                basic_agent.sense_smoothing_map_bool = sense_smoothing_map_bool
                basic_agent.build_probability_smoothing = build_probability_smoothing
                basic_agent.smoothing_density_limits = smoothing_density_limits
                basic_agent.min_density = min_density

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
                    radius=basic_agent.build_radius * 3, height=1, z_lift=0
                )
                basic_agent.sense_smoothing_map = index_map_cylinder(
                    radius=basic_agent.build_radius * 2, height=1, z_lift=0
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
                    bp_build_next_to = 0
                else:
                    print(f"built_density = {built_density}")
                    bp_build_next_to = agent.build_probability_next_to
            else:
                bp_build_next_to = 0

            # BUILD BY WALL RADAR
            if agent.sense_wall_radar_bool:
                ground = state.grids["ground"]
                wall_radar_map = agent.orient_sense_wall_radar_map()
                radar_density = agent.get_array_density_by_oriented_index_map(
                    ground.array, wall_radar_map, nonzero=True
                )
                if radar_density < agent.max_radar_density:  # walking on wall
                    bp_wall_radar = agent.build_probability_wall_radar[0]
                else:
                    bp_wall_radar = agent.build_probability_wall_radar[1]
            else:
                bp_wall_radar = 0

            # BUILD BY SMOOTHING_RADAR
            if agent.sense_smoothing_map_bool:
                ground = state.grids["ground"]
                smoothing_map = agent.orient_index_map(
                    agent.sense_smoothing_map, normal=[0, 0, 1]
                )
                smooth_density = agent.get_array_density_by_oriented_index_map(
                    ground.array, smoothing_map, nonzero=True
                )
                # print(f'smoothing_density: {smooth_density}')
                if smooth_density < agent.min_density:
                    bp_smoothing = 0
                if (
                    agent.smoothing_density_limits[0]
                    <= smooth_density
                    <= agent.smoothing_density_limits[1]
                ):  # walking on wall
                    bp_smoothing = agent.build_probability_smoothing[0]
                else:
                    bp_smoothing = agent.build_probability_smoothing[1]
            else:
                bp_smoothing = 0

            build_probability = (
                bp_random + bp_build_next_to + bp_wall_radar + bp_smoothing
            )
        # print(f"build_probability:{build_probability}")
        return build_probability

    def get_move_values(self, agent: Agent, state: Environment):
        follow_grid = state.grids["follow"]
        built_grid = state.grids["built"]
        return (
            agent.calculate_move_values_random__z_based__follow_with_low_density_weight(
                follow_grid,
                built_grid,
                low_density_threshold=0.1,
                low_density_weight=1e7,
            )
        )
