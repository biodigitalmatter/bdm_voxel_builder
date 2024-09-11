import random as r
from dataclasses import dataclass

import compas.geometry as cg
import numpy as np
from compas.colors import Color

from bdm_voxel_builder import REPO_DIR
from bdm_voxel_builder.agent import Agent
from bdm_voxel_builder.agent_algorithms.base import AgentAlgorithm
from bdm_voxel_builder.agent_algorithms.common import make_ground_mockup
from bdm_voxel_builder.environment import Environment
from bdm_voxel_builder.grid import DiffusiveGrid, Grid
from bdm_voxel_builder.helpers import (
    get_nth_newest_file_in_folder,
    get_surrounding_offset_region,
    index_map_cylinder,
    index_map_sphere,
)

# CREATE AGENT TYPES

agent = Agent()
# TYPE A
# movement settings
# agent.pose = [0, 0, 0]
agent.walk_radius = 4
agent.move_mod_z = 0.2
agent.move_mod_random = 0.5
# build settings
agent.build_radius = 3
agent.build_h = 3
agent.reset_after_build = True
agent.inactive_step_count_limit = None
# sensor settings
agent.sense_radius = 3
agent.build_random_factor = 0.1
agent.pref_build_angle = 25
agent.pref_build_angle_gain = 0.1
agent.max_shell_thickness = 5
# create shape maps
agent.move_map = index_map_sphere(agent.walk_radius, agent.min_walk_radius)
agent.build_map = index_map_cylinder(agent.build_radius, agent.build_h, 0, -1)
agent.sense_map = index_map_sphere(agent.sense_radius)
agent.sense_inplane_map = index_map_cylinder(radius=3, height=2, min_radius=0, z_lift=1)
agent.sense_depth_map = index_map_cylinder(1, agent.max_shell_thickness * 2, 0, 1)
# __dict__
agent_dict_A = agent.__dict__

# TYPE B
agent.build_random_factor = 0.5
agent.build_radius = 2
agent.build_map = index_map_sphere(agent.build_radius)
agent.move_radius = 6
# __dict__
agent_dict_B = agent.__dict__

# dict list
agent_type_dicts = [agent_dict_A, agent_dict_B]
agent_type_distribution = [0.1, 0.6]


@dataclass
class Algo20_Build(AgentAlgorithm):
    """
    # Voxel Builder Algorithm: Algo 20

    the agents randomly build until a density is reached in a radius

    """

    agent_count: int
    clipping_box: cg.Box
    name: str = "algo_20"
    relevant_data_grids: str = "density"
    grid_to_dump: str = "density"

    # TODO
    vdb_to_dump: str = "density"  # not implemented
    point_cloud_to_dump: str = "centroids"  # not implemented

    seed_iterations: int = 1

    # global settings

    walk_region_thickness = 1

    # import scan
    import_scan = True
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
        centroids = DiffusiveGrid(
            name="centroids",
            grid_size=self.grid_size,
            color=Color.from_rgb255(252, 25, 0),
            flip_colors=True,
            decay_ratio=1 / 10000,
            decay_linear_value=1 / (iterations * 10),
        )
        self.print_dot_counter = 0
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
            "density": density,
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
        # grids["density"].decay()
        # diffuse_diffusive_grid(grids.follow_grid, )

    def setup_agents(self, grids: dict[str, DiffusiveGrid]):
        agent_space = grids["agent"]
        track = grids["track"]
        ground_grid = grids["ground"]
        agents: list[Agent] = []

        # generate agents based on agent_type_dicts
        for id in range(self.agent_count):
            # select type
            d = np.array(agent_type_distribution)
            sum_ = np.sum(d)
            d_normalized = d * 1 / sum_
            for i, n in enumerate(d_normalized):
                type_size = int(n * self.agent_count)
                for _j in range(type_size):
                    data_dict = agent_type_dicts[i]
                    # create object

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
        return agents

    def build(self, agent: Agent, state: Environment):
        """fill built volume in built_shape if agent.build_probability >= build_limit"""

        self.print_dot_counter += 1

        density = state.grids["density"]
        centroids = state.grids["centroids"]
        ground = state.grids["ground"]
        sense_maps_grid = state.grids["sense_maps_grid"]
        move_map_grid = state.grids["move_map_grid"]

        x, y, z = agent.pose

        # update print dot array
        centroids.array[x, y, z] = self.print_dot_counter

        # orient shape map
        build_map = agent.orient_build_map()
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
        nozzle_access_condition = True
        build_angle_condition = agent.normal_angle > agent.max_build_angle
        if not nozzle_access_condition and build_angle_condition:
            build_probability = 0
        else:
            # RANDOM FACTOR
            bp_random = r.random() * agent.build_random_factor

            # NORMAL ANGLE PREFERENCE
            if agent.normal_angle <= agent.pref_build_angle:
                bp_angle_factor = agent.pref_build_angle_gain
            else:
                bp_angle_factor = agent.pref_build_angle_gain * -2

            # TOPOLOGY SENSATION:
            topology_gain_inplane = 0.8
            topology_gain_edge = 0.8
            # topology sensor values
            shell_planarity_max_fill = 0.75
            shell_thickness_max_fill = 0.6
            edge_depth_min_fill = 0.2
            # wall thickness and shell edge
            density = state.grids["density"]
            ground = state.grids["ground"]
            sense_depth_map = agent.orient_sense_depth_map()
            depth_density = agent.get_array_density_by_oriented_index_map(
                density.array + ground.array, sense_depth_map, nonzero=True
            )
            sense_inplane_map = agent.orient_sense_inplane_map()
            inplane_density = agent.get_array_density_by_oriented_index_map(
                density.array + ground.array, sense_inplane_map, nonzero=True
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

            build_probability = bp_random + bp_angle_factor + bp_shell_topology

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
            print(
                f"""\n## agent_{agent.id} - {agent.pose} ##\n {agent.normal_vector}"""
            )
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
