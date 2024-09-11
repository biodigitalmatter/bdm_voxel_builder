import random as r
from dataclasses import dataclass

import compas.geometry as cg
from compas.colors import Color

from bdm_voxel_builder.agent import Agent
from bdm_voxel_builder.agent_algorithms.base import AgentAlgorithm
from bdm_voxel_builder.agent_algorithms.common import (
    make_ground_mockup,
)
from bdm_voxel_builder.environment import Environment
from bdm_voxel_builder.grid import DiffusiveGrid, Grid
from bdm_voxel_builder.helpers import (
    get_surrounding_offset_region,
    index_map_sphere,
)


@dataclass
class Algo14_Build_DensRange(AgentAlgorithm):
    """
    # Voxel Builder Algorithm: Algo_12 just go and build:

    ## Summary

    random walk
    probability is not gained, but flat
    build in shape

    multiple agents types act parallel. for example:

    agent_settings_B = {
        "build_prob_rand_range": [0, 0.2],
        "walk_radius": 6,
        "min_walk_radius": 3,
        "build_radius": 3.5,
        "inactive_step_count_limit": None,
        "reset_after_build": False,
        "move_mod_z": 0.05,
        "move_mod_random": 0.5,
    """

    agent_count: int
    clipping_box: cg.Box
    name: str = "algo_12_random_builder"
    relevant_data_grids: str = "built_volume"
    grid_to_dump: str = "built_volume"

    seed_iterations: int = 1

    # Agent deployment
    legal_move_region_thickness = 1

    print_dot_counter = 0
    legal_move_region = None

    walk_region_thickness = 1

    build_by_density = True
    build_by_density_mod = 1.2

    # agent settings

    # settings
    agent_settings_A = {
        "build_probability": 0.5,
        "walk_radius": 2,
        "min_walk_radius": 1,
        "build_radius": 1.2,
        "inactive_step_count_limit": None,
        "reset_after_build": False,
        "move_mod_z": 0.05,
        "move_mod_random": 0.5,
        "min_build_density": 0.4,
        "max_build_density": 0.8,
        "build_limit_mod_by_density": [0.25, -0.25, 0.25],
        "build_by_density": True,
    }
    agent_settings_B = {
        "build_probability": 0.25,
        "walk_radius": 6,
        "min_walk_radius": 3,
        "build_radius": 3.5,
        "inactive_step_count_limit": None,
        "reset_after_build": False,
        "move_mod_z": 0.05,
        "move_mod_random": 0.5,
        "min_build_density": 0.4,
        "max_build_density": 0.6,
        "build_limit_mod_by_density": [0.25, -0.2, 0.5],
        "build_by_density": True,
    }
    settings_split = 0.7  # A/B

    def __post_init__(self):
        """Initialize values held in parent class.

        Run in __post_init__ since @dataclass creates __init__ method"""
        super().__init__(
            agent_count=self.agent_count,
            clipping_box=self.clipping_box,
            grid_to_dump=self.grid_to_dump,
            name=self.name,
            grids_to_decay=["built_centroids", "built_volume"],
        )

    def initialization(self, **kwargs):
        """
        creates the simulation environment setup
        with preset values in the definition

        returns: grids
        """
        iterations = float(kwargs["iterations"])

        xform = kwargs.get("xform")
        assert isinstance(xform, cg.Transformation | None)

        # CREATE MOCK UP VOLUME
        mockup_ground = make_ground_mockup(self.clipping_box)
        ground = Grid.from_numpy(
            mockup_ground,
            name="ground",
            xform=xform,
            color=Color.from_rgb255(97, 92, 97),
        )
        agent_space = Grid(
            name="agent_space",
            clipping_box=self.clipping_box,
            xform=xform,
            color=Color.from_rgb255(34, 116, 240),
        )
        track = DiffusiveGrid(
            name="track",
            clipping_box=self.clipping_box,
            xform=xform,
            color=Color.from_rgb255(34, 116, 240),
            decay_ratio=1 / 10000,
        )
        built_centroids = DiffusiveGrid(
            name="built_centroids",
            clipping_box=self.clipping_box,
            xform=xform,
            color=Color.from_rgb255(252, 25, 0),
            flip_colors=True,
            decay_ratio=1 / 10000,
            decay_linear_value=1 / (iterations * 10),
        )
        built_volume = DiffusiveGrid(
            name="built_volume",
            clipping_box=self.clipping_box,
            xform=xform,
            color=Color.from_rgb255(219, 26, 206),
            flip_colors=True,
            decay_ratio=1 / 10000,
        )
        follow_grid = DiffusiveGrid(
            name="follow_grid",
            clipping_box=self.clipping_box,
            xform=xform,
            color=Color.from_rgb255(232, 226, 211),
            flip_colors=True,
            decay_ratio=1 / 10000,
        )

        # init legal_move_mask
        self.region_legal_move = get_surrounding_offset_region(
            [ground.to_numpy()], self.walk_region_thickness
        )

        # WRAP ENVIRONMENT
        grids = {
            "agent": agent_space,
            "ground": ground,
            "track": track,
            "built_centroids": built_centroids,
            "built_volume": built_volume,
            "follow_grid": follow_grid,
        }
        return grids

    def setup_agents(self, state: Environment):
        agent_space = state.grids["agent"]
        track = state.grids["track"]

        agents: list[Agent] = []

        for i in range(self.agent_count):
            # create object
            agent = Agent(
                space_grid=agent_space,
                track_grid=track,
                leave_trace=True,
            )

            # deploy agent
            agent.deploy_in_region(self.region_legal_move)

            # agent settings
            if i < self.agent_count * self.settings_split:
                d = self.agent_settings_A
            else:
                d = self.agent_settings_B

            agent.build_probability = d["build_probability"]
            agent.walk_radius = d["walk_radius"]
            agent.min_walk_radius = d["min_walk_radius"]
            agent.build_radius = d["build_radius"]
            agent.inactive_step_count_limit = d["inactive_step_count_limit"]
            agent.reset_after_build = d["reset_after_build"]
            agent.reset_after_erase = False
            agent.move_mod_z = d["move_mod_z"]
            agent.move_mod_random = d["move_mod_random"]

            agent.min_build_density = d["min_build_density"]
            agent.max_build_density = d["max_build_density"]
            agent.build_limit_mod_by_density = d["build_limit_mod_by_density"]
            agent.build_by_density = d["build_by_density"]

            # create shape maps
            agent.move_shape_map = index_map_sphere(
                agent.walk_radius, min_radius=agent.min_walk_radius
            )
            agent.built_shape_map = index_map_sphere(agent.build_radius)

            agents.append(agent)
        return agents

    def build(self, agent: Agent, state: Environment, build_limit=0.5):
        """fill built volume in built_shape if agent.build_probability >= build_limit"""

        built_volume: Grid = state.grids["built_volume"]
        built_centroids: Grid = state.grids["built_centroids"]

        # build

        # update print dot array
        built_centroids.set_value(agent.pose, 1)
        self.print_dot_counter += 1

        # update built_volume_volume_array
        built_volume.set_value_using_map_and_origin(agent.built_shape_map, agent.pose)

        print(f"built at: {agent.pose}")

    # ACTION FUNCTION
    def agent_action(self, agent: Agent, state: Environment):
        """
        BUILD
        MOVE
        *RESET
        """

        # BUILD
        if agent.build_by_density:
            mod = agent.modify_limit_in_density_range(
                array=state.grids["built_volume"].to_numpy(),
                radius=agent.build_radius,
                min_density=agent.min_build_density,
                max_density=agent.max_build_density,
                mod_below_range=agent.build_limit_mod_by_density[0],
                mod_in_range=agent.build_limit_mod_by_density[1],
                mod_above_range=agent.build_limit_mod_by_density[2],
                nonzero=True,
            )
            build_limit = r.random() + mod
        else:
            build_limit = r.random()

        if agent.build_probability >= build_limit:
            # build
            self.build(agent, state, build_limit)

            # update walk region
            self.region_legal_move = get_surrounding_offset_region(
                [
                    state.grids["ground"].to_numpy(),
                    state.grids["built_volume"].to_numpy(),
                ],
                self.walk_region_thickness,
            )
            # reset if
            if agent.reset_after_build:
                agent.deploy_in_region(self.region_legal_move)

        # MOVE
        # check collision
        collision = agent.check_solid_collision(
            [state.grids["built_volume"], state.grids["ground"]]
        )
        # move
        if not collision:
            move_values = agent.calculate_move_values_random__z_based()
            localized_move_map = agent.get_localized_move_map()

            legal_move_mask = localized_move_map == 1

            agent.move_by_index_map(
                index_map_in_place=localized_move_map[legal_move_mask],
                move_values=move_values[legal_move_mask],
                random_batch_size=1,
            )

        # RESET
        else:
            # reset if stuck
            agent.deploy_in_region(self.region_legal_move)

        # reset if inactive
        if agent.inactive_step_count_limit:  # noqa: SIM102
            if len(agent.move_history) >= agent.inactive_step_count_limit:
                agent.deploy_in_region(self.region_legal_move)
