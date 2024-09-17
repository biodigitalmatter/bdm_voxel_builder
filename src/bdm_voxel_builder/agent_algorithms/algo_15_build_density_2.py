import random as r
from dataclasses import dataclass

import compas.geometry as cg
from compas.colors import Color

from bdm_voxel_builder import REPO_DIR
from bdm_voxel_builder.agent import Agent
from bdm_voxel_builder.agent_algorithms.base import AgentAlgorithm
from bdm_voxel_builder.environment import Environment
from bdm_voxel_builder.grid import DiffusiveGrid, Grid
from bdm_voxel_builder.helpers import (
    get_nth_newest_file_in_folder,
    get_surrounding_offset_region,
    index_map_cylinder,
    index_map_sphere,
)


@dataclass
class Algo15_Build(AgentAlgorithm):
    """
    # Voxel Builder Algorithm: Algo 15

    the agents randomly build until a density is reached in a radius

    """

    agent_count: int
    clipping_box: cg.Box
    name: str = "algo_15"
    relevant_data_grids: str = "built_volume"
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
    }
    agent_types = [agent_type_A, agent_type_A]
    agent_type_dividers = [0, 0.5]

    def __post_init__(self):
        """Initialize values held in parent class.

        Run in __post_init__ since @dataclass creates __init__ method"""
        super().__init__(
            agent_count=self.agent_count,
            clipping_box=self.clipping_box,
            grid_to_dump=self.grid_to_dump,
            name=self.name,
            grids_to_decay=[],
        )

    def initialization(self, **kwargs):
        """
        creates the simulation environment setup
        with preset values in the definition

        returns: grids

        """
        iterations = int(kwargs["iterations"])
        xform = kwargs.get("xform")

        assert isinstance(xform, cg.Transformation | None)

        file_path = get_nth_newest_file_in_folder(self.dir_save_solid_npy)
        scan = Grid.from_numpy(
            file_path,
            clipping_box=self.clipping_box,
            name="scan",
            color=Color.from_rgb255(210, 220, 230),
        )

        ground = scan.copy()
        ground.name = "ground"
        ground.color = Color.from_rgb255(97, 92, 97)

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
        density = DiffusiveGrid(
            name="density",
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
        ground_grid = state.grids["ground"]

        agents: list[Agent] = []

        for i in range(self.agent_count):
            # agent settings
            div = self.agent_type_dividers + [1]
            for j in range(len(self.agent_types)):
                u, v = div[j], div[j + 1]
                if u <= i / self.agent_count < v:
                    d = self.agent_types[j]

            # create object
            agent = Agent(
                space_grid=agent_space,
                track_grid=track,
                ground_grid=ground_grid,
            )

            # deploy agent
            agent.deploy_in_region(self.region_legal_move)

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
            agent.move_map = index_map_sphere(agent.walk_radius, agent.min_walk_radius)
            agent.build_map = index_map_cylinder(agent.build_radius, agent.build_h)
            agent.sense_map = index_map_sphere(agent.sense_radius, 0)

            agents.append(agent)
        return agents

    def build(self, agent: Agent, state: Environment, build_limit=0.5):
        """fill built volume in built_shape if agent.build_probability >= build_limit"""

        self.print_dot_counter += 1

        density = state.grids["density"]
        built_centroids = state.grids["built_centroids"]

        # update print dot array
        self.print_dot_counter += 1
        built_centroids.set_value(agent.pose, self.print_dot_counter)

        # orient shape map
        build_map = agent.orient_build_map()
        # update density_volume_array
        density.set_values(build_map, self.print_dot_counter)

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
            # check density
            arr = state.grids["built_volume"].array
            build_limit = agent.get_build_limit_by_density_range(
                arr, self.density_check_radius, nonzero=True
            )

        else:
            build_limit = r.random()

        if agent.build_probability >= build_limit:
            # build
            self.build(agent, state, build_limit)

            # update walk region
            self.region_legal_move = get_surrounding_offset_region(
                [
                    state.grids["ground"].to_numpy(),
                    state.grids["density"].to_numpy(),
                ],
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
            [state.grids["density"], state.grids["ground"]]
        )
        # move
        if not collision:
            move_values = agent.calculate_move_values_random__z_based()
            localized_move_map = agent.get_localized_move_map()

            legal_move_mask = agent.get_localized_move_mask(self.region_legal_move)

            agent.move_by_index_map(
                index_map_in_place=localized_move_map[legal_move_mask],
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
