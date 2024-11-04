from dataclasses import dataclass

import compas.geometry as cg

from bdm_voxel_builder import REPO_DIR
from bdm_voxel_builder.agent_algorithms.base import AgentAlgorithm
from bdm_voxel_builder.agents import Agent
from bdm_voxel_builder.environment import Environment
from bdm_voxel_builder.helpers import (
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
    relevant_data_grids: str = "built"
    grids_to_dump: list[str] = ("built",)

    seed_iterations: int = 1

    import_scan: bool = False

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
            grids_to_dump=self.grids_to_dump,
            name=self.name,
            grids_to_decay=[],
        )

    def initialization(self, state: Environment):
        """
        creates the simulation environment setup
        with preset values in the definition

        returns: grids

        """

        # init legal_move_mask
        self.region_legal_move = get_surrounding_offset_region(
            [state.grids["ground"].to_numpy()], self.walk_region_thickness
        )

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
