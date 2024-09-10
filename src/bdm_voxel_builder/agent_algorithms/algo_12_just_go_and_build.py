from dataclasses import dataclass

import numpy as np
from compas.colors import Color

from bdm_voxel_builder.agent import Agent
from bdm_voxel_builder.agent_algorithms.base import AgentAlgorithm
from bdm_voxel_builder.agent_algorithms.common import make_ground_mockup
from bdm_voxel_builder.environment import Environment
from bdm_voxel_builder.grid import DiffusiveGrid, Grid
from bdm_voxel_builder.helpers import (
    get_indices_from_map_and_origin,
    get_values_by_index_map_and_origin,
    index_map_cylinder,
    index_map_sphere,
    offset_array_radial,
)


@dataclass
class Algo12_Random_builder(AgentAlgorithm):
    """
    # Voxel Builder Algorithm: Algo_12 just go and build:

    ## Summary

    random walk
    gain random chance
    build in shape

    multiple agents types act parallel
        build probability
        build radius
        walk radius
    """

    agent_count: int
    grid_size: int | tuple[int, int, int]
    name: str = "algo_12_random_builder"
    relevant_data_grids: str = "built_volume"
    grid_to_dump: str = "built_volume"

    seed_iterations: int = 1

    # Agent deployment
    legal_move_region_thickness = 1

    print_dot_counter = 0
    legal_move_region = None

    # agent settings

    # settings
    agent_settings_A = {
        "build_prob_rand_range": [1, 1],
        "walk_radius": 2,
        "min_walk_radius": 1,
        "build_radius": 1.2,
        "inactive_step_count_limit": 20,
        "reset_after_build": False,
        "move_mod_z": 0.05,
        "move_mod_random": 0.5,
    }
    agent_settings_B = {
        "build_prob_rand_range": [0, 0.2],
        "walk_radius": 6,
        "min_walk_radius": 3,
        "build_radius": 3.5,
        "inactive_step_count_limit": 100,
        "reset_after_build": False,
    }
    settings_split = 0.75  # A/B

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
        clipping_box = kwargs.get("clipping_box")
        xform = kwargs.get("xform")

        # CREATE MOCK UP VOLUME
        mockup_ground = make_ground_mockup(clipping_box)

        ground = Grid.from_numpy(
            mockup_ground,
            name="ground",
            xform=xform,
            color=Color.from_rgb255(97, 92, 97),
        )
        ground = Grid(
            name="ground",
            grid_size=self.grid_size,
            color=Color.from_rgb255(97, 92, 97),
        )
        agent_space = Grid(
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
        built_centroids = DiffusiveGrid(
            name="built_centroids",
            grid_size=self.grid_size,
            color=Color.from_rgb255(252, 25, 0),
            flip_colors=True,
            decay_ratio=1 / 10000,
            decay_linear_value=1 / (iterations * 10),
        )
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

        # init legal_move_mask
        walk_on_array = np.clip(ground.to_numpy() + built_volume.to_numpy(), 0, 1)
        walk_on_array_offset = offset_array_radial(walk_on_array, 2)
        self.region_legal_move = walk_on_array_offset - walk_on_array

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

    def update_environment(self, state: Environment):
        grids = state.grids
        grids["built_centroids"].decay()
        grids["built_volume"].decay()
        # diffuse_diffusive_grid(grids.follow_grid, )

    def setup_agents(self, grids: dict[str, DiffusiveGrid]):
        agent_space = grids["agent"]
        track = grids["track"]

        agents: list[Agent] = []

        for i in range(self.agent_count):
            # create object
            agent = Agent(
                space_grid=agent_space,
                track_grid=track,
                leave_trace=True,
                save_move_history=True,
            )

            # deploy agent
            agent.deploy_in_region(self.region_legal_move)

            # agent settings
            if i < self.agent_count * self.settings_split:
                d = self.agent_settings_A
            else:
                d = self.agent_settings_B

            agent.build_prob_rand_range = d["build_prob_rand_range"]
            agent.walk_radius = d["walk_radius"]
            agent.min_walk_radius = d["min_walk_radius"]
            agent.build_radius = d["build_radius"]
            agent.inactive_step_count_limit = d["inactive_step_count_limit"]
            agent.reset_after_build = d["reset_after_build"]
            agent.reset_after_erase = False
            agent.move_mod_z = d["move_mod_z"]
            agent.move_mod_random = d["move_mod_random"]

            # create shape maps
            agent.move_shape_map = index_map_sphere(
                agent.walk_radius, agent.min_walk_radius
            )
            agent.built_shape_map = index_map_sphere(agent.build_radius)

            agents.append(agent)
        return agents

    def move_agent_simple(self, agent: Agent, state: Environment):
        """moves agents in a calculated direction
        calculate weighted sum of slices of grids makes the direction_cube
        check and excludes illegal moves by replace values to -1
        move agent
        return True if moved, False if not or in ground
        """
        move_map_in_place = agent.get_localized_move_map()
        ground = state.grids["ground"]
        # follow_grid = state.grids["follow_grid"]
        # check solid volume inclusion
        gv = ground.get_value(agent.pose)
        if gv != 0:
            return False

        # legal move mask
        legal_move_mask = self.get_legal_move_mask(agent)

        # random map
        map_size = len(move_map_in_place)
        random_map_values = np.random.random(map_size) + 0.5

        # global direction preference
        dir_map = agent.get_localized_move_map()

        # follow_map = get_values_by_index_map(
        #     follow_grid, agent.move_shape_map, agent.pose
        # )
        follow_map = []

        # follow pheromones

        # print(dir_map)
        move_z_coordinate = np.array(dir_map, dtype=np.float64)[:, 2] - agent.pose[2]

        # MOVE PREFERENCE SETTINGS
        move_z_coordinate *= agent.move_mod_z
        random_map_values *= agent.move_mod_random
        follow_map *= agent.move_mod_follow
        move_values = move_z_coordinate + random_map_values  # + follow_map

        # filter legal moves
        move_values_masked = move_values[legal_move_mask]
        move_map_in_place_masked = move_map_in_place[legal_move_mask]

        ############################################################################

        moved = agent.move_by_index_map(
            index_map_in_place=move_map_in_place_masked,
            move_values=move_values_masked,
            random_batch_size=1,
        )

        agent.step_counter += 1

        return moved

    def move_agent(self, agent: Agent, state: Environment):
        """moves agents in a calculated direction
        calculate weighted sum of slices of grids makes the direction_cube
        check and excludes illegal moves by replace values to -1
        move agent
        return True if moved, False if not or in ground
        """
        move_map_in_place = get_indices_from_map_and_origin(
            agent.move_shape_map, agent.pose, agent.space_grid.grid_size
        )
        pheromone_grid_move = state.grids["pheromone_grid_move"]
        ground = state.grids["ground"]
        design = state.grids["design"]
        pheromone_build_flags = state.grids["pheromone_build_flags"]

        gv = ground.get_value(agent.pose)
        if gv != 0:
            return False

        # ph attraction towards design
        pheromone_grid_map = pheromone_grid_move.get_values_by_index_map(
            agent.move_shape_map, agent.pose
        )

        # ph attraction toward build track start
        build_track_flag_map = pheromone_build_flags.get_values_by_index_map(
            agent.move_shape_map, agent.pose
        )

        # legal move mask
        legal_move_region_map = get_values_by_index_map_and_origin(
            self.region_legal_move, agent.move_shape_map, agent.pose
        )
        legal_move_mask = legal_move_region_map == 1

        # random map
        map_size = len(pheromone_grid_map)
        random_map_values = np.random.random(map_size) + 0.5
        # global direction preference
        dir_map = get_indices_from_map_and_origin(
            agent.move_shape_map, agent.pose, agent.space_grid.grid_size
        )
        # print(dir_map)
        move_z_coordinate = np.array(dir_map, dtype=np.float64)[:, 2] - agent.pose[2]
        # print(f"move_z_coordinate {move_z_coordinate}")

        density_in_move_map = agent.get_array_density_by_index_map(
            design.array, agent.move_shape_map, nonzero=True
        )

        # MOVE PREFERENCE SETTINGS

        # outside design boundary - direction toward design
        if density_in_move_map < 0.2:
            random_mod = 1
            move_values = pheromone_grid_map * 1 + move_z_coordinate * -0

        # inside design space >> direction down
        else:
            random_mod = 3
            move_z_coordinate *= -1
            random_map_values *= 0.1
            pheromone_grid_map *= 0
            build_track_flag_map *= 10
            move_values = (
                move_z_coordinate
                + random_map_values
                + pheromone_grid_map
                + build_track_flag_map
            )

        # filter legal moves
        legal_move_mask = legal_move_region_map == 1
        move_values_masked = move_values[legal_move_mask]
        move_map_in_place = move_map_in_place[legal_move_mask]

        ############################################################################

        moved = agent.move_by_index_map(
            index_map_in_place=move_map_in_place,
            move_values=move_values_masked,
            random_batch_size=random_mod,
        )

        # double check if in bounds
        if any(np.array(agent.pose) < 0) or any(
            np.array(agent.pose) >= np.array(self.grid_size)
        ):
            moved = False
            print(f"not in bounds at{agent.pose}")

        agent.step_counter += 1

        return moved

    def check_print_chance_gain_flat_rate(
        self, agent: Agent, state: Environment, rate=0.6
    ):
        agent.build_chance += rate

    def check_print_chance_gain_flat_rate_check_above(
        self, agent: Agent, state: Environment, rate=0.6
    ):
        built_volume = state.grids["built_volume"]
        ground = state.grids["ground"]
        printed_and_ground = np.clip((built_volume.array + ground.array), 0, 1)
        check_above = index_map_cylinder(self.build_radius, 15)
        printed_density_above = agent.get_array_density_by_index_map(
            printed_and_ground,
            check_above,
            agent.pose + [0, 0, +1],
            nonzero=True,
        )
        print(f"printed_density_above {printed_density_above}")
        if printed_density_above > 0.1:
            agent.build_chance = 0
        else:
            agent.build_chance += rate

    def build_simple(self, agent: Agent, state: Environment):
        """add index the print_dot list, and fill either:
        - one_voxel
        - voxels in cross shape
        - or voxels in 3x3 square
        of the built_volume grid"""
        built = False
        built_volume: Grid = state.grids["built_volume"]
        built_centroids: Grid = state.grids["built_centroids"]

        # build
        if agent.build_chance >= agent.build_limit:
            # update print dot array
            built_centroids.set_value(agent.pose, 1)
            self.print_dot_counter += 1

            # update built_volume_volume_array
            built_volume.set_value_by_index_map(agent.built_shape_map, agent.pose, 1)
            built = True
        else:
            built = False
        if built:
            self.update_legal_move_region(state, self.legal_move_region_thickness)
            agent.build_chance = 0
        return built

    def update_legal_move_region(self, state: Environment, offset_steps=2):
        """returns offsetted shell next to the union of ground and built_volume grids"""
        ground: Grid = state.grids["ground"]
        built_volume: Grid = state.grids["built_volume"]
        walk_on_array = np.clip(ground.to_numpy() + built_volume.to_numpy(), 0, 1)
        walk_on_array_offset = offset_array_radial(walk_on_array, offset_steps)
        self.region_legal_move = walk_on_array_offset - walk_on_array

    # ACTION FUNCTION
    def agent_action(self, agent, state: Environment):
        """
        check print chance
        BUILD
        MOVE
        *RESET
        """

        agent.update_build_chance_random()

        # build
        built = self.build_simple(agent, state)

        if built:
            print(f"built {agent.pose}")
            if agent.reset_after_build:
                agent.deploy_in_region(self.region_legal_move)
            else:
                agent.step_counter = 0
        # MOVE
        moved = self.move_agent_simple(agent, state)

        # RESET IF STUCK
        if not moved:
            agent.deploy_in_region(self.region_legal_move)

        # if agent.step_counter >= agent.inactive_step_count_limit:
        #     agent.deploy_in_region(self.region_legal_move)
