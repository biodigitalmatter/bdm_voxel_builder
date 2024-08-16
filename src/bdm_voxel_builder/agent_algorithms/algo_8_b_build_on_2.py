from dataclasses import dataclass

import numpy as np
from bdm_voxel_builder.agent import Agent
from bdm_voxel_builder.agent_algorithms.base import AgentAlgorithm
from bdm_voxel_builder.agent_algorithms.common import (
    diffuse_diffusive_grid,
    get_random_index_in_zone_xxyy_on_Z_level,
)
from bdm_voxel_builder.environment import Environment
from bdm_voxel_builder.grid import DiffusiveGrid
from compas.colors import Color

"""
Algorithm structure overview:

settings
initialization
    make_grid
    setup_agents
        reset_agents

iterate:
    move_agents
        reset_agents
    calculate_build_chances
    build/erase
        reset_agents
    update_environment
"""

"""
Algorithm Objectives:

initial stage algorithm - start to grow on attractive features of 
existing/scanned volumes

Find scan:
- an initially defined volume attracts the agents
- move there

Search for build
- 
Build on:
- recognize features to start building

"""


@dataclass
class Algo8b(AgentAlgorithm):
    """
    basic build_on existing algorithm

    extend with erase if too dense

    ...
    agent is attracted toward existing + newly built geomoetry by 'move_to_ph_grid'
    build_chance is rewarded if within the given ph limits
    if enough chances gained, agent builds / erases
    erase: explosive :)
    6 or 26 voxels are cleaned

    if below sg > just build
    if more then half around > erase

    """

    agent_count: int
    grid_size: int | tuple[int, int, int]
    name: str = "build_on_and_erase6nbs"
    relevant_data_grids: str = "ground"
    seed_iterations: int = 10

    # EXISTING GEOMETRY
    box_template = [20, 25, 22, 25, 1, 6]
    ground_level_Z = 0

    ################### Main control: ##################
    # MOVE SETTINGS
    move_ph_random_strength = 0.000000007
    move_ph_attractor_strength = 10000
    move_up = 1
    move_side = 0.9
    move_down = 0.1
    move_dir_pref_weigth = 0.5

    # Built Chance Reward if in pheromon limits
    built_ph__min_to_build: float = 0.005
    built_ph__max_to_build: float = 5
    built_ph__build_chance_reward = 0

    reach_to_build: int = 10
    reach_to_erase: int = 10

    # slice below:
    check_d1 = True
    # built volumes density below the agent in a disc shape
    slice_shape_1__ = [1, 1, 0, 0, 0, -1]  # radius x,y,z , offset x,y,z
    density_1__build_if_over = 0.1
    density_1__build_if_below = 1
    density_1__build_chance_reward = 5
    density_1__erase_if_over = 1
    density_1__erase_if_below = 1
    density_1__erase_chance_reward = 0

    # slice around:
    check_d2 = True
    # built volumes density below the agent in a disc shape
    slice_shape_2__ = [2, 2, 0, 0, 0, 0]  # radius x,y,z , offset x,y,z
    density_2__build_if_over = 0.01
    density_2__build_if_below = 0.4
    density_2__build_chance_reward = 3

    density_2__erase_if_over = 0.7
    density_2__erase_if_below = 1
    density_2__erase_chance_reward = 3

    # slice above:
    check_d3 = True
    # built volumes density below the agent in a disc shape
    slice_shape_3__ = [1, 1, 0, 0, 0, 1]  # radius x,y,z , offset x,y,z
    density_3__build_if_over = 0
    density_3__build_if_below = 1
    density_3__build_chance_reward = 0

    density_3__erase_if_over = 0.4
    density_3__erase_if_below = 1
    density_3__erase_chance_reward = 5

    decay_clay_bool: bool = False
    ####################################################

    stacked_chances: bool = True
    reset_after_build: bool = True
    reset_after_erased: bool = True

    # Agent deployment
    deployment_zone_xxyy = [0, 60, 0, 60]

    check_self_collision = True
    keep_in_bounds = True

    grid_to_dump: str = "clay_grid"

    def __post_init__(self):
        """Initialize values held in parent class."""
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
        ### GRIDS OF THE ENVIRONMENT
        rgb_agents = [34, 116, 240]
        rgb_ground = [100, 100, 100]
        rgb_queen = [232, 226, 211]
        rgb_existing = [207, 179, 171]
        ground = DiffusiveGrid(
            name="ground",
            grid_size=self.grid_size,
            color=Color.from_rgb255(*rgb_ground),
        )
        agent_space = DiffusiveGrid(
            name="agent_space",
            grid_size=self.grid_size,
            color=Color.from_rgb255(*rgb_agents),
        )
        move_to_ph_grid = DiffusiveGrid(
            name="move_to_ph_grid",
            grid_size=self.grid_size,
            color=Color.from_rgb255(*rgb_queen),
            flip_colors=True,
            diffusion_ratio=1,
            decay_ratio=1 / 10000000,
            gradient_resolution=100000,
        )
        clay_grid = DiffusiveGrid(
            name="clay_grid",
            grid_size=self.grid_size,
            color=Color.from_rgb255(*rgb_existing),
            flip_colors=True,
        )
        # decay_linear_value=1/(self.agent_count * 3000 * 1000)

        ### CREATE GROUND ARRAY *could be imported from scan
        ground.set_values_in_zone_xxyyzz(
            [0, self.grid_size, 0, self.grid_size, 0, self.ground_level_Z], 1
        )
        ground.set_values_in_zone_xxyyzz(self.box_template, 1)
        move_to_ph_grid.set_values_in_zone_xxyyzz(self.box_template, 1)
        clay_grid.set_values_in_zone_xxyyzz(self.box_template, 1)
        # move_to_ph_grid.array += clay_grid.array

        # WRAP ENVIRONMENT
        grids = {
            "agent_space": agent_space,
            "ground": ground,
            "move_to_ph_grid": move_to_ph_grid,
            "clay_grid": clay_grid,
        }
        return grids

    def update_environment(self, state: Environment):
        grids = state.grids
        emission_array_for_move_ph = grids["clay_grid"].array
        diffuse_diffusive_grid(
            grids["move_to_ph_grid"],
            emmission_array=emission_array_for_move_ph,
            blocking_grid=grids["ground"],
            gravity_shift_bool=False,
            decay=True,
        )
        if self.decay_clay_bool:
            grids["clay_grid"].decay_linear()
        # print(
        #     "ph bounds:",
        #     np.amax(move_to_ph_grid.array),
        #     np.amin(move_to_ph_grid.array),
        # )

    def setup_agents(self, grids: dict[str, DiffusiveGrid]):
        agent_space = grids["agent_space"]
        ground = grids["ground"]

        agents = []

        for _ in range(self.agent_count):
            # create object
            agent = Agent(
                space_grid=agent_space,
                ground_grid=ground,
                track_grid=None,
                leave_trace=False,
                save_move_history=True,
            )

            # deploy agent
            self.reset_agent(agent)
            agents.append(agent)

        return agents

    def reset_agent(self, agent: Agent):
        pose = get_random_index_in_zone_xxyy_on_Z_level(
            self.deployment_zone_xxyy, agent.space_grid.grid_size, self.ground_level_Z
        )
        agent.space_grid.set_value_at_index(agent.pose, 0)
        agent.pose = pose
        agent.build_chance = 0
        agent.erase_chance = 0
        agent.move_history = []
        # print('agent reset')

    def move_agent(self, agent: Agent, state: Environment):
        """moves agents in a calculated direction
        calculate weigthed sum of slices of grids makes the direction_cube
        check and excludes illegal moves by replace values to -1
        move agent
        return True if moved, False if not or in ground
        """
        move_to_ph_grid = state.grids["move_to_ph_grid"]
        ground = state.grids["ground"]

        gv = agent.get_grid_value_at_pose(ground)
        # print('ground value before move:', gv, agent.pose)
        if gv != 0:
            return False

        # move by move_to_ph_grid
        ph_cube = agent.get_direction_cube_values_for_grid(
            move_to_ph_grid, self.move_ph_attractor_strength
        )

        # get random directions cube
        random_cube = np.random.random(26) * self.move_ph_random_strength
        ph_cube += random_cube

        # # get move dir pref
        # dir_cube = agent.direction_preference_26_pheromones(
        #     self.move_up, self.move_side, self.move_down
        # )
        # ph_cube += dir_cube * self.move_dir_pref_weigth
        moved = agent.move_on_ground_by_ph_cube(
            ground=ground,
            pheromon_cube=ph_cube,
            grid_size=self.grid_size,
            fly=False,
            only_bounds=self.keep_in_bounds,
            check_self_collision=self.check_self_collision,
        )

        # check if in bounds
        if np.min(agent.pose) < 0 or np.max(agent.pose) >= self.grid_size:
            # print(agent.pose)
            moved = False

        # print('agent pose:', agent.pose)
        # print('agent moved flag', moved)
        return moved

    def calculate_build_chances(self, agent, state: Environment):
        """simple build chance getter

        returns build_chance, erase_chance
        """
        clay_grid = state.grids["clay_grid"]
        build_chance = agent.build_chance
        erase_chance = agent.erase_chance

        # # pheromone density in place
        # v = agent.get_chance_by_pheromone_strength(
        #     move_to_ph_grid,
        #     limit1 = self.built_ph__min_to_build,
        #     limit2 = self.built_ph__max_to_build,
        #     strength = self.built_ph__build_chance_reward,
        #     flat_value = True,
        # )
        # build_chance += v
        # erase_chance += 0

        # built volumes density below the agent
        if self.check_d1:
            b, e = agent.get_chances_by_density_normal_by_slice(
                clay_grid,
                self.slice_shape_1__,
                self.density_1__build_if_over,
                self.density_1__build_if_below,
                self.density_1__erase_if_below,
                self.density_1__erase_if_over,
                self.density_1__build_chance_reward,
                self.density_1__erase_chance_reward,
            )
            # print(b,e)
            build_chance += b
            erase_chance += e

        # built volumes density around the agent
        if self.check_d2:
            b, e = agent.get_chances_by_density_normal_by_slice(
                clay_grid,
                self.slice_shape_2__,
                self.density_2__build_if_over,
                self.density_2__build_if_below,
                self.density_2__erase_if_below,
                self.density_2__erase_if_over,
                self.density_2__build_chance_reward,
                self.density_2__erase_chance_reward,
            )
            build_chance += b
            erase_chance += e

        # built volumes density above the agent
        if self.check_d3:
            b, e = agent.get_chances_by_density_normal_by_slice(
                clay_grid,
                self.slice_shape_3__,
                self.density_3__build_if_over,
                self.density_3__build_if_below,
                self.density_3__erase_if_below,
                self.density_3__erase_if_over,
                self.density_3__build_chance_reward,
                self.density_3__erase_chance_reward,
            )
            build_chance += b
            erase_chance += e

        # update probabilities
        if self.stacked_chances:
            # print(erase_chance)
            agent.build_chance += build_chance
            agent.erase_chance += erase_chance
        else:
            agent.build_chance = build_chance
            agent.erase_chance = erase_chance

    def build_by_chance(self, agent: Agent, state: Environment):
        """agent builds on construction_grid, if pheromon value in cell hits limit
        chances are either momentary values or stacked by history
        return bool"""
        built = False
        erased = False
        ground = state.grids["ground"]
        clay_grid = state.grids["clay_grid"]
        build_condition = agent.check_build_conditions(ground)
        if build_condition:
            # build
            if agent.build_chance >= self.reach_to_build:
                built = agent.build_on_grid(ground)
                agent.build_on_grid(clay_grid)
                # print('built', agent.pose, agent.build_chance)
            # erase
            elif agent.erase_chance >= self.reach_to_erase:
                erased = agent.erase_6(ground)
                erased = agent.erase_6(clay_grid)
                # print('erased', agent.pose, agent.erase_chance)
            if erased or built:
                agent.erase_chance = 0
                agent.build_chance = 0
        return built, erased

    # ACTION FUNCTION - move first
    def agent_action(self, agent, state: Environment):
        """first build, then move
        to allow continous movement"""
        # MOVE
        moved = self.move_agent(agent, state)
        if not moved:
            self.reset_agent(agent)

        # get move probabilty
        self.calculate_build_chances(agent, state)

        # BUILD
        built, erased = self.build_by_chance(agent, state)
        if built or erased and self.reset_after_build:
            self.reset_agent(agent)
