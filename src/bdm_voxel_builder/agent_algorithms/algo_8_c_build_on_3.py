from dataclasses import dataclass

import numpy as np
from bdm_voxel_builder.agent import Agent
from bdm_voxel_builder.agent_algorithms.base import AgentAlgorithm
from bdm_voxel_builder.agent_algorithms.common import (
    diffuse_diffusive_grid,
    get_random_index_in_zone_xxyy_on_ground,
)
from bdm_voxel_builder.environment import Environment
from bdm_voxel_builder.grid import DiffusiveGrid
from compas.colors import Color


@dataclass
class Algo8c(AgentAlgorithm):
    """
    # Voxel Builder Algorithm: Algo_8_c_build_on:

    ## Summary

    default voxel builder algorithm
    agents build on and around an initial 'clay' volume on a 'ground' surface
    inputs: solid_ground_volume, clay_volume
    output:
        A >> flat growth around 'clay' covering the ground
        B >> tower growth on 'clay'


    ## Agent behaviour

    1. find the built clay
    2. climb <up> on it
    3. build after a while of climbing
    4. reset or not

    ## Features

    - move on solid array
    - move direction is controlled with the mix of the pheromon environment
      and a global direction preference
    - move randomness controlled by setting the number of best directions for
      the random choice
    - build on existing volume
    - build and erase is controlled by gaining rewards
    - move and build both is regulated differently at different levels of
      environment grid density

    ## Observations:

    resetting the agents after build results in a flat volume, since the agent
    generally climbs upwards for the same amount of steps

    not resetting the agent after build results in towerlike output

    more agents > wider tower, less agent > thinner tower. because of creating
    obstacles while simultanously trimbing up
    """

    agent_count: int
    grid_size: int | tuple[int, int, int]
    name: str = "algo_8c_build_on"
    relevant_data_grid: str = "clay"

    seed_iterations: int = 100

    # EXISTING GEOMETRY
    add_box = True
    box_template = [20, 22, 20, 22, 1, 4]
    ground_level_Z = 0

    reach_to_build: int = 1
    reach_to_erase: int = 1

    decay_clay_bool: bool = False
    ####################################################

    stacked_chances: bool = True
    reset_after_build: bool = True
    reset_after_erased: bool = False

    # Agent deployment
    deployment_zone_xxyy = [5, 50, 5, 50]

    check_self_collision = True
    keep_in_bounds = True

    grid_to_dump: str = "clay"

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
        number_of_iterations = kwargs.get("iterations")
        clay_decay_linear_value = max(
            1 / (self.agent_count * number_of_iterations * 100), 0.00001
        )
        rgb_agents = (34, 116, 240)
        rgb_trace = (17, 60, 120)
        rgb_ground = (100, 100, 100)
        rgb_queen = (232, 226, 211)
        rgb_existing = (207, 179, 171)
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
        track_grid = DiffusiveGrid(
            name="track",
            grid_size=self.grid_size,
            color=Color.from_rgb255(*rgb_trace),
        )
        pheromon_move = DiffusiveGrid(
            name="pheromon_move",
            grid_size=self.grid_size,
            color=Color.from_rgb255(*rgb_queen),
            flip_colors=True,
            diffusion_ratio=1 / 7,
            decay_ratio=1 / 10000000000,
            gradient_resolution=0,
        )
        clay_grid = DiffusiveGrid(
            name="clay",
            grid_size=self.grid_size,
            color=Color.from_rgb255(*rgb_existing),
            flip_colors=True,
            decay_linear_value=clay_decay_linear_value,
        )

        ### CREATE GROUND ARRAY *could be imported from scan
        ground.set_values_in_zone_xxyyzz(
            [0, ground.grid_size[0], 0, ground.grid_size[1], 0, self.ground_level_Z], 1
        )

        if self.add_box:
            # ground.set_values_in_zone_xxyyzz(self.box_template, 1)
            clay_grid.set_values_in_zone_xxyyzz(self.box_template, 1)

        # WRAP ENVIRONMENT
        grids = {
            "agent": agent_space,
            "ground": ground,
            "pheromon_move": pheromon_move,
            "clay": clay_grid,
            "track": track_grid,
        }
        return grids

    def update_environment(self, state: Environment):
        grids = state.grids
        emission_array_for_move_ph = grids["clay"].array
        diffuse_diffusive_grid(
            grids["pheromon_move"],
            emmission_array=emission_array_for_move_ph,
            blocking_grid=grids["ground"],
            gravity_shift_bool=False,
            grade=False,
            decay=True,
        )
        if self.decay_clay_bool:
            grids["clay"].decay_linear()
        # # print to examine
        # ph_array = grid['pheromon_move'].array
        # print('ph bounds:', np.amax(ph_array),np.amin(ph_array))

    def setup_agents(self, grids: dict[str, DiffusiveGrid]):
        agent_space = grids["agent"]
        ground = grids["ground"]
        track_grid = grids["track"]

        agents = []

        for _ in range(self.agent_count):
            # create object
            agent = Agent(
                space_grid=agent_space,
                ground_grid=ground,
                track_grid=track_grid,
                leave_trace=True,
                save_move_history=True,
            )

            # deploy agent
            self.reset_agent(agent)
            agents.append(agent)
        # print('agent_setup')
        return agents

    def reset_agent(self, agent: Agent):
        pose = get_random_index_in_zone_xxyy_on_ground(
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
        pheromon_move = state.grids["pheromon_move"]
        ground = state.grids["ground"]
        clay_grid = state.grids["clay"]

        # check solid volume collision
        gv = agent.get_grid_value_at_pose(ground)
        if gv != 0:
            # print("""agent in the ground""")
            return False

        # print clay density for examination
        clay_density = agent.get_grid_density(clay_grid)

        # move by pheromon_move
        move_pheromon_cube = agent.get_direction_cube_values_for_grid(pheromon_move, 1)
        directional_bias_cube = agent.direction_preference_26_pheromones(1, 0.8, 0.2)

        ############################################################################
        # CHANGE MOVE BEHAVIOUR ####################################################
        ############################################################################
        ############# randomize ##########

        if clay_density < 0.1:
            """far from the clay, agents are aiming to get there"""
            direction_cube = move_pheromon_cube
            random_mod = 0.1

        elif 0.1 <= clay_density < 0.7:
            """clay isnt that attractive anymore, they prefer climbing or random move"""
            move_pheromon_cube *= 0.01
            directional_bias_cube *= 1
            direction_cube = move_pheromon_cube + directional_bias_cube
            random_mod = 0.2

        elif clay_density >= 0.7:
            """clay is super dense, they really climb up"""
            move_pheromon_cube *= 0.001
            directional_bias_cube *= 100
            direction_cube = move_pheromon_cube + directional_bias_cube
            random_mod = 0.2

        ############################################################################

        # move by pheromons avoid collision
        collision_array = clay_grid.array + ground.array
        random_mod = int(random_mod * 26)
        moved = agent.move_by_pheromons(
            solid_array=collision_array,
            pheromon_cube=direction_cube,
            grid_size=self.grid_size,
            fly=False,
            only_bounds=self.keep_in_bounds,
            check_self_collision=self.check_self_collision,
            random_batch_size=random_mod,
        )

        # doublecheck if in bounds
        if any(np.array(agent.pose) < 0) or any(
            np.array(agent.pose) >= np.array(self.grid_size)
        ):
            moved = False
            print(f"not in bounds at{agent.pose}")

        return moved

    def calculate_build_chances(self, agent: Agent, state: Environment):
        """simple build chance getter

        returns build_chance, erase_chance
        """
        clay_grid = state.grids["clay"]
        build_chance = 0
        erase_chance = 0

        ##########################################################################
        # build probability settings #############################################
        ##########################################################################
        low_density__build_reward = 0.1
        low_density__erase_reward = 0

        normal_density__build_reward = 0.3
        normal_density__erase_reward = 0

        high_density__build_reward = 0
        high_density__erase_reward = 0.8
        ##########################################################################

        # get clay density
        clay_density = agent.get_grid_density(clay_grid)

        # set chances
        if 0 <= clay_density < 1 / 26:
            # extrem low ph density
            pass
        elif 1 / 26 <= clay_density < 3 / 26:
            build_chance += low_density__build_reward
            erase_chance += low_density__erase_reward
        elif 3 / 26 <= clay_density < 4 / 5:
            build_chance += normal_density__build_reward
            erase_chance += normal_density__erase_reward
        elif clay_density >= 4 / 5:
            build_chance += high_density__build_reward
            erase_chance += high_density__erase_reward

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
        # ground = state.data_layers["ground"]
        clay_grid = state.grids["clay"]
        has_nb_voxel = agent.check_build_conditions(clay_grid, only_face_nbs=True)

        if has_nb_voxel:
            # build
            if agent.build_chance >= self.reach_to_build:
                # built = agent.build_on_layer(ground)
                built = agent.build_on_grid(clay_grid)
                # print('built', agent.pose, agent.build_chance)
            # erase
            elif agent.erase_chance >= self.reach_to_erase:
                # erased = agent.erase_26(ground)
                erased = agent.erase_26(clay_grid)
                # print('erased', agent.pose, agent.erase_chance)
            if erased or built:
                agent.erase_chance = 0
                agent.build_chance = 0
        return built, erased

    # ACTION FUNCTION
    def agent_action(self, agent, state: Environment):
        """MOVE BUILD .RESET"""

        # MOVE
        moved = self.move_agent(agent, state)

        # BUILD
        if moved:
            self.calculate_build_chances(agent, state)
            built, erased = self.build_by_chance(agent, state)
            # print(f'built: {built}, erased: {erased}')
            if (built or erased) and self.reset_after_build:
                self.reset_agent(agent)

        # RESET IF STUCK
        if not moved:
            self.reset_agent(agent)
            # print('reset in move, couldnt move')
