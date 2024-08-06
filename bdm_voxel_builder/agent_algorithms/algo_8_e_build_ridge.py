from dataclasses import dataclass

import numpy as np
from compas.colors import Color

from bdm_voxel_builder.agent import Agent
from bdm_voxel_builder.agent_algorithms.base import AgentAlgorithm
from bdm_voxel_builder.agent_algorithms.common import diffuse_diffusive_grid
from bdm_voxel_builder.environment import Environment
from bdm_voxel_builder.grid import DiffusiveGrid


@dataclass
class Algo8eRidge(AgentAlgorithm):
    """
    # Voxel Builder Algorithm: Algo_8_d_build_fresh:

    ## Summary

    default voxel builder algorithm
    agents build on and around an initial 'clay' volume on a 'ground' surface
    inputs: solid_ground_volume, clay_volume
    output:


    ## Agent behaviour

    1. find the built clay
    2. climb <up> on it
    3. build after a while of climbing
    4. reset or not

    ## Features

    - move on solid array
    - move direction is controlled with the mix of the pheromon environment and
      a global direction preference
    - move randomness controlled by setting the number of best directions for
      the random choice
    - build on existing volume
    - build and erase is controlled by gaining rewards
    - move and build both is regulated differently at different levels of
      environment grid density

    ## NEW in 8_d
    agents aim more towards the freshly built volumes.
    the clay array values slowly decay

    ## Observations:

    """

    agent_count: int
    grid_size: int | tuple[int, int, int]
    name: str = "algo_8_e"
    relevant_data_grids: str = "clay"

    seed_iterations: int = 100

    # EXISTING GEOMETRY
    add_box = True
    # box_template = [15, 35, 15, 35, 1, 5]
    box_template = [20, 30, 25, 30, 1, 5]

    ground_level_Z = 0

    reach_to_build: int = 1
    reach_to_erase: int = 1
    agent_age_limit: float = 100

    decay_clay_bool: bool = True
    ####################################################
    stacked_chances: bool = True
    reset_after_build: bool = True
    reset_after_erased: bool = False

    build_overhang = False
    
    if not build_overhang:
        move_pattern = ['up', 'side']
        decay_clay_bool: bool = False
    else:
        move_pattern = ['up', 'up', 'up,']
        decay_clay_bool: bool = True
    
    # Agent deployment
    deployment_zone_xxyy = [10, 40, 8, 12]

    check_collision = True
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
        # clay_decay_linear_value = max(
        #     1 / (self.agent_count * number_of_iterations * 100), 0.00001
        # )
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
            color=Color.from_rgb255(*rgb_agents),
            decay_ratio=1 / 10000,
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
            decay_ratio=1 / 100,
        )

        ### CREATE GROUND ARRAY *could be imported from scan
        ground.add_values_in_zone_xxyyzz(
            [0, ground.grid_size[0], 0, ground.grid_size[1], 0, self.ground_level_Z], 1
        )

        if self.add_box:
            clay_grid.add_values_in_zone_xxyyzz(self.box_template, 1)

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
            blocking_grids=grids["ground"],
            gravity_shift_bool=False,
            grade=False,
            decay=True,
        )
        if self.decay_clay_bool:
            grids["clay"].decay()
        grids["track"].decay()

    def setup_agents(self, grids: dict[str, DiffusiveGrid]):
        agent_space = grids["agent"]
        ground = grids["ground"]
        track_grid = grids["track"]

        agents: list[Agent] = []

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
        return agents
    
    def reset_agent(self, agent: Agent):
        self._reset_agent_on_Z_plane(agent)


    def _reset_agent_on_Z_plane(self, agent: Agent):
        # TODO: make work with non square grids
        # centered setup
        grid_size = agent.space_grid.grid_size
        x1, x2, y1, y2 = self.deployment_zone_xxyy

        x1 = max(x1, 0)
        x2 = min(x2, grid_size[0] - 1)
        y1 = max(y1, 0)
        y2 = min(y2, grid_size[1] - 1)
        x = np.random.randint(x1, x2)
        y = np.random.randint(y1, y2)
        z = self.ground_level_Z + 1

        agent.space_grid.set_value_at_index(agent.pose, 0)
        agent.pose = [x, y, z]

        agent.build_chance = 0
        agent.erase_chance = 0
        agent.move_history = []
        # print('agent reset functioned')

    def move_agent(self, agent: Agent, state: Environment):
        """moves agents in a calculated direction
        calculate weigthed sum of slices of grids makes the direction_cube
        check and excludes illegal moves by replace values to -1
        move agent
        return True if moved, False if not or in ground
        """
        pheromon_grid_move = state.grids["pheromon_move"]
        ground = state.grids["ground"]
        clay_grid = state.grids["clay"]

        # check solid volume collision
        gv = agent.get_grid_value_at_pose(ground)
        if gv != 0:
            # print("""agent in the ground""")
            return False

        clay_density_filled = agent.get_grid_density(clay_grid, nonzero=True)

        # move by pheromon_grid_move
        move_pheromon_cube = agent.get_direction_cube_values_for_grid(
            pheromon_grid_move, 1
        )
        directional_bias_cube_up = agent.direction_preference_26_pheromones_v2(
            1, 0.5, 0.1
        )
        directional_bias_cube_side = agent.direction_preference_26_pheromones_v2(
            0.1, 1, 0.5
        )

        ############################################################################
        # CHANGE MOVE BEHAVIOUR ####################################################
        ############################################################################
        ############# randomize ##########

        if clay_density_filled < 0.1:
            """far from the clay, agents are aiming to get there"""
            move_pheromon_cube *= 10000
            directional_bias_cube_side *= 1
            direction_cube = move_pheromon_cube + directional_bias_cube_side
            random_mod = 3

        elif clay_density_filled >= 0.1:
            """clay isnt that attractive anymore, they prefer climbing or random move"""
            move_pheromon_cube *= 0.01
            directional_bias_cube_up *= 1
            direction_cube = move_pheromon_cube + directional_bias_cube_up
            random_mod = 2

        ############################################################################

        # move by pheromons, avoid collision
        collision_array = clay_grid.array + ground.array

        moved = agent.move_by_pheromons(
            solid_array=collision_array,
            pheromon_cube=direction_cube,
            grid_size=self.grid_size,
            fly=False,
            only_bounds=self.keep_in_bounds,
            check_self_collision=self.check_collision,
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

        ##########################################################################
        # build probability settings #############################################
        ##########################################################################
        step_on_ridge_reward = 1.25
        step_on_ridge_moves_pattern = self.move_pattern

        gain_reward_if_on_top_floor = 0.25
        lost_track_die_chance_gain = 0.2
        ##########################################################################

        clay_grid = state.grids["clay"]
        build_chance = 0
        erase_chance = 0

        # set chances based on movement pattern
        below = clay_grid.get_value_at_index(agent.pose + [0, 0, -1])
        if self.build_overhang == False and below == 0:
                pass
        else:
            clay_density_filled = agent.get_grid_density(clay_grid, nonzero=True)
            if 1/26 <= clay_density_filled:
                if self.build_overhang == False and below == 0:
                    pass
                else:
                    # print(f'below:{below}')
                    if agent.match_vertical_move_history(step_on_ridge_moves_pattern):
                        build_chance += step_on_ridge_reward
                    else:
                        if agent.match_vertical_move_history(['side', "side"]):
                             agent.die_chance += lost_track_die_chance_gain
                        elif agent.match_vertical_move_history(['side', "down"]):
                            agent.die_chance += lost_track_die_chance_gain


            # # set chance if it walks on topfloor
            # v = agent.get_nb_values_3x3_below_of_array(clay_grid.array)
            # d_below = np.count_nonzero(v) / len(v)
            # v = agent.get_nb_values_3x3_around_of_array(clay_grid.array)
            # d_in_level = np.count_nonzero(v) / len(v)
            # if d_below > 5 / 9 and d_in_level <= 1 / 8:
            #     build_chance += gain_reward_if_on_top_floor


        # update probabilities
        agent.build_chance += build_chance
        agent.erase_chance += erase_chance


    def build_by_chance(self, agent: Agent, state: Environment):
        """agent builds on construction_grid, if pheromon value in cell hits limit
        chances are either momentary values or stacked by history
        return bool"""
        built = False
        erased = False
        clay_grid = state.grids["clay"]
        has_nb_voxel = agent.check_build_conditions(clay_grid, only_face_nbs=True)

        if has_nb_voxel:
            # build
            if agent.build_chance >= self.reach_to_build:
                built = agent.build_on_grid(clay_grid)
                # print('built', agent.pose, agent.build_chance)
                if built:
                    agent.build_chance = 0
            # erase
            elif agent.erase_chance >= self.reach_to_erase:
                # erased = agent.erase_26(ground)
                erased = agent.erase_26(clay_grid)
                print("erased", agent.pose, agent.erase_chance)
                if erased:
                    agent.erase_chance = 0
        return built, erased

    # ACTION FUNCTION
    def agent_action(self, agent, state: Environment):
        """ BUILD .RESET MOVE .RESET"""



        # BUILD
        # if moved:
        self.calculate_build_chances(agent, state)
        built, erased = self.build_by_chance(agent, state)
        # print(f'built: {built}, erased: {erased}')
        if (built is True or erased is True) and self.reset_after_build:
            self.reset_agent(agent)
            # print("reset in built")
        
        if agent.die_chance >= self.agent_age_limit:
            self.reset_agent(agent)
        # MOVE
        moved = self.move_agent(agent, state)

        # RESET IF STUCK
        if not moved:
            self.reset_agent(agent)
            # print('reset in move, couldnt move')
