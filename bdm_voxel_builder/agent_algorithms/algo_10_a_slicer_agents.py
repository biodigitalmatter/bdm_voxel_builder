from dataclasses import dataclass

import numpy as np
from compas.colors import Color

from bdm_voxel_builder.agent import Agent
from bdm_voxel_builder.agent_algorithms.base import AgentAlgorithm
from bdm_voxel_builder.agent_algorithms.common import diffuse_diffusive_grid
from bdm_voxel_builder.environment import Environment
from bdm_voxel_builder.grid import DiffusiveGrid


@dataclass
class Algo10a_VoxelSlicer(AgentAlgorithm):
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
    name: str = "algo_8_d"
    relevant_data_grids: str = "design"

    seed_iterations: int = 100

    # PRINT SETTINGS
    # overhang options: '30' '45' '60'
    overhang = 45

    # IMPORTED GEOMETRY ----- PLACEHOLDER
    add_box = True
    box_template_1 = [20, 40, 20, 40, 1, 4]
    ground_stair_1 = [0, 50, 20, 50, 0, 2]
    ground_stair_2 = [20, 50, 0, 30, 0, 3]
    ground_level_Z = 0

    reach_to_build: int = 1
    reach_to_erase: int = 1

    # decay_clay_bool: bool = True
    ####################################################

    stacked_chances: bool = True
    reset_after_build: bool = False
    reset_after_erased: bool = False

    # Agent deployment
    deployment_zone__a = 5
    deployment_zone__b = 20

    check_collision = True
    keep_in_bounds = True

    grid_to_dump: str = "print_dots"

    print_dot_list = []
    print_dot_dict = {}
    print_dot_counter = 0

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
        design = DiffusiveGrid(
            name="design",
            grid_size=self.grid_size,
            color=Color.from_rgb255(*rgb_existing),
            flip_colors=True,
            decay_ratio=1 / 100,
        )
        print_dots = DiffusiveGrid(
            name="print_dots",
            grid_size=self.grid_size,
            color=Color.from_rgb255(*rgb_existing),
            flip_colors=True,
            decay_ratio=1 / 100,
        )
        printed_clay = DiffusiveGrid(
            name="printed_clay",
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
            ground.add_values_in_zone_xxyyzz(self.ground_stair_1, 1)
            ground.add_values_in_zone_xxyyzz(self.ground_stair_2, 1)
            design.add_values_in_zone_xxyyzz(self.box_template_1, 1)

        # WRAP ENVIRONMENT
        grids = {
            "agent": agent_space,
            "ground": ground,
            "pheromon_move": pheromon_move,
            "design": design,
            "track": track_grid,
            "print_dots" : print_dots,
            "printed_clay" : printed_clay
        }
        return grids

    def update_environment(self, state: Environment):
        
        grids = state.grids
        emission_array_for_move_ph = grids["design"].array
        diffuse_diffusive_grid(
            grids["pheromon_move"],
            emmission_array=emission_array_for_move_ph,
            blocking_grids=[grids["ground"], grids['printed_clay']],
            gravity_shift_bool=False,
            grade=False,
            decay=True,
        )

        # grids["design"].decay()
        # grids["track"].decay()

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
        # TODO: make work with non square grids
        # centered setup
        grid_size = agent.space_grid.grid_size
        a, b = [
            self.deployment_zone__a,
            grid_size[0] + self.deployment_zone__b,
        ]

        a = max(a, 0)
        b = min(b, grid_size[0] - 1)
        x = np.random.randint(a, b)
        y = np.random.randint(a, b)
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
        design = state.grids["design"]
        printed_clay = state.grids['printed_clay']

        # check solid volume inclusion
        gv = agent.get_grid_value_at_pose(ground, )
        if gv != 0:
            return False

        clay_density_filled = agent.get_grid_density(design, nonzero=True)

        # move by pheromon_grid_move
        move_pheromon_cube = agent.get_direction_cube_values_for_grid(
            pheromon_grid_move, 1
        )
        directional_bias_cube = agent.direction_preference_26_pheromones_v2(1, 0.8, 0.2)
        random_cube = np.random.random(26)

        ############################################################################
        # CHANGE MOVE BEHAVIOUR ####################################################
        ############################################################################
        ############# randomize ##########

        if clay_density_filled < 0.1:
            """far from the clay, agents are aiming to get there"""
            direction_cube = move_pheromon_cube
            random_mod = 2

        elif clay_density_filled >= 0.1:
            """clay isnt that attractive anymore, they prefer climbing or random move"""
            move_pheromon_cube *= 0.01
            directional_bias_cube *= 1
            random_cube *= 2
            direction_cube = move_pheromon_cube + directional_bias_cube + random_cube
            random_mod = 1

        ############################################################################

        # move by pheromons, avoid collision
        collision_array = printed_clay.array + ground.array

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
        design = state.grids["design"]
        printed_clay = state.grids['printed_clay']
        ground = state.grids['ground']

        # reset build chance
        agent.build_chance = 0

        # check if in design volume
        design_grid_value = agent.get_grid_value_at_pose(design, )
        if design_grid_value == 0:
            return None
        
        elif design_grid_value >= 0:
            printed_clay_value = agent.get_grid_value_at_pose(printed_clay, )
            if printed_clay_value > 0:
                return None
            else:
                array_to_check = printed_clay.array + ground.array
                if self.overhang == 45:
                    check_index = np.asarray(agent.pose) - np.asarray([0,0,-1])
                    value = agent.get_array_value_at_index(array_to_check, check_index )
                    if value > 0:
                        agent.build_chance = 1
                        return None
                elif self.overhang == 60:
                    check_index_1 = np.asarray(agent.pose) - np.asarray([0,0,-1])
                    check_index_2 = np.asarray(agent.pose) - np.asarray([0,0,-2])

                    value_1 = agent.get_grid_value_at_index(array_to_check, check_index_1)
                    value_2 = agent.get_grid_value_at_index(array_to_check, check_index_2 )
                    
                    if value_1 > 0 and value_2 > 0:
                        agent.build_chance = 1
                        return None
                elif self.overhang == 30:
                    pass
        return None

    def build_the_print_grids(self, agent: Agent, state: Environment):
        """agent builds on construction_grid, if pheromon value in cell hits limit
        chances are either momentary values or stacked by history
        return bool"""
        built = False
        printed_clay = state.grids["printed_clay"]
        print_dots = state.grids["print_dots"]
        
        # build
        if agent.build_chance >= self.reach_to_build:
            # get pose
            x,y,z = agent.pose
            # update print dot array
            print_dots.array[x,y,z] = print_dots
            self.print_dot_list.append([x,y,z])
            self.print_dot_dict[self.print_dot_counter] = [x,y,z]
            self.print_dot_counter += 1
            # update printed_clay_volume_array
            # TODO def agent.set_nb_values_in_slice():
            printed_clay.array[x - 1 : x + 2][y - 1 : y + 2][z - 1 : z + 2] = 1
            built = True
        else:
            built = False
        return built

    # CHECK END STATEs
    def check_end_state_agent(self, agent: Agent, state: Environment):
        """TODO"""

        stop_agent = False
        if stop_agent:
            self.reset_agent(agent)
        else:
            pass
    
    def check_end_state_simulation(self, agent: Agent, state: Environment):
        """TODO"""
        its_over = False

        if its_over:
            state.end_state == True
            return True
        else:
            return False


    # ACTION FUNCTION
    def agent_action(self, agent, state: Environment):
        """MOVE BUILD .RESET"""

        # MOVE
        moved = self.move_agent(agent, state)

        # BUILD
        if moved:
            self.calculate_build_chances(agent, state)
            built = self.build_the_print_grids(agent, state)
            # print(f'built: {built}, erased: {erased}')
            if (built is True) and self.reset_after_build:
                self.reset_agent(agent)

        # RESET IF STUCK
        if not moved:
            self.reset_agent(agent)
            # print('reset in move, couldnt move')
        
        # check end states:
        self.check_end_state_agent(agent, state)
        self.check_end_state_simulation(agent, state)
