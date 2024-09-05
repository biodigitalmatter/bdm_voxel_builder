from dataclasses import dataclass

import numpy as np
from compas.colors import Color

from bdm_voxel_builder.agent import Agent
from bdm_voxel_builder.agent_algorithms.base import AgentAlgorithm
from bdm_voxel_builder.agent_algorithms.common import (
    diffuse_diffusive_grid,
    get_random_index_in_zone_xxyy_on_Z_level,
)
from bdm_voxel_builder.environment import Environment
from bdm_voxel_builder.grid import DiffusiveGrid


@dataclass
class Algo10b_VoxelSlicer(AgentAlgorithm):
    """
    # Voxel Builder Algorithm: Algo_10 SLICER BASIC:

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
    name: str = "algo_10_slicer"
    relevant_data_grids: str = "printed_dots"

    seed_iterations: int = 50

    # PRINT SETTINGS
    overhang_limit = 3 / 9
    print_goal_density = 0.5
    print_one_voxel = True
    print_cross_shape = False
    print_3x3 = False

    track_length = 20
    track_flag = None

    # IMPORTED GEOMETRY ----- PLACEHOLDER
    add_simple_design = False
    add_complex_design = True
    box_add_1 = [3, 8, 3, 10, 1, 5]
    box_add_2 = [15, 20, 15, 18, 1, 40]
    box_add_3 = [0, 12, 0, 10, 4, 25]
    box_add_4 = [0, 18, 0, 15, 15, 40]

    # box_template_2 = [20, 35, 6, 10, 4, 8]
    box_subtract_1 = [8, 15, 8, 16, 3, 20]
    # box_subtract_1 = [8, 20, 8, 18, 10, 10]

    ground_stair_1 = [0, 50, 20, 50, 0, 2]
    # ground_stair_2 = [20, 50, 0, 30, 0, 3]
    ground_level_Z = 0

    reach_to_build: int = 1
    reach_to_erase: int = 1

    # decay_clay_bool: bool = True
    ####################################################

    stacked_chances: bool = True
    reset_after_build: bool = False
    reset_after_erased: bool = False

    # Agent deployment

    check_self_collision = True
    keep_in_bounds = True

    grid_to_dump: str = "print_dots"

    print_dot_list = []
    print_dot_dict = {}
    print_dot_counter = 0
    step_counter = 0
    passive_counter = 0
    passive_limit = 15
    deployment_zone_xxyy = [0, 50, 0, 50]

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
        agent_space = DiffusiveGrid(
            name="agent_space",
            grid_size=self.grid_size,
            color=Color.from_rgb255(34, 116, 240),
        )
        track_grid = DiffusiveGrid(
            name="track",
            grid_size=self.grid_size,
            color=Color.from_rgb255(34, 116, 240),
            decay_ratio=1 / 10000,
        )
        pheromon_move = DiffusiveGrid(
            name="pheromon_move",
            grid_size=self.grid_size,
            color=Color.from_rgb255(232, 226, 211),
            flip_colors=True,
            diffusion_ratio=1 / 7,
            decay_ratio=1 / 10000000000,
            gradient_resolution=0,
        )
        pheromon_build_flags = DiffusiveGrid(
            name="pheromon_move",
            grid_size=self.grid_size,
            color=Color.from_rgb255(232, 226, 211),
            flip_colors=True,
            diffusion_ratio=1 / 7,
            decay_ratio=1 / 10000000000,
            gradient_resolution=0,
        )
        design = DiffusiveGrid(
            name="design",
            grid_size=self.grid_size,
            color=Color.from_rgb255(207, 179, 171),
            flip_colors=True,
            decay_ratio=1 / 100,
        )
        print_dots = DiffusiveGrid(
            name="print_dots",
            grid_size=self.grid_size,
            color=Color.from_rgb255(252, 25, 0),
            flip_colors=True,
            decay_ratio=1 / 10000,
            decay_linear_value=1 / (iterations * 10),
        )
        printed_clay = DiffusiveGrid(
            name="printed_clay",
            grid_size=self.grid_size,
            color=Color.from_rgb255(219, 26, 206),
            flip_colors=True,
            decay_ratio=1 / 10000,
        )

        ### CREATE GROUND ARRAY *could be imported from scan
        ground.set_values_in_zone_xxyyzz(
            [0, ground.grid_size[0], 0, ground.grid_size[1], 0, self.ground_level_Z], 1
        )

        # imported design TEMP
        if self.add_simple_design:
            design.set_values_in_zone_xxyyzz(self.box_add_1, 1)
        if self.add_complex_design:
            ground.set_values_in_zone_xxyyzz(self.ground_stair_1, 1)
            design.set_values_in_zone_xxyyzz(self.box_add_1, 1)
            design.set_values_in_zone_xxyyzz(self.box_add_2, 1)
            design.set_values_in_zone_xxyyzz(self.box_add_3, 1)
            design.set_values_in_zone_xxyyzz(self.box_add_4, 1)
            design.set_values_in_zone_xxyyzz(self.box_subtract_1, 0)
            design.set_values_in_zone_xxyyzz(self.ground_stair_1, 0)
        # WRAP ENVIRONMENT

        grids = {
            "agent": agent_space,
            "ground": ground,
            "pheromon_move": pheromon_move,
            "pheromon_build_flags": pheromon_build_flags,
            "design": design,
            "track": track_grid,
            "print_dots": print_dots,
            "printed_clay": printed_clay,
        }
        return grids

    def update_environment(self, state: Environment):
        grids = state.grids
        emission_array_for_move_ph = (
            grids["design"].array * 0.0001 + grids["printed_clay"].array * 0.33
        )
        diffuse_diffusive_grid(
            grids["pheromon_move"],
            emmission_array=emission_array_for_move_ph,
            blocking_grids=[grids["ground"], grids["printed_clay"]],
            gravity_shift_bool=False,
            grade=False,
            decay=True,
        )
        grids["print_dots"].decay_linear()
        grids["printed_clay"].decay()

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

    # def deploy_agent_airborne_min(self, agent: Agent, state: Environment):
    #     printed_clay = state.grids["printed_clay"]
    #     design = state.grids["design"]
    #     pose = get_lowest_free_voxel_above_array(printed_clay.array, design.array)

    #     if not isinstance(pose, np.ndarray | list):
    #         pose = get_random_index_in_zone_xxyy_on_Z_level(
    #             [0, 50, 0, 50], design.grid_size, self.ground_level_Z
    #         )

    #     agent.reset_at_pose(pose, reset_move_history=True)
    #     agent.passive_counter = 0
    #     return pose

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
        pheromon_grid_move = state.grids["pheromon_move"]
        ground = state.grids["ground"]
        design = state.grids["design"]
        printed_clay = state.grids["printed_clay"]
        pheromon_build_flags = state.grids["pheromon_build_flags"]
        # check solid volume inclusion
        gv = agent.get_grid_value_at_pose(
            ground,
        )
        if gv != 0:
            return False

        clay_density_filled = agent.get_grid_density(design, nonzero=True)
        # move by pheromon_grid_move
        move_pheromon_cube = agent.get_direction_cube_values_for_grid(
            pheromon_grid_move, 1
        )
        directional_bias_cube = agent.direction_preference_26_pheromones(
            0.000, 0.5, 0.48
        )
        random_cube = np.random.random(26) + 0.5
        random_cube[:9] = 0

        design_value_nbs = agent.get_grid_nb_values_26(design, agent.pose)

        ############################################################################
        # CHANGE MOVE BEHAVIOUR ####################################################
        ############################################################################

        if clay_density_filled < 0.25:
            """far from the clay, agents are aiming to get there"""
            direction_cube = move_pheromon_cube
            random_mod = 2

        elif clay_density_filled >= 0.25:
            """clay isnt that attractive anymore, they prefer climbing or random move"""
            move_pheromon_cube_build = agent.get_direction_cube_values_for_grid(
                pheromon_build_flags, 1
            )
            directional_bias_cube *= 1
            random_cube *= 0.1
            move_pheromon_cube_build *= 2
            move_pheromon_cube *= 0.1
            design_value_nbs *= 0.8
            direction_cube = (
                directional_bias_cube
                + random_cube
                + move_pheromon_cube_build
                + design_value_nbs
            )
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

    def move_agent_up(self, agent: Agent, state: Environment):
        """moves agents in a calculated direction
        calculate weigthed sum of slices of grids makes the direction_cube
        check and excludes illegal moves by replace values to -1
        move agent
        return True if moved, False if not or in ground
        """
        pheromon_grid_move = state.grids["pheromon_move"]
        ground = state.grids["ground"]
        design = state.grids["design"]
        printed_clay = state.grids["printed_clay"]
        pheromon_build_flags = state.grids["pheromon_build_flags"]
        # check solid volume inclusion
        gv = agent.get_grid_value_at_pose(
            ground,
        )
        if gv != 0:
            return False
        # move by pheromon_grid_move
        move_pheromon_cube = agent.get_direction_cube_values_for_grid(
            pheromon_grid_move, 1
        )
        directional_bias_cube = agent.direction_preference_26_pheromones(0.9, 0.5, 0.48)
        random_cube = np.random.random(26) + 0.5
        random_cube[:9] = 0

        design_value_nbs = agent.get_grid_nb_values_26(design, agent.pose)

        ############################################################################
        move_pheromon_cube_build = agent.get_direction_cube_values_for_grid(
            pheromon_build_flags, 1
        )
        directional_bias_cube *= 6
        random_cube *= 0.1
        move_pheromon_cube_build *= 0.2
        move_pheromon_cube *= 0.1
        design_value_nbs *= 0.8
        direction_cube = (
            directional_bias_cube
            + random_cube
            + move_pheromon_cube_build
            + design_value_nbs
        )
        random_mod = 3

        ############################################################################

        # move by pheromons, avoid collision
        collision_array = printed_clay.array + ground.array

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

    def update_env__track_flag_emmision(
        self, agent: Agent, state: Environment, repeat=2
    ):
        pheromon_build_flags = state.grids["pheromon_build_flags"]
        x, y, z = agent.track_flag
        for _i in range(repeat):
            # emission_intake
            pheromon_build_flags.array[x, y, z] = 2
            pheromon_build_flags.diffuse()
            pheromon_build_flags.decay()

    def check_print_chance_one_voxel(self, agent: Agent, state: Environment):
        design = state.grids["design"]
        printed_clay = state.grids["printed_clay"]
        ground = state.grids["ground"]

        # reset build chance
        agent.build_chance = 0

        # merge ground with design and ground w printed
        print_and_ground = np.clip((printed_clay.array + ground.array), 0, 1)

        v_in_design = agent.get_array_value_at_index(design.array, agent.pose)
        pose_below = agent.pose + [0, 0, -1]
        v_print_below = agent.get_array_value_at_index(print_and_ground, pose_below)

        if v_in_design > 0:
            # agent in design
            # check overhang
            if v_print_below > 0:
                # no overhang
                agent.build_chance = 1
            else:
                v = agent.get_nb_values_3x3_below_of_array(print_and_ground)
                print_density_below = np.count_nonzero(v) / 9
                if print_density_below >= self.overhang_limit:
                    # print(f"OVERHANG: {print_density_below}")
                    agent.build_chance = 1
                else:
                    agent.build_chance = 0

    def check_print_chance_cross(self, agent: Agent, state: Environment):
        design = state.grids["design"]
        printed_clay = state.grids["printed_clay"]
        ground = state.grids["ground"]

        # reset build chance
        agent.build_chance = 0

        # merge ground with design and ground w printed
        print_and_ground = np.clip((printed_clay.array + ground.array), 0, 1)
        design_and_ground = np.clip((design.array + ground.array), 0, 1)

        v = agent.get_nb_values_3x3_around_of_array(design.array)
        design_density = sum(v) / (len(v))
        v = agent.get_nb_values_3x3_below_of_array(design_and_ground)
        design_density_below = sum(v) / len(v)
        v = agent.get_nb_values_3x3_below_of_array(print_and_ground)

        print_density_below = sum(v) / len(v)
        if design_density >= self.print_goal_density:  # agent in design
            # check overhang
            if design_density_below >= self.print_goal_density:  # no overhang
                if print_density_below >= self.print_goal_density:  # no undercut
                    agent.build_chance = 1
            else:
                # overhang check
                if self.overhang_limit <= 30:
                    if print_density_below >= 0.33:
                        agent.build_chance = 1
                elif self.overhang_limit <= 45 and print_density_below >= 0.5:
                    agent.build_chance = 1
                # elif self.overhang_limit <= 60:
                #     if print_density_below >= 0.5 and print_density_below_2 >= 0.5:
                #         self.build_chance = 1
        # print(f"""pose: {agent.pose}, \ndesign density: {design_density},\n
        #     design density below: {design_density_below},\n
        #     print_density_below: {print_density_below},\n
        #     build chance: {agent.build_chance}.
        # """)

    def print_build(self, agent: Agent, state: Environment):
        """add index the print_dot list, and fill either:
        - one_voxel
        - voxels in cross shape
        - or voxels in 3x3 square
        of the printed_clay grid"""
        built = False
        printed_clay = state.grids["printed_clay"]
        print_dots = state.grids["print_dots"]

        # build
        if agent.build_chance >= self.reach_to_build:
            # get pose
            x, y, z = agent.pose

            # update print dot array
            print_dots.array[x, y, z] = 1
            self.print_dot_counter += 1

            # update printed_clay_volume_array
            if self.print_one_voxel:
                printed_clay.array[x][y][z] = 1
            elif self.print_cross_shape:
                agent.set_grid_value_cross_shape(printed_clay, 1)
            elif self.print_3x3:
                zone = [x - 1, x + 1, y - 1, y + 1, z, z]
                printed_clay.set_values_in_zone_xxyyzz(zone, 1)

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
        return state.end_state

    # ACTION FUNCTION
    def agent_action(self, agent, state: Environment):
        """MOVE BUILD .RESET"""
        moved = True
        # BUILD
        if moved:
            # check print chance
            if self.print_cross_shape:
                self.check_print_chance_cross(agent, state)
            elif self.print_one_voxel:
                self.check_print_chance_one_voxel(agent, state)
            built = self.print_build(agent, state)

            # create new flag if there isnt any yet
            if built and not isinstance(agent.track_flag, np.ndarray):
                agent.track_flag = agent.pose
                agent.move_history = []
                self.step_counter = 0
                print(f"new flag:{agent.pose}")

            # update env if there is a flag
            if isinstance(agent.track_flag, np.ndarray):
                self.step_counter += 1
                self.update_env__track_flag_emmision(agent, state)
            if built:
                print(f"built {agent.pose}")
            if not built:
                self.passive_counter += 1

        # MOVE
        moved = self.move_agent(agent, state)
        # print(agent.pose)
        # RESET IF STUCK
        if not moved:
            self.reset_agent(agent)
            print("reset in move, couldnt move")

        elif self.step_counter == self.track_length:
            agent.move_history = []
            agent.track_flag = None

        if self.passive_counter > self.passive_limit:
            printed_clay = state.grids["printed_clay"]
            design = state.grids["design"]
            agent.deploy_airborne_min(
                printed_clay, design, ground_level_Z=self.ground_level_Z
            )
            print(f"passive reset{agent.pose}")

        # check end states:
        self.check_end_state_agent(agent, state)
        self.check_end_state_simulation(agent, state)
