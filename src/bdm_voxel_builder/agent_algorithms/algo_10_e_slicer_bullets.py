from dataclasses import dataclass

import numpy as np
from bdm_voxel_builder.agent import Agent
from bdm_voxel_builder.agent_algorithms.base import AgentAlgorithm
from bdm_voxel_builder.agent_algorithms.common import (
    diffuse_diffusive_grid,
    get_lowest_free_voxel_above_array,
)
from bdm_voxel_builder.environment import Environment
from bdm_voxel_builder.grid import DiffusiveGrid
from bdm_voxel_builder.helpers.array import (
    extrude_array_linear,
    get_mask_zone_xxyyzz,
    get_value_by_index_map,
    index_map_cylinder,
    index_map_move_and_clip,
    index_map_sphere,
    index_map_sphere_scale_NU,
    offset_array_radial,
    set_value_by_index_map,
)
from compas.colors import Color


@dataclass
class Algo10e_VoxelSlicer(AgentAlgorithm):
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
    relevant_data_grids: str = "printed_clay"

    seed_iterations: int = 50

    # PRINT SETTINGS
    overhang_limit = 6 / 9  # only 1 voxel
    print_goal_density = 0.2
    max_print_goal_density = 0.4
    print_goal_density_below = 7 / 9

    walk_in_region_thickness = 1

    # move_index_map = index_map_sphere_scale_NU(
    #     radius=3.8, min_radius=2, scale_NU=[1, 1, 0.5]
    # )
    radius = 4
    min_radius = 3
    move_index_map = index_map_sphere(radius, min_radius)
    bullet_radius = 2.5
    bullet_h = 1
    bullet_index_map = index_map_cylinder(bullet_radius, bullet_h)

    build_chance_flat_rate = 1

    print_one_voxel = False
    print_cross_shape = False
    print_3x3 = False
    print_5x5 = True

    track_length = 1
    track_flag = None

    reach_to_build: int = 1
    reach_to_erase: int = 1

    # decay_clay_bool: bool = True
    ####################################################

    stacked_chances: bool = True
    reset_after_build: bool = True
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
    passive_limit = 500
    deployment_zone_xxyy = [0, 50, 0, 50]

    ground_level_Z = 0
    walk_in_region = None

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

        # IMPORT DESIGN TODO

        # CREATE MOCK UP DESIGN

        box_1 = [5, 30, 5, 25, 1, 10]
        # box_2 = [15, 20, 15, 18, 1, 40]
        # box_3 = [0, 12, 0, 10, 4, 25]
        # box_4 = [0, 18, 0, 15, 15, 40]
        ground_box = [
            0,
            self.grid_size[0],
            0,
            self.grid_size[1],
            0,
            self.ground_level_Z,
        ]

        zones = [box_1]
        # zones = [box_1, box_2, box_3, box_4]
        zones_not = [ground_box]
        ground_zones = [ground_box]
        mockup_design = np.zeros(self.grid_size)  # noqa: F821
        for zone in zones:
            mask = get_mask_zone_xxyyzz(self.grid_size, zone, return_bool=True)
            mockup_design[mask] = 1
        for zone in zones_not:
            mask = get_mask_zone_xxyyzz(self.grid_size, zone, return_bool=True)
            mockup_design[mask] = 0

        mockup_ground = np.zeros(self.grid_size)  # noqa: F821
        for zone in ground_zones:
            mask = get_mask_zone_xxyyzz(self.grid_size, zone, return_bool=True)
            mockup_ground[mask] = 1

        # imported design TEMP
        design.array = mockup_design
        ground.array = mockup_ground
        # WRAP ENVIRONMENT

        walk_on_array = np.clip(ground.array + printed_clay.array, 0, 1)
        walk_on_array_offset = offset_array_radial(walk_on_array, 2)
        self.walk_in_region = walk_on_array_offset - walk_on_array

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
        grids["print_dots"].decay()
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
            self.reset_agent(agent, grids)
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

    def reset_agent(self, agent: Agent, grids):
        # pose = get_random_index_in_zone_xxyy_on_Z_level(
        #     self.deployment_zone_xxyy, agent.space_grid.grid_size, self.ground_level_Z
        # )
        printed_clay = grids["printed_clay"]
        ground = grids["ground"]
        design = grids["design"]
        array_on = printed_clay.array + ground.array
        pose = get_lowest_free_voxel_above_array(array_on, design.array)
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
        move_index_map = self.move_index_map
        move_index_map_clipped_oriented = index_map_move_and_clip(
            move_index_map, agent.pose, agent.space_grid.grid_size
        )
        pheromon_grid_move = state.grids["pheromon_move"]
        ground = state.grids["ground"]
        design = state.grids["design"]
        printed_clay = state.grids["printed_clay"]
        # pheromon_build_flags = state.grids["pheromon_build_flags"]
        # check solid volume inclusion
        walk_on_array = np.clip(ground.array + printed_clay.array, 0, 1)
        walk_on_array_offset = offset_array_radial(walk_on_array, 1)
        # walk_on_array_offset = extrude_array_linear(walk_on_array.copy(), [0, 0, 1], 1)
        walk_in_region = walk_on_array_offset - walk_on_array

        gv = agent.get_grid_value_at_pose(
            ground,
        )
        if gv != 0:
            return False

        # move by pheromon_grid_move

        move_pheromon_map = get_value_by_index_map(
            pheromon_grid_move.array,
            self.move_index_map,
            agent.pose,
        )
        walk_in_region_map = get_value_by_index_map(
            walk_in_region, self.move_index_map, agent.pose, dtype=np.float64
        )

        print(f"INDEX: {agent.pose}")

        l = len(move_pheromon_map)
        random_map_values = np.random.random(l) + 0.5
        # print(f"random_map_values_shape {random_map_values.shape}")
        dir_map = index_map_move_and_clip(
            self.move_index_map, agent.pose, agent.space_grid.grid_size
        )
        directional_bias_map = np.array(dir_map, dtype=np.float64)[:, 2] - agent.pose[2]
        # print(f"directional_bias_map.shape {directional_bias_map.shape}")

        ############################################################################
        # CHANGE MOVE BEHAVIOUR ####################################################
        ############################################################################
        clay_density_filled = agent.get_array_density_by_index_map(
            design.array, self.move_index_map, nonzero=True
        )
        if clay_density_filled < 0.25:
            random_mod = 2
            pheromon_map = random_map_values * 0.1
            mask = walk_in_region_map == 1
            pheromon_map = pheromon_map[mask]
            move_index_map_clipped_oriented = move_index_map_clipped_oriented[mask]
        else:
            directional_bias_map *= -0.5
            random_map_values *= 0.001
            move_pheromon_map *= 0.1
            pheromon_map = +directional_bias_map + random_map_values + move_pheromon_map

            mask = walk_in_region_map == 1
            pheromon_map = pheromon_map[mask]
            move_index_map_clipped_oriented = move_index_map_clipped_oriented[mask]
            random_mod = 1

        ############################################################################

        moved = agent.move_by_index_map(
            move_index_map_absolute_locations=move_index_map_clipped_oriented,
            pheromon_values_map=pheromon_map,
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

    def check_print_chance_flat_rate(self, agent: Agent, state: Environment, rate=0.6):
        printed_clay = state.grids["printed_clay"]
        ground = state.grids["ground"]
        print_and_ground = np.clip((printed_clay.array + ground.array), 0, 1)
        check_above = index_map_cylinder(self.bullet_radius, 15)
        printed_density_above = agent.get_array_density_by_index_map(
            print_and_ground,
            check_above,
            agent.pose + [0, 0, +1],
            nonzero=True,
        )
        print(f"printed_density_above {printed_density_above}")
        if printed_density_above > 0.1:
            agent.build_chance = 0
        else:
            agent.build_chance += rate

    def check_print_chance(self, agent: Agent, state: Environment):
        design = state.grids["design"]
        printed_clay = state.grids["printed_clay"]
        ground = state.grids["ground"]

        # reset build chance

        agent.build_chance = 0

        # merge ground with design and ground w printed
        print_and_ground = np.clip((printed_clay.array + ground.array), 0, 1)
        # design_and_ground = np.clip((design.array + ground.array), 0, 1)

        design_density_around = agent.get_array_density_by_index_map(
            design.array, self.bullet_index_map, agent.pose, nonzero=True
        )
        printed_density_below = agent.get_array_density_by_index_map(
            print_and_ground,
            self.bullet_index_map,
            agent.pose - [0, 0, -1],
            nonzero=True,
        )
        printed_density_above = agent.get_array_density_by_index_map(
            print_and_ground,
            self.bullet_index_map,
            agent.pose - [0, 0, +1],
            nonzero=True,
        )

        print(f"agent pose: {agent.pose}, array sum: {np.sum(printed_clay.array)}")
        print(
            f"printed d below: {printed_density_below}, design d around {design_density_around}"
        )
        in_design = agent.get_array_value_at_index(design.array, agent.pose)

        if in_design:
            if printed_density_above > 0.05:
                agent.build_chance = 0
            elif printed_density_below > 0.75:
                # no overhang
                agent.build_chance = 1
                # if design_density_around < max_print_goal_density:
                #     agent.build_chance = 1
            elif printed_density_below > 0.33 and design_density_around < 0.6:
                # overhang on the edge
                agent.build_chance = 1
        else:
            pass

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
            print(f"BUILD HERE: {agent.pose}")
            printed_clay.array = set_value_by_index_map(
                printed_clay.array, self.bullet_index_map, agent.pose
            )
            built = True
        else:
            built = False
        if built:
            self.update_walk_in_region(state, self.walk_in_region_thickness)
        return built

    def update_walk_in_region(self, state: Environment, offset_steps=2):
        ground = state.grids["ground"]
        printed_clay = state.grids["printed_clay"]
        walk_on_array = np.clip(ground.array + printed_clay.array, 0, 1)
        walk_on_array_offset = offset_array_radial(walk_on_array, offset_steps)
        self.walk_in_region = walk_on_array_offset - walk_on_array

    # CHECK END STATEs
    def check_end_state_agent(self, agent: Agent, state: Environment):
        """TODO"""

        stop_agent = False
        if stop_agent:
            self.reset_agent(agent, state.grids)
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
            # self.check_print_chance(agent, state)
            if not self.build_chance_flat_rate:
                self.check_print_chance(agent, state)
            else:
                self.check_print_chance_flat_rate(
                    agent, state, rate=self.build_chance_flat_rate
                )

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
                if self.reset_after_build:
                    # agent.deploy_airborne_min(
                    #     state.grids["printed_clay"],
                    #     state.grids["design"],
                    #     ground_level_Z=self.ground_level_Z,
                    # )
                    self.reset_agent(agent, state.grids)
            if not built:
                self.passive_counter += 1

        # MOVE
        moved = self.move_agent(agent, state)
        # print(agent.pose)
        # RESET IF STUCK
        if not moved:
            self.reset_agent(agent, state.grids)
            print("reset in move, couldnt move")

        elif self.step_counter == self.track_length:
            agent.move_history = []
            agent.track_flag = None

        # if self.passive_counter > self.passive_limit:
        #     # printed_clay = state.grids["printed_clay"]
        #     # design = state.grids["design"]
        #     # agent.deploy_airborne(
        #     #     printed_clay, design, ground_level_Z=self.ground_level_Z
        #     # )
        #     self.reset_agent(agent, state.grids)

        #     print(f"passive reset{agent.pose}")

        # check end states:
        self.check_end_state_agent(agent, state)
        self.check_end_state_simulation(agent, state)
