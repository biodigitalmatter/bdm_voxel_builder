from dataclasses import dataclass

import numpy as np
from bdm_voxel_builder.agent import Agent
from bdm_voxel_builder.agent_algorithms.base import AgentAlgorithm
from bdm_voxel_builder.agent_algorithms.common import (
    diffuse_diffusive_grid,
    get_any_voxel_in_region,
)
from bdm_voxel_builder.environment import Environment
from bdm_voxel_builder.grid import DiffusiveGrid
from bdm_voxel_builder.helpers.array import (
    get_mask_zone_xxyyzz,
    get_values_by_index_map,
    index_map_cylinder,
    index_map_move_and_clip,
    index_map_sphere,
    offset_array_radial,
    set_value_by_index_map,
)
from compas.colors import Color


@dataclass
class Algo10f_VoxelSlicer(AgentAlgorithm):
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

    seed_iterations: int = 10

    # PRINT SETTINGS
    minimum_design_density_around = 0.6
    minimum_printed_density_below = 0.2
    overhang_limit = 0.1
    maximum_printed_density_above = 0.01

    # MOVE SETTINGS
    walk_in_region_thickness = 2

    walk_radius = 4
    min_walk_radius = 2
    move_shape_map = index_map_sphere(walk_radius, min_walk_radius)
    bullet_radius = 2.5
    bullet_h = 1
    bullet_index_map = index_map_cylinder(bullet_radius, bullet_h)

    track_length = 20

    reach_to_build: int = 1
    reach_to_erase: int = 1

    reset_after_build: bool = True
    reset_after_erased: bool = False

    # Agent deployment

    grid_to_dump: str = "print_dots"

    print_dot_counter = 0
    passive_counter = 0
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
        print(self.grid_size)
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
        pheromon_grid_move = DiffusiveGrid(
            name="pheromon_grid_move",
            grid_size=self.grid_size,
            color=Color.from_rgb255(232, 226, 211),
            flip_colors=True,
            diffusion_ratio=1 / 7,
            decay_ratio=1 / 10000000000,
            gradient_resolution=0,
        )
        pheromon_build_flags = DiffusiveGrid(
            name="pheromon_build_flags",
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

        box_1 = [10, 60, 10, 40, 1, 15]
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
            "pheromon_grid_move": pheromon_grid_move,
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
            grids["design"].array * 0.33 + grids["printed_clay"].array * 0.33
        )
        diffuse_diffusive_grid(
            grids["pheromon_grid_move"],
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

    def reset_agent(self, agent: Agent, grids):
        # pose = get_random_index_in_zone_xxyy_on_Z_level(
        #     self.deployment_zone_xxyy, agent.space_grid.grid_size, self.ground_level_Z
        # )
        # printed_clay = grids["printed_clay"]
        # ground = grids["ground"]
        # design = grids["design"]
        # array_on = printed_clay.array + ground.array
        arr = self.walk_in_region
        pose = get_any_voxel_in_region(arr)
        # pose = get_any_free_voxel_above_array(array_on, design.array)
        agent.space_grid.set_value_at_index(agent.pose, 0)
        print(f"RESET POSE pose: {pose}")
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
        move_map_oriented = index_map_move_and_clip(
            self.move_shape_map, agent.pose, agent.space_grid.grid_size
        )
        pheromon_grid_move = state.grids["pheromon_grid_move"]
        ground = state.grids["ground"]
        design = state.grids["design"]
        printed_clay = state.grids["printed_clay"]
        pheromon_build_flags = state.grids["pheromon_build_flags"]
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

        # ph attraction towards design
        pheromon_grid_map = get_values_by_index_map(
            pheromon_grid_move.array,
            self.move_shape_map,
            agent.pose,
        )
        # ph attraction toward build track start
        build_track_flag_map = get_values_by_index_map(
            pheromon_build_flags.array,
            self.move_shape_map,
            agent.pose,
        )
        # legal move mask
        walk_in_region_map = get_values_by_index_map(
            walk_in_region, self.move_shape_map, agent.pose, dtype=np.float64
        )
        legal_move_mask = walk_in_region_map == 1
        # random map
        map_size = len(pheromon_grid_map)
        random_map_values = np.random.random(map_size) + 0.5
        # global direction preference
        dir_map = index_map_move_and_clip(
            self.move_shape_map, agent.pose, agent.space_grid.grid_size
        )
        # print(dir_map)
        move_z_coordinate = np.array(dir_map, dtype=np.float64)[:, 2] - agent.pose[2]
        # print(f"move_z_coordinate {move_z_coordinate}")

        density_in_move_map = agent.get_array_density_by_index_map(
            design.array, self.move_shape_map, nonzero=True
        )

        # MOVE PREFERENCE SETTINGS

        # outside design boundary - direction toward design
        if density_in_move_map < 0.2:
            random_mod = 1
            pheromon_map = pheromon_grid_map * 1 + move_z_coordinate * -0

        # inside design space >> direction down
        else:
            random_mod = 3
            move_z_coordinate *= -1
            random_map_values *= 0.1
            pheromon_grid_map *= 0
            build_track_flag_map *= 0
            pheromon_map = (
                move_z_coordinate
                + random_map_values
                + pheromon_grid_map
                + build_track_flag_map
            )

        # filter legal moves
        legal_move_mask = walk_in_region_map == 1
        pheromons_masked = pheromon_map[legal_move_mask]
        move_map_oriented = move_map_oriented[legal_move_mask]

        ############################################################################

        moved = agent.move_by_index_map(
            index_map_in_place=move_map_oriented,
            move_values=pheromons_masked,
            random_batch_size=random_mod,
        )

        # doublecheck if in bounds
        if any(np.array(agent.pose) < 0) or any(
            np.array(agent.pose) >= np.array(self.grid_size)
        ):
            moved = False
            print(f"not in bounds at{agent.pose}")

        agent.step_counter += 1

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

    def check_print_chance_flat_rate(self, agent: Agent, state: Environment, rate=0.6):
        printed_clay = state.grids["printed_clay"]
        ground = state.grids["ground"]
        printed_and_ground = np.clip((printed_clay.array + ground.array), 0, 1)
        check_above = index_map_cylinder(self.bullet_radius, 15)
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

    def check_print_chance_indesign(self, agent: Agent, state: Environment):
        design = state.grids["design"]
        printed_clay = state.grids["printed_clay"]
        ground = state.grids["ground"]

        # reset build chance

        agent.build_chance = 0

        # merge ground with design and ground w printed
        printed_and_ground = np.clip((printed_clay.array + ground.array), 0, 1)
        # design_and_ground = np.clip((design.array + ground.array), 0, 1)

        design_density_around = agent.get_array_density_by_index_map(
            design.array, self.bullet_index_map, agent.pose, nonzero=True
        )
        printed_density_below = agent.get_array_density_by_index_map(
            printed_and_ground,
            self.bullet_index_map,
            pose=agent.pose + [0, 0, -1],
            nonzero=True,
        )

        printed_density_around = agent.get_array_density_by_index_map(
            printed_and_ground,
            self.bullet_index_map,
            pose=agent.pose,
            nonzero=True,
        )

        printed_density_above = agent.get_array_density_by_index_map(
            printed_and_ground,
            self.bullet_index_map,
            pose=agent.pose + [0, 0, +1],
            nonzero=True,
        )

        # print(
        #     f"""agent pose: {agent.pose}
        #     design_density_around: {design_density_around}
        #     printed_density_below: {printed_density_below}
        #     printed_density_above: {printed_density_above}"""
        # )

        if design_density_around >= self.minimum_design_density_around:
            if printed_density_above > self.maximum_printed_density_above:
                # collision
                agent.build_chance = 0
            elif printed_density_around <= self.maximum_printed_density_around:
                if printed_density_below >= self.minimum_printed_density_below:
                    # no overhang
                    agent.build_chance = 1
                elif printed_density_below >= self.overhang_limit:
                    # overhang
                    agent.build_chance = 1
        else:
            # out of the design boundary
            agent.build_chance = 0
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
            printed_clay.array = set_value_by_index_map(
                printed_clay.array, self.bullet_index_map, agent.pose
            )
            built = True
        else:
            built = False
        if built:
            self.update_legal_move_region(state, self.walk_in_region_thickness)
        return built

    def update_legal_move_region(self, state: Environment, offset_steps=2):
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
        """
        check print chance
        BUILD
        MOVE
        *RESET
        """

        # check print chance
        self.check_print_chance_indesign(agent, state)

        # build
        built = self.print_build(agent, state)

        if built:
            print(f"built {agent.pose}")
            if self.reset_after_build:
                self.reset_agent(agent, state.grids)

        # MOVE
        moved = self.move_agent(agent, state)
        # print(agent.pose)
        # RESET IF STUCK
        if not moved:
            self.reset_agent(agent, state.grids)
            print("reset in move, couldnt move")

        elif agent.step_counter == self.track_length:
            agent.move_history = []
            # if self.reset_after_track:
            #     agent
            agent.track_flag = agent.pose
            agent.step_counter = 0

        # check end states:
        self.check_end_state_agent(agent, state)
        self.check_end_state_simulation(agent, state)
