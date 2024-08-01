from dataclasses import dataclass

import numpy as np
from compas.colors import Color

from bdm_voxel_builder.agent import Agent
from bdm_voxel_builder.agent_algorithms.base import AgentAlgorithm
from bdm_voxel_builder.agent_algorithms.common import diffuse_diffusive_layer
from bdm_voxel_builder.data_layer.diffusive_layer import DiffusiveGrid
from bdm_voxel_builder.environment import Environment


@dataclass
class Algo9a(AgentAlgorithm):
    """
    # Voxel Builder Algorithm: Algo_8_d_build_fresh:

    ## Summary

    default voxel builder algorithm
    agents build on and around an initial 'clay' volume on a 'ground' surface
    agents move towards and build more at newly built clay
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
      environment layer density
    - agents aim more towards the freshly built volumes.
    the clay array values slowly decay

    ## NEW in 9_a

    erase if more than 6/9 density above

    ## Observations:

    """

    agent_count: int
    grid_size: int | tuple[int, int, int]
    name: str = "algo_9_a"
    relevant_data_layers: str = "clay"

    seed_iterations: int = 100

    # EXISTING GEOMETRY
    add_box = True
    box_template = [0, 50, 0, 50, 1, 6]
    ground_level_Z = 0

    reach_to_build: int = 1
    reach_to_erase: int = 1

    decay_clay_bool: bool = True
    ####################################################

    stacked_chances: bool = True
    reset_after_build: bool = False
    reset_after_erased: bool = False

    # Agent deployment
    deployment_zone__a = 5
    deployment_zone__b = -1

    check_collision = True
    keep_in_bounds = True

    layer_to_dump: str = "clay_layer"

    def __post_init__(self):
        """Initialize values held in parent class.

        Run in __post_init__ since @dataclass creates __init__ method"""
        super().__init__(
            agent_count=self.agent_count,
            grid_size=self.grid_size,
            layer_to_dump=self.layer_to_dump,
            name=self.name,
        )

    def initialization(self, **kwargs):
        """
        creates the simulation environment setup
        with preset values in the definition

        returns: layers
        """
        number_of_iterations = kwargs.get("iterations")
        # clay_decay_linear_value = max(
        #     1 / (self.agent_count * number_of_iterations * 100), 0.00001
        # )
        ### LAYERS OF THE ENVIRONMENT
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
        track_layer = DiffusiveGrid(
            name="track_layer",
            grid_size=self.grid_size,
            color=Color.from_rgb255(*rgb_agents),
            decay_ratio=1 / 10000,
        )
        pheromon_layer_move = DiffusiveGrid(
            name="pheromon_layer_move",
            grid_size=self.grid_size,
            color=Color.from_rgb255(*rgb_queen),
            flip_colors=True,
            diffusion_ratio=1 / 7,
            decay_ratio=1 / 10000000000,
            gradient_resolution=0,
        )
        clay_layer = DiffusiveGrid(
            name="clay_layer",
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
            # ground.add_values_in_zone_xxyyzz(self.box_template, 1)
            clay_layer.add_values_in_zone_xxyyzz(self.box_template, 1)

        # WRAP ENVIRONMENT
        layers = {
            "agent_space": agent_space,
            "ground": ground,
            "pheromon_layer_move": pheromon_layer_move,
            "clay_layer": clay_layer,
            "track_layer": track_layer,
        }
        return layers

    def update_environment(self, state: Environment):
        layers = state.data_layers
        emission_array_for_move_ph = layers["clay_layer"].array
        diffuse_diffusive_layer(
            layers["pheromon_layer_move"],
            emmission_array=emission_array_for_move_ph,
            blocking_layer=layers["ground"],
            gravity_shift_bool=False,
            grade=False,
            decay=True,
        )
        if self.decay_clay_bool:
            layers["clay_layer"].decay()
        layers['track_layer'].decay()
        # # print to examine
        # ph_array = layers['pheromon_layer_move'].array
        # print('ph bounds:', np.amax(ph_array),np.amin(ph_array))

    def setup_agents(self, data_layers: dict[str, DiffusiveGrid]):
        agent_space = data_layers["agent_space"]
        ground = data_layers["ground"]
        track_layer = data_layers["track_layer"]

        agents = []

        for _ in range(self.agent_count):
            # create object
            agent = Agent(
                space_layer=agent_space,
                ground_layer=ground,
                track_layer=track_layer,
                leave_trace=True,
                save_move_history=True,
            )

            # deploy agent
            self.reset_agent(agent)
            agents.append(agent)
        # print('agent_setup')
        return agents

    def reset_agent(self, agent: Agent):
        # TODO: make work with non square grids
        # centered setup
        grid_size = agent.space_layer.grid_size
        a, b = [
            self.deployment_zone__a,
            grid_size[0] + self.deployment_zone__b,
        ]

        a = max(a, 0)
        b = min(b, grid_size[0] - 1)
        x = np.random.randint(a, b)
        y = np.random.randint(a, b)
        z = self.ground_level_Z + 1

        agent.space_layer.set_layer_value_at_index(agent.pose, 0)
        agent.pose = [x, y, z]

        agent.build_chance = 0
        agent.erase_chance = 0
        agent.move_history = []
        # print('agent reset functioned')

    def move_agent(self, agent: Agent, state: Environment):
        """moves agents in a calculated direction
        calculate weigthed sum of slices of layers makes the direction_cube
        check and excludes illegal moves by replace values to -1
        move agent
        return True if moved, False if not or in ground
        """
        pheromon_layer_move = state.data_layers["pheromon_layer_move"]
        ground = state.data_layers["ground"]
        clay_layer = state.data_layers["clay_layer"]

        # check solid volume collision
        gv = agent.get_layer_value_at_pose(ground, print_=False)
        if gv != 0:
            # print("""agent in the ground""")
            return False

        # print clay density for examination
        # clay_density = agent.get_layer_density(clay_layer)

        clay_density_filled = agent.get_layer_density(clay_layer, nonzero=True)

        # move by pheromon_layer_move
        move_pheromon_cube = agent.get_direction_cube_values_for_layer(
            pheromon_layer_move, 1
        )
        directional_bias_cube = agent.direction_preference_26_pheromones_v2(1, 0.8, 0.2)

        ############################################################################
        # CHANGE MOVE BEHAVIOUR ####################################################
        ############################################################################
        ############# randomize ##########

        if clay_density_filled < 0.1:
            """far from the clay, agents are aiming to get there"""
            direction_cube = move_pheromon_cube
            random_mod = 2

        elif 0.1 <= clay_density_filled:
            """clay isnt that attractive anymore, they prefer climbing or random move"""
            move_pheromon_cube *= 0.1
            directional_bias_cube *= 10
            direction_cube = move_pheromon_cube + directional_bias_cube
            random_mod = 5

        ############################################################################

        # move by pheromons, avoid collision
        collision_array = clay_layer.array + ground.array

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

    def calculate_build_chances(self, agent, state: Environment):
        """simple build chance getter

        returns build_chance, erase_chance
        """
        clay_layer = state.data_layers["clay_layer"]
        build_chance = 0
        erase_chance = 0

        ##########################################################################
        # build probability settings #############################################
        ##########################################################################
        low_density__build_reward = 0.1
        low_density__erase_reward = 0

        normal_density__build_reward = 0.8
        normal_density__erase_reward = 0

        high_density__build_reward = 0.3
        high_density__erase_reward = 0

        ##########################################################################

        # get clay density
        clay_density = agent.get_layer_density(clay_layer)
        # dense_mod = clay_density + 0.2
        dense_mod = 1
        clay_density_filled = agent.get_layer_density(clay_layer, nonzero = True)
        # set chances
        if 0 <= clay_density < 1 / 26:
            # extrem low density
            pass
        elif 1 / 26 <= clay_density_filled < 3 / 26:
            build_chance += low_density__build_reward * dense_mod
            erase_chance += low_density__erase_reward
        elif 3 / 26 <= clay_density_filled < 4 / 5:
            build_chance += normal_density__build_reward * dense_mod
            erase_chance += normal_density__erase_reward
        elif clay_density_filled >= 4 / 5:
            build_chance += high_density__build_reward * dense_mod
            erase_chance += high_density__erase_reward

        # check density above
        dense_above__erase_reward = 2
        slice_shape = [1,1,0,0,0,1]
        density_filled_above = agent.get_layer_density_in_slice_shape(clay_layer, slice_shape, True)
        if 2/3 <= density_filled_above:
            erase_chance += dense_above__erase_reward
        print(f'density: {clay_density_filled*26}, density_above: {density_filled_above*9}')

        # update probabilities
        if self.stacked_chances:
            # print(erase_chance)
            agent.build_chance += build_chance
            agent.erase_chance += erase_chance
        else:
            agent.build_chance = build_chance
            agent.erase_chance = erase_chance

    def build_by_chance(self, agent: Agent, state: Environment):
        """agent builds on construction_layer, if pheromon value in cell hits limit
        chances are either momentary values or stacked by history
        return bool"""
        built = False
        erased = False
        # ground = state.data_layers["ground"]
        clay_layer = state.data_layers["clay_layer"]
        has_nb_voxel = agent.check_build_conditions(clay_layer, only_face_nbs=True)

        if has_nb_voxel:
            # build
            if agent.build_chance >= self.reach_to_build:
                # built = agent.build_on_layer(ground)
                built = agent.build_on_layer(clay_layer)
                # print('built', agent.pose, agent.build_chance)
                if built: agent.build_chance = 0
            # erase
            elif agent.erase_chance >= self.reach_to_erase:
                # erased = agent.erase_26(ground)
                erased = agent.erase_26(clay_layer)
                print('erased', agent.pose, agent.erase_chance)
                if erased: agent.erase_chance = 0
        return built, erased

    # ACTION FUNCTION - build first!
    def agent_action(self, agent, state: Environment):
        """BUILD /reset > MOVE /reset"""

        # BUILD
        self.calculate_build_chances(agent, state)
        built, erased = self.build_by_chance(agent, state)
        # print(f'built: {built}, erased: {erased}')
        if (built is True or erased is True) and self.reset_after_build:
            self.reset_agent(agent)
            # print("reset in built")

        # MOVE
        moved = self.move_agent(agent, state)
        
        # RESET IF STUCK
        if not moved:
            self.reset_agent(agent)
            # print('reset in move, couldnt move')

