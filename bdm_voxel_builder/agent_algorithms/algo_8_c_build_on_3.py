from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import numpy.typing as npt
from compas.colors import Color

from bdm_voxel_builder.agent import Agent
from bdm_voxel_builder.agent_algorithms.base import AgentAlgorithm
from bdm_voxel_builder.agent_algorithms.common import diffuse_diffusive_layer
from bdm_voxel_builder.data_layer.diffusive_layer import DiffusiveLayer
from bdm_voxel_builder.helpers.numpy import make_solid_box_xxyyzz
from bdm_voxel_builder.simulation_state import SimulationState

"""
Algorithm structure overview:

settings
initialization
    make_layers
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

initial stage algorithm - start to grow on attractive features of existing/scanned volumes

Find scan:
- an initially defined volume attracts the agents
- move there

Search for build
- 
Build on:
- recognize features to start building

"""


@dataclass
class Algo8c(AgentAlgorithm):
    """
    basic build_on existing algorithm

    extend with erase if too dense

    ...
    agent is attracted toward existing + newly built geomoetry by 'pheromon_layer_move'
    build_chance is rewarded if within the given ph limits
    if enough chances gained, agent builds / erases
    erase: explosive :) 
    6 or 26 voxels are cleaned

    if below sg > just build
    if more then half around > erase

    """

    name: str = "algo_8c_grownup"
    relevant_data_layers: Tuple[str] = "clay"
    seed_iterations: int = 40

    # EXISTING GEOMETRY
    add_box = True
    box_template = [20, 25, 22, 25, 1, 6]
    ground_level_Z = 0

    ################### Main control: ##################
    # MOVE SETTINGS
    move_ph_random_strength = 0.000000007
    move_ph_attractor_strength = 10000
    move_up = 1
    move_side = 0.9 
    move_down = 0.1 
    move_dir_pref_weigth = 0

    # Built Chance Reward if in pheromon limits 
    built_ph__min_to_build: float = 0.005
    built_ph__max_to_build: float = 5
    built_ph__build_reward = 0

    reach_to_build: int = 1
    reach_to_erase: int = 1

    # # slice below:
    # check_1__below = True
    # # built volumes density below the agent in a disc shape
    # slice_shape_1__ = [1,1,0,0,0,-1] # radius x,y,z , offset x,y,z
    # density_1__build_if_over = 0.1
    # density_1__build_if_below = 1
    # density_1__build_reward = 5
    # density_1__erase_if_over = 1
    # density_1__erase_if_below = 1
    # density_1__erase_reward = 0

    # # slice around:
    # check_2__around = True
    # # built volumes density below the agent in a disc shape
    # slice_shape_2__ = [2,2,0,0,0,0] # radius x,y,z , offset x,y,z
    # density_2__build_if_over = 0.01
    # density_2__build_if_below = 0.4
    # density_2__build_reward = 3

    # density_2__erase_if_over = 0.7
    # density_2__erase_if_below = 1
    # density_2__erase_reward = 3

    # # slice above:
    # check_3__above = True
    # # built volumes density below the agent in a disc shape
    # slice_shape_3__ = [1,1,0,0,0,1] # radius x,y,z , offset x,y,z
    # density_3__build_if_over = 0
    # density_3__build_if_below = 1
    # density_3__build_reward = 0

    # density_3__erase_if_over = 0.4
    # density_3__erase_if_below = 1
    # density_3__erase_reward = 5

    decay_clay_bool : bool = False
    ####################################################

    stacked_chances: bool = True
    reset_after_build: bool = True
    reset_after_erased: bool = True

    # Agent deployment
    deployment_zone__a = 5
    deployment_zone__b = -1

    check_collision = True
    keep_in_bounds = True


    layer_to_dump : str = 'clay_layer'



    def initialization(self, **kwargs):
        """
        creates the simulation environment setup
        with preset values in the definition

        returns: layers
        """
        number_of_iterations = kwargs.get('iterations')
        clay_decay_linear_value = max(1 / (self.agent_count * number_of_iterations * 100), 0.00001)
        ### LAYERS OF THE ENVIRONMENT
        rgb_agents = [34, 116, 240]
        rgb_trace = [17, 60, 120]
        rgb_ground = [100, 100, 100]
        rgb_queen = [232, 226, 211]
        rgb_existing = [207, 179, 171]
        ground = DiffusiveLayer(
            name="ground",
            voxel_size=self.voxel_size,
            color=Color.from_rgb255(*rgb_ground),
        )
        agent_space = DiffusiveLayer(
            name="agent_space",
            voxel_size=self.voxel_size,
            color=Color.from_rgb255(*rgb_agents),
        )
        track_layer = DiffusiveLayer(
            name="track_layer",
            voxel_size=self.voxel_size,
            color=Color.from_rgb255(*rgb_agents),
        )
        pheromon_layer_move = DiffusiveLayer(
            name="pheromon_layer_move",
            voxel_size=self.voxel_size,
            color=Color.from_rgb255(*rgb_queen),
            flip_colors=True,
            diffusion_ratio= 1/7,
            decay_ratio=1/10000000,
            gradient_resolution=0
        )
        clay_layer = DiffusiveLayer(
            name="clay_layer",
            voxel_size=self.voxel_size,
            color=Color.from_rgb255(*rgb_existing),
            flip_colors=True, 
            decay_linear_value=clay_decay_linear_value
        )

        ### CREATE GROUND ARRAY *could be imported from scan
        ground.add_values_in_zone_xxyyzz([0,self.voxel_size, 0, self.voxel_size, 0, self.ground_level_Z], 1)
        if self.add_box:
            ground.add_values_in_zone_xxyyzz(self.box_template, 1)
            pheromon_layer_move.add_values_in_zone_xxyyzz(self.box_template, 1)
            clay_layer.add_values_in_zone_xxyyzz(self.box_template, 1)


        

        # WRAP ENVIRONMENT
        layers = {
            "agent_space": agent_space,
            "ground": ground,
            "pheromon_layer_move": pheromon_layer_move,
            'clay_layer' : clay_layer,
            'track_layer' : track_layer
        }
        return layers


    def update_environment(self, state: SimulationState):
        layers = state.data_layers
        emission_array_for_move_ph = layers['clay_layer'].array
        diffuse_diffusive_layer(
            layers["pheromon_layer_move"],
            emmission_array=emission_array_for_move_ph,
            blocking_layer=layers["ground"],
            gravity_shift_bool=False,
            decay=True
        )
        if self.decay_clay_bool:
            layers['clay_layer'].decay_linear()
        # print('ph bounds:', np.amax(pheromon_layer_move.array),np.amin(pheromon_layer_move.array))
  


    def setup_agents(self, data_layers: Dict[str, DiffusiveLayer]):
        agent_space = data_layers["agent_space"]
        ground = data_layers["ground"]
        track_layer = data_layers["track_layer"]

        agents = []

        for i in range(self.agent_count):
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

        return agents

    def reset_agent(self, agent):
        # centered setup
        a, b = [self.deployment_zone__a, self.voxel_size + self.deployment_zone__b]
        a = max(a, 0)
        b = min(b, self.voxel_size - 1)
        x = np.random.randint(a, b)
        y = np.random.randint(a, b)
        z = self.ground_level_Z + 1

        agent.space_layer.set_layer_value_at_index(agent.pose, 0)
        agent.pose = [x, y, z]

        agent.build_chance = 0
        agent.erase_chance = 0
        agent.move_history = []

    def move_agent(self, agent, state: SimulationState):
        """moves agents in a calculated direction
        calculate weigthed sum of slices of layers makes the direction_cube
        check and excludes illegal moves by replace values to -1
        move agent
        return True if moved, False if not or in ground
        """
        # layers = state.data_layers
        pheromon_layer_move = state.data_layers["pheromon_layer_move"]
        ground = state.data_layers['ground']
        clay_layer = state.data_layers['clay_layer']

        gv = agent.get_layer_value_at_pose(ground, print_=False)
        cv = agent.get_layer_value_at_pose(clay_layer, print_=False)
        # print('ground value before move:', gv, agent.pose)
        if gv != 0 or cv != 0:
            return False
        
        # check clay density

        clay_density = agent.get_layer_density(clay_layer, trunc_decimals=False, print_=False)
        if clay_density > 0:
            txt = 'clay_density = {}'
            print(txt.format(clay_density))
        # print('move pheromon density:{}'.format(agent.get_layer_density(pheromon_layer_move, )))

        """idea is that when agent are 'on' the clay, they start to move more randomly, or more towards direction preference"""
        # move by pheromon_layer_move
        move_pheromon_cube = agent.get_direction_cube_values_for_layer(pheromon_layer_move, self.move_ph_attractor_strength)
        # get random directions cube
        random_cube = np.random.random(26)
        # # get move dir pref
        up_direction_cube = agent.direction_preference_26_pheromones_v2(1, 0.8, 0.2)

        ############################################################################
        # CHANGE MOVE BEHAVIOUR ####################################################
        ############################################################################
        if clay_density < 0.1: 
            """pheromon values are really small here"""
            move_pheromon_cube *= 2000
            random_cube *= 0.00000000000001
            direction_cube = move_pheromon_cube + random_cube

        elif 0.1 <= clay_density < 0.7:
            """clay isnt that attractive anymore, they prefer climbing"""
            move_pheromon_cube *= 0.001
            random_cube *= 0.5
            up_direction_cube *= 1
            direction_cube = move_pheromon_cube + random_cube + up_direction_cube

        elif 0.7 <= clay_density:
            """clay is super dense, they really climb up"""
            move_pheromon_cube *= 0.001
            random_cube *= 0.0001
            up_direction_cube *= 100
            direction_cube = move_pheromon_cube + random_cube + up_direction_cube
        ############################################################################

        # move by pheromons avoid collision
        collision_array = clay_layer.array + ground.array
        moved = agent.move_by_pheromons(
            solid_array=collision_array,
            pheromon_cube=direction_cube,
            voxel_size=self.voxel_size,
            fly=False,
            only_bounds=self.keep_in_bounds,
            check_self_collision=self.check_collision,
        )

        # doublecheck if in bounds
        if 0 > np.min(agent.pose) or np.max(agent.pose) >= self.voxel_size:
            # print(agent.pose)
            moved = False

        return moved

    def calculate_build_chances(self, agent, state: SimulationState):
        """simple build chance getter

        returns build_chance, erase_chance
        """
        pheromon_layer_move = state.data_layers["pheromon_layer_move"]
        clay_layer = state.data_layers['clay_layer']
        build_chance = agent.build_chance
        erase_chance = agent.erase_chance

        # print('calculate chances: Build_chance: {:_}, Erase_chance: {:_}'.format(agent.build_chance, agent.erase_chance))

        

        ##########################################################################
        # build probability settings #############################################
        ##########################################################################
        low_density__build_reward = 0
        low_density__erase_reward = 0

        normal_density__build_reward = 2
        normal_density__erase_reward = 0

        high_density__build_reward = 1
        high_density__erase_reward = 1

        slice_shape_above = [1,1,0, 0,0,1] # radius x,y,z , offset x,y,z
        high_density__roof__build_reward = -1
        high_density__roof__erase_reward = 20
        ##########################################################################

        # check clay density
        clay_density = agent.get_layer_density(clay_layer, trunc_decimals = False)
        # print(clay_density)
        if clay_density < 0.2:
            build_chance += low_density__build_reward
            erase_chance += low_density__erase_reward
        elif 0.2 <= clay_density < 0.8:
            build_chance += normal_density__build_reward
            erase_chance += normal_density__erase_reward
        elif 0.8 <= clay_density:
            build_chance += high_density__build_reward
            erase_chance += high_density__erase_reward
        
        # # check clay density above
        # clay_density_above = agent.get_layer_density_in_slice_shape(
        #         clay_layer,
        #         slice_shape_above
        #     )
        # if clay_density_above > 0.85:
        #     build_chance += high_density__roof__build_reward
        #     erase_chance += high_density__roof__erase_reward

        # # built volumes density below the agent
        # if self.check_1__below: 
        #     b, e = agent.get_chances_by_density_normal_by_slice(
        #         clay_layer,
        #         self.slice_shape_1__,   
        #         self.density_1__build_if_over,
        #         self.density_1__build_if_below,
        #         self.density_1__erase_if_below,
        #         self.density_1__erase_if_over,
        #         self.density_1__build_reward,
        #         self.density_1__erase_reward
        #     )
        #     # print(b,e)
        #     build_chance += b
        #     erase_chance += e
        
        # # built volumes density around the agent
        # if self.check_2__around:
        #     b, e = agent.get_chances_by_density_normal_by_slice(
        #         clay_layer,
        #         self.slice_shape_2__,
        #         self.density_2__build_if_over,
        #         self.density_2__build_if_below,
        #         self.density_2__erase_if_below,
        #         self.density_2__erase_if_over,
        #         self.density_2__build_reward,
        #         self.density_2__erase_reward,
        #     )
        #     build_chance += b
        #     erase_chance += e

        # # built volumes density above the agent
        # if self.check_3__above:
        #     b, e = agent.get_chances_by_density_normal_by_slice(
        #         clay_layer,
        #         self.slice_shape_3__,   
        #         self.density_3__build_if_over,
        #         self.density_3__build_if_below,
        #         self.density_3__erase_if_below,
        #         self.density_3__erase_if_over,
        #         self.density_3__build_reward,
        #         self.density_3__erase_reward,
        #     )
        #     build_chance += b
        #     erase_chance += e

        # update probabilities
        if self.stacked_chances:
            # print(erase_chance)
            agent.build_chance += build_chance
            agent.erase_chance += erase_chance
        else:
            agent.build_chance = build_chance
            agent.erase_chance = erase_chance
        
        # MANUAL TEST
        # agent.build_chance += 0.3
        # agent.erase_chance += 0

        # print('chances calculated: Build_chance: {:_}, Erase_chance: {:_}'.format(agent.build_chance, agent.erase_chance))


    def build_by_chance(
        self, agent, state: SimulationState
    ):
        """agent builds on construction_layer, if pheromon value in cell hits limit
        chances are either momentary values or stacked by history
        return bool"""
        built = False
        erased = False
        ground = state.data_layers["ground"]
        clay_layer = state.data_layers['clay_layer']
        build_condition = agent.check_build_conditions(ground)
        if build_condition:
            # build
            if agent.build_chance >= self.reach_to_build:
                # built = agent.build_on_layer(ground)
                built = agent.build_on_layer(clay_layer)
                # print('built', agent.pose, agent.build_chance)
            # erase
            elif agent.erase_chance >= self.reach_to_erase:
                # erased = agent.erase_26(ground)
                erased = agent.erase_26(clay_layer)
                # print('erased', agent.pose, agent.erase_chance)
            if erased or built:
                agent.erase_chance = 0
                agent.build_chance = 0
        return built, erased