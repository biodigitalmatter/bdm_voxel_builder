from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import numpy.typing as npt
from compas.colors import Color

from bdm_voxel_builder.agent import Agent
from bdm_voxel_builder.agent_algorithms.base import AgentAlgorithm
from bdm_voxel_builder.agent_algorithms.common import pheromon_loop
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
class Algo8(AgentAlgorithm):
    """
    basic build_on existing algorithm

    agent is attracted toward existing + newly built geomoetry by 'built_ph_layer'
    build_chance is rewarded if within the given ph limits
    if enough chances gained, agent builds

    """

    name: str = "build_on_existing_no_decay"
    relevant_data_layers: Tuple[str] = "ground"
    seed_iterations: int = 10

    # EXISTING GEOMETRY
    box_template = [15, 16, 15, 16, 0, 5]
    ground_level_Z = 0



    ################### Main control: ##################
    # Built Chance Reward if in pheromon limits 
    built_ph__min_to_build: float = 0.005
    built_ph__max_to_build: float = 5
    built_ph__build_chance_reward = 0.4

    reach_to_build: int = 10
    reach_to_erase: int = 1
    ####################################################

    stacked_chances: bool = True
    reset_after_build: bool = True

    # Agent deployment
    deployment_zone__a = 5
    deployment_zone__b = -5

    # MOVE SETTINGS
    move_ph_random_strength = 0.000000007
    move_ph_attractor_strength = 10000

    check_collision = True
    keep_in_bounds = True


    layer_to_dump : str = 'existing_geo'

    def initialization(self):
        """
        creates the simulation environment setup
        with preset values in the definition

        returns: layers
        """
        ### LAYERS OF THE ENVIRONMENT
        rgb_agents = [34, 116, 240]
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
        built_ph_layer = DiffusiveLayer(
            name="built_ph_layer",
            voxel_size=self.voxel_size,
            color=Color.from_rgb255(*rgb_queen),
            flip_colors=True,
            diffusion_ratio= 1,
            decay_ratio=1/10000000,
            gradient_resolution=100000
        )
        existing_geo = DiffusiveLayer(
            name="existing_geo",
            voxel_size=self.voxel_size,
            color=Color.from_rgb255(*rgb_existing),
            flip_colors=True, 
        )
            # decay_linear_value=1/(self.agent_count * 3000 * 1000)


        ### CREATE GROUND ARRAY *could be imported from scan
        ground.add_values_in_zone_xxyyzz([0,self.voxel_size, 0, self.voxel_size, 0, self.ground_level_Z], 1)
        ground.add_values_in_zone_xxyyzz(self.box_template, 1)
        built_ph_layer.add_values_in_zone_xxyyzz(self.box_template, 1)
        existing_geo.add_values_in_zone_xxyyzz(self.box_template, 1)
        # built_ph_layer.array += existing_geo.array


        

        # WRAP ENVIRONMENT
        layers = {
            "agent_space": agent_space,
            "ground": ground,
            "built_ph_layer": built_ph_layer,
            'existing_geo' : existing_geo
        }
        return layers


    def update_environment(self, state: SimulationState):
        layers = state.data_layers

        ground = layers["ground"]
        built_ph_layer = layers["built_ph_layer"]
        existing_geo = layers['existing_geo']
        pheromon_loop(
            built_ph_layer,
            emmission_array=existing_geo.array * 1,
            blocking_layer=ground,
            gravity_shift_bool=False,
            decay=True
        )
        # existing_geo.decay_linear()
        # print('ph bounds:', np.amax(built_ph_layer.array),np.amin(built_ph_layer.array))
  


    def setup_agents(self, data_layers: Dict[str, DiffusiveLayer]):
        agent_space = data_layers["agent_space"]
        ground = data_layers["ground"]

        agents = []

        for i in range(self.agent_count):
            # create object
            agent = Agent(
                space_layer=agent_space,
                ground_layer=ground,
                track_layer=None,
                leave_trace=False,
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
        built_ph_layer = state.data_layers["built_ph_layer"]
        ground = state.data_layers['ground']

        gv = agent.get_layer_value_at_pose(ground, print_=False)
        # print('ground value before move:', gv, agent.pose)
        if gv != 0:
            return False

        # move by built_ph_layer
        ph_cube = agent.get_direction_cube_values_for_layer(built_ph_layer, self.move_ph_attractor_strength)

        # get random directions cube
        random_cube = np.random.random(26) * self.move_ph_random_strength
        ph_cube += random_cube

        moved = agent.move_on_ground_by_ph_cube(
            ground=ground,
            pheromon_cube=ph_cube,
            voxel_size=self.voxel_size,
            fly=False,
            only_bounds=self.keep_in_bounds,
            check_self_collision=self.check_collision,
        )

        # check if in bounds
        if 0 > np.min(agent.pose) or np.max(agent.pose) >= self.voxel_size:
            # print(agent.pose)
            moved = False

        # print('agent pose:', agent.pose)
        # print('agent moved flag', moved)
        return moved

    def calculate_build_chances(self, agent, state: SimulationState):
        """simple build chance getter

        returns build_chance, erase_chance
        """
        built_ph_layer = state.data_layers["built_ph_layer"]

        build_chance = agent.build_chance
        erase_chance = agent.erase_chance

        # pheromone density in position
        v = agent.get_chance_by_pheromone_strength(
            built_ph_layer,
            limit1 = self.built_ph__min_to_build,
            limit2 = self.built_ph__max_to_build,
            strength = self.built_ph__build_chance_reward,
            flat_value = True,
        )

        build_chance += v
        erase_chance += 0

        # update probabilities
        if self.stacked_chances:
            # print(erase_chance)
            agent.build_chance += build_chance
            agent.erase_chance += erase_chance
        else:
            agent.build_chance = build_chance
            agent.erase_chance = erase_chance

        # return build_chance, erase_chance

    def build_by_chance(
        self, agent, state: SimulationState
    ):
        """agent builds on construction_layer, if pheromon value in cell hits limit
        chances are either momentary values or stacked by history
        return bool"""
        built = False
        erased = False
        ground = state.data_layers["ground"]
        existing_geo = state.data_layers['existing_geo']
        build_condition = agent.check_build_conditions(ground)
        if build_condition:
            # build
            if agent.build_chance >= self.reach_to_build:
                built = agent.build()
                agent.build_on_layer(existing_geo)
            # erase
            elif agent.erase_chance >= self.reach_to_erase:
                erased = agent.erase(ground)
                erased = agent.erase(existing_geo)
        return built, erased