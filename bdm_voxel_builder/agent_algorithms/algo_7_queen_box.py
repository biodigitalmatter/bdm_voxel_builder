from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import numpy.typing as npt

from bdm_voxel_builder.agent import Agent
from bdm_voxel_builder.agent_algorithms.base import AgentAlgorithm
from bdm_voxel_builder.agent_algorithms.common import pheromon_loop
from bdm_voxel_builder.data_layer import DataLayer
from bdm_voxel_builder.helpers.numpy import make_solid_box_xxyyzz
from bdm_voxel_builder.simulation_state import SimulationState


@dataclass
class Algo7QueenBox(AgentAlgorithm):
    """SETUP GOAL
    play with build by pheromon definition
    and edit with density build later

    """

    name: str = "queen_box"
    relevant_data_layers: Tuple[str] = "ground"
    seed_iterations: int = 25

    # BUILD SETTINGS
    reach_to_build: int = 1
    reach_to_erase: int = 1
    stacked_chances: bool = True
    reset_after_build: bool = True

    # pheromon sensitivity
    queen_pheromon_min_to_build: float = 0.005
    queen_pheromon_max_to_build: float = 0.05
    queen_pheromon_build_strength = 1
    queen_ph_build_flat_strength: bool = True

    # Agent deployment
    deployment_zone__a = 5
    deployment_zone__b = 35

    # MOVE SETTINGS
    # pheromon layers
    move_ph_random_strength = 0.0001
    move_ph_queen_bee_strength = 2
    moisture_ph_strength = 0

    # direction preference
    move_dir_prefer_to_side = 0
    move_dir_prefer_to_up = 0
    move_dir_prefer_to_down = 0
    move_dir_prefer_strength = 0

    # general
    check_collision = True
    keep_in_bounds = True

    # PHEROMON SETTINGS
    # queen bee:
    queen_box_1 = [10, 11, 10, 11, 1, 4]
    queens_place_array: npt.NDArray = None
    queen_bee_pheromon_gravity_ratio = 0

    # ENVIRONMENT GEO
    ground_level_Z = 1
    solid_box = None
    # solid_box = [25,26,0,30,ground_level_Z,12]
    # solid_box = [10,20,10,20,0,6]
    # solid_box = [0,1,0,1,0,1]

    def initialization(self):
        """
        creates the simulation environment setup
        with preset values in the definition

        returns: [settings, layers, clai_moisture_layer]
        layers = [agent_space, air_moisture_layer, build_boundary_pheromon, clay_moisture_layer,  ground, queen_bee_pheromon, sky_ph_layer]
        settings = [agent_count, voxel_size]
        """
        ### LAYERS OF THE ENVIRONMENT
        rgb_agents = [34, 116, 240]
        rgb_ground = [207, 179, 171]
        rgb_queen = [232, 226, 211]
        rgb_queen = [237, 190, 71]

        self.queens_place_array = make_solid_box_xxyyzz(
            self.voxel_size, *self.queen_box_1
        )
        self.queens_place_array *= 1 / (2 * 2 * 3)

        ground = DataLayer(
            voxel_size=self.voxel_size, name="ground", rgb=[i / 255 for i in rgb_ground]
        )
        agent_space = DataLayer(
            "agent_space", voxel_size=self.voxel_size, rgb=[i / 255 for i in rgb_agents]
        )
        queen_bee_pheromon = DataLayer(
            "queen_bee_pheromon",
            voxel_size=self.voxel_size,
            rgb=[i / 255 for i in rgb_queen],
            flip_colors=True,
        )

        queen_bee_pheromon.diffusion_ratio = 1 / 7
        queen_bee_pheromon.decay_ratio = 1 / 1000
        queen_bee_pheromon.gradient_resolution = 0
        queen_bee_pheromon.gravity_dir = 5
        queen_bee_pheromon.gravity_ratio = self.queen_bee_pheromon_gravity_ratio

        ### CREATE GROUND
        ground.array[:, :, : self.ground_level_Z] = 1
        # print(ground.array)
        if self.solid_box:
            wall = make_solid_box_xxyyzz(self.voxel_size, *self.solid_box)
            ground.array += wall

        # set ground moisture
        # clay_moisture_layer.array = ground.array.copy()

        # WRAP ENVIRONMENT
        layers = {
            "agent_space": agent_space,
            "ground": ground,
            "queen_bee_pheromon": queen_bee_pheromon,
        }
        return layers

    def update_environment(self, state: SimulationState):
        layers = state.data_layers

        ground = layers["ground"]
        queen_bee_pheromon = layers["queen_bee_pheromon"]

        pheromon_loop(
            queen_bee_pheromon,
            emmission_array=self.queens_place_array,
            blocking_layer=ground,
            gravity_shift_bool=False,
        )

    def setup_agents(self, data_layers: Dict[str, DataLayer]):
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

            # drop in the middle
            self.reset_agent(agent)

            agents.append(agent)

        return agents

    def reset_agent(self, agent):
        # centered setup
        a, b = [self.deployment_zone__a, self.deployment_zone__b]

        x = np.random.randint(a, b)
        y = np.random.randint(a, b)
        z = self.ground_level_Z

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
        layers = state.data_layers

        # # check layer value
        gv = agent.get_layer_value_at_pose(layers["ground"], print_=False)
        if gv != 0:
            return False

        # move by queen_ph
        layer = layers["queen_bee_pheromon"]
        domain = [self.queen_pheromon_min_to_build, self.queen_pheromon_max_to_build]
        strength = self.move_ph_queen_bee_strength
        ph_cube_1 = agent.get_direction_cube_values_for_layer_domain(
            layer, domain, strength
        )
        # ph_cube_1 = agent.get_direction_cube_values_for_layer(layer, strength)
        # get random directions cube
        random_cube = np.random.random(26) * self.move_ph_random_strength

        cube = ph_cube_1 + random_cube

        # global direction preference cube
        move_dir_preferences = [
            self.move_dir_prefer_to_up,
            self.move_dir_prefer_to_side,
            self.move_dir_prefer_to_down,
        ]
        if move_dir_preferences:
            up, side, down = move_dir_preferences
            cube += (
                agent.direction_preference_26_pheromones_v2(up, side, down)
                * self.move_dir_prefer_strength
            )

        moved = agent.move_on_ground_by_ph_cube(
            ground=layers["ground"],
            pheromon_cube=cube,
            voxel_size=self.voxel_size,
            fly=False,
            only_bounds=self.keep_in_bounds,
            check_self_collision=self.check_collision,
        )

        # check if in bounds
        if 0 > np.min(agent.pose) or np.max(agent.pose) >= self.voxel_size:
            # print(agent.pose)
            moved = False

        return moved

    def calculate_build_chances(self, agent, state: SimulationState):
        """simple build chance getter

        returns build_chance, erase_chance
        """
        queen_bee_pheromon = state.data_layers["queen_bee_pheromon"]

        build_chance = agent.build_chance
        erase_chance = agent.erase_chance

        v = agent.get_pheromone_strength(
            queen_bee_pheromon,
            self.queen_pheromon_min_to_build,
            self.queen_pheromon_max_to_build,
            self.queen_pheromon_build_strength,
            self.queen_ph_build_flat_strength,
        )
        build_chance += v
        erase_chance += 0

        return build_chance, erase_chance

    def build_over_limits(
        self, agent, state: SimulationState, build_chance, erase_chance
    ):
        """agent builds on construction_layer, if pheromon value in cell hits limit
        chances are either momentary values or stacked by history
        return bool"""
        ground = state.data_layers["ground"]

        if self.stacked_chances:
            # print(erase_chance)
            agent.build_chance += build_chance
            agent.erase_chance += erase_chance
        else:
            agent.build_chance = build_chance
            agent.erase_chance = erase_chance

        # check is there is any solid neighbors
        build_condition = agent.check_build_conditions(ground)

        built = False
        erased = False
        if build_condition:
            # build
            if agent.build_chance >= self.reach_to_build:
                built = agent.build()
            # erase
            elif agent.erase_chance >= self.reach_to_erase:
                erased = agent.erase()
        return built, erased

    def build(self, agent, state, build_chance, erase_chance):
        """build - select build style here"""
        return self.build_over_limits(agent, state, build_chance, erase_chance)
