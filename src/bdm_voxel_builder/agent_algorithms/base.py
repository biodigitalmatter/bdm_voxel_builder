import abc
import random

import compas.geometry as cg
import numpy as np
import numpy.typing as npt
from compas import json_dumps

from bdm_voxel_builder.agent import Agent
from bdm_voxel_builder.environment import Environment
from bdm_voxel_builder.grid import DiffusiveGrid
from bdm_voxel_builder.helpers import (
    get_surrounding_offset_region,
    get_values_using_index_map,
)


class AgentAlgorithm(abc.ABC):
    def __init__(
        self,
        agent_count: int,
        grid_to_dump: str,
        clipping_box: cg.Box,
        name: str | None = None,
        grids_to_decay: list[str] | None = None,
    ) -> None:
        self.agent_count = agent_count
        self.grid_to_dump = grid_to_dump
        self.clipping_box = clipping_box
        self.name = name
        self.grids_to_decay = grids_to_decay

    @abc.abstractmethod
    def setup_agents(self, state: Environment):
        raise NotImplementedError

    def initialization(self, state: Environment, **kwargs):
        """Setup that needs to be after environment is created"""
        ground = state.grids["ground"].to_numpy()
        scan = state.grids["scan"].to_numpy()
        self.update_offset_regions(ground, scan)

    @abc.abstractmethod
    def update_environment(self, state: Environment):
        raise NotImplementedError

    def decay_environment(self, state: Environment):
        for grid_name in self.grids_to_decay or []:
            grid = state.grids[grid_name]
            assert isinstance(grid, DiffusiveGrid)
            grid.decay()

    def diffuse_follow_grid(
        self, state: Environment, emission_array: npt.NDArray[np.float_]
    ):
        follow_grid: DiffusiveGrid = state.grids["follow"]
        blocking_grids = [state.grids["ground"]]

        # update goal_density
        follow_grid.diffuse_diffusive_grid(
            emission_array=emission_array,
            blocking_grids=blocking_grids,
        )

    def get_legal_move_mask(self, agent: Agent, state: Environment):
        """moves agents in a calculated direction
        calculate weigthed sum of slices of grids makes the direction_cube
        check and excludes illegal moves by replace values to -1
        move agent
        return True if moved, False if not or in ground
        """

        # legal move mask
        filter = get_values_using_index_map(
            self.region_legal_move, agent.move_map, agent.pose
        )
        legal_move_mask = filter == 1
        return legal_move_mask

    def update_offset_regions(self, ground_array, scan_array):
        self.region_legal_move = get_surrounding_offset_region(
            arrays=[ground_array],
            offset_thickness=self.walk_region_thickness,
        )
        if self.deploy_anywhere:
            self.region_deploy_agent = self.region_legal_move
        else:
            self.region_deploy_agent = get_surrounding_offset_region(
                arrays=[scan_array],
                offset_thickness=self.walk_region_thickness,
                exclude_arrays=[ground_array],
            )

    def get_move_values(self, agent: Agent, state: Environment):
        follow_grid = state.grids["follow"]
        return agent.calculate_move_values_random__z_based__follow(
            follow_grid,
        )

    def build(self, agent: Agent, state: Environment):
        # update print dot array
        centroids = state.grids["centroids"]
        self.print_dot_counter += 1
        centroids.set_value(agent.pose, self.print_dot_counter)

        built = state.grids["built"]

        # orient shape map
        build_map = agent.orient_build_map()

        # update built_volume_volume_array
        built.set_value_using_index_map(build_map, values=1)

        print(f"built at: {agent.pose}")

        # TODO: Why change ground here?
        ground = state.grids["ground"]
        ground.set_value_using_index_map(build_map, value=1)

        # update grids for index_map visualisation
        move_map_grid = state.grids.get("move_map")
        if move_map_grid:
            move_map_grid.set_value_using_index_map(agent.orient_sense_map(), values=1)

        sense_maps_grid = state.grids.get("sense_maps")

        if sense_maps_grid:
            sense_maps_grid.set_value_using_index_map(
                agent.orient_sense_inplane_map(), values=1
            )

            sense_maps_grid.set_value_using_index_map(
                agent.orient_sense_depth_map(), values=1
            )

        self.update_fab_planes_file(agent.get_plane())

    def update_fab_planes_file(self, plane):
        self.fab_planes.append(plane)

        if not self.fab_planes_file_path.exists():
            self.fab_planes_file_path.parent.mkdir(parents=True, exist_ok=True)
            self.fab_planes_file_path.touch()

        with self.fab_planes_file_path.open(mode="a") as f:
            f.write(json_dumps(plane) + "\n")

    # ACTION FUNCTION
    def agent_action(self, agent: Agent, state: Environment):
        """
        GET_PROBABILITY
        BUILD
        MOVE
        *RESET
        """
        # BUILD
        if hasattr(agent, "build_probability"):
            build_probability = agent.build_probability
        else:
            build_probability = self.get_agent_build_probability(agent, state)

        if build_probability > random.random():
            print(f"""## agent_{agent.id} built at {agent.pose}""")
            print(f"## build_probability = {build_probability}")

            # build
            self.build(agent, state)

            # update offset regions
            self.update_offset_regions(
                ground_array=state.grids["ground"].to_numpy(),
                scan_array=state.grids["scan"].to_numpy(),
            )

            # reset if
            agent.step_counter = 0
            if agent.reset_after_build:
                if isinstance(agent.reset_after_build, float):
                    if random.random() < agent.reset_after_build:
                        agent.deploy_in_region(self.region_deploy_agent)
                elif agent.reset_after_build is True:
                    agent.deploy_in_region(self.region_deploy_agent)
                else:
                    pass

        # check collision
        if not agent.check_solid_collision([state.grids["built"]]):
            # move
            move_values = self.get_move_values(agent, state)

            legal_move_mask = self.get_legal_move_mask(agent, state)

            agent.move_by_index_map(
                index_map_in_place=agent.get_localized_move_map(),
                move_values=move_values[legal_move_mask],
                random_batch_size=4,
            )
            agent.step_counter += 1

        # RESET
        else:
            # reset if stuck
            agent.deploy_in_region(self.region_deploy_agent)

        # reset if inactive
        if agent.inactive_step_count_limit:  # noqa: SIM102
            if agent.step_counter >= agent.inactive_step_count_limit:
                agent.deploy_in_region(self.region_deploy_agent)
