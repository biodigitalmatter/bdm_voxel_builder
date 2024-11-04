import math
from dataclasses import dataclass

from compas.colors import Color

from bdm_voxel_builder.agent_algorithms.base import AgentAlgorithm
from bdm_voxel_builder.agents import Agent
from bdm_voxel_builder.environment import Environment
from bdm_voxel_builder.grid import DiffusiveGrid
from bdm_voxel_builder.helpers.geometry import tpms_gyroid


@dataclass
class Make_Gyroid(AgentAlgorithm):
    """ """

    agent_count: int
    grid_size: int | tuple[int, int, int]
    name: str = "gyroid"
    relevant_data_grids: str = "density"
    grids_to_dump: list[str] = ("density",)

    seed_iterations: int = 1

    scale = 1 / math.pi
    scale = math.pi * 2 / 50
    print(scale)

    thickness_in = 1
    thickness_out = 0.5

    def __post_init__(self):
        """Initialize values held in parent class.

        Run in __post_init__ since @dataclass creates __init__ method"""
        super().__init__(
            agent_count=self.agent_count,
            grid_size=self.grid_size,
            grids_to_dump=self.grids_to_dump,
            name=self.name,
        )

    def initialization(self, **kwargs):
        """
        creates the simulation environment setup
        with preset values in the definition

        returns: grids
        """
        agent_space = DiffusiveGrid(
            name="agent_space",
            grid_size=self.grid_size,
            color=Color.from_rgb255(34, 116, 240),
        )
        track = DiffusiveGrid(
            name="track",
            grid_size=self.grid_size,
            color=Color.from_rgb255(34, 116, 240),
            decay_ratio=1 / 10000,
        )

        density = DiffusiveGrid(
            name="density",
            grid_size=self.grid_size,
            color=Color.from_rgb255(219, 26, 206),
            flip_colors=True,
            decay_ratio=1 / 10000,
        )
        array = tpms_gyroid(
            self.grid_size, self.scale, self.thickness_out, self.thickness_in
        )
        density.array = array
        # print(array)
        # WRAP ENVIRONMENT
        grids = {
            "agent": agent_space,
            "track": track,
            "density": density,
        }
        return grids

    def update_environment(self, state: Environment):
        pass

    def setup_agents(self, grids: dict[str, DiffusiveGrid]):
        agent_space = grids["agent"]
        track = grids["track"]

        agents: list[Agent] = []

        for _i in range(self.agent_count):
            # create object
            agent = Agent(
                space_grid=agent_space,
                track_grid=track,
                leave_trace=True,
                save_move_history=True,
            )
            agents.append(agent)
        return agents

    # ACTION FUNCTION
    def agent_action(self, agent, state: Environment):
        pass
