from dataclasses import dataclass

from compas.colors import Color

from bdm_voxel_builder.agent import Agent
from bdm_voxel_builder.agent_algorithms.base import AgentAlgorithm
from bdm_voxel_builder.environment import Environment
from bdm_voxel_builder.grid import DiffusiveGrid


@dataclass
class Make_Gyroid(AgentAlgorithm):
    """ """

    agent_count: int
    grid_size: int | tuple[int, int, int]
    name: str = "algo_12_random_builder"
    relevant_data_grids: str = "built_volume"
    grid_to_dump: str = "built_volume"

    seed_iterations: int = 1

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

        for _ in range(self.agent_count):
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
