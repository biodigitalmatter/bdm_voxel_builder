import abc
import numpy as np


class AgentAlgorithm(abc.ABC):
    def __init__(
        self,
        agent_count: int,
        grid_size: int | tuple[int, int, int],
        grid_to_dump: str,
        name: str = None,
    ) -> None:
        self.agent_count = agent_count
        self.grid_to_dump = grid_to_dump
        self.name = name

        if isinstance(grid_size, int | float):
            grid_size = np.array([grid_size, grid_size, grid_size], dtype=np.int32)
        elif isinstance(grid_size, list | tuple):
            grid_size = np.array(grid_size, dtype=np.int32)
        if np.min(grid_size) < 1:
            raise ValueError("grid_size must be nonzero and positive")
        self.grid_size = grid_size.tolist()

    @abc.abstractmethod
    def setup_agents(self):
        raise NotImplementedError
