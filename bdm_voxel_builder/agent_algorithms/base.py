import abc


class AgentAlgorithm(abc.ABC):
    def __init__(
        self,
        agent_count: int,
        grid_size: int | tuple[int, int, int],
        layer_to_dump: str,
        name: str = None,
    ) -> None:
        self.agent_count = agent_count
        self.grid_size = grid_size
        self.layer_to_dump = layer_to_dump
        self.name = name

    @abc.abstractmethod
    def move_agent(self):
        raise NotImplementedError

    @abc.abstractmethod
    def reset_agent(self):
        raise NotImplementedError

    @abc.abstractmethod
    def setup_agents(self):
        raise NotImplementedError
