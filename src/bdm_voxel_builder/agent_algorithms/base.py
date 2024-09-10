import abc


class AgentAlgorithm(abc.ABC):
    def __init__(
        self,
        agent_count: int,
        grid_to_dump: str,
        name: str = None,
    ) -> None:
        self.agent_count = agent_count
        self.grid_to_dump = grid_to_dump
        self.name = name

    @abc.abstractmethod
    def setup_agents(self):
        raise NotImplementedError
