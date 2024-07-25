from dataclasses import dataclass


@dataclass
class AgentAlgorithm:
    agent_count: int
    voxel_size: int
    layer_to_dump: str

    def move_agents(self):
        raise NotImplementedError

    def reset_agents(self):
        raise NotImplementedError

    def setup_agents(self):
        raise NotImplementedError

    def calculate_build_chances(self):
        raise NotImplementedError

    def initialization(self):
        raise NotImplementedError

    def build(self):
        raise NotImplementedError

    def update_environment(self):
        raise NotImplementedError
