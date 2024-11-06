import random as r

from bdm_voxel_builder.agent_algorithms.base import AgentAlgorithm
from bdm_voxel_builder.agents import Agent
from bdm_voxel_builder.environment import Environment


class Algo21PrintPath(AgentAlgorithm):
    def setup_agents(self, state: Environment):
        agent_space = state.grids["agent"]
        track = state.grids["track"]
        ground_grid = state.grids["ground"]

        agents = []
        for n in range(self.agent_count):
            agent = Agent(
                initial_pose=(0, 0, 0),
                space_grid=agent_space,
                track_grid=track,
                ground_grid=ground_grid,
            )
            agent.id = n
            agent.deploy_in_region(self.region_deploy_agent)
            agents.append(agent)

        print("build_random_chance", agents[0].build_random_chance)

        return agents

    def get_agent_build_probability(self, agent: Agent, state: Environment):
        # RANDOM FACTOR
        if r.random() < agent.build_random_chance:  # noqa: SIM108
            return agent.build_random_gain
        else:
            return 0

    def update_environment(self, state: Environment):
        """Do nothing"""
        pass
