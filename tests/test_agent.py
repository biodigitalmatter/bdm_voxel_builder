import compas.geometry as cg
import numpy as np
import pytest

from bdm_voxel_builder.agents import OrientableAgent


class TestOrientableAgent:
    @pytest.fixture
    def agent(self):
        return OrientableAgent()

    def test_get_plane_default(self, agent):
        plane = agent.get_plane()
        expected_plane = cg.Plane(agent.pose, cg.Vector.Zaxis().inverted())
        assert plane == expected_plane

    def test_get_frame_custom_pose(self, agent):
        agent.pose = np.array([1, 2, 3])
        plane = agent.get_plane()
        expected_plane = cg.Frame(agent.pose, cg.Vector.Zaxis().inverted())
        assert plane == expected_plane

    def test_get_frame_custom_normal_vector(self, agent):
        agent.normal_vector = cg.Vector(0, 1, 0)
        plane = agent.get_plane()
        expected_plane = cg.Plane(agent.pose, cg.Vector.Yaxis())
        assert plane == expected_plane
