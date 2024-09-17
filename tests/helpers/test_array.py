# ruff: noqa: E712
import numpy as np
import pytest

from bdm_voxel_builder.agent_algorithms.common import get_any_index_of_mask
from bdm_voxel_builder.helpers.array import get_mask_zone_xxyyzz, index_map_sphere


def test_get_mask_zone_xxyyzz():
    grid_size = (10, 10, 10)
    zone_xxyyzz = (2, 7, 3, 8, 1, 6)

    # Test return type
    mask = get_mask_zone_xxyyzz(grid_size, zone_xxyyzz, return_bool=True)
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == np.bool_
    assert mask[3, 5, 1] == True
    assert mask[2, 7, 6] == True
    assert mask[7, 4, 5] == True

    assert mask[1, 5, 1] == False
    assert mask[3, 9, 1] == False
    with pytest.raises(IndexError):
        assert mask[3, 11, 1] == False

    mask = get_mask_zone_xxyyzz(grid_size, zone_xxyyzz, return_bool=False)
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == np.int32


def test_get_any_voxel_in_region():
    # test 1
    array_1 = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]])
    x, y = get_any_index_of_mask(array_1)
    assert x == 2 and y == 1

    # test 2
    array_2 = np.array([[1, 1, 0, 1, 1], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0]])
    index = get_any_index_of_mask(array_2)
    result = array_2[*index]
    assert result == 1


class TestIndexMapSphere:
    @pytest.mark.parametrize("radius", [1.5, 2.5, 3.5])
    def test_index_map_sphere(self, radius):
        radius = 1.5
        indices = index_map_sphere(radius, min_radius=None)
        assert isinstance(indices, np.ndarray)
        assert indices.shape[1] == 3
        assert np.all(np.linalg.norm(indices, axis=1) <= radius)

    @pytest.mark.parametrize("radius", [1.5, 2.5, 3.5])
    @pytest.mark.parametrize("min_radius", [1.5, 2.5, 3.5])
    def test_index_map_sphere_with_min_radius(self, radius, min_radius):
        indices = index_map_sphere(radius, min_radius=min_radius)
        assert isinstance(indices, np.ndarray)
        assert indices.shape[1] == 3
        assert np.all(
            np.logical_and(
                np.linalg.norm(indices, axis=1) <= radius,
                np.linalg.norm(indices, axis=1) >= min_radius,
            )
        )

    @pytest.fixture
    def sphere_1_5(self):
        return np.array(
            [
                [-1, -1, 0],
                [-1, 0, -1],
                [-1, 0, 0],
                [-1, 0, 1],
                [-1, 1, 0],
                [0, -1, -1],
                [0, -1, 0],
                [0, -1, 1],
                [0, 0, -1],
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, -1],
                [0, 1, 0],
                [0, 1, 1],
                [1, -1, 0],
                [1, 0, -1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
            ]
        )

    @pytest.fixture
    def sphere_3(self):
        return np.array(
            [
                [-3, 0, 0],
                [-2, -2, -1],
                [-2, -2, 0],
                [-2, -2, 1],
                [-2, -1, -2],
                [-2, -1, -1],
                [-2, -1, 0],
                [-2, -1, 1],
                [-2, -1, 2],
                [-2, 0, -2],
                [-2, 0, -1],
                [-2, 0, 0],
                [-2, 0, 1],
                [-2, 0, 2],
                [-2, 1, -2],
                [-2, 1, -1],
                [-2, 1, 0],
                [-2, 1, 1],
                [-2, 1, 2],
                [-2, 2, -1],
                [-2, 2, 0],
                [-2, 2, 1],
                [-1, -2, -2],
                [-1, -2, -1],
                [-1, -2, 0],
                [-1, -2, 1],
                [-1, -2, 2],
                [-1, -1, -2],
                [-1, -1, -1],
                [-1, -1, 0],
                [-1, -1, 1],
                [-1, -1, 2],
                [-1, 0, -2],
                [-1, 0, -1],
                [-1, 0, 0],
                [-1, 0, 1],
                [-1, 0, 2],
                [-1, 1, -2],
                [-1, 1, -1],
                [-1, 1, 0],
                [-1, 1, 1],
                [-1, 1, 2],
                [-1, 2, -2],
                [-1, 2, -1],
                [-1, 2, 0],
                [-1, 2, 1],
                [-1, 2, 2],
                [0, -3, 0],
                [0, -2, -2],
                [0, -2, -1],
                [0, -2, 0],
                [0, -2, 1],
                [0, -2, 2],
                [0, -1, -2],
                [0, -1, -1],
                [0, -1, 0],
                [0, -1, 1],
                [0, -1, 2],
                [0, 0, -3],
                [0, 0, -2],
                [0, 0, -1],
                [0, 0, 0],
                [0, 0, 1],
                [0, 0, 2],
                [0, 0, 3],
                [0, 1, -2],
                [0, 1, -1],
                [0, 1, 0],
                [0, 1, 1],
                [0, 1, 2],
                [0, 2, -2],
                [0, 2, -1],
                [0, 2, 0],
                [0, 2, 1],
                [0, 2, 2],
                [0, 3, 0],
                [1, -2, -2],
                [1, -2, -1],
                [1, -2, 0],
                [1, -2, 1],
                [1, -2, 2],
                [1, -1, -2],
                [1, -1, -1],
                [1, -1, 0],
                [1, -1, 1],
                [1, -1, 2],
                [1, 0, -2],
                [1, 0, -1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 0, 2],
                [1, 1, -2],
                [1, 1, -1],
                [1, 1, 0],
                [1, 1, 1],
                [1, 1, 2],
                [1, 2, -2],
                [1, 2, -1],
                [1, 2, 0],
                [1, 2, 1],
                [1, 2, 2],
                [2, -2, -1],
                [2, -2, 0],
                [2, -2, 1],
                [2, -1, -2],
                [2, -1, -1],
                [2, -1, 0],
                [2, -1, 1],
                [2, -1, 2],
                [2, 0, -2],
                [2, 0, -1],
                [2, 0, 0],
                [2, 0, 1],
                [2, 0, 2],
                [2, 1, -2],
                [2, 1, -1],
                [2, 1, 0],
                [2, 1, 1],
                [2, 1, 2],
                [2, 2, -1],
                [2, 2, 0],
                [2, 2, 1],
                [3, 0, 0],
            ]
        )

    @pytest.fixture
    def sphere_3__2(self):
        return np.array(
            [
                [-3, 0, 0],
                [-2, -2, -1],
                [-2, -2, 0],
                [-2, -2, 1],
                [-2, -1, -2],
                [-2, -1, -1],
                [-2, -1, 0],
                [-2, -1, 1],
                [-2, -1, 2],
                [-2, 0, -2],
                [-2, 0, -1],
                [-2, 0, 0],
                [-2, 0, 1],
                [-2, 0, 2],
                [-2, 1, -2],
                [-2, 1, -1],
                [-2, 1, 0],
                [-2, 1, 1],
                [-2, 1, 2],
                [-2, 2, -1],
                [-2, 2, 0],
                [-2, 2, 1],
                [-1, -2, -2],
                [-1, -2, -1],
                [-1, -2, 0],
                [-1, -2, 1],
                [-1, -2, 2],
                [-1, -1, -2],
                [-1, -1, 2],
                [-1, 0, -2],
                [-1, 0, 2],
                [-1, 1, -2],
                [-1, 1, 2],
                [-1, 2, -2],
                [-1, 2, -1],
                [-1, 2, 0],
                [-1, 2, 1],
                [-1, 2, 2],
                [0, -3, 0],
                [0, -2, -2],
                [0, -2, -1],
                [0, -2, 0],
                [0, -2, 1],
                [0, -2, 2],
                [0, -1, -2],
                [0, -1, 2],
                [0, 0, -3],
                [0, 0, -2],
                [0, 0, 2],
                [0, 0, 3],
                [0, 1, -2],
                [0, 1, 2],
                [0, 2, -2],
                [0, 2, -1],
                [0, 2, 0],
                [0, 2, 1],
                [0, 2, 2],
                [0, 3, 0],
                [1, -2, -2],
                [1, -2, -1],
                [1, -2, 0],
                [1, -2, 1],
                [1, -2, 2],
                [1, -1, -2],
                [1, -1, 2],
                [1, 0, -2],
                [1, 0, 2],
                [1, 1, -2],
                [1, 1, 2],
                [1, 2, -2],
                [1, 2, -1],
                [1, 2, 0],
                [1, 2, 1],
                [1, 2, 2],
                [2, -2, -1],
                [2, -2, 0],
                [2, -2, 1],
                [2, -1, -2],
                [2, -1, -1],
                [2, -1, 0],
                [2, -1, 1],
                [2, -1, 2],
                [2, 0, -2],
                [2, 0, -1],
                [2, 0, 0],
                [2, 0, 1],
                [2, 0, 2],
                [2, 1, -2],
                [2, 1, -1],
                [2, 1, 0],
                [2, 1, 1],
                [2, 1, 2],
                [2, 2, -1],
                [2, 2, 0],
                [2, 2, 1],
                [3, 0, 0],
            ]
        )

    def test_index_map_sphere_1_5(self, sphere_1_5):
        indices = index_map_sphere(1.5)
        assert np.array_equal(indices, sphere_1_5)

    def test_index_map_sphere_3(self, sphere_3):
        indices = index_map_sphere(3)
        assert np.array_equal(indices, sphere_3)

    def test_index_map_sphere_3__2(self, sphere_3__2):
        indices = index_map_sphere(3, min_radius=2)
        assert np.array_equal(indices, sphere_3__2)
