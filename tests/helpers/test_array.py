# ruff: noqa: E712
import compas.geometry as cg
import numpy as np
import pytest

from bdm_voxel_builder.agent_algorithms.common import get_any_index_of_mask
from bdm_voxel_builder.helpers.array import (
    get_array_average_using_index_map,
    get_mask_zone_xxyyzz,
    get_values_using_index_map,
    index_map_sphere,
)
from bdm_voxel_builder.helpers.geometry import box_from_corner_frame


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


class TestGetValuesUsingIndexMap:
    @pytest.fixture
    def array(self):
        return np.arange(27).reshape((3, 3, 3))

    @pytest.fixture
    def index_map(self):
        return np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])

    def test_with_valid_index_map(self, array, index_map):
        values = get_values_using_index_map(array, index_map)
        assert np.array_equal(values, [0, 13, 26])

    def test_with_empty_index_map(self, array):
        with pytest.raises(ValueError):
            get_values_using_index_map(array, np.array([]))

    def test_with_empty_list_as_index_map(self, array):
        with pytest.raises(ValueError):
            get_values_using_index_map(array, [])

    def test_with_different_origin(self, array, index_map):
        origin = (1, 1, 1)
        values = get_values_using_index_map(array, index_map, origin)
        assert np.array_equal(values, [13, 26])

    def test_with_clipping_box(self, array, index_map):
        clipping_box = cg.Box.from_diagonal(((0, 0, 0), (2, 2, 2)))
        values = get_values_using_index_map(array, index_map, clipping_box=clipping_box)
        assert np.array_equal(values, [0, 13])


class TestGetArrayAverageUsingIndexMap:
    @pytest.fixture
    def array(self):
        arr = np.zeros((3, 3, 3))
        arr[0, 0, 0] = 1
        arr[1, 1, 1] = 2
        arr[2, 2, 2] = 3
        return arr

    @pytest.fixture
    def index_map(self):
        return np.array([[1, 1, 1], [2, 2, 2], [2, 0, 2]])

    @pytest.fixture
    def clipping_box1(self):
        return box_from_corner_frame(cg.Frame.worldXY(), 3, 3, 3)

    @pytest.fixture
    def clipping_box2(self):
        return box_from_corner_frame(cg.Frame.worldXY(), 2, 2, 2)

    def test_without_index_map(self, array, clipping_box1):
        with pytest.raises(ValueError):
            get_array_average_using_index_map(
                array, [], (0, 0, 0), clipping_box1, nonzero=False
            )

    def test_with_all_indicies(self, array, clipping_box1):
        index_map = list(np.ndindex(array.shape))
        avg = get_array_average_using_index_map(
            array, index_map, (0, 0, 0), clipping_box1, nonzero=False
        )
        assert avg == 6.0 / array.size

    def test_without_nonzero(self, array, index_map, clipping_box1):
        avg = get_array_average_using_index_map(
            array, index_map, (0, 0, 0), clipping_box1, nonzero=False
        )
        assert avg == 5.0 / 3

    def test_with_nonzero(self, array, index_map, clipping_box1):
        avg_nonzero = get_array_average_using_index_map(
            array, index_map, (0, 0, 0), clipping_box1, nonzero=True
        )
        assert avg_nonzero == 2.0 / 3

    def test_with_different_origin(self, array, index_map, clipping_box1):
        origin = (1, 1, 1)
        avg = get_array_average_using_index_map(
            array, index_map, origin, clipping_box1, nonzero=False
        )
        assert avg == 1.5

    def test_with_different_clipping_box(self, array, index_map, clipping_box2):
        # Test with different clipping box
        avg = get_array_average_using_index_map(
            array, index_map, (1, 1, 1), clipping_box2, nonzero=False
        )
        assert avg == 2
