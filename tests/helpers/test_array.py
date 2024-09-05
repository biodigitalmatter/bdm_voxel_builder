# ruff: noqa: E712
import numpy as np
import pytest
from bdm_voxel_builder.agent_algorithms.common import get_any_voxel_in_region
from bdm_voxel_builder.helpers.array import get_mask_zone_xxyyzz


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
    x, y = get_any_voxel_in_region(array_1)
    assert x == 2 and y == 1

    # test 2
    array_2 = np.array([[1, 1, 0, 1, 1], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0]])
    index = get_any_voxel_in_region(array_2)
    result = array_2[*index]
    assert result == 1
