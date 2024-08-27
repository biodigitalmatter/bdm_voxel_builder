# ruff: noqa: E712
import numpy as np
import pytest
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
    assert mask.dtype == np.int8
