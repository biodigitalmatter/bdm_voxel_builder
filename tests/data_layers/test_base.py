import numpy as np
import pytest

from bdm_voxel_builder.data_layer.base import DataLayer


@pytest.fixture
def random_int_array():
    shape = (10, 10, 10)
    return np.random.default_rng().integers(0, 2, size=shape)


def test_to_grid(random_int_array):
    layer = DataLayer(array=random_int_array, grid_size=random_int_array.shape)

    grid = layer.to_vdb_grid()

    assert grid.name == "layer_None"


def test_save_vdb(random_int_array):
    layer = DataLayer(array=random_int_array, grid_size=random_int_array.shape)

    path = layer.save_vdb()

    assert path.exists()
