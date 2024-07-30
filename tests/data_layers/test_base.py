import compas.geometry as cg
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


def test_get_local_bbox():
    layer = DataLayer(grid_size=(10, 10, 10))
    bbox = layer.get_local_bbox()
    assert bbox.diagonal == ((0, 0, 0), (10, 10, 10))


def test_get_local_bbox_with_non_square_grid_size():
    with pytest.raises(NotImplementedError):
        layer = DataLayer(grid_size=(5, 10, 15))
        layer.get_local_bbox()


def test_get_world_bbox():
    layer = DataLayer(grid_size=(10, 10, 10))
    bbox = layer.get_world_bbox()
    assert bbox.diagonal == ((0, 0, 0), (10, 10, 10))


def test_get_world_bbox_with_non_square_grid_size():
    T = cg.Translation.from_vector([15, 10, 5])
    layer = DataLayer(grid_size=(5, 5, 5), xform=T)
    bbox = layer.get_world_bbox()
    assert bbox.diagonal == cg.Line((15, 10, 5), (20, 15, 10))
