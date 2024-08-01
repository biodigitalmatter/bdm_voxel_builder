import compas.geometry as cg
import numpy as np
import pyopenvdb as vdb
import pytest

from bdm_voxel_builder import get
from bdm_voxel_builder.grid import Grid


@pytest.fixture
def random_int_array():
    shape = (10, 10, 10)
    return np.random.default_rng().integers(0, 2, size=shape)


def test_to_vdb_grid(random_int_array):
    grid = Grid(array=random_int_array, grid_size=random_int_array.shape)

    grid = grid.to_vdb_grid()

    assert grid.name == "None"


def test_save_vdb(random_int_array):
    grid = Grid(array=random_int_array, grid_size=random_int_array.shape)

    path = grid.save_vdb()

    assert path.exists()


def test_get_local_bbox():
    grid = Grid(grid_size=(10, 10, 10))
    bbox = grid.get_local_bbox()
    assert bbox.diagonal == ((0, 0, 0), (10, 10, 10))


def test_get_local_bbox_with_non_square_grid_size():
    with pytest.raises(NotImplementedError):
        grid = Grid(grid_size=(5, 10, 15))
        grid.get_local_bbox()


def test_get_world_bbox():
    grid = Grid(grid_size=(10, 10, 10))
    bbox = grid.get_world_bbox()
    assert bbox.diagonal == ((0, 0, 0), (10, 10, 10))


def test_get_world_bbox_with_non_square_grid_size():
    T = cg.Translation.from_vector([15, 10, 5])
    grid = Grid(grid_size=(5, 5, 5), xform=T)
    bbox = grid.get_world_bbox()
    assert bbox.diagonal == cg.Line((15, 10, 5), (20, 15, 10))


def test_from_vdb_w_file():
    grid = Grid.from_vdb(get("sphere.vdb"))

    assert grid.name == "ls_sphere"
    assert grid.array.shape == (124, 124, 124)


def test_from_vdb_w_grid():
    grid = vdb.read(str(get("sphere.vdb")), "ls_sphere")
    grid = Grid.from_vdb(grid)

    assert grid.name == "ls_sphere"
    assert grid.array.shape == (124, 124, 124)
