import pyopenvdb as vdb

from bdm_voxel_builder import get
from bdm_voxel_builder.grid import ImportedLayer


def test_from_vdb_w_file():
    layer = ImportedLayer.from_vdb(get("sphere.vdb"))

    assert layer.name == "ls_sphere"
    assert layer.array.shape == (124, 124, 124)


def test_from_vdb_w_grid():
    grid = vdb.read(str(get("sphere.vdb")), "ls_sphere")
    layer = ImportedLayer.from_vdb(grid)

    assert layer.name == "ls_sphere"
    assert layer.array.shape == (124, 124, 124)
