import compas.geometry as cg
import numpy as np
import pyopenvdb as vdb
import pytest
from numpy.testing import assert_allclose

from bdm_voxel_builder import DATA_DIR, get
from bdm_voxel_builder.grid import Grid


def test__activate_random_indices(activate_random_indices, random_generator):
    array = np.zeros(shape=(10, 10, 10), dtype=np.float64)
    array, activated = activate_random_indices(array, random_generator)
    assert activated == 499
    assert array[5, 2, 2] == 1
    assert array[4, 4, 1] == 0


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
    assert bbox.diagonal == cg.Line((0, 0, 0), (10, 10, 10))


def test_get_local_bbox_with_non_square_grid_size():
    grid = Grid(grid_size=(5, 10, 15))
    local_bbox = grid.get_local_bbox()

    assert local_bbox.xsize == cg.Box(xsize=5, ysize=10, zsize=15).xsize
    assert local_bbox.ysize == cg.Box(xsize=5, ysize=10, zsize=15).ysize
    assert local_bbox.zsize == cg.Box(xsize=5, ysize=10, zsize=15).zsize


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


def test_set_value_at_index():
    grid = Grid(grid_size=(10, 10, 10))
    grid.set_value_at_index((5, 5, 5), 1)
    assert grid.get_value_at_index((5, 5, 5)) == 1


def test_get_active_voxels():
    grid = Grid(grid_size=(10, 10, 10))
    grid.set_value_at_index((5, 5, 5), 1)
    active_voxels = grid.get_active_voxels()
    assert len(active_voxels) == 3
    assert active_voxels[0][0] == 5
    assert active_voxels[1][0] == 5
    assert active_voxels[2][0] == 5


def test_get_index_pts(activate_random_indices, random_generator):
    grid = Grid(grid_size=(10, 10, 10))

    grid.array, activated = activate_random_indices(grid.array, random_generator)

    index_pts = grid.get_index_pts()
    assert len(index_pts) == activated
    assert len(index_pts[0]) == 3


def test_get_index_pointcloud(activate_random_indices, random_generator):
    grid = Grid(grid_size=(10, 10, 10))

    grid.array, activated = activate_random_indices(grid.array, random_generator)

    pointcloud = grid.get_index_pointcloud()

    assert len(pointcloud.points) == activated


def test_get_world_pts(activate_random_indices, random_generator):
    grid = Grid(grid_size=(10, 10, 10), xform=cg.Scale.from_factors([2, 2, 2]))

    grid.array, activated = activate_random_indices(grid.array, random_generator)

    world_pts = grid.get_world_pts()
    assert len(world_pts) == activated
    assert len(world_pts[0]) == 3
    assert grid.get_local_bbox().contains_points(world_pts)


def test_get_world_pointcloud(activate_random_indices, random_generator):
    grid = Grid(grid_size=(10, 10, 10), xform=cg.Translation.from_vector([2, 2, 4]))

    grid.array, activated = activate_random_indices(grid.array, random_generator)

    pointcloud = grid.get_world_pointcloud()
    assert len(pointcloud.points) == activated
    assert pointcloud.closest_point([0, 0, 0]) == [3, 2, 4]


def test_get_world_pointcloud_messy(activate_random_indices, random_generator):
    grid = Grid(
        grid_size=(10, 10, 10),
        xform=cg.Translation.from_vector([2.23, 2.11, 4.13])
        * cg.Scale.from_factors([1.1, 1.1, 1.1]),
    )

    grid.array, activated = activate_random_indices(grid.array, random_generator)

    pointcloud = grid.get_world_pointcloud()
    assert len(pointcloud.points) == activated
    assert pointcloud.closest_point([0, 0, 0]) == [3.33, 2.11, 4.13]


def test_get_merged_array_with():
    grid1 = Grid(grid_size=(10, 10, 10))
    grid2 = Grid(grid_size=(10, 10, 10))
    merged_array = grid1.get_merged_array_with(grid2)
    assert merged_array.shape == (10, 10, 10)


def test_from_pointcloud():
    pointcloud = cg.Pointcloud([[0, 0, 0], [1, 1, 1], [2, 2, 2]])

    grid = Grid.from_pointcloud(pointcloud, grid_size=3)

    assert grid.array[0, 0, 0] == 1
    assert grid.array[1, 1, 1] == 1
    assert grid.array[2, 2, 2] == 1


def test_from_pointcloud_large(random_pts, random_generator):
    pointcloud = cg.Pointcloud(random_pts(1000, random_generator))
    grid = Grid.from_pointcloud(pointcloud, grid_size=25)
    assert grid.get_number_of_active_voxels() == 967  # rounding loses us some points


# TODO: Revisit after vdb backed grid is implemented
@pytest.mark.skip("Need to look into transform.")
class TestFromPcd:
    @pytest.fixture
    def ascii_pcd(self):
        return get("ascii.pcd")

    @pytest.fixture
    def binary_pcd(self):
        return get("binary.pcd")

    @pytest.fixture
    def binary_compressed_pcd(self):
        return get("binary_compressed.pcd")

    @pytest.fixture
    def parsed_ascii_pcd(self, ascii_pcd):
        pt_lines = ascii_pcd.read_text().splitlines()[11:]
        parts = (line.split(" ") for line in pt_lines)

        pts = [[float(p) * 1000 for p in part[:3]] for part in parts]
        pts.sort(key=lambda x: (x[0], x[1], x[2]))
        return pts

    def test_from_pcd_ascii(self, ascii_pcd, parsed_ascii_pcd):
        grid = Grid.from_pcd(ascii_pcd, grid_size=100)

        assert grid.array.shape == (100, 100, 100)

        assert_allclose(grid.get_world_pts(), parsed_ascii_pcd)

    def test_from_pcd_binary(self, binary_pcd, parsed_ascii_pcd):
        grid = Grid.from_pcd(binary_pcd, grid_size=100)

        assert grid.array.shape == (100, 100, 100)
        assert_allclose(grid.get_world_pts(), parsed_ascii_pcd)

    def test_from_pcd_binary_compressed(self, binary_compressed_pcd, parsed_ascii_pcd):
        grid = Grid.from_pcd(binary_compressed_pcd, grid_size=100)

        assert grid.array.shape == (100, 100, 100)
        assert_allclose(grid.get_world_pts(), parsed_ascii_pcd)
