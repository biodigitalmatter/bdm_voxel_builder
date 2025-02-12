import compas.geometry as cg
import numpy as np
import pyopenvdb as vdb
import pytest
from numpy.testing import assert_allclose

from bdm_voxel_builder import get
from bdm_voxel_builder.grid import Grid


def test__activate_random_indices(activate_random_voxels, random_generator):
    """Fixture test"""
    grid = vdb.FloatGrid()
    indices = activate_random_voxels(grid, random_generator)
    assert len(indices) == grid.activeVoxelCount()
    assert len(indices) == 20

    accessor = grid.getConstAccessor()

    assert accessor.probeValue(indices[0]) == (1.0, True)
    assert accessor.probeValue(indices[-1]) == (1.0, True)


@pytest.fixture
def small_grid():
    return Grid(clipping_box=cg.Box(xsize=10, ysize=10, zsize=10), name="test")


@pytest.fixture
def large_grid(activate_random_voxels, random_generator):
    grid = Grid(name="large_grid")

    activate_random_voxels(grid.vdb, random_generator)

    return grid


def test_init():
    grid = Grid(name="test_init")
    assert grid.vdb.name == "test_init"


class TestClippingBox:
    def test_getter(self, large_grid):
        grid = Grid()
        assert grid.clipping_box.dimensions == [4294967295] * 3

        grid.set_value((5, 5, 5), 1)
        assert grid.clipping_box.dimensions == [4294967295] * 3

        assert large_grid.clipping_box.dimensions == [290, 254, 295]

    def test_setter(self):
        grid = Grid(clipping_box=5)
        assert grid.clipping_box.dimensions == [4.0, 4.0, 4.0]

        grid = Grid(clipping_box=(10, 10, 10))

        assert grid.clipping_box.dimensions == [9, 9, 9]

        grid = Grid(clipping_box=(5, 12, 10))

        assert grid.clipping_box.dimensions == [4.0, 11.0, 9.0]

        grid = Grid(clipping_box=cg.Box(xsize=10, ysize=10, zsize=10))

        assert grid.clipping_box.dimensions == [10.0, 10.0, 10.0]


@pytest.mark.skip("TODO.")
def test_get_value():
    # TODO: Implement test
    pass


def test_get_values():
    grid = Grid(name="test_get_values")
    grid.set_value((0, 0, 0), 1)
    grid.set_value((1, 1, 1), 2)
    values = grid.get_values([(0, 0, 0), (1, 1, 1)])
    assert values == [1, 2]


def test_set_value():
    grid = Grid("set_value_test")
    grid.set_value((5, 5, 3), 1)
    grid.set_value((1, 4, 0), 1)
    grid.set_value((2, 3, 1), 1)

    assert grid.get_number_of_active_voxels() == 3
    assert grid.get_value((5, 5, 3)) == 1


def test_set_values():
    grid = Grid(name="test_set_values")
    indices = [(0, 0, 0), (1, 1, 1)]
    values = [1, 2]
    grid.set_values(indices, values)
    assert grid.get_value((0, 0, 0)) == 1
    assert grid.get_value((1, 1, 1)) == 2


def test_set_value_using_index_map():
    grid = Grid(name="test_set_value_using_map_and_origin")
    map = np.array([[0, 0, 0], [1, 1, 1]])
    origin = (2, 2, 2)
    grid.set_value_using_index_map(map, origin, values=1)
    assert grid.get_value((1, 1, 1)) == 0
    assert grid.get_value((2, 2, 2)) == 1
    assert grid.get_value((3, 3, 3)) == 1


def test_set_values_in_zone_xxyyzz():
    grid = Grid(name="test_set_values_in_zone_xxyyzz")
    zone = (2, 5, 1, 3, 3, 7)
    grid.set_values_in_zone_xxyyzz(zone, value=1)
    assert grid.get_value((0, 0, 0)) == 0
    assert grid.get_value((3, 2, 4)) == 1
    assert grid.get_value((2, 1, 3)) == 1
    assert grid.get_value((4, 2, 6)) == 1
    assert grid.get_value((10, 10, 10)) == 0


class TestSetValuesUsingArray:
    @pytest.fixture
    def array(self):
        arr = np.zeros((3, 3, 3))
        arr[2, 1, 0] = 1
        arr[0, 2, 1] = 1
        arr[1, 0, 2] = 1
        arr[1, 1, 0] = 1
        return arr

    def test_set_values_using_array(self, array):
        grid = Grid(name="test_set_values_by_array")
        grid.set_values_using_array(array)

        assert grid.get_number_of_active_voxels() == 4
        assert grid.get_value((2, 1, 0)) == 1
        assert grid.get_value((0, 2, 1)) == 1
        assert grid.get_value((1, 0, 2)) == 1
        assert grid.get_value((1, 1, 0)) == 1

    def test_set_values_by_array_origin(self, array, large_grid):
        assert large_grid.get_number_of_active_voxels() == 20
        origin = (-10, -20, -30)

        large_grid.set_values_using_array(array, origin=origin)

        assert large_grid.get_number_of_active_voxels() == 24
        assert large_grid.get_value((-9, -19, -30)) == 1  # + [1, 1, 0]
        assert large_grid.get_value((-9, -19, -29)) == 0  # + [1, 1, 1]
        assert large_grid.get_value((-9, -20, -28)) == 1  # + [1, 0, 2]
        assert large_grid.get_value((84, 78, 226)) == 1
        assert large_grid.get_value((88, 57, 149)) == 1
        assert large_grid.get_value((96, 38, 282)) == 1


def test_get_active_voxels(small_grid):
    small_grid.set_value((5, 5, 3), 1)
    small_grid.set_value((1, 4, 0), 1)
    small_grid.set_value((2, 3, 1), 1)

    index_pts = small_grid.get_active_voxels()
    assert len(index_pts) == 3
    assert len(index_pts[0]) == 3


def test_get_number_of_active_voxels(large_grid):
    assert large_grid.get_number_of_active_voxels() == 20


def test_get_pointcloud(activate_random_voxels, random_generator):
    grid = Grid(name="test_get_pointcloud")
    activated_indices = activate_random_voxels(grid.vdb, random_generator)

    pointcloud = grid.get_pointcloud()

    assert pointcloud == cg.Pointcloud(activated_indices)


@pytest.mark.skip("TODO: Need to look into transform.")
def test_get_world_pointcloud(activate_random_voxels, random_generator):
    grid = Grid(grid_size=(10, 10, 10), xform=cg.Translation.from_vector([2, 2, 4]))

    grid.array, activated = activate_random_voxels(grid.array, random_generator)

    pointcloud = grid.get_world_pointcloud()
    assert len(pointcloud.points) == activated
    assert pointcloud.closest_point([0, 0, 0]) == [3, 2, 4]


@pytest.mark.skip("TODO: Need to look into transform.")
def test_get_world_pointcloud_messy(activate_random_voxels, random_generator):
    grid = Grid(
        grid_size=(10, 10, 10),
        xform=cg.Translation.from_vector([2.23, 2.11, 4.13])
        * cg.Scale.from_factors([1.1, 1.1, 1.1]),
    )

    grid.array, activated = activate_random_voxels(grid.array, random_generator)

    pointcloud = grid.get_world_pointcloud()
    assert len(pointcloud.points) == activated
    assert pointcloud.closest_point([0, 0, 0]) == [3.33, 2.11, 4.13]


@pytest.mark.skip("TODO.")
def test_merge_with():
    # TODO: Implement test
    pass


def test_merged_with():
    grid1 = Grid(name="grid1")
    grid1.set_value((0, 0, 0), 1)
    grid2 = Grid(name="grid2")
    grid2.set_value((1, 1, 1), 2)
    merged_grid = grid1.merged_with(grid2)
    assert merged_grid.get_value((0, 0, 0)) == 1
    assert merged_grid.get_value((1, 1, 1)) == 2


@pytest.mark.skip("TODO.")
def to_numpy():
    # TODO: Implement test
    pass


@pytest.mark.skip("TODO.")
def test_save_vdb(small_grid, large_grid):
    path = small_grid.save_vdb()

    assert path.exists()

    path.delete()

    path = large_grid.save_vdb()

    assert path.exists()


@pytest.mark.skip("TODO.")
def test_from_numpy():
    # TODO: Implement test
    pass


def test_from_vdb_w_file():
    grid = Grid.from_vdb(get("sphere.vdb"))

    assert grid.name == "ls_sphere"
    assert grid.vdb.evalActiveVoxelBoundingBox() == ((-62, -62, -62), (62, 62, 62))


def test_from_vdb_w_grid():
    grid = vdb.read(str(get("sphere.vdb")), "ls_sphere")
    grid = Grid.from_vdb(grid, name="test")

    assert grid.name == "test"
    assert grid.vdb.evalActiveVoxelBoundingBox() == ((-62, -62, -62), (62, 62, 62))


def test_from_pointcloud():
    pointcloud = cg.Pointcloud([[0, 0, 0], [1, 1, 1], [2, 2, 2]])

    grid = Grid.from_pointcloud(pointcloud)

    assert grid.get_value((0, 0, 0)) == 1
    assert grid.get_value((1, 1, 1)) == 1
    assert grid.get_value((2, 2, 2)) == 1


def test_from_pointcloud_large(random_pts, random_generator):
    pointcloud = cg.Pointcloud(random_pts(1000, random_generator))
    grid = Grid.from_pointcloud(pointcloud)
    assert grid.get_number_of_active_voxels() == 998  # rounding loses us some points


@pytest.mark.skip("TODO.")
def test_from_ply():
    # TODO: Implement test
    pass


# TODO: Revisit after vdb backed grid is implemented
@pytest.mark.skip("Need to look into transform.")
class TestFromPcd:
    @pytest.fixture
    def test_ascii_pcd(self):
        return get("ascii.pcd")

    @pytest.fixture
    def test_binary_pcd(self):
        return get("binary.pcd")

    @pytest.fixture
    def test_binary_compressed_pcd(self):
        return get("binary_compressed.pcd")

    @pytest.fixture
    def test_parsed_ascii_pcd(self, ascii_pcd):
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


def test_copy():
    grid = Grid(name="original")
    grid_copy = grid.copy(name="copy")
    assert grid_copy.name == "copy"
    assert grid_copy is not grid
    assert grid_copy.vdb is not grid.vdb


def test_get_neighbors():
    expected_neighbors = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]

    grid = Grid(name="test_get_neighbors")
    neighbors = set(list(grid.get_neighbors((0, 0, 0))))

    assert neighbors == set(expected_neighbors)

    idx = (5, 5, 5)
    neighbors = set(list(grid.get_neighbors(idx)))

    expected_neighbors = np.array(expected_neighbors) + idx
    expected_neighbors = tuple(tuple(ijk) for ijk in expected_neighbors)

    assert neighbors == set(expected_neighbors)


def test_get_active_neighbors():
    grid = Grid(name="test_get_active_neighbors")
    grid.set_value((1, 0, 0), 1)
    active_neighbors = list(grid.get_active_neighbors((0, 0, 0)))
    assert active_neighbors == [(1, 0, 0)]
