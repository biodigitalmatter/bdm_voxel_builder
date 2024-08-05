import compas.geometry as cg
import numpy as np
import pytest

from bdm_voxel_builder import get
from bdm_voxel_builder.helpers.pointcloud import (
    ply_to_compas,
    ply_to_numpy,
    pointcloud_to_grid_array,
)


@pytest.fixture
def stone_ply():
    path = get("stone_scan_1mm.ply")

    pts: tuple[float, float, float] = []
    with path.open(mode="r") as fp:
        for line in fp:
            if line[0].isdigit() or line.startswith("-"):
                pt = [float(coord) for coord in line.split()]
                pts.append(pt[:3])

    return get("stone_scan_1mm.ply"), pts


def test_ply_to_np(stone_ply):
    path, pts = stone_ply
    arr = ply_to_numpy(path)

    assert isinstance(arr, np.ndarray)
    assert arr.shape[1] == 3
    assert np.allclose(arr, pts)


def test_ply_to_compas(stone_ply):
    path, pts = stone_ply
    pointcloud = ply_to_compas(path)

    assert isinstance(pointcloud, cg.Pointcloud)
    assert len(pointcloud.points) == len(pts)
    assert pointcloud == pts


def test_pointcloud_to_grid():
    pointcloud = cg.Pointcloud([(0, 0, 0), (0, 1, 0), (0, 2, 0), (2, 1, 0)])
    grid_size = (3, 3, 3)

    expected_arr = np.zeros(grid_size)
    expected_arr[0, 0, 0] = 1
    expected_arr[0, 1, 0] = 1
    expected_arr[0, 2, 0] = 1
    expected_arr[2, 1, 0] = 1

    grid_array = pointcloud_to_grid_array(pointcloud, grid_size)

    assert isinstance(grid_array, np.ndarray)
    assert np.array_equal(grid_array, expected_arr)


def test_pointcloud_to_grid_messy(random_pts, random_generator):
    pointcloud = cg.Pointcloud(random_pts(1000, random_generator))
    grid_size = (3, 3, 3)

    with pytest.raises(
        ValueError,
        match="Pointcloud contains negative values, needs to be transformed to index grid",  # noqa: E501
    ):
        pointcloud_to_grid_array(pointcloud, grid_size)
