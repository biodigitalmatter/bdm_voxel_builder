import numpy as np
import pytest
from bdm_voxel_builder import get
import compas.geometry as cg

from bdm_voxel_builder.helpers.pointcloud import ply_to_compas, ply_to_numpy


@pytest.fixture
def stone_ply():
    path = get("stone_scan_1mm.ply")

    first_three_pts: tuple[float, float, float] = []

    with path.open(mode="r") as fp:
        pts: tuple[float, float, float] = []
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
