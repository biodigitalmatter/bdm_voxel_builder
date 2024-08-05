import math

import compas.geometry as cg
import numpy as np
import pytest

from bdm_voxel_builder import get
from bdm_voxel_builder.helpers.geometry import (
    box_from_corner_frame,
    convert_array_to_pts,
    get_xform_box2grid,
    ply_to_array,
    ply_to_compas,
    pointcloud_from_ndarray,
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


class TestBoxFromCornerFrame:
    def test_box_from_corner_frame(self):
        # Test case 1: Box with positive dimensions
        frame = cg.Frame.worldXY()
        xsize = 1.0
        ysize = 2.0
        zsize = 3.0
        expected_box = cg.Box(
            xsize=xsize,
            ysize=ysize,
            zsize=zsize,
            frame=cg.Frame(
                point=cg.Point(0.5, 1.0, 1.5),
                xaxis=cg.Vector(1.0, 0.0, 0.0),
                yaxis=cg.Vector(0.0, 1.0, 0.0),
            ),
        )

        box = box_from_corner_frame(frame, xsize, ysize, zsize)

        assert box.xsize == expected_box.xsize
        assert box.ysize == expected_box.ysize
        assert box.zsize == expected_box.zsize
        assert box.frame == expected_box.frame


    def test_box_from_corner_frame_zero_dimensions(self):
        frame = cg.Frame.worldXY()
        xsize = 0.0
        ysize = 0.0
        zsize = 0.0
        expected_box = cg.Box(
            xsize=xsize,
            ysize=ysize,
            zsize=zsize,
            frame=cg.Frame(
                point=cg.Point(0.0, 0.0, 0.0),
                xaxis=cg.Vector(1.0, 0.0, 0.0),
                yaxis=cg.Vector(0.0, 1.0, 0.0),
            ),
        )

        box = box_from_corner_frame(frame, xsize, ysize, zsize)

        assert box.xsize == expected_box.xsize
        assert box.ysize == expected_box.ysize
        assert box.zsize == expected_box.zsize
        assert box.frame == expected_box.frame


    def test_box_from_corner_frame_custom_frame(self):
        frame = cg.Frame(
            point=cg.Point(1.0, 2.0, 3.0),
            xaxis=cg.Vector(1.0, 0.0, 0.0),
            yaxis=cg.Vector(0.0, 1.0, 0.0),
        )
        xsize = 2.0
        ysize = 3.0
        zsize = 4.0
        expected_box = cg.Box(
            xsize=xsize,
            ysize=ysize,
            zsize=zsize,
            frame=cg.Frame(
                point=cg.Point(2.0, 3.5, 5.0),
                xaxis=cg.Vector(1.0, 0.0, 0.0),
                yaxis=cg.Vector(0.0, 1.0, 0.0),
            ),
        )

        box = box_from_corner_frame(frame, xsize, ysize, zsize)

        assert box.xsize == expected_box.xsize
        assert box.ysize == expected_box.ysize
        assert box.zsize == expected_box.zsize
        assert box.frame == expected_box.frame


class TestConvertArrayToPts:
    def test_nonzero(self):
        arr = np.array([[1, 0, 0, 0.7], [0, 1, 0, 0.5], [0, 0, 1, 0.2]])
        expected_pts = [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
        expected_data = [0.7, 0.5, 0.2]

        pts, data = convert_array_to_pts(arr)

        assert pts == expected_pts
        assert data == expected_data

    def test_zeroes(self):
        arr = np.zeros((3, 3, 3))
        expected_pts = []

        pts = convert_array_to_pts(arr)

        assert isinstance(pts, list)
        assert pts == expected_pts

    def test_non_zero_wo_data(self):
        # Test case 3: Array with non-zero values and get_data=False
        arr = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        expected_pts = [[0, 0, 0], [1, 1, 1], [2, 2, 2]]

        pts = convert_array_to_pts(arr, get_data=False)

        assert isinstance(pts, list)
        assert pts == expected_pts

    def test_zero_wo_data(self):
        arr = np.zeros((3, 3, 3))
        expected_pts = []

        pts = convert_array_to_pts(arr, get_data=False)

        assert isinstance(pts, list)
        assert pts == expected_pts

class TestGetXformBox2Grid:
    def test_identical(self):
        box = box_from_corner_frame(cg.Frame.worldXY(), 1, 1, 1)

        expected_transformation = cg.Transformation()

        assert get_xform_box2grid(box, [2, 2, 2]) == expected_transformation

    def test_different_dimensions(self):
        from_size = (2, 3, 4)
        grid_size = 3
        box = box_from_corner_frame(cg.Frame.worldXY(), *from_size)

        xform = get_xform_box2grid(box, grid_size=[grid_size] * 3)

        # transform not implemented on box
        pts = cg.transform_points(box.points, xform)

        assert pts[0] == [0, 0, 0]
        assert pts[6] == [n / 2 for n in from_size]

    def test_different_positions(self):
        grid_size = 5
        box_frame = cg.Frame([1, 2, 3])

        box = box_from_corner_frame(
            frame=box_frame, xsize=grid_size, ysize=grid_size, zsize=grid_size
        )
        xform = get_xform_box2grid(box, [grid_size] * 3)

        # transform not implemented on box
        pts = cg.transform_points(box.points, xform)

        assert pts[0] == [0, 0, 0]
        np.testing.assert_allclose(pts[6], [grid_size - 1] * 3)

    def test_different_orientations(self):
        grid_size = 2
        frame = cg.Frame.worldXY()

        R = cg.Rotation.from_frame_to_frame(cg.Frame.worldXY(), cg.Frame.worldZX())

        frame.transform(R)

        box = box_from_corner_frame(
            frame=frame, xsize=grid_size, ysize=grid_size, zsize=grid_size
        )

        xform = get_xform_box2grid(box, [grid_size] * 3)

        # transform not implemented on box
        pts = cg.transform_points(box.points, xform)

        np.testing.assert_allclose(pts[0], [0, 0, 0], atol=1e-15)
        np.testing.assert_allclose(pts[6], [grid_size - 1] * 3)

    def test_different_positions_and_orientations(self):
        grid_size = 6
        frame = cg.Frame.worldXY()

        frame.point += cg.Vector(1, 2, 3)

        R = cg.Rotation.from_axis_and_angle(
            axis=cg.Vector.Zaxis(), angle=0.5 * math.pi, point=frame.point
        )

        frame.transform(R)

        box = box_from_corner_frame(
            frame=frame, xsize=grid_size, ysize=grid_size, zsize=grid_size
        )

        xform = get_xform_box2grid(box, grid_size=[grid_size] * 3)

        # transform not implemented on box
        pts = cg.transform_points(box.points, xform)

        np.testing.assert_allclose(pts[0], [0, 0, 0], atol=1e-15)
        np.testing.assert_allclose(pts[6], [grid_size - 1] * 3)


def test_ply_to_array(stone_ply):
    path, pts = stone_ply
    arr = ply_to_array(path)

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


def test_pointcloud_from_ndarray():
    arr = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    expected_pts = [[0, 0, 0], [1, 1, 1], [2, 2, 2]]

    pointcloud = pointcloud_from_ndarray(arr)

    assert isinstance(pointcloud, cg.Pointcloud)
    assert pointcloud.points == expected_pts


def test_pointcloud_from_ndarray_with_values():
    arr = np.array([[0, 0, 0, 1], [1, 1, 1, 2], [2, 2, 2, 3]])
    expected_pts = [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
    expected_values = [1, 2, 3]

    pointcloud, values = pointcloud_from_ndarray(arr, return_values=True)

    assert isinstance(pointcloud, cg.Pointcloud)
    assert pointcloud.points == expected_pts
    assert values == expected_values
