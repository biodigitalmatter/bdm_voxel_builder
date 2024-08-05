import math

import compas.geometry as cg
import numpy as np

from bdm_voxel_builder.helpers import (
    box_from_corner_frame,
    get_xform_box2grid,
)


def test_box_from_corner_frame():
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


def test_box_from_corner_frame_zero_dimensions():
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


def test_box_from_corner_frame_custom_frame():
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
