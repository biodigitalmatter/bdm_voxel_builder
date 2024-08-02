import math

import compas.geometry as cg
import numpy as np

from bdm_voxel_builder.helpers import (
    box_from_corner_frame,
    get_linear_xform_between_2_boxes,
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

class TestGetLinearTransformationBetweenTwoBoxes:
    def test_identical(self):
        from_box = cg.Box(1, 1, 1, frame=cg.Frame.worldXY())
        to_box = from_box.copy()

        expected_transformation = cg.Transformation()

        assert (
            get_linear_xform_between_2_boxes(from_box, to_box)
            == expected_transformation
        )

    def test_different_dimensions(self):
        from_size = (2, 3, 4)
        to_size = (1, 2, 3)
        from_box = box_from_corner_frame(cg.Frame.worldXY(), *from_size)
        to_box = box_from_corner_frame(cg.Frame.worldXY(), *to_size)

        xform = get_linear_xform_between_2_boxes(from_box, to_box)

        moved_box = from_box.transformed(xform)

        assert moved_box.xsize == to_box.xsize
        assert moved_box.ysize == to_box.ysize
        assert moved_box.zsize == to_box.zsize
        assert moved_box.frame.normal == to_box.frame.normal
        assert moved_box.frame == to_box.frame

    def test_different_positions(self):
        xsize = ysize = zsize = 5
        from_frame = cg.Frame.worldXY()

        Tr = cg.Translation.from_vector(cg.Vector(1, 2, 3))

        to_frame = from_frame.transformed(Tr)

        from_box = box_from_corner_frame(
            frame=from_frame, xsize=xsize, ysize=ysize, zsize=zsize
        )
        to_box = box_from_corner_frame(
            frame=to_frame, xsize=xsize, ysize=ysize, zsize=zsize
        )

        xform = get_linear_xform_between_2_boxes(from_box, to_box)

        moved_box = from_box.transformed(xform)

        assert moved_box.points == to_box.points
        assert moved_box.xsize == to_box.xsize
        assert moved_box.ysize == to_box.ysize
        assert moved_box.zsize == to_box.zsize

    def test_different_orientations(self):
        xsize = ysize = zsize = 2
        from_frame = cg.Frame.worldXY()

        # R = cg.Rotation.from_axis_and_angle(
        #     axis=cg.Vector.Zaxis(), angle=0.5 * math.pi, point=from_frame.point
        # )
        R = cg.Rotation.from_frame_to_frame(cg.Frame.worldXY(), cg.Frame.worldZX())

        to_frame = from_frame.transformed(R)

        from_box = box_from_corner_frame(
            frame=from_frame, xsize=xsize, ysize=ysize, zsize=zsize
        )
        to_box = box_from_corner_frame(
            frame=to_frame, xsize=xsize, ysize=ysize, zsize=zsize
        )

        xform = get_linear_xform_between_2_boxes(from_box, to_box)

        moved_box = from_box.transformed(xform)

        assert moved_box.front == to_box.front
        assert moved_box.xsize == to_box.xsize
        assert moved_box.ysize == to_box.ysize
        assert moved_box.zsize == to_box.zsize

    def test_different_positions_and_orientations(self):
        from_frame = cg.Frame.worldXY()
        to_frame = from_frame.copy()

        to_frame.point += cg.Vector(1, 2, 3)

        R = cg.Rotation.from_axis_and_angle(
            axis=cg.Vector.Zaxis(), angle=0.5 * math.pi, point=to_frame.point
        )

        to_frame.transform(R)

        from_box = box_from_corner_frame(frame=from_frame, xsize=1, ysize=1, zsize=1)
        to_box = box_from_corner_frame(frame=to_frame, xsize=1, ysize=1, zsize=1)

        xform = get_linear_xform_between_2_boxes(from_box, to_box)

        moved_box = from_box.transformed(xform)

        assert moved_box.front == to_box.front
        assert moved_box.xsize == to_box.xsize
        assert moved_box.ysize == to_box.ysize
        assert moved_box.zsize == to_box.zsize
