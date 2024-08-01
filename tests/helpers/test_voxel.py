import compas.geometry as cg

from bdm_voxel_builder.helpers.voxel import get_linear_xform_between_2_boxes


class TestGetLinearTransformationBetweenTwoBoxes:
    def test_identical(self):
        # Test case 1: Boxes with same dimensions
        from_box = cg.Box(1, 1, 1, frame=cg.Frame.worldXY())
        to_box = cg.Box(1, 1, 1, frame=cg.Frame.worldXY())
        expected_transformation = cg.Transformation()
        assert (
            get_linear_xform_between_2_boxes(from_box, to_box)
            == expected_transformation
        )

    def test_different_dimensions(self):
        # Test case 2: Boxes with different dimensions
        from_box = cg.Box(2, 3, 4, frame=cg.Frame.worldXY())
        to_box = cg.Box(1, 2, 3, frame=cg.Frame.worldXY())
        expected_transformation = cg.Transformation.from_matrix(
            [
                [0.5, 0.0, 0.0, 0.0],
                [0.0, 2 / 3, 0.0, 0.0],
                [0.0, 0.0, 0.75, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        assert (
            get_linear_xform_between_2_boxes(from_box, to_box)
            == expected_transformation
        )

    def test_different_positions_and_orientations(self):
        from_box = cg.Box(1, 1, 1, frame=cg.Frame.worldXY())
        to_box = cg.Box(1, 1, 1, frame=cg.Frame.worldXY())
        to_box.transform(
            cg.Transformation.from_matrix(
                [
                    [1.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 2.0],
                    [0.0, 0.0, 1.0, 3.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        )
        expected_transformation = cg.Transformation.from_matrix(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 2.0],
                [0.0, 0.0, 1.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        assert (
            get_linear_xform_between_2_boxes(from_box, to_box)
            == expected_transformation
        )
