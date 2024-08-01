import compas.geometry as cg


def get_linear_xform_between_2_boxes(from_box: cg.Box, to_box: cg.Box):
    """Get the linear transformation between two bounding boxes."""
    T = cg.Transformation.from_frame_to_frame(
        frame_from=from_box.frame, frame_to=to_box.frame
    )

    T *= cg.Scale.from_factors(to_box.diagonal.vector / from_box.diagonal.vector)

    return T


def pointcloud_to_grid(pointcloud, grid_size):
    bbox_pts = cg.Box.from_bounding_box(cg.bounding_box(pointcloud))
    bbox_grid = cg.Box.from_width_height_depth(grid_size)
