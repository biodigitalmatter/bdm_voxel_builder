import compas.geometry as cg
import numpy as np


def box_from_corner_frame(frame: cg.Frame, xsize: float, ysize: float, zsize: float):
    """Create a box from origin and size."""
    center_pt = frame.point.copy()
    center_pt += cg.Vector(xsize / 2, ysize / 2, zsize / 2)

    center_frame = cg.Frame(center_pt, xaxis=frame.xaxis, yaxis=frame.yaxis)

    return cg.Box(xsize=xsize, ysize=ysize, zsize=zsize, frame=center_frame)


def get_xform_box2grid(
    box: cg.Box, grid_size: tuple[int, int, int]
) -> cg.Transformation:
    """Get the linear transformation between two bounding boxes with uniform scaling."""
    v = cg.Vector(box.xmin, box.ymin, box.zmin)

    Tr = cg.Translation.from_vector(v.inverted())

    R = cg.Rotation.from_frame_to_frame(box.frame, cg.Frame.worldXY())

    factor = float(max(grid_size) - 1) / max(box.dimensions)

    S = cg.Scale.from_factors([factor] * 3, frame=box.frame)

    return Tr * S * R
