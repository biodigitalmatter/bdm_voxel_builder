import compas.geometry as cg
import numpy as np


def box_from_corner_frame(frame: cg.Frame, xsize: float, ysize: float, zsize: float):
    """Create a box from origin and size."""
    center_pt = frame.point.copy()
    center_pt += cg.Vector(xsize / 2, ysize / 2, zsize / 2)

    center_frame = cg.Frame(center_pt, xaxis=frame.xaxis, yaxis=frame.yaxis)

    return cg.Box(xsize=xsize, ysize=ysize, zsize=zsize, frame=center_frame)


def get_linear_xform_between_2_boxes(from_box: cg.Box, to_box: cg.Box) -> cg.Transformation:
    """Get the linear transformation between two bounding boxes."""
    v = to_box.corner(0) - from_box.corner(0)

    Tr = cg.Translation.from_vector(v)

    R = cg.Rotation.from_change_of_basis(
        frame_from=from_box.frame, frame_to=to_box.frame
    )

    factors = np.divide(to_box.dimensions, from_box.dimensions).tolist()

    S = cg.Scale.from_factors(factors, frame=from_box.frame)

    return Tr * R * S
