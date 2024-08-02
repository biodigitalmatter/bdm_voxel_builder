import compas.geometry as cg


def box_from_corner_frame(frame: cg.Frame, xsize:float, ysize:float, zsize: float):
    """Create a box from origin and size."""
    center_pt = frame.point.copy()
    center_pt += cg.Vector(xsize / 2, ysize / 2, zsize / 2)

    center_frame = cg.Frame(center_pt, xaxis=frame.xaxis, yaxis=frame.yaxis)

    return cg.Box(xsize=xsize, ysize=ysize, zsize=zsize, frame=center_frame)    


def get_linear_xform_between_2_boxes(from_box: cg.Box, to_box: cg.Box):
    """Get the linear transformation between two bounding boxes."""
    f_pt = from_box.frame.point.copy()
    f_pt -= cg.Point(from_box.width / 2, from_box.height / 2, from_box.depth / 2)

    t_pt = to_box.frame.point.copy()
    t_pt -= cg.Point(to_box.width / 2, to_box.height / 2, to_box.depth / 2)

    Tr = cg.Translation.from_vector(t_pt - f_pt)

    R = cg.Rotation.from_frame_to_frame(
        frame_from=to_box.frame, frame_to=from_box.frame
    )

    xfactor = from_box.width / to_box.width
    yfactor =from_box.height / to_box.height
    zfactor = from_box.depth / to_box.depth
    S = cg.Scale.from_factors([xfactor, yfactor, zfactor])

    return Tr * R * S
