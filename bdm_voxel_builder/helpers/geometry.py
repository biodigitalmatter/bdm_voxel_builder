import os
from collections.abc import Sequence

import compas.geometry as cg
import numpy as np
import numpy.typing as npt
from compas.files import PLY

from bdm_voxel_builder.helpers import sort_pts_by_values


def box_from_corner_frame(frame: cg.Frame, xsize: float, ysize: float, zsize: float):
    """Create a box from origin and size."""
    center_pt = frame.point.copy()
    center_pt += cg.Vector(xsize / 2, ysize / 2, zsize / 2)

    center_frame = cg.Frame(center_pt, xaxis=frame.xaxis, yaxis=frame.yaxis)

    return cg.Box(xsize=xsize, ysize=ysize, zsize=zsize, frame=center_frame)


def _convert_array_to_pts_wo_data(arr: npt.NDArray) -> list[list[float]]:
    pts = []
    for i, j, k in zip(*np.nonzero(arr), strict=False):
        pts.append([i, j, k])
    return pts


def convert_array_to_pts(
    arr: npt.NDArray, get_data=True
) -> list[list[float] | npt.NDArray]:
    if not get_data:
        return _convert_array_to_pts_wo_data(arr)

    indicies = np.indices(arr.shape)
    pt_location = np.logical_not(arr == 0)

    coordinates = []
    for i in range(3):
        c = indicies[i][pt_location]
        coordinates.append(c)

    return np.vstack(coordinates).transpose()


def get_xform_box2grid(
    box: cg.Box, grid_size: tuple[int, int, int]
) -> cg.Transformation:
    """Get the linear transformation between two bounding boxes with uniform
    scaling."""
    R = cg.Rotation.from_frame_to_frame(box.frame, cg.Frame.worldXY())

    scale_factor = float(max(grid_size) - 1) / max(box.dimensions)

    S = cg.Scale.from_factors([scale_factor] * 3)

    v = cg.Vector(box.xmin, box.ymin, box.zmin) * scale_factor

    Tr = cg.Translation.from_vector(v.inverted())

    return Tr * R * S


def ply_to_array(ply_path: os.PathLike, precision: str = None):
    """Convert a PLY file to a numpy array."""
    ply = PLY(filepath=ply_path, precision=precision)

    return np.array(ply.parser.vertices)


def ply_to_compas(ply_path: os.PathLike, precision: str = None):
    """Convert a PLY file to a compas Pointcloud."""
    ply = PLY(filepath=ply_path, precision=precision)

    return cg.Pointcloud(ply.parser.vertices)


def pointcloud_from_ndarray(arr: npt.NDArray, return_values=False):
    pts, values = sort_pts_by_values(arr)

    pointcloud = cg.Pointcloud(pts)

    if return_values:
        return pointcloud, values

    return pointcloud


def pointcloud_to_grid_array(
    pointcloud: cg.Pointcloud, grid_size: tuple[int, int, int], dtype=int, value=1
):
    """Convert a pointcloud to a grid."""
    if not isinstance(grid_size, Sequence):
        grid_size = (grid_size, grid_size, grid_size)

    grid_array = np.zeros(grid_size)

    pts = np.array(pointcloud).floor().astype(dtype=dtype)

    if pts.min() < 0:
        raise ValueError(
            "Pointcloud contains negative values, needs to be transformed to index grid."
        )  # noqa: E501

    for i, j, k in pts:
        if (i, j, k) > grid_size:
            raise ValueError(
                "Pointcloud contains values that are larger than grid size."
            )
        grid_array[i, j, k] = value
    return grid_array
