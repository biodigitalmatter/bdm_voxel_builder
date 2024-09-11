import os
from collections.abc import Sequence
from math import sin

import compas.geometry as cg
import numpy as np
import numpy.typing as npt
from compas.files import PLY
from compas.geometry import Frame, Plane, Point, Pointcloud, Transformation, Vector

from bdm_voxel_builder.helpers import sort_pts_by_values


def box_from_corner_frame(frame: cg.Frame, xsize: float, ysize: float, zsize: float):
    """Create a box from origin and size."""
    center_pt = frame.point.copy()
    center_pt += cg.Vector(xsize / 2, ysize / 2, zsize / 2)

    center_frame = cg.Frame(center_pt, xaxis=frame.xaxis, yaxis=frame.yaxis)

    return cg.Box(xsize=xsize, ysize=ysize, zsize=zsize, frame=center_frame)


def convert_grid_array_to_pts(arr: npt.NDArray) -> npt.NDArray:
    indices = np.indices(arr.shape)
    pt_location = np.logical_not(arr == 0)

    coordinates = []
    for i in range(3):
        c = indices[i][pt_location]
        coordinates.append(c)

    return np.vstack(coordinates).transpose()


def get_xform_box2grid(
    box: cg.Box, grid_size: tuple[int, int, int]
) -> cg.Transformation:
    """Get the linear transformation between two bounding boxes with uniform
    scaling."""
    R = cg.Rotation.from_frame_to_frame(box.frame, cg.Frame.worldXY())

    scale_factor = float(max(grid_size) - 1) / max(box.dimensions)

    Sc = cg.Scale.from_factors([scale_factor] * 3)

    v = cg.Vector(box.xmin, box.ymin, box.zmin) * scale_factor

    Tl = cg.Translation.from_vector(v.inverted())

    return Tl * R * Sc


def _get_xform_box2grid(
    box: cg.Box, grid_size: tuple[int, int, int]
) -> cg.Transformation:
    """Get the linear transformation between two bounding boxes with uniform
    scaling."""
    Sc = get_scaling_box2grid(box, grid_size)
    R = get_rotation_box2grid(box)
    Tl = get_translation_box2grid(box, grid_size)

    return Sc, R, Tl


def get_translation_box2grid(
    box: cg.Box, grid_size: tuple[int, int, int]
) -> cg.Translation:
    """Get the translation between bounding box and grid."""
    v = cg.Vector(box.xmin, box.ymin, box.zmin)

    return cg.Translation.from_vector(v.inverted())


def get_scaling_box2grid(box: cg.Box, grid_size: tuple[int, int, int]) -> cg.Scale:
    """Get the scaling needed between bounding box and grid."""
    return cg.Scale.from_factors([float(max(grid_size) - 1) / max(box.dimensions)] * 3)


def get_rotation_box2grid(box: cg.Box) -> cg.Rotation:
    """Get the rotation needed between bounding box and grid."""
    return cg.Rotation.from_frame_to_frame(box.frame, cg.Frame.worldXY())


def ply_to_array(ply_path: os.PathLike, precision: str = None):
    """Convert a PLY file to a numpy array."""
    ply = PLY(filepath=ply_path, precision=precision)

    return np.array(ply.parser.vertices)


def ply_to_compas(ply_path: os.PathLike, precision: str = None):
    """Convert a PLY file to a compas Pointcloud."""
    ply = PLY(filepath=ply_path, precision=precision)

    return cg.Pointcloud(ply.parser.vertices)


def pointcloud_from_grid_array(arr: npt.NDArray, return_values=False):
    pts, values = sort_pts_by_values(arr)

    pointcloud = cg.Pointcloud(pts)

    if return_values:
        return pointcloud, values

    return pointcloud


def pointcloud_to_grid_array(
    pointcloud: cg.Pointcloud,
    grid_size: tuple[int, int, int],
    dtype=np.float64,
    value=1,
):
    """Convert a pointcloud to a grid."""
    if not isinstance(grid_size, Sequence):
        grid_size = (grid_size, grid_size, grid_size)

    grid_array = np.zeros(grid_size)

    indices = np.array(pointcloud.points).round().astype(np.int_)
    indices = np.array(pointcloud.points).round().astype(np.int_)

    for i, j, k in indices:
        if np.max((i, j, k)) >= np.max(grid_size) or np.min((i, j, k)) < 0:
            raise IndexError(f"Index out of bounds: {i,j,k} in {grid_size}")
        grid_array[i, j, k] = value
    return grid_array.astype(dtype)


### INDEX map transformations using compas.geometry.Transform


def transfrom_index_map_to_plane(
    index_map: np.array = None,
    new_origin: tuple[float, float, float] = None,
    normal_vector: tuple[float, float, float] = None,
):
    """
    transform an index map from World_XY to Plane(new_origin, normal_vector)

    index map: np.array [a 4D array representing a point list]
    normal_vector: list of floats | compas.geometry.Vector
    new_origin: list of floats | compas.geometry.Point
    """

    if isinstance(index_map, np.ndarray):
        index_map.tolist()
    elif isinstance(index_map, list):
        pass
    else:
        raise TypeError
    index_map_pointcloud = Pointcloud(index_map)

    p = Plane(new_origin, normal_vector)
    f = Frame.from_plane(p)
    T = Transformation.from_frame(f)

    index_map_pointcloud.transform(T)
    index_map_oriented = np.array(index_map_pointcloud.points, dtype=np.int32)
    return index_map_oriented


def transfrom_index_map_to_frame(
    index_map: np.array = None,
    frame: Frame = None,
):
    """
    transform an index map from World_XY to Frame

    index map: np.array [a 4D array representing a point list]
    frame: compas.geometry.Frame
    """

    if isinstance(index_map, np.ndarray):
        index_map.tolist()
    elif isinstance(index_map, list):
        pass
    else:
        raise TypeError
    index_map_pointcloud = Pointcloud(index_map)

    f = frame
    T = Transformation.from_frame(f)

    index_map_pointcloud.transform(T)
    index_map_oriented = np.array(index_map_pointcloud.points, dtype=np.int32)
    return index_map_oriented


def translate_index_map(
    index_map: np.array = None,
    vector: tuple[float, float, float] | Point | Vector = None,
):
    """
    move index map to point
    translate an index map from [0,0,0] to [x,y,z]

    index map: np.array [a 4D array representing a point list]
    point: compas.geometry.Point or compas.geometry.Vector
    """

    if isinstance(index_map, np.ndarray):
        index_map.tolist()
    elif isinstance(index_map, list):
        pass
    else:
        raise TypeError
    index_map_pointcloud = Pointcloud(index_map)

    p = Plane(vector, [0, 0, 1])
    f = Frame.from_plane(p)

    T = Transformation.from_frame(f)

    index_map_pointcloud.transform(T)
    index_map_oriented = np.array(index_map_pointcloud.points, dtype=np.int32)
    return index_map_oriented


### generate array fills by trigonometric functions


def gyroid_array(grid_size, scale=1, thickness_out=1, thickness_in=0):
    x, y, z = np.indices(grid_size) * scale

    isovalue = np.cos(x) * np.sin(y) + np.cos(y) * np.sin(z) + np.cos(z) * np.sin(z)
    isovalue *= scale
    gyroid = np.where(isovalue <= thickness_out, 1, 0)
    return gyroid


def lidinoid_array(grid_size, thickness=1, scale=1):
    x, y, z = np.indices(grid_size) * scale
    isovalue = 0.5 * (
        sin(2 * x) * np.cos(y) * np.sin(z)
        + np.sin(2 * y) * np.cos(z) * np.sin(x)
        + np.sin(2 * z) * np.cos(x) * np.sin(y)
    )
    (
        -0.5
        * (
            np.cos(2 * x) * np.cos(2 * y)
            + np.cos(2 * y) * np.cos(2 * z)
            + np.cos(2 * z) * np.cos(2 * x)
        )
        + 0.15
    )

    isovalue = thickness - isovalue
    mask = isovalue >= 0
    gyroid = np.where(mask is True, 1, 0)
    return gyroid
