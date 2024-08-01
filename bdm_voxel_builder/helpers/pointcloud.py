import os

import compas.geometry as cg
import numpy as np
import numpy.typing as npt
from compas.data import json_dump
from compas.files import PLY

from bdm_voxel_builder import TEMP_DIR
from bdm_voxel_builder.helpers.numpy import sort_pts_by_values
from bdm_voxel_builder.helpers.savepaths import get_savepath


def ply_to_numpy(ply_path: os.PathLike, precision: str = None):
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


def save_pointcloud(
    pointcloud: cg.Pointcloud, values: list[float] = None, note: str = None
):
    dict_ = {"pointcloud": pointcloud, "values": values}

    json_dump(dict_, get_savepath(TEMP_DIR, ".json", note=note))
