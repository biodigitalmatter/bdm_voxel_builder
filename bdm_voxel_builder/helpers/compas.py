
import numpy.typing as npt
from compas.data import json_dump
from compas.geometry import Pointcloud

from bdm_voxel_builder import TEMP_DIR
from bdm_voxel_builder.helpers.numpy import sort_pts_by_values
from bdm_voxel_builder.helpers.savepaths import get_savepath


def pointcloud_from_ndarray(arr: npt.NDArray, return_values=False):
    pts, values = sort_pts_by_values(arr)

    pointcloud = Pointcloud(pts)

    if return_values:
        return pointcloud, values

    return pointcloud


def save_pointcloud(
    pointcloud: Pointcloud, values: list[float] = None, note: str = None
):
    dict_ = {"pointcloud": pointcloud, "values": values}

    json_dump(dict_, get_savepath(TEMP_DIR, ".json", note=note))
