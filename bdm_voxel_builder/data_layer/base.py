import math
from typing import Sequence

import numpy as np
import numpy.typing as npt
import pyopenvdb as vdb
from compas.colors import Color
from compas.geometry import Box, Pointcloud, Transformation, transform_points_numpy

from bdm_voxel_builder import TEMP_DIR
from bdm_voxel_builder.helpers.numpy import convert_array_to_pts
from bdm_voxel_builder.helpers.savepaths import get_savepath
from bdm_voxel_builder.helpers.vdb import xform_to_vdb


class DataLayer:
    def __init__(
        self,
        name: str = None,
        bbox: int | tuple[int, int, int] | Box = None,
        voxel_size: int = 20,
        color: Color = None,
        array: npt.NDArray = None,
        xform: Transformation = None,
    ):
        self.name = name

        if not bbox and not voxel_size:
            raise ValueError("either bbox or voxel_size must be provided")

        if not bbox:
            self.local_bbox = Box(voxel_size)
        elif isinstance(bbox, float):
            self.local_bbox = Box(bbox)
        elif isinstance(bbox, Sequence):
            self.local_bbox = Box(*bbox)
        elif isinstance(bbox, Box):
            self.local_bbox = bbox
        else:
            raise ValueError("bbox not understood")

        self.color = color or Color.black()

        if array is None:
            self.array = np.zeros([int(d) for d in self.local_bbox.dimensions])
        else:
            self.array = array

        if not xform:
            self.xform = Transformation()
        else:
            self.xform = xform

    @property
    def voxel_size(self):
        return int(self.local_bbox.dimensions[0])

    def get_world_bbox(self) -> Box:
        return self.local_bbox.transformed(self.xform)

    def to_grid(self):
        grid = vdb.FloatGrid()
        grid.copyFromArray(self.array)

        grid.name = f"layer_{self.name}"

        grid.transform = xform_to_vdb(self.xform)

        return grid

    def save_vdb(self):
        path = get_savepath(TEMP_DIR, ".vdb", note=f"layer_{self.name}")

        grid = self.to_grid()

        # rotate the grid to make Y up for vdb_view and houdini
        grid.transform.rotate(-math.pi / 2, vdb.Axis.X)
        vdb.write(str(path), grids=[grid])

        return path

    def get_index_pts(self) -> list[list[float]]:
        return convert_array_to_pts(self.array, get_data=False)

    def get_index_pointcloud(self):
        return Pointcloud(self.get_index_pts())

    def get_world_pts(self) -> list[list[float]]:
        return transform_points_numpy(self.get_index_pts(), self.xform).tolist()

    def get_world_pointcloud(self) -> Pointcloud:
        return self.get_index_pointcloud().transformed(self.xform)
