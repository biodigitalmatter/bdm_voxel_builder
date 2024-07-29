import math
from typing import Sequence

import numpy as np
import numpy.typing as npt
import pyopenvdb as vdb
from compas.colors import Color
from compas.geometry import Box, Pointcloud

from bdm_voxel_builder import TEMP_DIR
from bdm_voxel_builder.helpers.numpy import convert_array_to_pts
from bdm_voxel_builder.helpers.savepaths import get_savepath


class DataLayer:
    def __init__(
        self,
        name: str = None,
        bbox: int | tuple[int, int, int] | Box = None,
        voxel_size: int = 20,
        color: Color = None,
        array: npt.NDArray = None,
    ):
        self.name = name

        if not bbox and not voxel_size:
            raise ValueError("either bbox or voxel_size must be provided")

        if not bbox:
            self.bbox = Box(voxel_size)
        elif isinstance(bbox, float):
            self.bbox = Box(bbox)
        elif isinstance(bbox, Sequence):
            self.bbox = Box(*bbox)
        elif isinstance(bbox, Box):
            self.bbox = bbox
        else:
            raise ValueError("bbox not understood")

        self.color = color or Color.black()

        if array is not None:
            self.array = array
        else:
            self.array = np.zeros([int(d) for d in self.bbox.dimensions])

    @property
    def voxel_size(self):
        return int(self.bbox.dimensions[0])

    def to_grid(self):
        grid = vdb.FloatGrid()
        grid.copyFromArray(self.array)

        grid.name = f"layer_{self.name}"

        return grid

    def save_vdb(self):
        path = get_savepath(TEMP_DIR, ".vdb", note=f"layer_{self.name}")

        grid = self.to_grid()

        # rotate the grid to make Y up for vdb_view and houdini
        grid.transform.rotate(-math.pi / 2, vdb.Axis.X)
        vdb.write(str(path), grids=[grid])

        return path

    def get_pts(self):
        return convert_array_to_pts(self.array, get_data=False)

    def get_pointcloud(self):
        return Pointcloud(self.get_pts())
