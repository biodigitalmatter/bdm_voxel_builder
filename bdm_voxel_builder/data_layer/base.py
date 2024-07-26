from typing import Tuple

import numpy as np
import numpy.typing as npt
from compas.geometry import Box
from compas.colors import Color
from compas.geometry import Pointcloud

from bdm_voxel_builder.helpers.numpy import convert_array_to_pts


class DataLayer:
    def __init__(
        self,
        name: str = None,
        bbox: int | Tuple[int, int, int] | Box = None,
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
        elif isinstance(bbox, Tuple):
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

    def get_pts(self):
        return convert_array_to_pts(self.array, get_data=False)

    def get_pointcloud(self):
        return Pointcloud(self.get_pts())
