from dataclasses import dataclass
from enum import Enum, auto

import numpy.typing as npt
from compas.colors import Color
from compas.geometry import Pointcloud

from bdm_voxel_builder.helpers.numpy import convert_array_to_pts, create_zero_array


class AxisOrder(Enum):
    XYZ = auto()
    ZYX = auto()


@dataclass
class DataLayer:
    name: str = None
    voxel_size: int = 20
    color: Color = None
    axis_order: AxisOrder = AxisOrder.ZYX

    def __post_init__(self):
        if self.color is None:
            self.color = Color.black()

        self.array: npt.NDArray = create_zero_array(self.voxel_size)

    def get_pts(self):
        return convert_array_to_pts(self.array, get_data=False)

    def get_pointcloud(self):
        return Pointcloud(self.get_pts())
