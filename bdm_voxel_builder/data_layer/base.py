from enum import Enum, auto
from compas.colors import Color
from dataclasses import dataclass

class AxisOrder(Enum):
    XYZ = auto()
    ZYX = auto()


@dataclass
class DataLayer:
    name: str = None
    voxel_size: int = 20
    color: Color = None
    axis_order:AxisOrder = AxisOrder.ZYX

    def __post_init__(self):
        if self.color is None:
            self.color = Color.black()

