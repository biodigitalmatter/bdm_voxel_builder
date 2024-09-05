import math
import os
from abc import ABCMeta, abstractmethod
from typing import Self

import compas.geometry as cg
import numpy.typing as npt
import pyopenvdb as vdb
import pypcd4
from compas.colors import Color

from bdm_voxel_builder import TEMP_DIR
from bdm_voxel_builder.helpers import (
    get_savepath,
)


class Grid(metaclass=ABCMeta):
    def __init__(
        self,
        bbox: cg.Box = None,
        name: str = None,
        color: Color = None,
        xform: cg.Transformation = None,
    ):
        self.name = name

        self.color = color or Color.black()

        self.bbox = bbox

        self.xform = xform or cg.Transformation()

    def get_world_bbox(self) -> cg.Box:
        return self.bbox.transformed(self.xform)

    @abstractmethod
    def to_vdb_grid(self):
        raise NotImplementedError

    def save_vdb(self, dir: os.PathLike = TEMP_DIR):
        path = get_savepath(dir, ".vdb", note=f"grid_{self.name}")

        grid = self.to_vdb_grid()

        # rotate the grid to make Y up for vdb_view and houdini
        grid.transform.rotate(-math.pi / 2, vdb.Axis.X)
        vdb.write(str(path), grids=[grid])

        return path

    @abstractmethod
    def set_value_at_index(
        self, index: tuple[int, int, int], value: float, **kwargs
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_value_at_index(self, index: tuple[int, int, int]) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_active_voxels(self) -> npt.NDArray:
        """returns indices of nonzero values
        shape = [3,n]"""
        raise NotImplementedError

    @abstractmethod
    def get_number_of_active_voxels(self) -> int:
        """returns indices of nonzero values"""
        raise NotImplementedError

    @abstractmethod
    def get_index_pts(self) -> npt.NDArray:
        raise NotImplementedError

    def get_index_pointcloud(self):
        return cg.Pointcloud(self.get_index_pts())

    def get_world_pts(self) -> npt.NDArray:
        return cg.transform_points_numpy(self.get_index_pts(), self.xform)

    def get_world_pointcloud(self) -> cg.Pointcloud:
        return self.get_index_pointcloud().transformed(self.xform)

    @abstractmethod
    def merged_with(self, grid: Self) -> Self:
        raise NotImplementedError

    def offset_radial(self, radius: int):
        raise NotImplementedError

    def offset_along_axes(self, direction, steps):
        raise NotImplementedError

    def extrude_along_vector(self, vector: tuple[float, float, float], length: int):
        raise NotImplementedError

    def extrude_unit(self, vector: tuple[int, int, int], steps: int):
        raise NotImplementedError

    def extrude_from_point(self, point: tuple[int, int, int], steps: int):
        raise NotImplementedError

    def extrude_tapered(self, direction: tuple[int, int, int], steps: int):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_pointcloud(
        cls,
        pointcloud: cg.Pointcloud | os.PathLike,
        grid_size: list[int, int, int] | int = None,
        voxel_edge_length: int = None,
        xform=None,
        name: str = None,
    ):
        raise NotImplementedError

    @classmethod
    def from_ply(
        cls,
        ply_path: os.PathLike,
        grid_size: list[int, int, int] | int = None,
        voxel_edge_length: int = None,
        name: str = None,
    ):
        pointcloud = cg.Pointcloud.from_ply(ply_path)
        return cls.from_pointcloud(
            pointcloud,
            grid_size=grid_size,
            voxel_edge_length=voxel_edge_length,
            name=name,
        )

    @classmethod
    def from_pcd(
        cls,
        pcd_path: os.PathLike,
        grid_size: list[int, int, int] | int = None,
        voxel_edge_length: int = None,
        name: str = None,
    ):
        pc = pypcd4.PointCloud.from_path(pcd_path)
        xyz = pc.metadata.viewpoint[:3]
        Tl = cg.Translation.from_vector(xyz)

        q = pc.metadata.viewpoint[3:]
        R = cg.Rotation.from_quaternion(q)

        pts = [cg.Point(*pt[:3]) for pt in pc.numpy()]
        pointcloud = cg.Pointcloud(pts)

        pointcloud.scale(1000)

        return cls.from_pointcloud(
            pointcloud,
            grid_size=grid_size,
            voxel_edge_length=voxel_edge_length,
            name=name,
            xform=Tl * R,
        )
