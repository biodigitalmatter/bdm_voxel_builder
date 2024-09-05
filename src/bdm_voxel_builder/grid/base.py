import math
import os
from collections.abc import Sequence
from typing import Self

import compas.geometry as cg
import numpy as np
import numpy.typing as npt
import pyopenvdb as vdb
import pypcd4
from compas.colors import Color

from bdm_voxel_builder import TEMP_DIR
from bdm_voxel_builder.helpers import (
    convert_grid_array_to_pts,
    get_savepath,
    pointcloud_to_grid_array,
    xform_to_compas,
    xform_to_vdb,
)
from bdm_voxel_builder.helpers.array import (
    extrude_array_along_vector,
    extrude_array_from_point,
    extrude_array_in_direction_expanding,
    extrude_array_linear,
    offset_array_radial,
)
from bdm_voxel_builder.helpers.geometry import (
    _get_xform_box2grid,
)


class Grid:
    def __init__(
        self,
        grid_size: int | tuple[int, int, int],
        name: str = None,
        color: Color = None,
        array: npt.NDArray = None,
        xform: cg.Transformation = None,
    ):
        self.name = name

        if isinstance(grid_size, int):
            grid_size = [grid_size] * 3

        self.grid_size = grid_size

        self.color = color or Color.black()

        if array is None:
            self.array = np.zeros(self.grid_size)
        else:
            self.array = array

        if not xform:
            self.xform = cg.Transformation()
        else:
            self.xform = xform

    @property
    def grid_size(self):
        value = self._grid_size

        if not isinstance(value, Sequence):
            return (value, value, value)

        return value

    @grid_size.setter
    def grid_size(self, value):
        if isinstance(value, int | float):
            value = np.array([value, value, value], dtype=np.int32)
        elif isinstance(value, list | tuple):
            value = np.array(value, dtype=np.int32)
        if np.min(value) < 1:
            raise ValueError("grid_size must be nonzero and positive")
        self._grid_size = value.tolist()

    def get_local_bbox(self) -> cg.Box:
        """Returns a bounding box containing the grid, 0, 0, 0 to ijk"""
        xsize, ysize, zsize = self.grid_size
        return cg.Box.from_diagonal(cg.Line((0, 0, 0), (xsize, ysize, zsize)))

    def get_world_bbox(self) -> cg.Box:
        return self.get_local_bbox().transformed(self.xform)

    def to_vdb_grid(self):
        grid = vdb.FloatGrid()
        # f_array = np.float_(self.array)
        grid.copyFromArray(self.array)

        name = self.name or "None"
        try:
            grid.name = name
        except TypeError:
            grid.__setitem__("name", name)

        grid.transform = xform_to_vdb(self.xform)

        return grid

    def save_vdb(self, dir: os.PathLike = TEMP_DIR):
        path = get_savepath(dir, ".vdb", note=f"grid_{self.name}")

        grid = self.to_vdb_grid()

        # rotate the grid to make Y up for vdb_view and houdini
        grid.transform.rotate(-math.pi / 2, vdb.Axis.X)
        vdb.write(str(path), grids=[grid])

        return path

    def set_value_at_index(
        self, index=(0, 0, 0), value=1, wrapping: bool = True, clipping: bool = False
    ):
        if wrapping:
            index = np.mod(index, self.grid_size)
        elif clipping:
            index = np.clip(index, [0, 0, 0], self.array.shape - np.asarray([1, 1, 1]))
        i, j, k = index
        self.array[i][j][k] = value
        return self.array

    def get_value_at_index(self, index=(0, 0, 0)):
        i, j, k = index
        return self.array[i][j][k]

    def get_active_voxels(self):
        """returns indicies of nonzero values
        list of coordinates
            shape = [3,n]"""
        return np.nonzero(self.array)

    def get_number_of_active_voxels(self):
        """returns indicies of nonzero values
        list of coordinates
            shape = [3,n]"""
        return len(self.array[self.get_active_voxels()])

    def get_index_pts(self) -> list[list[float]]:
        pts = convert_grid_array_to_pts(self.array).tolist()
        pts.sort(key=lambda x: (x[0], x[1], x[2]))
        return pts

    def get_index_pointcloud(self):
        return cg.Pointcloud(self.get_index_pts())

    def get_world_pts(self) -> list[list[float]]:
        return cg.transform_points_numpy(self.get_index_pts(), self.xform).tolist()

    def get_world_pointcloud(self) -> cg.Pointcloud:
        return self.get_index_pointcloud().transformed(self.xform)

    def get_merged_array_with(self, grid: Self):
        a1 = self.array
        a2 = grid.array
        return a1 + a2

    def pad_array(self, pad_width: int, values=0):
        """pad self.array uniform
        updates self.grid_size = array.shape
        return self.grid_size"""

        array = np.pad(
            self.array,
            [[pad_width, pad_width], [pad_width, pad_width], [pad_width, pad_width]],
            "constant",
            constant_values=values,
        )
        print(array.shape)
        self.array = array
        self.grid_size = list(array.shape)
        return self.grid_size

    def shrink_array(self, width: int):
        pad = width
        self.array = self.array[pad:-pad, pad:-pad, pad:-pad]

    def offset_radial(self, radius: int):
        self.array = offset_array_radial(self.array, radius, True)

    def offset_along_axes(self, direction, steps):
        self.array = extrude_array_linear(self.array, direction, steps, True)

    def extrude_along_vector(self, vector: tuple[float, float, float], length: int):
        self.array = extrude_array_along_vector(self.array, vector, length, True)

    def extrude_unit(self, vector: tuple[int, int, int], steps: int):
        self.array = extrude_array_linear(self.array, vector, steps, True)

    def extrude_from_point(self, point: tuple[int, int, int], steps: int):
        self.array = extrude_array_from_point(self.array, point, steps, True)

    def extrude_tapered(self, direction: tuple[int, int, int], steps: int):
        self.array = extrude_array_in_direction_expanding(
            self.array, direction, steps, True
        )

    @classmethod
    def from_npy(cls, path: os.PathLike, name: str = None):
        arr = np.load(path)
        return cls(name=name, array=arr, grid_size=arr.shape)

    @classmethod
    def from_vdb(cls, grid: os.PathLike | vdb.GridBase, name: str = None):
        if isinstance(grid, os.PathLike):
            grids = vdb.readAllGridMetadata(str(grid))

            if not name and len(grids) > 1:
                print(
                    "File contains more than one grid, ",
                    f"only processing first named {grids[0].name}",
                )

            name = name or grids[0].name
            grid = vdb.read(str(grid), name)

        bbox_min = grid.metadata["file_bbox_min"]
        bbox_max = grid.metadata["file_bbox_max"]

        shape = np.array(bbox_max) - np.array(bbox_min)
        arr = np.zeros(shape)

        # rotate the grid to make Z up
        grid.transform.rotate(math.pi / 2, vdb.Axis.X)

        grid.copyToArray(arr, ijk=bbox_min)

        return cls(
            grid_size=arr.shape,
            name=name or grid.name,
            array=arr,
            xform=xform_to_compas(grid.transform),
        )

    @classmethod
    def from_pointcloud(
        cls,
        pointcloud: cg.Pointcloud | os.PathLike,
        grid_size: list[int, int, int] | int = None,
        voxel_edge_length: int = None,
        xform=None,
        name: str = None,
    ):
        if grid_size is None and voxel_edge_length is None:
            raise ValueError("Either grid_size or unit_in_mm must be provided")

        if isinstance(pointcloud, os.PathLike):
            pointcloud = cg.Pointcloud.from_json(pointcloud)

        if not grid_size:
            grid_size = int(max(pointcloud.aabb.dimensions)) // voxel_edge_length + 1
        # print(f"pointcloud.aabb.dimensions{pointcloud.aabb.dimensions}")
        if isinstance(grid_size, int):
            grid_size = [grid_size] * 3
        # print(f"grid_size in from pointcloud{grid_size}")

        # TODO: Replace with multiplied version
        Sc, R, Tl = _get_xform_box2grid(pointcloud.aabb, grid_size=grid_size)

        pointcloud_transformed = pointcloud.copy()

        pointcloud_transformed.transform(Tl)
        pointcloud_transformed.transform(R)
        pointcloud_transformed.transform(Sc)

        array = pointcloud_to_grid_array(pointcloud_transformed, grid_size)

        new_xform = xform * Tl * R * Sc if xform else Tl * R * Sc

        return cls(grid_size=grid_size, name=name, xform=new_xform, array=array)

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
