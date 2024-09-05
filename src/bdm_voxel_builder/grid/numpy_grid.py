import math
import os
from typing import Self

import compas.geometry as cg
import numpy as np
import numpy.typing as npt
import pyopenvdb as vdb
from compas.colors import Color

from bdm_voxel_builder.grid.base import Grid
from bdm_voxel_builder.helpers import (
    box_from_corner_frame,
    convert_grid_array_to_pts,
    extrude_array_along_vector,
    extrude_array_from_point,
    extrude_array_in_direction_expanding,
    extrude_array_linear,
    offset_array_radial,
    pointcloud_to_grid_array,
    xform_to_compas,
    xform_to_vdb,
)
from bdm_voxel_builder.helpers.geometry import (
    _get_xform_box2grid,
)


class NumpyGrid(Grid):
    def __init__(
        self,
        grid_size: int | tuple[int, int, int],
        name: str = None,
        color: Color = None,
        array: npt.NDArray = None,
        xform: cg.Transformation = None,
    ):
        if isinstance(grid_size, int):
            grid_size = [grid_size] * 3

        bbox = box_from_corner_frame(cg.Frame.worldXY(), *grid_size)

        super().__init__(
            bbox=bbox,
            name=name,
            color=color,
            xform=xform,
        )

        if array is None:
            self.array = np.zeros(self.grid_size)
        else:
            self.array = array

    @property
    def grid_size(self):
        xsize = int(self.bbox.xsize)
        ysize = int(self.bbox.ysize)
        zsize = int(self.bbox.zsize)

        return (xsize, ysize, zsize)

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

    def set_value_at_index(
        self,
        index: tuple[0, 0, 0],
        value: float,
    ) -> None:
        i, j, k = index
        self.array[i, j, k] = value

    def get_value_at_index(self, index: tuple[0, 0, 0]) -> float:
        i, j, k = index
        return self.array[i, j, k]

    def get_active_voxels(self):
        """returns indices of nonzero values
        list of coordinates
            shape = [3,n]"""
        return np.nonzero(self.array)

    def get_number_of_active_voxels(self):
        """returns indices of nonzero values
        list of coordinates
            shape = [3,n]"""
        return len(self.array[self.get_active_voxels()])

    def get_index_pts(self) -> npt.NDArray:
        pts = convert_grid_array_to_pts(self.array).tolist()
        pts.sort(key=lambda x: (x[0], x[1], x[2]))
        return np.array(pts)

    def get_world_pointcloud(self) -> cg.Pointcloud:
        return self.get_index_pointcloud().transformed(self.xform)

    def merged_with(self, other: Self):
        return self.array + other.array

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
