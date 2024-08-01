import math
import os
from collections.abc import Sequence
from typing import Self

import compas.geometry as cg
import numpy as np
import numpy.typing as npt
import pyopenvdb as vdb
from compas.colors import Color
from compas.geometry import Box, Pointcloud, Transformation, transform_points_numpy

from bdm_voxel_builder import TEMP_DIR
from bdm_voxel_builder.helpers.numpy import convert_array_to_pts
from bdm_voxel_builder.helpers.savepaths import get_savepath
from bdm_voxel_builder.helpers.vdb import xform_to_compas, xform_to_vdb


class Grid:
    def __init__(
        self,
        grid_size: int | tuple[int, int, int],
        name: str = None,
        color: Color = None,
        array: npt.NDArray = None,
        xform: Transformation = None,
    ):
        self.name = name

        self.grid_size = grid_size

        self.color = color or Color.black()

        if array is None:
            self.array = np.zeros(self.grid_size)
        else:
            self.array = array

        if not xform:
            self.xform = Transformation()
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
        value = np.array(value, dtype=np.intp)
        if value.min() < 1:
            raise ValueError("grid_size must be nonzero and positive")
        if np.unique(value).size != 1:
            raise NotImplementedError("Non square grid not supported yet")

        self._grid_size = value.tolist()

    def get_local_bbox(self) -> Box:
        """Returns a bounding box containing the grid, 0, 0, 0 to ijk"""
        return Box.from_diagonal(((0, 0, 0), self.grid_size))

    def get_world_bbox(self) -> Box:
        return self.get_local_bbox().transformed(self.xform)

    def to_vdb_grid(self):
        grid = vdb.FloatGrid()
        grid.copyFromArray(self.array)

        name = self.name or "None"
        try:
            grid.name = name
        except TypeError:
            grid.__setitem__("name", name)

        grid.transform = xform_to_vdb(self.xform)

        return grid

    def save_vdb(self):
        path = get_savepath(TEMP_DIR, ".vdb", note=f"grid_{self.name}")

        grid = self.to_vdb_grid()

        # rotate the grid to make Y up for vdb_view and houdini
        grid.transform.rotate(-math.pi / 2, vdb.Axis.X)
        vdb.write(str(path), grids=[grid])

        return path

    def set_value_at_index(self, index=(0, 0, 0), value=1, wrapping: bool = True):
        if wrapping:
            index = np.mod(index, self.grid_size)
        i, j, k = index
        self.array[i][j][k] = value
        return self.array

    def get_value_at_index(self, index=(0, 0, 0)):
        i, j, k = index
        return self.array[i][j][k]

    def get_active_voxels(self, array):
        """returns indicies of nonzero values
        list of coordinates
            shape = [3,n]"""
        return np.nonzero(self.array)

    def get_index_pts(self) -> list[list[float]]:
        return convert_array_to_pts(self.array, get_data=False)

    def get_index_pointcloud(self):
        return Pointcloud(self.get_index_pts())

    def get_world_pts(self) -> list[list[float]]:
        return transform_points_numpy(self.get_index_pts(), self.xform).tolist()

    def get_world_pointcloud(self) -> Pointcloud:
        return self.get_index_pointcloud().transformed(self.xform)

    def get_merged_array_with(self, grid: Self):
        a1 = self.array
        a2 = grid.array
        return a1 + a2

    @classmethod
    def from_npy(cls, path: os.PathLike, name: str = None):
        arr = np.load(path)
        return cls(name=name, array=arr)

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
    def from_pointcloud(cls, pointcloud: cg.Pointcloud, name: str = None):
        vdb.grid.createLevelSetFromPolygons(pointcloud.points)
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
